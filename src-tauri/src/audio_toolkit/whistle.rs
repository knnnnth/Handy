//! Cactus Whistle — the `EngineType::Whistle` speech-to-text backend.
//!
//! Whistle is a 16.9 MB `.cact` model (Apache-2.0, `Cactus-Compute/whistle`)
//! served by Cactus's prebuilt `libneedle.a`. Handy links that archive
//! statically (see `link_needle()` in `src-tauri/build.rs`) and calls the small
//! C API declared in `vendor/needle/needle.h` through the FFI module below.
//!
//! ## Target support
//!
//! Upstream ships no `macos-x86_64` build, so only five triples have an archive:
//! `aarch64-apple-darwin`, `x86_64`/`aarch64-pc-windows-msvc` and
//! `x86_64`/`aarch64-unknown-linux-gnu`. `link_needle()` in `build.rs` maps a
//! target to its archive folder and emits `cfg(whistle_linked)` for exactly
//! those five — that cfg is the single source of truth for the module split
//! below, because a `cfg` attribute cannot expand a macro and duplicating the
//! predicate here would let the linked archives and the compiled code drift
//! apart silently. Everything else (Intel Macs, musl, any future triple)
//! compiles the stub, whose `load`/`transcribe` both fail with an "unsupported
//! platform" error. The same cfg backs [`is_supported_target`], which hides the
//! model from the catalog on unsupported builds — so a user can never select a
//! model that could only fail to load.
//!
//! ## Threading
//!
//! The header documents the engine as *process-global and non-thread-safe*: one
//! model per kind, no locking of its own. `NEEDLE_LOCK` serialises every FFI
//! call (load / transcribe / reset) so a model swap racing a transcription is
//! safe. That is the only synchronisation needed because `TranscriptionManager`
//! already drops its own engine mutex around both operations, so two Whistle
//! calls never overlap from Handy's side either.
//!
//! ## Audio contract
//!
//! [`WhistleEngine::transcribe`] takes 16 kHz mono `f32` PCM in `[-1, 1]` —
//! exactly what Handy's recording pipeline produces — and caps a single call at
//! 30 s ([`MAX_CHUNK_SAMPLES`]), the engine's hard limit. Longer input is split
//! into consecutive chunks; [`join_chunks`] folds the per-chunk results back
//! into one.

// The pure half of the engine (chunking, argument coercion, output parsing) is
// only reachable from the FFI module and from these tests, so a stub-only build
// compiles it out entirely rather than carrying it as dead code.
#[cfg(any(whistle_linked, test))]
mod pure {
    use super::WhistleResult;

    /// Longest clip handed to a single `needle_transcribe` call: 30 s at 16 kHz,
    /// the header's documented hard cap.
    pub const MAX_CHUNK_SAMPLES: usize = 480_000;

    /// The only language codes `needle_transcribe` accepts. `language` is passed
    /// as NULL (auto-detect) for anything else — a NULL is a documented input,
    /// an unknown code is not, so we translate one into the other rather than
    /// letting the engine reject the call. Defence in depth on top of
    /// `effective_language`'s coercion to the model's advertised set.
    pub const SUPPORTED_LANGUAGES: &[&str] = &["en", "de", "fr", "es", "it", "nl", "pl"];

    /// The exact shape `needle_transcribe` documents:
    /// `{"text":"...","language":"en","ttft_ms":0.0,"decode_tps":0.0}`.
    ///
    /// The numeric fields are `#[serde(default)]`-ed only as belt-and-braces —
    /// the header says all four keys are always present, so a missing one is a
    /// runtime surprise that should degrade to "timing unknown" rather than
    /// fail the whole transcription.
    #[derive(Debug, serde::Deserialize)]
    struct TranscribeOutput {
        text: String,
        language: String,
        #[serde(default)]
        ttft_ms: f64,
        #[serde(default)]
        decode_tps: f64,
    }

    pub fn parse_output(json: &str) -> anyhow::Result<WhistleResult> {
        let parsed: TranscribeOutput = serde_json::from_str(json)
            .map_err(|e| anyhow::anyhow!("invalid whistle engine output: {e}"))?;
        Ok(WhistleResult {
            text: parsed.text,
            language: parsed.language,
            ttft_ms: parsed.ttft_ms,
            decode_tps: parsed.decode_tps,
        })
    }

    /// An `opts.language` reduced to a code the engine actually accepts.
    pub fn coerce_language(language: Option<&str>) -> Option<&str> {
        language.filter(|code| SUPPORTED_LANGUAGES.contains(code))
    }

    /// `Some(s)` for a string with content, `None` for `""`. The FFI layer uses
    /// it to turn an empty keywords list into a NULL rather than an empty C
    /// string.
    pub fn non_empty(s: &str) -> Option<&str> {
        if s.is_empty() {
            None
        } else {
            Some(s)
        }
    }

    /// Folds the per-chunk results of one long utterance back into a single
    /// pass's worth of output: non-empty transcripts joined by a single space,
    /// and the language/timings of the last chunk (the only ones we have).
    pub fn join_chunks(chunks: Vec<WhistleResult>) -> WhistleResult {
        let mut texts: Vec<String> = Vec::new();
        let mut last: Option<WhistleResult> = None;

        for chunk in chunks {
            if !chunk.text.trim().is_empty() {
                texts.push(chunk.text.clone());
            }
            last = Some(chunk);
        }

        match last {
            Some(mut r) => {
                r.text = texts.join(" ");
                r
            }
            // No chunks at all (empty input): the same empty result the engine
            // documents for silence, so callers need no separate case.
            None => WhistleResult::default(),
        }
    }
}

#[cfg(whistle_linked)]
mod native {
    use super::pure::{coerce_language, join_chunks, non_empty, parse_output, MAX_CHUNK_SAMPLES};
    use super::{WhistleOptions, WhistleResult};
    use anyhow::{anyhow, Context, Result};
    use std::ffi::{c_char, c_int, c_uchar, c_ulonglong, CStr, CString};
    use std::path::Path;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Mutex;

    /// `NEEDLE_SPEECH` from `needle.h` — the bit `needle_models` reports for a
    /// loaded speech model.
    const NEEDLE_SPEECH: c_int = 2;

    /// Capacity of the JSON output buffer handed to `needle_transcribe`. The C
    /// API has no size query, so this is a fixed allocation; 1 MiB is far more
    /// than 30 s of transcript (worst case a few tens of kilobytes).
    const OUT_CAPACITY: usize = 1024 * 1024;

    /// Serialises every call into the process-global, non-thread-safe runtime.
    /// Held across `needle_load` / `needle_transcribe` / `needle_reset`.
    static NEEDLE_LOCK: Mutex<()> = Mutex::new(());

    /// Monotonic id handed to each successfully loaded engine.
    static NEXT_GENERATION: AtomicU64 = AtomicU64::new(1);

    /// Generation of the engine currently occupying the process-global slot, or
    /// 0 when nothing is loaded. Guards [`WhistleEngine`]'s `Drop` against
    /// unloading a *newer* model that superseded it.
    static LOADED_GENERATION: AtomicU64 = AtomicU64::new(0);

    // The subset of `vendor/needle/needle.h` that Handy's STT path uses. The
    // header's text/tool-calling entry points (`needle_init`,
    // `needle_complete`, `needle_set_audio`, `needle_embed`) are deliberately
    // absent: a static archive only pulls in the objects a symbol reference
    // drags in, so declaring only what we call keeps the rest of the runtime
    // out of the link.
    extern "C" {
        /// Loads a `.cact` blob into the process-global slot for whichever kind
        /// (text or speech) it holds. Negative on failure.
        fn needle_load(cact: *const c_uchar, n: c_ulonglong) -> c_int;
        /// Bitmask of the kinds loaded in this process.
        fn needle_models() -> c_int;
        /// Last error message, owned by the runtime and valid until the next
        /// API call.
        fn needle_last_error() -> *const c_char;
        /// Transcribes 16 kHz mono float PCM (`samples`, at most 30 s) into a
        /// caller-provided JSON buffer. Negative on failure.
        fn needle_transcribe(
            pcm: *const f32,
            samples: c_int,
            language: *const c_char,
            keywords: *const c_char,
            word_timestamps: c_int,
            out: *mut c_char,
            out_capacity: c_int,
        ) -> c_int;
        /// Drops the loaded model.
        fn needle_reset();
    }

    /// The last runtime error, or a generic marker when the runtime has no
    /// message to give.
    unsafe fn last_error() -> String {
        let ptr = needle_last_error();
        if ptr.is_null() {
            return "no error message available".to_string();
        }
        unsafe { CStr::from_ptr(ptr) }
            .to_string_lossy()
            .into_owned()
    }

    /// A `needle_transcribe` output buffer. Zeroed rather than merely allocated
    /// so the C side can treat it as a plain C string even if it returns without
    /// writing.
    fn out_buffer() -> Vec<u8> {
        vec![0u8; OUT_CAPACITY]
    }

    /// The engine for a loaded `.cact` speech model.
    ///
    /// The model itself lives in the C runtime's process-global slot, not here;
    /// this type is a handle that records "something is loaded" and unloads on
    /// drop.
    pub struct WhistleEngine {
        /// This handle's id; see [`LOADED_GENERATION`].
        generation: u64,
    }

    impl WhistleEngine {
        /// Loads `path` (a `.cact` file) into the engine, replacing whatever was
        /// loaded before.
        pub fn load(path: &Path) -> Result<Self> {
            let bytes = std::fs::read(path)
                .with_context(|| format!("failed to read Whistle model {}", path.display()))?;

            let generation = NEXT_GENERATION.fetch_add(1, Ordering::Relaxed);
            let _guard = NEEDLE_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            let rc = unsafe { needle_load(bytes.as_ptr(), bytes.len() as c_ulonglong) };
            if rc < 0 {
                return Err(anyhow!(
                    "needle_load rejected {} ({} bytes): {}",
                    path.display(),
                    bytes.len(),
                    unsafe { last_error() }
                ));
            }

            // A `.cact` may hold a text model, a speech model, or both. Whistle
            // is the speech one; without this bit `needle_transcribe` would fail
            // later with a far less obvious message.
            let kinds = unsafe { needle_models() };
            if kinds & NEEDLE_SPEECH == 0 {
                unsafe { needle_reset() };
                return Err(anyhow!(
                    "{} is not a Cactus speech model (loaded kinds: {kinds:#x})",
                    path.display()
                ));
            }

            LOADED_GENERATION.store(generation, Ordering::Release);
            log::info!(
                "Loaded Cactus Whistle model '{}' ({} bytes, generation {generation})",
                path.display(),
                bytes.len()
            );
            Ok(Self { generation })
        }

        /// Transcribes 16 kHz mono `f32` PCM in `[-1, 1]`.
        ///
        /// Input longer than [`MAX_CHUNK_SAMPLES`] is split into consecutive
        /// chunks; [`join_chunks`] folds the results back together.
        pub fn transcribe(&self, audio: &[f32], opts: &WhistleOptions) -> Result<WhistleResult> {
            let language = coerce_language(opts.language.as_deref())
                .map(|code| CString::new(code).expect("language codes never contain a NUL"));
            let keywords = opts
                .keywords
                .as_deref()
                .and_then(non_empty)
                .map(|k| CString::new(k).expect("keywords contain an interior NUL"));

            let results = audio
                .chunks(MAX_CHUNK_SAMPLES)
                .map(|chunk| self.transcribe_chunk(chunk, language.as_deref(), keywords.as_deref()))
                .collect::<Result<Vec<_>>>()?;

            Ok(join_chunks(results))
        }

        /// One `needle_transcribe` call on a chunk already known to fit the
        /// engine's 30 s limit.
        fn transcribe_chunk(
            &self,
            pcm: &[f32],
            language: Option<&CStr>,
            keywords: Option<&CStr>,
        ) -> Result<WhistleResult> {
            let mut out = out_buffer();
            let _guard = NEEDLE_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            let rc = unsafe {
                needle_transcribe(
                    pcm.as_ptr(),
                    pcm.len() as c_int,
                    language.map_or(std::ptr::null(), |c| c.as_ptr()),
                    keywords.map_or(std::ptr::null(), |c| c.as_ptr()),
                    // word_timestamps: Handy's dictation path only needs the
                    // text, and the engine skips building the array without it.
                    0,
                    out.as_mut_ptr() as *mut c_char,
                    OUT_CAPACITY as c_int,
                )
            };
            if rc < 0 {
                return Err(anyhow!(
                    "needle_transcribe failed on {} samples: {}",
                    pcm.len(),
                    unsafe { last_error() }
                ));
            }

            // The buffer is pre-zeroed, so this is a valid C string even if the
            // engine wrote nothing at all.
            let json = unsafe { CStr::from_ptr(out.as_ptr() as *const c_char) };
            parse_output(json.to_str().unwrap_or_default())
        }
    }

    impl Drop for WhistleEngine {
        fn drop(&mut self) {
            let _guard = NEEDLE_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Only unload if this handle still owns the global slot: a newer
            // model may have been loaded since, and resetting would destroy it.
            if LOADED_GENERATION
                .compare_exchange(self.generation, 0, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                unsafe { needle_reset() };
            }
        }
    }
}

/// Fallback for targets with no vendored `libneedle.a` (see the module docs).
#[cfg(not(whistle_linked))]
mod native {
    use super::{WhistleOptions, WhistleResult};
    use anyhow::{anyhow, Result};
    use std::path::Path;

    const UNSUPPORTED: &str =
        "Whistle is not supported on this platform (no Cactus engine for this target)";

    pub struct WhistleEngine;

    impl WhistleEngine {
        pub fn load(_path: &Path) -> Result<Self> {
            Err(anyhow!(UNSUPPORTED))
        }

        pub fn transcribe(&self, _audio: &[f32], _opts: &WhistleOptions) -> Result<WhistleResult> {
            Err(anyhow!(UNSUPPORTED))
        }
    }
}

pub use native::WhistleEngine;

/// One transcription pass. Mirrors the JSON `needle_transcribe` writes into its
/// output buffer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct WhistleResult {
    pub text: String,
    /// Language the engine reported for the audio it decoded. Empty for the
    /// documented silence/steady-noise case, where there is nothing to detect.
    pub language: String,
    /// Milliseconds to the first decoded token.
    pub ttft_ms: f64,
    /// Decoder tokens per second after the first token.
    pub decode_tps: f64,
}

/// Per-call knobs for [`WhistleEngine::transcribe`]. `Default` is what every
/// caller wants: detect the language, apply no keyword biasing.
#[derive(Debug, Clone, Default)]
pub struct WhistleOptions {
    /// A language code the engine accepts, or `None`/anything else to
    /// auto-detect. Whisper-family models take a full BCP-47 tag here; Whistle
    /// takes one of seven ISO 639-1 codes, and anything else is downgraded to
    /// auto-detect rather than passed through.
    pub language: Option<String>,
    /// Newline-separated words and phrases to bias the decoder toward.
    pub keywords: Option<String>,
}

/// Whether this build can run Whistle at all — i.e. whether `build.rs` found an
/// archive for the current target, linked it, and set `cfg(whistle_linked)`, so
/// the real [`WhistleEngine`] was compiled instead of the stub. The
/// catalog-seeding path uses it to hide the model on targets that could only
/// ever fail to load it.
pub fn is_supported_target() -> bool {
    cfg!(whistle_linked)
}

#[cfg(test)]
mod tests {
    use super::pure::{
        coerce_language, join_chunks, non_empty, parse_output, MAX_CHUNK_SAMPLES,
        SUPPORTED_LANGUAGES,
    };
    use super::WhistleResult;
    use std::path::Path;

    fn chunk(text: &str, language: &str) -> WhistleResult {
        WhistleResult {
            text: text.to_string(),
            language: language.to_string(),
            ttft_ms: 10.0,
            decode_tps: 30.0,
        }
    }

    #[test]
    fn parses_documented_transcribe_output() {
        // Verbatim shape from needle.h.
        let json = r#"{"text":"And so my fellow Americans","language":"en","ttft_ms":131.5,"decode_tps":42.25}"#;
        let parsed = parse_output(json).unwrap();
        assert_eq!(
            parsed,
            WhistleResult {
                text: "And so my fellow Americans".to_string(),
                language: "en".to_string(),
                ttft_ms: 131.5,
                decode_tps: 42.25,
            }
        );
    }

    #[test]
    fn parses_documented_silence_output() {
        let json = r#"{"text":"","language":"","ttft_ms":0.0,"decode_tps":0.0}"#;
        assert_eq!(parse_output(json).unwrap(), WhistleResult::default());
    }

    #[test]
    fn missing_timings_default_instead_of_failing() {
        let parsed = parse_output(r#"{"text":"hi","language":"en"}"#).unwrap();
        assert_eq!(parsed.text, "hi");
        assert_eq!((parsed.ttft_ms, parsed.decode_tps), (0.0, 0.0));
    }

    #[test]
    fn rejects_malformed_output() {
        assert!(parse_output("not json").is_err());
        // A missing required field must not silently transcribe to nothing.
        assert!(parse_output(r#"{"language":"en"}"#).is_err());
    }

    #[test]
    fn chunk_boundaries_land_on_the_thirty_second_limit() {
        // 480,000 samples == 30 s @ 16 kHz, the header's hard cap.
        assert_eq!(MAX_CHUNK_SAMPLES, 480_000);

        let chunk_count = |samples: usize| samples.div_ceil(MAX_CHUNK_SAMPLES);
        assert_eq!(chunk_count(0), 0);
        assert_eq!(chunk_count(1), 1);
        assert_eq!(chunk_count(MAX_CHUNK_SAMPLES - 1), 1);
        assert_eq!(chunk_count(MAX_CHUNK_SAMPLES), 1);
        assert_eq!(chunk_count(MAX_CHUNK_SAMPLES + 1), 2);
        // A 5-minute dictation clip is 10 engine passes.
        assert_eq!(chunk_count(5 * 60 * 16_000), 10);
    }

    #[test]
    fn joined_chunks_concatenate_with_single_spaces() {
        let joined = join_chunks(vec![
            chunk("first thirty seconds", "en"),
            chunk("next thirty seconds", "en"),
        ]);
        assert_eq!(joined.text, "first thirty seconds next thirty seconds");
        // Language/timings come from the last chunk — the only ones available.
        assert_eq!(joined.language, "en");
    }

    #[test]
    fn joined_chunks_skip_empty_segments() {
        // Silence mid-dictation (VAD gaps, a cough) must not leave a double
        // space or a leading separator.
        let joined = join_chunks(vec![
            chunk("hello", "en"),
            chunk("  ", "en"),
            chunk("world", "en"),
        ]);
        assert_eq!(joined.text, "hello world");
    }

    #[test]
    fn joining_nothing_yields_the_empty_result() {
        assert_eq!(join_chunks(Vec::new()), WhistleResult::default());
    }

    #[test]
    fn language_allowlist_passes_supported_codes() {
        for code in SUPPORTED_LANGUAGES {
            assert_eq!(coerce_language(Some(code)), Some(*code));
        }
    }

    #[test]
    fn language_allowlist_downgrades_anything_else_to_detect() {
        // NULL is a documented input (auto-detect); an unknown code is not, so
        // it must be coerced away rather than handed to the engine.
        for code in ["auto", "", "zh", "ja", "en-US", "EN", "klingon"] {
            assert_eq!(
                coerce_language(Some(code)),
                None,
                "{code} must coerce to None"
            );
        }
        assert_eq!(coerce_language(None), None);
    }

    #[test]
    fn empty_keywords_become_a_null_pointer() {
        assert_eq!(non_empty(""), None);
        assert_eq!(non_empty("Handy"), Some("Handy"));
        assert_eq!(non_empty("Handy\nCactus"), Some("Handy\nCactus"));
    }

    #[test]
    fn support_flag_matches_the_link_state_build_script_reported() {
        // The catalog only lists Whistle when this is true; a target that
        // reported support but compiled the stub would surface a model that can
        // only fail to load.
        let linked = matches!(
            (std::env::consts::OS, std::env::consts::ARCH),
            ("macos", "aarch64")
                | ("windows", "x86_64")
                | ("windows", "aarch64")
                | ("linux", "x86_64")
                | ("linux", "aarch64")
        );
        assert_eq!(super::is_supported_target(), linked);
    }

    #[test]
    fn every_mapped_target_has_a_vendored_archive() {
        // Mirrors `link_needle()`'s target -> folder map. A typo or a dropped
        // archive breaks the build for that target (build.rs asserts), but this
        // catches it here too — in the common case — without a cross build.
        const MAPPED: &[(&str, &str)] = &[
            ("aarch64-apple-darwin", "macos-arm64"),
            ("x86_64-pc-windows-msvc", "windows-x86_64"),
            ("aarch64-pc-windows-msvc", "windows-arm64"),
            ("x86_64-unknown-linux-gnu", "linux-x86_64"),
            ("aarch64-unknown-linux-gnu", "linux-arm64"),
        ];
        let vendor = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/needle");
        for (target, folder) in MAPPED {
            let archive = vendor.join(folder).join("libneedle.a");
            assert!(
                archive.exists(),
                "{target}: {} is missing",
                archive.display()
            );
            assert!(
                vendor.join("needle.h").exists(),
                "vendored C API header is missing at {}",
                vendor.join("needle.h").display()
            );
        }
    }

    #[test]
    fn vendor_dir_has_no_unmapped_platform_folders() {
        // A folder nobody links is dead weight in the repo; one upstream drops
        // should be removed here rather than left to rot.
        let vendor = Path::new(env!("CARGO_MANIFEST_DIR")).join("vendor/needle");
        let mut folders: Vec<String> = std::fs::read_dir(&vendor)
            .expect("vendor/needle must be committed")
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        folders.sort();
        assert_eq!(
            folders,
            [
                "linux-arm64",
                "linux-x86_64",
                "macos-arm64",
                "windows-arm64",
                "windows-x86_64"
            ]
        );
    }
}
