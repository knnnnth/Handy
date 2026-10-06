# Vendored Cactus Whistle engine (`libneedle.a`)

This directory holds Handy's only vendored native binary: the prebuilt Cactus
Needle runtime that backs the `EngineType::Whistle` speech-to-text engine (model
`Cactus-Compute/whistle`). Every other inference dependency in Handy
(`transcribe-cpp`, `transcribe-rs` / ONNX Runtime) arrives as a crate, so this is
the first place a prebuilt, upstream-compiled archive is committed to the repo.

The Rust side is `src-tauri/src/audio_toolkit/whistle.rs`; the link wiring is
`link_needle()` in `src-tauri/build.rs`.

## Source

| What            | Where                                                                                                                                                                                                          |
| --------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Runtime repo    | [`Cactus-Compute/needle3`](https://huggingface.co/Cactus-Compute/needle3)                                                                                                                                      |
| Pinned revision | `c7c415a3d1b3d929014bc6e866d51ebb971f7089` (a commit sha — the resolve URLs are immutable and CDN-cacheable)                                                                                                   |
| C API header    | `needle.h` (verbatim copy of `<revision>/macos-arm64/needle.h`; the header is byte-identical across all five platform folders)                                                                                 |
| Weights         | [`Cactus-Compute/whistle`](https://huggingface.co/Cactus-Compute/whistle) @ `b358ddadd89b7a713b5aa131f23032d3cca1b251`, file `whistle.cact` — **not** vendored, downloaded at runtime into the shared HF cache |
| License         | Apache-2.0 — see [Attribution](#attribution) below                                                                                                |

## Attribution

The archives and `needle.h` here are **unmodified upstream builds** of the Cactus
Needle engine, redistributed under the Apache License 2.0. `LICENSE` in this
directory is a verbatim copy of the one upstream ships (sha256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`).

> Needle: Foundation Tool-Calling Model for Tiny Devices
> Copyright the Cactus Compute team — Cactus Compute, Inc.
> Ndubuaku, Henry; Mosoyan, Karen; Mroz, Jakub; Cylich, Noah; Kumar, Satyajit;
> Sandhu, Parkirat; Shemet, Roman; Lee, Justin H. (2026).
> <https://github.com/cactus-compute/needle>
>
> Licensed under the Apache License, Version 2.0. The upstream `LICENSE` file
> carries the stock `[yyyy] [name of copyright owner]` placeholder — it names no
> copyright holder — so the attribution above uses Cactus's own requested citation
> from the [needle3 model card](https://huggingface.co/Cactus-Compute/needle3)
> rather than a line we invented. Upstream contact: founders@cactuscompute.com.

Two notes for anyone refreshing or relicensing this:

- **Only the runtime is vendored.** The `whistle.cact` weights are fetched from
  Hugging Face on demand and sha256-verified against the hash in
  `src-tauri/src/catalog/catalog.json`; they are Apache-2.0 as well, but they are
  not in this directory and no licence copy is owed here.
- **Apache-2.0 §4(a)** requires passing a copy of the licence to anyone we
  distribute the archives to. That is why `LICENSE` sits beside them rather than
  only being linked from this file. Keep it if the archives move.

## Files

`sha256sum */libneedle.a needle.h`:

```
d6e33724cab170f0a25bd943d3a8ed5da925fcb793e9fcb1f201bf2478a50f91  linux-arm64/libneedle.a
5c0ff309dcf9c2238bd07986d9d100069d568596c2378f0bfedc1677cc02f756  linux-x86_64/libneedle.a
a3b9163abe7b4bd52487c4005506cb35c5163bb9b9587af4ae07ed7587de697d  macos-arm64/libneedle.a
e6c479a4094896785a22c2a54e3c8bfd8f60903488651ad6d4b47685512cec61  windows-arm64/libneedle.a
a2501f416a562bf0c2ad734821dd190571d22ced22c104372b8cd41b40245114  windows-x86_64/libneedle.a
90f347f9dca1199de79967ab199a56e0588bd473fe306051f8d80d146976a324  needle.h
```

| Folder           | Size (bytes) | Rust target                 |
| ---------------- | -----------: | --------------------------- |
| `macos-arm64`    |      1503976 | `aarch64-apple-darwin`      |
| `linux-x86_64`   |      2143656 | `x86_64-unknown-linux-gnu`  |
| `linux-arm64`    |      1966732 | `aarch64-unknown-linux-gnu` |
| `windows-x86_64` |      2292578 | `x86_64-pc-windows-msvc`    |
| `windows-arm64`  |      2078628 | `aarch64-pc-windows-msvc`   |

## Platform coverage caveat

**Upstream publishes no `macos-x86_64` build**, so Apple-silicon is the only
macOS target that can link this engine. Intel Macs are gated off: the
whistle module compiles to a stub there (`is_supported_target() == false`),
`build.rs` emits no link directives, and `seed_catalog_models` never lists
Whistle, so no user on Intel can select a model that cannot run. The same stub
path covers every other target — `x86_64-apple-darwin`, musl Linux, and any
future triple — which is why the FFI symbols live behind a `cfg`-split module
rather than at the top level.

All five archives are built against libc++ (`nm -u` shows `St3__1` symbols and
zero `__cxx11`), so `build.rs` emits `cargo:rustc-link-lib=c++` on every
supported target, not just Apple.

## Refreshing to a new upstream revision

1. Re-download each archive and the header at the new sha and verify them
   before touching the code:

   ```sh
   cd src-tauri/vendor/needle
   REV=<new-revision-sha>
   for f in macos-arm64 windows-x86_64 windows-arm64 linux-x86_64 linux-arm64; do
     mkdir -p "$f"
     curl -sLo "$f/libneedle.a" \
       "https://huggingface.co/Cactus-Compute/needle3/resolve/$REV/$f/libneedle.a"
   done
   curl -sLo needle.h \
     "https://huggingface.co/Cactus-Compute/needle3/resolve/$REV/macos-arm64/needle.h"
   sha256sum */libneedle.a needle.h
   ```

2. Sanity-check that the C API this repo binds is still exported and still
   C++-free of surprises:

   ```sh
   nm -g --defined-only macos-arm64/libneedle.a | grep -E 'needle_(load|models|reset|transcribe|last_error)'
   nm -u macos-arm64/libneedle.a | grep -c St3__1   # expect non-zero → libc++ needed
   ```

3. Update the hash table above and re-verify: `cargo build`, `cargo test --lib
whistle`, then silence and speech end-to-end through `--transcribe-file`
   (silence must yield empty text with exit code 0; speech must yield a correct
   decode). A C API change means editing the `extern` block in
   `src-tauri/src/audio_toolkit/whistle.rs`.
4. Commit the new archives **and** the updated hashes together — an unreviewed
   swap of a vendored binary is unreviewable.
