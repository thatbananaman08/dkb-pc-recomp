# DKB Recomp

Static recompilation of *Donkey Kong Bananza* (Switch 2, ARM64, base 1.0.0) into a native Windows x86-64 exe with a Vulkan renderer, built on TotK Recomp (*Tears of the Kingdom*, Switch 1). Most of the lifter, host and Vulkan back end is TotK Recomp's; the Bananza-specific parts are the NVN 2 front end (`runtime/nvn_api_nvn2.c` and the NVN 2 paths in `runtime/gfx_shader.c`), the Ampere shader lowering (`runtime/nvn2_lower.cpp`, `runtime/third_party/eglantine/`) and the `nn::codec` movie decoder (`runtime/ov_codec.c`). Design: [docs/architecture.md](docs/architecture.md). Plan: [docs/bananza-plan.md](docs/bananza-plan.md). Shader notes: [docs/bananza-shaders.md](docs/bananza-shaders.md). Toolchain, build commands and tests: [docs/building.md](docs/building.md). Running and testing Bananza: [docs/dkb-testing.md](docs/dkb-testing.md).

## Game content boundary

This repository is public. Game data lives outside it and every derived file stays in ignored folders.

- The user's game and key paths are in `config.local.toml` (ignored). Pass the keys file path to tools; the tools read it themselves. Key values stay out of your output, logs, commands and files.
- Everything extracted or generated goes under `build/` or `generated/`.
- Every file the code generator emits carries the marker defined as `GENERATED_MARKER` in `tools/check_content.py`.
- Run `python tools/check_content.py --staged` before proposing a commit. It is the same check CI runs.

## Reusing code

`refs/` holds the upstream checkouts (`tools/fetch_refs.sh` recreates it). Read-only.

- Copy only MIT, BSD, ISC, zlib or Apache-2.0 code, keep its header, and add a row to `THIRD_PARTY_NOTICES.md`.
- DKB Recomp is GPL-3.0-or-later as a whole (LICENSE), because it carries Eglantine's GPL Ampere lowering
  (`runtime/third_party/eglantine/`, with the owner's go-ahead). GPL-3.0-compatible code (Eden, Eglantine, yuzu and
  the like) may be copied with its header kept, a row in `THIRD_PARTY_NOTICES.md` and the upstream commit. Files that
  came from TotK Recomp keep their MIT terms (`LICENSE-MIT`); a fix meant for TotK Recomp itself must not use GPL code.
- `tools/vendor/` stays byte-identical to upstream; Windows changes go in `tools/extract.py`.

## Code

- Build C with MSYS2 UCRT64 Clang (`C:\msys64\ucrt64\bin`). Generated C uses GNU extensions, so `cl.exe` is out. Call Ninja from that directory by full path or put it first on PATH.
- Two Python environments: `.venv` (3.12) for the lifter and the build, `.venv-py314` for the extraction and shader scripts, which need `compression.zstd`.
- Generated C is never edited by hand: change the generator (`recomp/`).
- Every lifter change passes the differential tests against Unicorn ([docs/building.md](docs/building.md), "Checks"). After changing `recomp/jumptables.py`, `tests\jumptables_check.py` reports 0 disagreements on 1.2.1 and 1.4.3.
- `recomp/layout.py` alone decides module bases.
- Graphics bugs: [docs/maxwell.md](docs/maxwell.md) lists the Maxwell behaviour the Vulkan path must match and the TotK bugs other ports fixed.
- Bananza bugs (NVN 2, the GPU, Switch 2 services such as `nn::codec`): first read how `refs/eglantine` handles the same point. That emulator plays the game, so it is the default reference. Where it departs from the hardware on purpose (Eden's default GPU accuracy skips texture downloads on CPU reads, `RasterizerVulkan::MustFlushRegion`), say so instead of copying the shortcut.
- Comments say what the code does and why, in words that stand on their own. Cite our code by file plus function (a line number may follow as a hint) and upstream code by project, file and function: `refs/` moves, so its line numbers rot. Commit hashes belong to the attribution of copied code.

## Testing and debugging

The playbook is [DEBUGGING.md](DEBUGGING.md): what each command answers, how to read its output, and which command to
run for a crash, a slowdown, a rendering fault or a CPU-translation suspicion. The rules:

- While working, check with `.venv\Scripts\python tools\check.py --quick --build build\<your build>` (~5 min);
  before a commit, `--standard` (~20-30 min). Report the summary of the runs before and after your change. Never call
  a change done with a FAIL or a TIMEOUT in the run, and never remove a test or add `known_failure` to make a run pass
  without saying so.
- A change the game must show (rendering, speed, a crash) is checked with `tools/scenario.py`: build it into its own
  folder (`tools/build_game.py ... --out build/<name>`) and run the scenario on the old and the new build. Do not judge
  the game from one run, from a run marked `[busy machine]`, or from a last frame reported as not in the scene.
- New behaviour gets a test in `tests/` and an entry in `tests/manifest.toml`; a new game situation to check gets a
  scenario in `tests/scenarios/` (its save stays in `build/saves/`, never in the repository).
- A crash report (`FATAL` in the log, `crash.txt` in a scenario run) names the lifted function or the DLL and the
  minidump. Read it before guessing.
- Logs, screenshots, dumps, minidumps and saves are game-derived: they stay under `build/` and out of commits, issues
  and pull requests.

## Writing

Code, comments, docs and commit messages are in English. Commit messages are plain sentences without type prefixes.
