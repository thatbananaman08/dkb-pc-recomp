# Debugging playbook

How to check a change and how to chase a bug without playing the game. Every command runs from the repository root
with the project's Python, `.venv\Scripts\python` (Windows) or `.venv/bin/python`. Everything a command writes goes
under `build/`, which git ignores. Logs, screenshots, minidumps and saves hold game-derived data: they stay on your
machine and never go into a commit, an issue or a pull request (see "Game content boundary" in `AGENTS.md`).

## The three commands

| Command | What it answers | Time |
|---|---|---|
| `python tools/test.py` | Do the checks that need no game still pass? (tiers 0-2: Python, compiler, GPU) | ~3 min |
| `python tools/test.py --tier 0-3` | Also the checks that read the dump or run the built game's probes | ~4 min, ~10 with the dump |
| `python tools/scenario.py tutorial-cave` | Does the game load a save, run 75 s, keep its frame rate and draw the terrain? | ~2 min a run |

Run the first before and after every change. Run the second after a change to `runtime/`, `recomp/` or the translator.
Run a scenario after a change that the game itself must show (rendering, performance, crashes).

### Reading `tools/test.py`

| Status | Meaning | What to do |
|---|---|---|
| PASS | the check passed | nothing |
| FAIL | the check failed | read `build/test/runner/<name>.log`; `-v` prints its end |
| TIMEOUT | it ran longer than its limit | a hang or a much slower machine; read the log |
| SKIP | it cannot run here; the reason says what is missing | install the tool, or set `DKB_TEST_<NAME>`, or ignore it if you do not need that tier |
| XFAIL | a known failure, listed with its reason in `tests/manifest.toml` | nothing, unless you are fixing that failure |
| XPASS | a known failure now passes | remove its `known_failure` line from `tests/manifest.toml` |

The exit code is 0 only when nothing failed or timed out. `--list` shows what would run and why something would be
skipped; `--tools` shows where each tool was found. Paths for your machine (your ExeFS folder, another compiler) go in
`test.local.toml` at the repository root, which git ignores:

```toml
[tools]
exefs = 'C:/path/to/your/dump/exefs'   # turns on diff-lift, diff-func and jumptables
```

A new test gets an entry in `tests/manifest.toml`. A test that fails for a known reason you are not fixing gets
`known_failure = "<the reason>"`; never delete a failing test to make the run green.

### Reading `tools/scenario.py`

```
python tools/scenario.py --list                                   scenarios, and whether their private save is present
python tools/scenario.py tutorial-cave                            one run with build/dkb100
python tools/scenario.py tutorial-cave --repeat 3                 three runs
python tools/scenario.py tutorial-cave --ab native_gs --repeat 2  off, on, off, on
python tools/scenario.py tutorial-cave --build build/<other> --env TOTK_VK_VALIDATION=1
python tools/scenario.py tutorial-cave --warp HillLevelStage             start in another stage (any of the 232)
python tools/scenario.py tutorial-cave --sweep 'LevelStage$'             once per layer: the 20 main stages
python tools/save_warp.py --stages                                       the stage names
```

**One save reaches every area.** `--warp <stage>` rewrites the stage in that run's copy of the save
(`tools/save_warp.py`: three fixed fields of `progress.sav`, checked before anything is written); the private save in
`build/saves/` is never changed. `--sweep <regex>` runs the scenario once per matching stage and combines with `--ab`
and `--repeat`. The warped game keeps the source save's progress, items and story flags, so a stage it has not reached
may start as unvisited or differently from real play; a stage that will not load from a warp is itself worth a look.
The first run in a stage translates its shaders, so judge its speed on a second run.

Each run prints one line: PASS or FAIL, how it ended, the perf verdict with the cause of any major slowdown, the mean
frame rate, and the near-black share of the last frame. Its folder `build/scenarios/<stamp>-<name>/<variant>-<n>/` has
`game.log` (all output), `last.png`, `perf.json`, `result.json` and, after a crash, `crash.txt`.

| Ending | Meaning | Next step |
|---|---|---|
| completed | reached the scenario's game time and exited 0 | read the checks |
| crashed | the crash reporter wrote FATAL | `crash.txt`: host address, lifted function `f_<addr>`, guest registers and frames, minidump path |
| crashed, no report | died with no FATAL line | the exit code names it (0xC00000FD stack overflow, 0xC0000409 fast fail); rerun with `--env TOTK_VK_VALIDATION=1` if a Vulkan layer or driver is suspected |
| device lost | Vulkan lost the GPU | the `vk: device lost` lines in `game.log` (VK_EXT_device_fault report); compare with `TOTK_FIX_VS_FAULT` (#9) |
| hung | no exit within seconds x 6 + 180 s | rerun with `--env TOTK_FLIGHT=1 --env TOTK_FLIGHT_AT=<frame>` and read the flight dump |
| did not start | the exe could not start | build it (`tools/build_game.py`) or pass the right `--build` |

| Check failure | Meaning |
|---|---|
| `not in the scene at the end` | the top band of the last frame has no bright pixels: still a black loading screen or a fade. The run is not judged; give it more time with `--seconds`. On a busy machine loading takes longer. The check only catches dark screens: a bright sky fills the band too, so in outdoor stages it always passes. |
| `near-black N% > M%` | the last frame is mostly black below the HUD: the black destructible terrain (#45), or a missing pass |
| `perf SLOW` | a major slowdown (see the next section) |

`[busy machine]` after a run means other programs used over 1 GiB of memory or commit memory was low. The picture and
the crash checks still count; **the frame rates do not**. Close the other programs (editors, browsers, other games,
builds) and rerun before you draw any conclusion about speed. A scenario will not start while another game process runs.

## Symptom to command

| Symptom | First command | Then |
|---|---|---|
| A test fails after your change | `python tools/test.py <name> -v` | fix the cause; if the test itself is stale (it copies production code that moved), update the test and say why in the commit |
| The game crashes | `python tools/scenario.py <scenario>` | `crash.txt`. In lifted code: `f_<addr>` is the guest function. Outside it: the DLL is named. Open the minidump (`logs\crash-*.dmp` beside the exe) in WinDbg or Visual Studio with `totk.map` for host symbols |
| It is slow | `python tools/perf_check.py <game.log>` | the cause and its next tool, below |
| Something renders black or wrong | a scenario, then `TOTK_DUMP=<frame>` (DIAGNOSTICS.md) | compare `passes.txt` and the pass images of a good and a bad run; `DKB_DUMP_ORDERED=1` for draw order |
| NaN or a white/bright glow | `TOTK_DUMP=<frame> DKB_DUMP_PASS=<range>` | `images.txt` counts NaN/Inf per target; `DKB_DUMP_BOUNDARY=1` narrows it to a pass (DIAGNOSTICS.md) |
| The GPU is lost | a scenario with `--env TOTK_VK_VALIDATION=1` | the fault report; validation messages in `game.log` |
| CPU translation is suspected | `python tools/test.py --area cpu --tier 0-3` | `diff-lift` and `diff-func` compare lifted code with Unicorn on the game's own code; a lifter change must pass them |
| A shader fails to translate | `python tools/test.py --area shaders,geometry` | `tools/replay_shader_failure.py` on the saved request |

### Slowdowns

`python tools/perf_check.py <logs...>` reads the `TOTK_TRACE=perf` lines (every scenario run has them) and splits the
run into 60-frame windows. A window under `--min-fps` (50) is slow; consecutive slow windows are an episode, major when
it averages over 1.5 times the frame budget, lasts 3 windows or has a frame over a second. Verdicts: OK, HITCHY (minor
dips only, like the streaming load every run of the tutorial cave has at 34-43 s of game time), SLOW (a major
episode). `--group a='glob' --group b='glob'` compares sets of runs. Each episode gets a cause:

| Cause | What held the frame back | Next tool |
|---|---|---|
| compile | shader translation or pipeline builds | expected on the first run after a translator change; on a warm cache, find what is translated again |
| texture | texture uploads or decodes | `TOTK_TRACE=perf,gfx` for the uploads, or `TOTK_FLIGHT=1` and `tools/flight_view.py` |
| gpu | the GPU was busy for most of the frame | `TOTK_PERF_SLOW_GPU_MS=50`: the slow replay parts, their stages and indirect arguments |
| render-cpu | the render thread was busy itself | `TOTK_PROFILE=1 TOTK_PROFILE_RENDER=1`, then `tools/prof_report.py <profile> totk.map` |
| game-cpu | the game's own threads | `TOTK_CRIT=1`, then `tools/crit_path.py`: each frame's critical path (running, waiting for a core, in a host call, waiting on another thread) |

A slowdown is only real when it repeats on a quiet machine. Compare against a baseline build on the same save with
`--ab` or two `--build` runs, never against a number from another day or machine.

## Making a scenario

A scenario is `tests/scenarios/<name>.toml` plus a private save. See `tests/scenarios/README.md`. For another area you
usually need no new save: give the scenario `warp = "<stage>"` (or run with `--warp`). To find which game code opens a
file, run with `--env DKB_FS_BACKTRACE=<part of the path>`: the log gets the guest call stack of the first 64 matching
opens (`fs backtrace:` lines). Most stage files are opened by the resource loader thread, whose stack ends in the loader.

## Rules for changes

- One topic per change. Run `python tools/test.py` before and after, and report both summaries.
- A runtime change is checked in the game: rebuild into a separate folder so the baseline stays
  (`tools/build_game.py ... --out build/<name>`), then run the scenario with the old and the new `--build`.
- A lifter or `runtime/cpu.h` change passes `diff-simd`, `diff-atomic`, `excl-mt`, and with the dump `diff-lift`,
  `diff-func` and `jumptables` (`python tools/test.py --area cpu --tier 0-3`).
- `python tools/check_content.py --staged` before every commit.
- Say what you did not verify (another GPU vendor, a long session, other areas of the game).
