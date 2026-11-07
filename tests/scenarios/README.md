# Scenarios

A scenario is a scripted game run that `tools/scenario.py` checks without anyone playing. This folder holds its
description, `<name>.toml`; the save it starts from is private and lives in `build/saves/<save>/`, which git ignores.

## The file

```toml
description = "what the run shows, in one line"
save = "tutorial-cave"                       # build/saves/tutorial-cave/
input = "tools/input/dkb-title-to-save.txt"  # button timings from the title screen into the save (no game data)
seconds = 75                                 # game time to run; the last frame is checked
warp = "HillLevelStage"                      # optional: start in this stage (tools/save_warp.py)

[env]                                        # optional: settings for every run of this scenario
# TOTK_FIX_VS_FAULT = "0"

[checks]
min_fps = 50          # a 60-frame window under this is slow
hitch_ms = 100        # a frame over this is a hitch (counted, not failed)
skip_world = 5        # seconds of game time left out at the start (the scene loading)
perf = "SLOW"         # fail on a SLOW verdict; "HITCHY" fails on short dips too; leave it out to not judge speed
                      # (the verdict is still reported). In [env] and --env values, {run} is the run's folder
max_black = 0.10      # fail when more of the last frame below the HUD is near-black
hud_band = 0.15       # the top share of the frame that holds the HUD
min_hud_bright = 0.005  # under this the last frame is not gameplay (a black loading screen), and is not judged;
                        # a bright sky passes it too, so outdoors it does not tell gameplay from a bright menu
[variants.native_gs]  # --ab native_gs alternates these
off = { DKB_NATIVE_GS = "0" }
on = { DKB_NATIVE_GS = "1" }
```

## Another area without another save

`warp = "<stage>"` in the file (or `--warp <stage>` on the command line) starts each run in that stage: the runner
rewrites the stage in the run's copy of the save with `tools/save_warp.py`. `--sweep <regex>` runs once per matching
stage (`'LevelStage$'`: the 20 layers). `python tools/save_warp.py --stages` lists the names. Measured on 1.0.0: the
tutorial-cave save warped to `HillLevelStage` loads that stage and reaches gameplay. The source save's progress comes
along, so a later stage may not start as real play would; for those, a real save from that point is still better.

## The save

1. Get the game to the point you want, on your console or in a run, and save.
2. Copy that profile's `account.uid` and `save/` folder into `build/saves/<save>/`. A run's profile is kept with
   `python tools/scenario.py <name> --keep-profile` (in `build/scenarios/.../profile/`).
3. `python tools/scenario.py --list` says whether the save is found.

Without its save a scenario is skipped (exit 77, SKIP in `tools/test.py`), never failed.

## The input

`tools/input/*.txt`: one step per line, `<frame> <buttons> <frames held>`, for `TOTK_INPUT_SCRIPT`. Frames count from
the start of the process, so a slow start (a cold shader cache, a busy machine) shifts where they land. Record a session
with `TOTK_INPUT_RECORD` (see `tools/run_game.py`) to get one from real play.

For an interactive recording on a copy of a scenario's save, run:

```powershell
.venv\Scripts\python tools\input_capture.py record tutorial-cave --build build/dkb100-drawid-before --warp UnderLevelStage
```

Play in the game window, then close it. The tool writes the raw recording, validated replay script, copied profile and
game log under `build/input-captures/` and prints a replay command. It records from the title screen, so include the
input needed to load the save. Existing recordings can be checked with `input_capture.py convert <recording> --output
<script>`; both paths must be under `build/`, and the output must not already exist. Replay disables physical input.
Timings depend on loading and game state, and held input at an abnormal exit may be incomplete; inspect the result.

`under-tutorial` is an isolated replay for the tutorial-cave save warped to UnderLevelStage:
`python tools/scenario.py under-tutorial --build build/<build> --repeat 2`. It records the merged input in each run
and ignores physical controls (`TOTK_INPUT=none`). It compares the HUD region with a private reference frame under
`build/scenario-references/under-gameplay.png`: a missing image or mismatch fails the run. This detects the tested
title and tutorial-prompt frames but cannot establish movement, exact stage, or gameplay in all circumstances. The
reference is game-derived and must remain under `build/`. On another machine, supply a screenshot of the expected
gameplay state at that path, then recalibrate against both good and bad runs. On the tested build the world clock started at the title screen, so
this script uses timed events throughout; mixed tick and timed scripts suppress tick events once that clock starts.
The timings and warped save state can vary, and this is not a state-driven input or movement check.

For other scenarios, `[checks.reference_frame]` supports `reference` (an image under `build/`), `region` (normalized
left, top, right, bottom), and `max_mean_error` (0..255, mean absolute RGB error after resizing the region). Test both
expected and wrong states: a similar-looking HUD in a different stage can pass. The score is in `result.json`.

## Choosing the checks

Calibrate on a known-good and a known-bad run: run the scenario with `--repeat 2` on both builds, read
`result.json`'s `black_share`, `hud_bright` and `fps`, and put the limits between them with room on both sides. Note the
measured values in a comment, as `tutorial-cave.toml` does.

For a NaN check, add `dump_frame = 500` and `max_nan = 0` under `[checks]`; optionally add
`dump_passes = "35-40"` to limit attachment copies to those passes. The runner sums the NaN counts in the dumped
images and fails if the sum exceeds `max_nan` or the dump is missing. The frame and pass numbers can change between
runs, so inspect the dump and calibrate them for the scene. See [DIAGNOSTICS.md](../../DIAGNOSTICS.md) for the dump's
limits and timing effects.
