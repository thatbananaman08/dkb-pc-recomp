# Developer diagnostics

This document describes the individual diagnostic controls. For what to run when, start with [DEBUGGING.md](DEBUGGING.md): `tools/test.py` runs the checks, `tools/scenario.py` runs the game on a save and judges the run, and `tools/perf_check.py` finds and explains slowdowns. Full structured logging, configurable diagnostic levels, Vulkan draw labels, and a developer panel are not implemented. The controls below are independent and opt-in, except the crash reports, which are always on.

Outputs can contain game-derived data, memory contents, and screenshots. Keep them under ignored build directories and do not commit them. Record the source revision (`git rev-parse HEAD`), GPU and driver, settings, game version, scene, save and reproduction steps alongside a capture; do not include keys or proprietary content in public reports.

## Structured failure events

Set `TOTK_EVENT_LOG` (or `DKB_EVENT_LOG`) to a new JSONL file to record selected failures as they happen. For a
scenario, for example:

```powershell
python tools/scenario.py tutorial-cave --env 'TOTK_EVENT_LOG={run}/events.jsonl'
```

The parent directory must exist. Each process creates its own file and refuses to overwrite an existing one; a failed
open is reported in the text log. `{run}` gives repeated scenarios different paths. The file stays open for the process
lifetime and is not rotated or size-limited. It contains game-derived details and must remain private under `build/`.

Each JSON line has `version: 1`, `category`, `level`, `kind` and `time_unix_ms`. Categories currently cover shader
translation failures (`reason`, `shader`), Vulkan device loss reported through `vk_device_fault_report` (`result`),
and crashes: `fatal` has a message; `reported` covers platform crash reports, including exceptions and C runtime
aborts, and has the code, host address, module, module offset, access type, fault address and minidump path. Addresses
are hexadecimal strings; an unknown fault address or minidump is `null`. Exception access is 0 for read, 1 for write,
8 for execute, and -1 when unavailable. The lifted-function lookup and device-fault vendor details remain in the text
log. An absent event does not prove a run had no error: other categories are not yet emitted, and a fast fail, early
crash, damaged process or failed write can prevent an event or leave an incomplete line. Events use one unbuffered
append each and crash events do not take an event-log lock. Strings are bounded and set `truncated: true` when cut;
valid UTF-8 is preserved and invalid bytes are escaped as `\u00xx`. Timestamps use the system wall clock; concurrent
records are appended atomically but may appear in a different order from their timestamps.

For existing text logs, `python tools/log_events.py <game.log> --out build/diagnostics/events.jsonl` extracts selected
messages offline, including Vulkan validation messages. It recognizes only its documented text patterns and does not
produce the same schema as the runtime stream. Its output also stays private under `build/`.

## Vulkan frame capture

Install RenderDoc separately. Before launching the game in PowerShell, set `$env:TOTK_RDOC='500'` to capture presented frame 500, or use a comma-separated list (up to 64 frames). Set `TOTK_RDOC_DIR` for the output directory and `TOTK_RDOC_DLL` to override the library location. The runtime loads RenderDoc before Vulkan initialization and reports whether each capture completed. Frame 1 cannot be selected by this present-boundary mechanism. Captures may be large and are not automatically pruned. RenderDoc's unsupported vendor-extension option is used to expose external host memory; capture or replay may fail on a particular driver. Test without RenderDoc as well.

Loading the DLL alone does not register its Vulkan capture layer. On Windows with a portable RenderDoc distribution, its layer can be selected for one process with `VK_LAYER_PATH` set to the directory containing `renderdoc.json`, `VK_INSTANCE_LAYERS=VK_LAYER_RENDERDOC_Capture`, and `ENABLE_VULKAN_RENDERDOC_CAPTURE=1`. This does not guarantee a usable capture: RenderDoc 1.46 on the RTX 5060 / NVIDIA 617.42 exposed no `VK_EXT_external_memory_host` even with the unsupported-extension option enabled, so device selection failed before a frame could be captured. A capture-compatible memory import path or a RenderDoc setup exposing that extension is needed. Without the layer, the API may load while capture fails to start. Verify the file and replay rather than relying on API initialization.

For core validation, set `$env:TOTK_VK_VALIDATION='1'` with the Khronos validation layer installed. Validation warnings and errors go to the runtime log (the first 500 messages). It is expensive; unset the variable after the run. For GPU-assisted validation configured by the scenario runner, use `--safe-gpu` as described in [DEBUGGING.md](DEBUGGING.md); synchronization validation is not exposed as a separate runtime option. The same 500-message runtime cap applies in GPU-assisted mode, even when the layer's own message limit is disabled. Once the cap is reached, later reports are not available to the scenario parser, so a run without a reported error is inconclusive.

## Defective draw and shader evidence

A renderer-specific alternative is `$env:TOTK_DUMP='500'` and `$env:TOTK_DUMP_DIR='build/diagnostics/dump'`. It writes `passes.txt`, render target images and referenced SPIR-V files, subject to the caps documented at the top of `runtime/vk_dump.c`. The pass listing associates draws, pipeline/texture sets and shader hashes. `TOTK_DUMP_LOG` includes preceding frame listings. Buffer and extra texture snapshots are explicit additional options; see that file before enabling them. Dumps perform GPU copies and may be very large. They are not safe to initiate after device loss.

For ordered replay evidence set `DKB_DUMP_ORDERED=1` alongside `TOTK_DUMP`. `draws.txt` contains up to 16,384 emitted draw commands, in replay order, with frame, pass and per-pass ordinal, NVN operation kind, runtime pipeline/program identifiers, shader hashes and origins, target addresses and formats, and known descriptor handles and buffer ranges. Indirect arguments are GPU-written and are identified by address, not read speculatively. A runtime pipeline handle and a per-frame draw number are not stable across runs. A translated quad or line loop may be emitted as an indexed Vulkan draw; the operation field is the NVN replay operation, not a Vulkan event label. Skipped operations, original command-buffer offsets, Vulkan debug labels and error attribution are not recorded. The existing grouped listing can include more binding detail. The output contains game-derived identifiers and must stay private.

Each new ordered draw has `match_key=v1:<vs>:<gs>:<fs>:<op>:<occurrence>`: the SPIR-V hashes of its vertex,
geometry and fragment stages (zero for an absent stage), the NVN operation, and its one-based occurrence among draws
with that exact combination in the captured frame. Inserting a draw from a different group leaves the key unchanged.
Inserting, removing or reordering draws within the same group can shift occurrences or associate a key with a different
draw. Changing the translated shaders changes their hashes. Keys are candidate matches within comparable captures,
not proof of the same scene object or binding state; they are not Vulkan debug labels.

`python tools/compare_draws.py <before/draws.txt> <after/draws.txt>` compares draw counts grouped by vertex and
fragment shader hashes and operation. It warns when either trace is truncated. A group can contain many draws, and
matching group counts do not establish that their bindings or individual draws match; scene and frame differences can
also change the counts. The hashes are of translated SPIR-V, so changing the translator can change them even for the
same guest shader. Keep the comparison output private along with the dumps.

Add `--ids` to list candidate matches by `match_key`, alongside each run's draw ID, pass number and pass ordinal. It
marks matches ambiguous when the group counts differ or either trace is truncated. It requires both captures to
contain the new keys; older traces can still use the group-count comparison.

Set `DKB_DUMP_PASS=35-40` (or a single number) to copy attachments only for that inclusive range of pass and dispatch numbers. Sampled textures are omitted in this mode unless explicitly selected with `TOTK_DUMP_TEX`; shader files and logs still come from the existing dump. `images.txt` reports NaN and Inf channel counts for copied formats. The filter selects pass boundaries, not individual draws or pass entry. Pass numbers and scene state can change between runs; inspect `passes.txt` each time. Readbacks add synchronization and may perturb timing. They cannot establish the first invalid writer, especially where a target already contained invalid values at entry.

`DKB_DUMP_BOUNDARY=1` adds at most 32 snapshots of color target 0 before and after direct dispatches and immediately before beginning rendering, outside the rendering instance. Use it with a narrow `DKB_DUMP_PASS` range. `passes.txt` identifies each image, guest address, format, aspect, mip, layer and layout, or records why a snapshot was skipped; `images.txt` has the corresponding channel counts. Feedback-layout targets and unsupported subresources are skipped. Copies use GENERAL layout, an all-commands memory-write to copy-read dependency, and a copy-stage dependency to host reads and subsequent commands. Pass-end dumps also skip targets still in feedback layout. These copies alter synchronization and timing, so compare with an uninstrumented run and do not infer deterministic replay. This does not measure within a rendering instance.

For black terrain, capture a frame where the artifact is visible and inspect the terrain pass in `passes.txt`: compare the pipeline's shader hashes, bound textures and intermediate render targets with a known-good run. Identify the first pass where the expected terrain disappears; inspect its geometry stage and resource bindings before changing the translator. This is an investigation procedure, not a diagnosis of the defect. A RenderDoc capture, when supported, can supplement the pass dump with individual Vulkan draw state.

The independent fragment-output reporting fixture can be tested on the RTX 5060 without game content:

```powershell
.venv\Scripts\python tests\fragment_nonfinite_gpu_test.py --cc C:/msys64/ucrt64/bin/clang.exe --glslc C:/msys64/ucrt64/bin/glslc.exe --spirv-val C:/msys64/ucrt64/bin/spirv-val.exe --layer-path C:/msys64/ucrt64/bin --out build/test/fragment-gpu
```

It tests two synthetic ordinary descriptor-set arrangements with the Khronos validation layer, GPU output and report-buffer readback. It is not connected to game shader translation or replay; it does not instrument arbitrary SPIR-V, track components or exercise independent graphics pipeline library layouts. Successful fixture results do not identify a faulty game draw.

## Draw fast path switch

`TOTK_DRAW_FASTPATH` (default 1, restart): `0` restores the previous per-draw work for an A/B run. On, the draw path
remembers `light_quad_match` per (program, vertex shader, fragment shader) within a command buffer, answers
`nvn2_origin` from a lock-free cache instead of the shared mutex (the stage names are immutable), and skips an
index-buffer bind that repeats the buffer, offset and type already bound in the command buffer. The state clears with the
rest of the per-command-buffer draw state (`vk_exec_draw_new_cb`). Logic: `runtime/draw_cache.h`, `runtime/origin_cache.h`.

## Waits for shaders and pipelines

Every wait of the replay thread of 2 ms or more for a shader or a pipeline is logged (first 200 per kind), with its cause:
`vk_exec: frame N: a draw waited X ms for its pipeline (one-shot|full-screen ...)` (a one-shot or full-screen draw; other
draws are skipped while their pipeline builds), `one-shot draw waited X ms (shader translation), stage S origin O`, and
`dispatch|indirect dispatch waited X ms (compute shader translation or load|compute pipeline build)`: compute is
synchronous because a skipped dispatch changes results.

`--shot` runs (tools/scenario.py) build shaders and pipelines on the replay thread, so their `shader translation` and
`pipeline build` frame times are the worst case, draws are never skipped, and no census is written
(`pipeline-keys.bin`, `shader-answers.bin`, `shader-programs.bin` next to the pipeline cache; the next boot translates and
builds them before the guest starts: `vk_pipeline_prewarm`). `TOTK_PIPE_ASYNC=1` gives a `--shot` run the background builds,
skips and census of normal play (images can then depend on build timing).

## Boot prewarm and caches

`TOTK_PREWARM_BLOCK` (default 0): the boot prewarm (shaders and pipelines recorded by earlier runs: `pipeline-keys.bin`,
`shader-answers.bin`) goes on beside the guest at below-normal priority on half the cores; `=1` waits for it before the
guest starts, as before. The boot line says `guest starts N ms after process start (M ms of it waiting for the prewarm...)`,
and the prewarm's own line ends `done N ms after process start`. Draws whose pipeline is not ready take the ordinary
background path (fast link or skipped frame); one-shot draws wait for their own pipeline (its job moves to the front).

`TOTK_CACHE_DIR` (default: the folder of the exe): one folder for `pipeline-cache.bin`, the census files and `shader-cache/`,
so they survive a new exe. Stale entries are safe (see `cache_path_init`). `tools/scenario.py --async` passes `TOTK_PIPE_ASYNC=1`.

With `TOTK_TRACE=perf` a slow frame's line ends `process: page faults +N, working set +M MiB, write-watch X ms`: a hitch
with thousands of faults and a growing working set is the game touching new memory, not shader or replay work.

## Write-watch cost while a stage loads

An attempt to coalesce the pre-snapshot queries while GetWriteWatch calls are slow (TOTK_WW_ADAPT, 2026-11-09) was removed: it did not
reduce the write-watch time (6.0 against 6.1 s per Canyon run) and a verify run found 4 mismatches against none without it, so
scanning pools early or from a wider set cannot be assumed to keep the mirror current. Do not repeat it without a proof for the
mirror's epoch bookkeeping (`snap_at`, `mepoch`).

Verify run (is anything the GPU reads stale?): `TOTK_MEMORY=verify` (mirror mode, plus each submission copies up to 4 MiB of
mirror pages, round robin, back and compares them with guest memory). A page that differs and was not written since is logged as
`mem: verify mismatch: pool ... offset ...` and counted in `mem: mirror frame ...; verify: pages compared N, skipped M (written
since), mismatches K`: expect K = 0.

## Frame pacing

`TOTK_PACE_LOG=1` (diagnostic; also enables VK_KHR_present_id and present_wait when the driver has them): one `pace t=...` line
every 60 frames with the histogram of display intervals (`<12, 12-20, 20-28, 28-37, 37-50, >50` ms), p50/p95/p99, the mean and the
number of uneven frames (an interval more than 4 ms from the one before) for the real display time (`display (present_wait)`) and for the
host present, plus what the game asked for (present interval 1/2/3+), the pacer period and slot wait. Without present_wait the line
has the host present only.

Findings on Canyon (240 Hz monitor, mailbox, TOTK_PRESENT_HOLD off): Bananza has no `realtime` patch set, so `frame rate: no patch
table` and the game steps whole vsyncs by its own controller, which asks interval 1, 2 or 3 depending on how long its frames took;
presents leave when the GPU finishes them, so about a third of the frames are uneven. `TOTK_PACE_FLIP=1` (default off) holds each
host present for its pacer slot on a flip thread: the host present cadence gets even (uneven 34% -> 15%) but the added latency
(mean hold 31 ms) makes the game's controller ask interval 3 for 24% of the frames (3% without), p95 52 against 42 ms and 26.5
against 31.7 fps, so it stays off.

## Crash reports and device loss

Always on. A fault in the game's own code (lifted or host) is reported by the crash reporter (`runtime/main.c`
crash_handler): a `FATAL:` line with the host address, the lifted function `f_<addr>` it is in, the guest registers and
the guest frames, then the process exits with code 71. A fault in another module that nothing handles (a driver, a
Vulkan layer, the C runtime) is reported the same way from the last-chance filter (`runtime/platform/win32.c`
crash_unhandled), with the module named: `(ucrtbase.dll+0x48ca7)` in place of `(totk+...)`. `abort()` and invalid C
runtime parameters are reported with the host frames that called them. Each report is followed by `minidump:` and the
path of a minidump (`logs\crash-<pid>-<tick>.dmp` beside the exe: thread stacks, registers and modules, no heap, under
100 KiB) for WinDbg or Visual Studio. The minidump holds game stack memory and stays private like the logs. A stack
overflow or a fast fail that ends the process before any handler runs still leaves no report; `tools/scenario.py`
names it from the exit code.

On `VK_ERROR_DEVICE_LOST` the runtime logs what `VK_EXT_device_fault` reports, when the driver supports it
(`runtime/vk_device.c` vk_device_fault_report): its description, the faulting addresses and the vendor codes.

## Which code opens a file

`DKB_FS_BACKTRACE=<text>` logs, for the first 64 RomFS opens whose path contains text, the open and the guest call
stack (`fs backtrace:` lines; `runtime/romfs.c` fs_backtrace), independently of `TOTK_TRACE=fs`. Files the game
streams (stages, models) are opened on its resource loader thread, so their stacks end in the loader; the code that
asked for the file posted a request earlier.

## Crash reproduction and frame history

`TOTK_FLIGHT=1` enables a bounded in-memory frame history; F8 requests a dump, and `TOTK_FLIGHT_AT` accepts frame numbers for unattended runs. It can also trigger on slow frames. See `runtime/flight.c` for ring size, thumbnail and trigger settings. Dump files are not automatically retained or pruned. This is not a crash handler: it may not finish writing after a crash or device loss. Preserve logs and any existing completed capture, exception information and reproduction steps separately; never infer that a GPU capture completed from a crash alone.

## Screenshots and comparison

Set `TOTK_SHOTS=500-500:1`, `TOTK_SHOTS_DIR=build/diagnostics/shots`, and optionally `TOTK_SHOTS_FULL=1` to request frame 500 without a graphical interface. Frame indices depend on the run; pipeline compilation remains asynchronous. World-clock sampling is also available; see `runtime/boot.c`.

With Pillow installed in the project's Python environment:

```powershell
.venv\Scripts\python tools\compare_screenshots.py reference.png actual.png --diff build\diagnostics\diff.png --threshold 3 --max-different-fraction 0.01
```

The tool prints JSON metrics and returns 1 if the fraction of pixels with any RGB channel difference above the threshold exceeds the limit. The difference image contains absolute RGB differences. Choose tolerances for the scene and hardware; matching pixels do not prove rendering correctness. Keep references and reproduction metadata separate from game files.

The tool warns unless `--reference-state` and `--actual-state` name JSON objects with matching non-null `scene`, `event` and `camera` fields. These fields must be measured or independently verified by the investigator: this runtime does not yet capture a reliable gameplay event or camera transform automatically, and matching user-supplied labels alone do not prove alignment. The existing scripted input and present number are not deterministic scene markers.

## Automated agents

Request frames with the environment variables above, save the log, settings, revision and reproduction notes with outputs, and report missing or failed captures explicitly. Check tool exit codes and inspect the diff image and metrics. The existing controls do not yet provide a unified bundle, deterministic replay, guaranteed draw identity across scene or shader changes, or guaranteed crash recovery.

### Light-quad draw trace

`DKB_LQ_TRACE=1` (off by default, log only, draws unchanged) logs, for every draw the light-quad skip leaves out and
every other draw of the same fragment program family, the first appearance of each distinct key as
`lq-trace: t=<world clock s> key=<hash> ...`: vertex and fragment NVN 2 origins, topology, counts, render targets, the
fragment storage buffers, c0+0x530, and each texture and image the fragment stage reads (handle, address, format, size,
levels, layers, view type, sampler address modes and compare), and, outside the key and right after the `skip= vs= fs=` fields, the vertex stage's constant buffer c9
words at row 7 (`vs_c9_row7={x=.. y=..}`, the u32 flag words the family's vertex programs test with OpBitwiseAnd). Once per world-clock second (`frame_step_world_us`, the
clock `--seconds` ends on) it logs `lq-trace: sec=<n> skipped=<draws> keys=<distinct so far>`. At most 2000 lines.

### Light-quad fold

On Bananza with an NVIDIA GPU, seven light-shaped vertex stages lose the device 40 to 60 s into their scene or crash the driver's shader
compiler (issue #9, "instruction pointer (invalid)"): Ground's `67a23e6f132386f9` and `e2bf75f927acd249`, Tower's `3a10ce1a4f3cdf0b`,
`8475bd9dadb18f45` and `4645f0dd7b4502a2`, Ice's `7a76ae8de5016fd9`, and Fort's `1ee77eac998beea5`. By default the runtime changes those modules (`gfx_shader.c`, `lq_fold_rules`;
`spirv_lq_patch` kind 4096).

**The fold is not exact.** Each stage computes a selector r = (a computed index) & 3 that picks one of four constant-buffer words. The fault
goes away when the test-class `&3` ANDs that feed branch conditions become 0. What that changes depends on the stage:

- Ground, Tower and Ice stages (the whole fold; Ice's folded module compiled without the compiler crash and drew for 90 s without device loss): both ANDs become 0, so the per-instance value they export to the fragment stage (Location 4, 5 or 6),
  which the fragment stage uses as a texture array layer, becomes layer 0 for every instance. In the captured draws r was 0 to 3; only that
  exported value differed, the other outputs were identical.
- Fort's `1ee77eac998beea5` (construct rule): only the AND that decides selection construct 31 (the `r != 3` construct) becomes 0. That shifts a
  texture coordinate (Location 3) by up to 0.5 in about 46% of the captured invocations (92 of 200). Position and the exported layer are
  unchanged.

Every value-preserving rewrite of the same code was tried and still lost the device (`and3exact`, `rotexact`, `loadexact`, `selfix` and `mul2` in
`DKB_LQ_PATCH`, and a branch-free select form), so the cause is not understood and only removing r from the branch conditions has worked.

The stage is found by the NVN 2 stage name the shader request carries, not by the cache key. Two origins can share a module (`4645f0dd`
and `e2bf75f9` do); both are in the table, so the module is folded whichever loads first. A rule is applied only if the patcher finds the
expected pattern (the whole fold: exactly two test-class ANDs in a vertex module with a Location 9 input; the construct rule: an AND that reaches
that construct); otherwise the module is left as translated. `light_quad_fault_skip` (`TOTK_FIX_VS_FAULT`) no longer skips a light-quad draw whose
vertex module was folded; the log says once per stage `light-quad: drawing <origin> with folded vertex stage`. Fort's stage has no Location 9
input and is not a light-quad match, so nothing skips it, folded or not. Only Ice's fallback and `cae0ef8e9fe9147f` stay skipped by name: Ice's `7a76ae8de5016fd9` is skipped before its pipeline is built only if the fold
fails closed, because the folded flag is set when the shader loads, before that decision. Every other light-quad match stays skipped. `DKB_LQ_FOLD=0` turns the
fold off (default 1), and setting `DKB_LQ_PATCH` also turns it off. The diagnostics below override the table for the stages they name.

`DKB_LQ_FOLD_ORIGINS=<origin>[,<origin>...]` (whole 16-digit origins, off by default) also tries the fold on those vertex stages, with the same
fail-closed pattern check and the same skip lifting; it does not change the two default stages. `DKB_LQ_FOLD_SITE=export|branch` (diagnostic, only for stages named in `DKB_LQ_FOLD_ORIGINS`) nulls just one of the module's two test-class
`&3` ANDs: `branch` the one whose value reaches the condition of a branch (the `r != 3` construct), `export` the other one, which reaches
no branch condition. The ANDs are told apart by data flow through arithmetic, selects and function-local variables; unless exactly one
of each kind exists the module is left alone and its draws stay skipped. The log says which site was nulled.
`DKB_LQ_FOLD_SITE=construct` with `DKB_LQ_FOLD_CONSTRUCT=<n>` nulls every test-class `&3` AND that reaches the condition of selection construct `n`
(numbered as `DKB_LQ_VSBLOCK` numbers them, from 0 in module order), for a stage named in `DKB_LQ_FOLD_ORIGINS`. Unlike the other sites it also
works on a vertex module without a Location 9 input and with any number of test-class ANDs (Fort's VS has five), and it fails closed when no AND
reaches that construct. On Fort `1ee77eac998beea5` (cache module `d8e2c12e875a`) construct 31, the `r != 3` construct VSBLOCK=31 forced, is decided
by one AND whose nulling changes only Location 3 (a texture coordinate pair of the fragment stage, up to 0.5 in 92 of 200 captured invocations),
not Position; nulling all five ANDs also changes the exported Location 5 layer.
`DKB_LQ_FOLD_SITE=exact` nulls nothing: every selection construct whose branch condition the branch-reaching AND reaches (the `r <= 1` and
`r != 3` constructs) must be a triangle with a pure arm, and each is rewritten as ldcfix does it, running the arm unconditionally and keeping the
old value with a select on the true r. No branch condition depends on r any more and every output is bit-identical for every input. If any such
construct cannot be rewritten the module is left alone.

`DKB_LQ_MUL2_ORIGINS=<origin>[,<origin>...]` (whole 16-digit origins, off by default) applies the `mul2` rewrite to those stages instead of the
fold. The selector r the fold nulls is `cond ? (sum & 3) : 3` with `sum = shl(X, 16) + V`; the shifted term has no bits below 16, so
the `&3` ANDs are pointed at a twin select on V, and a product V = lo(P) * lo(Q) + C is reduced to `(P & 3) * (Q & 3) + (C & 3)`. r keeps
its value for every input (the fold fixes it at 0, which changes the exported selector where the data varies it), and the 16-bit-halves
multiply leaves r's path; it stays only where other values use the original select. It fails closed (the module is unchanged and its
light-quad draws stay skipped) unless every test-class `&3` of the module has that form. A stage in both lists gets mul2 only; a
patched stage is exempt from the light-quad skip like a folded one. Log: `gfx_shader: mul2 rewrite applied to stage <origin> (...)`.
`DKB_LQ_FOLD=0` turns the fold off (default 1). Setting `DKB_LQ_PATCH` also turns it off, so diagnostic runs see the
modules they ask for. The fold applies when the shader is loaded: a cached module is rebuilt from the cache each run, so no
cache needs clearing. The log shows `gfx_shader: light-quad fold applied to stage ...` or `... not applied ...; its draws
stay skipped`.

### Draw first-seen log and skip switches

For device faults whose draw is not a light quad (the Fort and Tower stages): three switches, all off by default, active only
on Bananza with an NVIDIA GPU (`vk_exec_draw.c`, `draw_diag`; the parsing is in `draw_filter.h`).

- `DKB_DRAW_FIRSTSEEN=1` logs `draw-firstseen: n=.. t=<world s> wall=<s since the first draw> vs=<origin> fs=<origin> prim=..
  attrs=.. streams=.. inst=.. loc9=..` the first time each (vertex origin, fragment origin) pair is drawn (`attrs`: inputs the
  vertex stage reads; `streams`: distinct vertex streams they come from; `inst` at least 1; `loc9`: reads Location 9), and
  once per world second `draw-firstseen: sec=N firstseen=K wall=..`, so the last new pair before a fault stands out. Each line
  is a separate write to the log, so it survives a device loss. At most 8192 pairs.
- `DKB_SKIP_ORIGINS=<hex prefixes>` (comma-separated, 1..16 hex digits each) skips every draw whose vertex or fragment origin
  starts with one of them, after the shaders are looked up and before the pipeline is built, and logs `draw-skip: origin ..`
  once per origin. A bad list is reported and ignored.
- `DKB_SKIP_BISECT=<mask>:<value>[,<mask>:<value>...]` (hex, up to 16 terms) skips the draws whose vertex origin satisfies
  `(origin & mask) == value` for any term; each value must lie inside its mask, and a malformed term ignores the whole list. Each second logs `draw-skip: sec=N bisect skipped=K`. For example `8000000000000000:0` skips the
  half of the vertex origins whose top bit is clear, and `c000000000000000:4000000000000000` one quarter.

- `DKB_SKIP_BISECT_FS=<mask>:<value>[,...]` is the same test on the fragment origin; with both set a draw is skipped when either
  matches. Each second logs `draw-skip: sec=N bisect-fs skipped=K` next to the vertex line.

`DKB_CAPTURE_ORIGINS=<hex prefixes>` (up to 64, 1..16 digits; combinable with `DKB_LQ_CAPTURE_VS`) makes the draw capture
(`lq_capture`, `DKB_LQ_CAPTURE_DIR`/`TAG`/`MAX`/`MAX_TOTAL`) work for any vertex stage, not only the light-quad family: instanced
draws, several streams with their divisors, no Location 9 input. It adds `fragment.spv` to the usual `vertex.spv`, `draw.txt`,
`support.bin`, `ubo-*.bin`, `stream-*.bin`, `ssbo-*.bin`, `index.bin`. CPU only; the draw is unchanged. Draws that
`DKB_SKIP_ORIGINS`/`DKB_SKIP_BISECT*` skip are not captured, so leave those unset while capturing. `tools/lq_vseval.py` runs
such a capture (instances, divisors and index buffers included); it does not model texture sampling, derivatives, atomics or
function calls.

`DKB_SHADER_DUMP=<folder>` is independent of these: it writes every stage the game creates as `<name>.ctl` and `<name>.bin`
when the program is made, so a run that faults still has its dump. The file names are the origins that the `vs=`/`fs=` fields of
these logs (and of `DKB_LQ_TRACE`) print. Dumps and logs are game
data and stay under `build/`.

### Locating a fault inside a vertex stage

With `DKB_LQ_PATCH_KEYS=<cache key prefix>` set and the module's key matching, the vertex kinds of `DKB_LQ_PATCH`
(`caps`, `cullvs`, `nullvs`, `passvs`, `flatten`, `notrig`, `noflags`, with `DKB_LQ_NOFLAGS_ONLY` / `DKB_LQ_AND3_CLASS`),
`DKB_LQ_VSMASK` and `DKB_LQ_VSCUT` apply to any vertex module, with or without an input at Location 9. Without
`DKB_LQ_PATCH_KEYS` they apply only to the light-quad vertex modules, as before. (`DKB_LQ_VSCUT` falls back to mask 0 on any
module that is not the captured Ground stage.) The default-on Ground fold is never widened this way.

`DKB_LQ_VSBLOCK=<a>-<b>[:t|:f]` halves a stage's branches. It numbers the module's selection constructs (`OpSelectionMerge`) from
0 in module order, and for the numbers a..b replaces the branch condition by a constant, so each such construct takes one
path; dead-branch elimination, block merging and dead-code elimination then remove the other path (and the code only it used).
Without a suffix a construct that has an arm straight to its merge block (an `if` without `else`) takes that way, which skips
the arm, and a construct with two real arms takes its false target. `:t` forces the true target and `:f` the false target of
every construct in the range. Nested constructs inside a forced one disappear with it. The patch log says how many constructs
the module has (`vsblock: 85 selection constructs in the module, ...`), so the range can be chosen by halving it. Outputs written
only on a removed path keep whatever the driver gives an unwritten output. Ignored while `DKB_LQ_VSMASK` is set.

`DKB_LQ_SELFIX=<a>[-<b>][,...]` (bit 14, with `DKB_LQ_PATCH_KEYS`) is the exact counterpart: each named selection construct that
is a triangle whose arm is one block of loads, pure arithmetic and stores to function-local variables runs both ways, every
store becoming a select of the new and the old value (`ldcfix`'s rewrite, without its `&3` condition test). The result is
bit-identical for every input. The log says `selfix: N data-dependent branches removed`.

For a stage with outputs at Locations 0..5 plus Position (Fort's `1ee77eac998beea5`, cache module `d8e2c12e875a`), the
`DKB_LQ_VSMASK` bits are: bit 0..5 keep the output at Location 0..5, bits 6..15 keep nothing (no such output), and bit 16 keeps
Position. Every other output is overwritten with null (Position with vec4(0, 0, 2, 1)) and the code that only fed it is removed.

### Slow-window evidence in the 60-frame perf lines

With `TOTK_TRACE=perf` (off by default; no change to behaviour) every 60 frames the log also has, next to the `perf: frame N:` lines:

- `perf: frame N: submit lock: S submits per frame, wait W ms per frame (max), held H ms per frame (max)`: how long submissions waited for the submit
  lock and how long they held it (the recording and the submit, i.e. the replay itself).
- `perf: frame N: process: page faults +F (per frame), working set ... (peak ...), private ...; system: available A of T MiB, commit ...`: a rise in
  page faults with the working set means first touches of new memory.
- `mem: write-watch timing frame N: C calls, T ms total (per call), slowest X ms (size, pages found, reset), K calls over 0.2 ms`: every
  `GetWriteWatch` call is timed (`plat_now_ns`, QueryPerformanceCounter on Windows); each of the first 20 calls over 0.2 ms in a window is also
  logged as `mem: write-watch call X ms: ...`.

- `perf: frame N: submit phases per frame (ms): pre …, slot (fence) …, record … [mirror begin …, usnap begin …, ops … (draws … for n, binds … for n), end …], post …, queue submit …; S submissions per frame`:
  where the time under the submit lock goes. `pre` is the work before the recording (dirty-page bookkeeping, command buffer begin), `slot` the wait for a free command
  slot (the GPU's fence), `record` the whole replay with the mirror and uniform snapshots at its start, the replayed ops (draws and binds timed one by one) and its
  end inside it, `post` what follows the recording (end of the buffer, uploads, the prologue) and `queue submit` vkQueueSubmit2 with the queue lock.

### Light-quad vertex fetch diagnostic

Set `DKB_LIGHT_QUAD_FETCH=1` (off by default) to log what the NVIDIA light-quad skip (`TOTK_FIX_VS_FAULT`) leaves out.
The skip stays active and the diagnostic submits nothing, so it runs from recorded state only. Each distinct draw shape
(vertex-stage origin, used inputs, attribute and stream state, draw kind; up to 256) is logged once as `light-fetch:`
records: a shape line (draw kind, instances, base instance, stream count, attributes that fell to the zero binding,
any divisor above one, whether vertex input is dynamic); one line per stream (rate, divisor, stride and stride modulo 4,
the raw resolved address and range, the available span, the bound range, the buffer end and allocation size, and
whether the zero buffer feeds it); the stream slots `bind_vertex_buffers` feeds from the zero buffer with the pipeline's
stride; and one line per attribute (offset, size, address modulo 4 and modulo the component size, stride below the
attribute end, the fetch end for the last instance and for one instance beyond, and overrun flags against the bound
range and the buffer). Device creation also logs the robustness features it enables. Divisors above one are treated as
one by the renderer. Indexed vertex-rate fetches and indirect draws need GPU data and are reported as unknown; a
host-visible index buffer is also snapshotted, which pending GPU writes can invalidate. The absence of an overrun report
does not establish GPU safety. Logs are game-derived and stay under `build/`.

## DKB_DELTA_SCAN (off by default)

`DKB_DELTA_SCAN=1` looks for the guest variable that carries Bananza's per-frame time step. It finds the game's
frame-rate object from the call to `nvnWindowSetPresentInterval`, then every present records the chosen interval
(object +0x88), the measured ticks (+0xe8) and the display interval. Every 600 presents it logs words of that object and
of the objects it points to that follow the interval, and (second scan, after 900 presents) every float of 12.5 to 60 ms
in main's data and the heap objects it points to, compared with interval/60, interval/30 and the display interval.
Nothing is written outside the log. Result on Canyon (WildLevelStage): only integer copies of the interval
turn up (about ten objects, as ints); no float equals interval/60, interval/30, the display interval or the measured
ticks in either scan. The step is probably derived from the integer interval inside the simulation, or kept in a
form (double, ticks, fixed point) these scans do not test.

`DKB_DELTA_PROBE=1` (patch set `deltaprobe`, off by default) counts, per guest caller and 60 presents, the calls of the framework's
measured-fps getter (main+0x138bb30, virtual slot 0xa8) and logs the fps the controller stores at G+0xe0 (main+0x1ead28). Canyon, 60 s:
the controller stored fps between 20 and 60 every 60 presents (60 calls per window, one per present); the getter was never called.

`DKB_DELTA_PROBE=1` also probes the update's frame context (`mov x1, sp` at main+0x1e9f10 and +0x1e9f28, before the calls into the
objects at G+0xf8 and G+0x170) and calls the measured-fps getter once from the host as a self-test. Canyon, 60 s: the getter stub works (the
self-test call is logged); no word of either 0x1c0-byte context follows interval/60, 1/interval or 60/interval, and none changes; the
only words that equal the interval are constant 1s. main+0x138b9f4 is the game's own frame limiter: for an interval of 2 or more it
spins until [G+0x88] * (a constant) / 60 s of ticks have passed since the frame start ([G+0xf0]).

Bananza's frame step (found with `DKB_DELTA_PROBE=1`): the controller (main+0x1eace8) stores fps = 60 / interval as an integer at its +0x50, and
the per-frame readers main+0x2da110 and main+0x34ec4c turn it into the step (interval / 60 s passed to the 0x68-stride updater main+0x2da5a0,
and the interval itself). Both equal 60 / interval on every present. Patch set `realtime` (`TOTK_REALTIME=1`, default off) replaces both with
the patch page's measured step; the probe at main+0x2da128 logs the step the game gets and the speed (sum of steps / wall time per 60 presents).

## DKB_CENSUS and DKB_DRAWSPLIT (off by default; runtime/census.c, build/dkb100-stage2a)

Measurement for BINDLESS_DESIGN.md phase 0 and R0; no behaviour change. Lines every 60 presents (perf.c calls census_frame), tag "census".
- `DKB_CENSUS=1`: per guest thread (TPIDR) recordings, ops, draws and the time between BeginRecording and EndRecording per frame, plus the draws
  inside CallCommands copies and whether the called recording was first seen in the current 60-present window (`census r0`); per frame the
  texture/sampler handle lookups, distinct and new handles, slot changes, uniform and storage range bindings and how often they differ from the
  slot's previous range, pushes, set allocations, pool creation and reuse, replay op counts (`census`); once the device's descriptor-indexing
  limits (`census limits`). The counters cost a hash lookup per binding, so fps drops a little while it is on.
- `DKB_DRAWSPLIT=1`: every 8th draw timed with the cycle counter in its parts: prepare, pipeline and dynamic state, bind_resources (signature
  check, buffers, textures, descriptor emit), vertex buffers, post, whole op (`census split`, ns per draw).
Limits of the numbers: a recording is "old" when its copy address was first seen in an earlier window (an address freed and reused counts as old);
draws inside called recordings are counted before the replay drops draws (empty scissor, pending pipeline), so they exceed the replayed draws.

## TOTK_BINDLESS_TEX (off by default; runtime/vk_bindless.c, bindless_table.h, tools/build_xlate_bindless.py)

Textures and samplers through one global descriptor-indexed table instead of descriptors per draw (BINDLESS_DESIGN.md phase 1).
- `TOTK_BINDLESS_TEX=1`: the shaders come from the bindless translator (`totk_xlate_bindless` next to the exe, or `build/xlate-bindless/`;
  `python tools/build_xlate_bindless.py` makes it; its identity differs from the default translator, so shaders have their own cache keys).
  Combined textures that are no arrays, not separate and not buffers are runtime arrays of combined image samplers in set 2, one binding per image
  class (dim | array 4 | multisample 8 | shadow 16), indexed by a slot read from push constants (vertex 8 slots at byte 0, fragment 32 slots at
  byte 64). The replay still resolves each handle on the CPU (tex_info and its memo, design option 1), enters the resolved (view, sampler,
  layout) in the table (a slot per distinct triple, written with update-after-bind, freed when its view is destroyed) and pushes the slots when they
  changed. All graphics layouts of the run have the table and both push ranges (one family: pipeline libraries need identical layouts).
  Needs descriptor indexing, update-after-bind for the table size, maxPushConstantsSize >= 192 and the graphics pipeline library path (stage-
  independent layouts); otherwise it logs why and stays off. Compute and geometry shaders keep ordinary bindings.
- `TOTK_BINDLESS_TEX=check`: the ordinary path, plus every texture use is entered in the table and compared with a lookup that skips the memo
  (`bindless check:` lines on a mismatch, counts in the `bindless <frame>:` line).
- `DKB_BL_DEBUG` bits: 1 no table bind, 2 no push constants (both break rendering), 4 log every table write, 16 log every pipeline layout made.
  The crash report prints the last two packets the recorder thread executed and its last BindDescriptorSets.

## TOTK_REPLAY_MT=<workers> (off by default; runtime/vk_exec_draw.c "TOTK_REPLAY_MT", vk_usnap.c, vk_memory.c vk_mem_use_ready)

Deferred fill of the bindless draw records on worker threads. Needs `TOTK_BINDLESS_TEX=1 TOTK_BINDLESS_BUF=1` (it logs and stays off otherwise);
0 or unset leaves the replay as it was.

- The walk thread still resolves in order everything that mutates shared state (c0, support, compiler constants, blit block, every storage buffer
  with its written-range bookkeeping and snapshot drops), reserves the record slot in transient memory, pushes its address at once and queues the
  draw in chunks of 256. Workers fill the unwritten uniform-buffer entries (slots 3-20) and the texture slots (tex_info plus table slot), each
  with its own texture memo and buffer cache. The walk joins (helping with the queue) before a part is submitted, before an ordered GPU read and at
  the end of the walk; draws whose range needs a mirror created or synced, or an unaligned copy, are flagged by the worker and resolved by the walk
  thread at the join.
- Snapshot drops are stamped (`vk_usnap.c` `dropped_at`): a draw queued at sequence s sees a copy dropped after s as intact, as it would have in
  order. Draws with a feedback-loop image bound, more than 32 textures or more than 18 uniform bindings in a stage stay on the serial path.
- `TOTK_TRACE=perf` and the log: `mt: per join ... us; draws deferred, same record, flagged entries, serial, chunks` every 600 joins.
- Not covered: storage buffers, c0 and the compiler constants stay on the walk thread; secondary command buffers per chunk are not used.

## TOTK_REPLAY_SPLIT=<threads> (off by default; runtime/vk_exec_draw.c "TOTK_REPLAY_SPLIT", vk_rec.c RecChunk, vk_usnap.c, vk_memory.c)

The walk itself split across threads (MT_DESIGN.md, version (ii)). Needs `TOTK_BINDLESS_TEX=1 TOTK_BINDLESS_BUF=1` and the recorder thread (it logs and
stays off otherwise); while on it replaces `TOTK_REPLAY_MT`. `TOTK_REPLAY_SPLIT_DRAWS=<n>` caps the draws of a chunk (default 1000; barriers end chunks first).

- The walk thread still walks the whole op stream and keeps the state (`vk_exec_split_bind`, `vk_exec_state_op`). Draws, binds and state ops go into a
  chunk: the op addresses plus a copy of `VkExec` at the chunk's first draw (the pass is begun by the walk thread before it). Anything else (SET_RT,
  clears, BARRIER, copies, counters, dispatches, indirect draws, uniform updates) closes the chunk and runs on the walk thread in order. Draws that need the
  walk thread run there too: render disabled, a pass without attachments, samples-passed counting, the indirect-reset program.
- A worker replays a chunk with the ordinary draw path on its copy of the state. B, RS, BS, DS, the shader/pipeline/texture memos and the descriptor
  pools are thread-local (its own execution context); the chunk starts as a new command buffer does. Its Vulkan calls go into a `RecChunk` (packet buffer,
  vk_rec.c) instead of the ring; the walk thread put an `R_Chunk` packet at the chunk's place in the ring when it closed the chunk and the recorder
  plays the chunk once the worker published it. So no secondary command buffers: the pass, barriers and layouts stay with the walk thread, and the
  recorder keeps one command buffer in order.
- Ordering of shared decisions: draws are numbered in the submission; `vk_usnap_find` and `vk_usnap_drop` work as of the thread's draw number
  (`vk_usnap_split_seq`, `dropped_at` stamps, `vk_usnap_gen` counts the drops a draw sees); snapshot addresses are not kept in the buffer cache. Written-range
  lists (`vk_mem_gpu_wrote`, the usnap write list) are per worker and handed over at the end of the chunk; mirror creation and sync take `vk_split_lock`
  (a worker that finds the mirror ready takes no lock). The inline-update ring (`UPD`) is read as of the chunk start.
- Fallbacks to the walk thread are counted in the `perf: split frame` line (every 60 frames with `TOTK_TRACE=perf`): serial draws by reason, the ops that
  cut chunks (BARRIER and SET_RT dominate), feedback events, refused alias copies, split-lock wait, run time per chunk, worker time and join wait per frame.
- Known gaps: (1) a feedback-loop restart in a worker cannot end the pass: the target is marked learned for the next pass and the draw gets a by-region
  barrier instead (4 events at start-up in Hill, none after); `fb_dirty` at a chunk start is assumed full. (2) an alias copy (`vk_exec_inline_begin`) in a worker
  is refused and the alias starts empty (0 seen). (3) a drop made by a draw on another worker (shader-written storage buffer over a snapshot uniform range)
  is seen only if it happened earlier in time than the lookup; drops are not verified against draw order (there is no verify mode). (4) traffic census
  counters and the slow-part notes skip worker draws. (5) the Hill indirect-reset draws (about 6 per frame) still submit and wait for the GPU on the walk thread
  and now also join the workers first: they are most of the remaining walk time there.
- Measurements (stage2k, tutorial cave warped, 90 s, `TOTK_TRACE=perf`, one run each, last 6 windows; `[busy machine]` flagged by the >1 GiB rule):

| stage | split | record ms | ops ms | chunks/frame (draws/chunk) | run ms/chunk | worker ms/frame | join wait ms | fps | near-black |
|---|---|---|---|---|---|---|---|---|---|
| Hill | off | 37.5 | 33.0 | - | - | - | - | 36.8 | 0.0% |
| Hill | 2 | 19.9 | 15.4 | 115 (192) | 0.19 | 22.2 | 6.8 | 40.4 | 0.0% |
| Hill | 4 | 16.9 | 12.2 | 117 (199) | 0.25 | 29.2 | 2.1 | 41.4 | 0.0% |
| Hill | 6 | 15.4-16.6 (one 23.4 window) | 11-12 | 113 (182) | 0.26-0.37 | 30-42 | 1-10 | 41.6 | 0.0% |
| Wild | off | 14.1 | 12.6 | - | - | - | - | 46.4 | 2.8% |
| Wild | 2 | 9.6 | 8.0 | 101 (80) | 0.11 | 10.7 | 3.6 | 48.6 | 2.9% |
| Wild | 4 | 7.8 | 6.4 | 100 (75) | 0.12 | 12.1 | 2.0 | 50.4 | 2.9% |
| Wild | 6 | 8.1 | 6.6 | 100 (74) | 0.14 | 13.5 | 1.9 | 49.5 | 2.9% |

  Scene check passed in every run, no `crash.txt`. fps moved less than the replay time because the frame is limited elsewhere (Hill: GPU about 12 ms, the
  indirect-reset waits, guest and present). Contention found on the way: the texture lock held per draw (23 ms/frame of waiting, fixed by locking only for a
  memo miss) and the split lock taken per draw for mirror use and written-range lists (32 ms/frame, fixed with per-worker lists).
