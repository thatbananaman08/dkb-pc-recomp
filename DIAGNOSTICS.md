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
