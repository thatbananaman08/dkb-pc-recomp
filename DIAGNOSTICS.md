# Developer diagnostics

This document describes the currently available diagnostics. The requested unified diagnostics system is not yet complete: structured event logging, configurable diagnostic levels, Vulkan draw labels, automatic crash bundles, synchronization/GPU-assisted validation controls, and a developer panel are not implemented here. Existing controls below are independent and opt-in.

Outputs can contain game-derived data, memory contents, and screenshots. Keep them under ignored build directories and do not commit them. Record the source revision (`git rev-parse HEAD`), GPU and driver, settings, game version, scene, save and reproduction steps alongside a capture; do not include keys or proprietary content in public reports.

## Vulkan frame capture

Install RenderDoc separately. Before launching the game in PowerShell, set `$env:TOTK_RDOC='500'` to capture presented frame 500, or use a comma-separated list (up to 64 frames). Set `TOTK_RDOC_DIR` for the output directory and `TOTK_RDOC_DLL` to override the library location. The runtime loads RenderDoc before Vulkan initialization and reports whether each capture completed. Frame 1 cannot be selected by this present-boundary mechanism. Captures may be large and are not automatically pruned. RenderDoc's unsupported vendor-extension option is used to expose external host memory; capture or replay may fail on a particular driver. Test without RenderDoc as well.

Loading the DLL alone does not register its Vulkan capture layer. On Windows with a portable RenderDoc distribution, its layer can be selected for one process with `VK_LAYER_PATH` set to the directory containing `renderdoc.json`, `VK_INSTANCE_LAYERS=VK_LAYER_RENDERDOC_Capture`, and `ENABLE_VULKAN_RENDERDOC_CAPTURE=1`. This does not guarantee a usable capture: RenderDoc 1.46 on the RTX 5060 / NVIDIA 617.42 exposed no `VK_EXT_external_memory_host` even with the unsupported-extension option enabled, so device selection failed before a frame could be captured. A capture-compatible memory import path or a RenderDoc setup exposing that extension is needed. Without the layer, the API may load while capture fails to start. Verify the file and replay rather than relying on API initialization.

For core validation, set `$env:TOTK_VK_VALIDATION='1'` with the Khronos validation layer installed. Validation warnings and errors go to the runtime log (the first 500 messages). It is expensive; unset the variable after the run. Synchronization and GPU-assisted validation are not exposed as separate options yet.

## Defective draw and shader evidence

A renderer-specific alternative is `$env:TOTK_DUMP='500'` and `$env:TOTK_DUMP_DIR='build/diagnostics/dump'`. It writes `passes.txt`, render target images and referenced SPIR-V files, subject to the caps documented at the top of `runtime/vk_dump.c`. The pass listing associates draws, pipeline/texture sets and shader hashes. `TOTK_DUMP_LOG` includes preceding frame listings. Buffer and extra texture snapshots are explicit additional options; see that file before enabling them. Dumps perform GPU copies and may be very large. They are not safe to initiate after device loss.

For ordered replay evidence set `DKB_DUMP_ORDERED=1` alongside `TOTK_DUMP`. `draws.txt` contains up to 16,384 emitted draw commands, in replay order, with frame, pass and per-pass ordinal, NVN operation kind, runtime pipeline/program identifiers, shader hashes and origins, target addresses and formats, and known descriptor handles and buffer ranges. Indirect arguments are GPU-written and are identified by address, not read speculatively. A runtime pipeline handle and a per-frame draw number are not stable across runs. A translated quad or line loop may be emitted as an indexed Vulkan draw; the operation field is the NVN replay operation, not a Vulkan event label. Skipped operations, original command-buffer offsets, Vulkan debug labels and error attribution are not recorded. The existing grouped listing can include more binding detail. The output contains game-derived identifiers and must stay private.

Set `DKB_DUMP_PASS=35-40` (or a single number) to copy attachments only for that inclusive range of pass and dispatch numbers. Sampled textures are omitted in this mode unless explicitly selected with `TOTK_DUMP_TEX`; shader files and logs still come from the existing dump. `images.txt` reports NaN and Inf channel counts for copied formats. The filter selects pass boundaries, not individual draws or pass entry. Pass numbers and scene state can change between runs; inspect `passes.txt` each time. Readbacks add synchronization and may perturb timing. They cannot establish the first invalid writer, especially where a target already contained invalid values at entry.

`DKB_DUMP_BOUNDARY=1` adds at most 32 snapshots of color target 0 before and after direct dispatches and immediately before beginning rendering, outside the rendering instance. Use it with a narrow `DKB_DUMP_PASS` range. `passes.txt` identifies each image, guest address, format, aspect, mip, layer and layout, or records why a snapshot was skipped; `images.txt` has the corresponding channel counts. Feedback-layout targets and unsupported subresources are skipped. Copies use GENERAL layout, an all-commands memory-write to copy-read dependency, and a copy-stage dependency to host reads and subsequent commands. Pass-end dumps also skip targets still in feedback layout. These copies alter synchronization and timing, so compare with an uninstrumented run and do not infer deterministic replay. This does not measure within a rendering instance.

For black terrain, capture a frame where the artifact is visible and inspect the terrain pass in `passes.txt`: compare the pipeline's shader hashes, bound textures and intermediate render targets with a known-good run. Identify the first pass where the expected terrain disappears; inspect its geometry stage and resource bindings before changing the translator. This is an investigation procedure, not a diagnosis of the defect. A RenderDoc capture, when supported, can supplement the pass dump with individual Vulkan draw state.

The independent fragment-output reporting fixture can be tested on the RTX 5060 without game content:

```powershell
.venv\Scripts\python tests\fragment_nonfinite_gpu_test.py --cc C:/msys64/ucrt64/bin/clang.exe --glslc C:/msys64/ucrt64/bin/glslc.exe --spirv-val C:/msys64/ucrt64/bin/spirv-val.exe --layer-path C:/msys64/ucrt64/bin --out build/test/fragment-gpu
```

It tests two synthetic ordinary descriptor-set arrangements with the Khronos validation layer, GPU output and report-buffer readback. It is not connected to game shader translation or replay; it does not instrument arbitrary SPIR-V, track components or exercise independent graphics pipeline library layouts. Successful fixture results do not identify a faulty game draw.

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

Request frames with the environment variables above, save the log, settings, revision and reproduction notes with outputs, and report missing or failed captures explicitly. Check tool exit codes and inspect the diff image and metrics. The existing controls do not yet provide a unified bundle, deterministic replay, stable draw identifiers, or guaranteed crash recovery.
