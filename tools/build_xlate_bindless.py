"""Build the bindless-texture shader translator: <out>/totk_xlate.exe (the default translator plus tools/xlate/bindless/*.patch).

The default translator (tools/build_xlate.py) is untouched, so its identity, and with it every cache key of the default
path, stays what it was. This build adds one option, a job field "bindless_tex": {"set": n, "push_offset": bytes}, with which
combined textures that are no arrays, not separate and not buffers are declared as runtime arrays in descriptor set n (binding =
texture class) and indexed by a slot read from the push constants (0001-bindless-sampled-textures.patch and 0003-bindless-buffers.patch on the vendored
Ryujinx sources, 0002-program-bindless-textures.patch on tools/xlate/Program.cs). Without the field its output is the default
translator's. Its identity file totk_xlate.id differs from the default's (it hashes the default identity, this script and the
bindless patches), so shaders it makes have their own keys; runtime/gfx_shader.c uses it only for TOTK_BINDLESS_TEX=1.

  python tools/build_xlate_bindless.py [--out DIR]     DIR defaults to build/xlate-bindless
"""
import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("build_xlate", ROOT / "tools" / "build_xlate.py")
bx = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bx)
BINDLESS = ROOT / "tools" / "xlate" / "bindless"


def apply(patch: Path, base: Path, directory: str) -> None:
    for extra in (["--check"], []):
        r = subprocess.run(["git", "apply", *extra, "--whitespace=nowarn", "--directory", directory, str(patch)],
                           cwd=base, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"patch does not apply: {patch.name}\n{r.stdout}{r.stderr}")
    print(f"applied {patch.name}")


def identity() -> str:
    h = hashlib.sha256()
    h.update(b"bindless\0" + bx.identity().encode())
    for f in [Path(__file__).resolve(), *sorted(BINDLESS.glob("*.patch"))]:
        h.update(f"{f.name}\0{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "xlate-bindless")
    a = parser.parse_args()
    build = a.out.resolve()
    bx.BUILD = build
    started = time.time()
    dotnet = shutil.which("dotnet")
    if not dotnet:
        sys.exit("missing: dotnet (.NET 10 SDK)")
    env = dict(os.environ)
    if bx.WIN:
        env["PATH"] = str(bx.VSWHERE_DIR) + os.pathsep + env["PATH"]
    build.mkdir(parents=True, exist_ok=True)
    bx.verify_vendor()
    src = bx.prepare_sources()                       # vendor copy with the default patches
    vendor_patches = [p for p in sorted(BINDLESS.glob("*.patch")) if "program" not in p.name]
    program_patches = [p for p in sorted(BINDLESS.glob("*.patch")) if "program" in p.name]
    for patch in vendor_patches:
        apply(patch, build, str(src.relative_to(build)))
    work = build / "xlate"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    for name in ("Program.cs", "totk_xlate.csproj"):
        shutil.copy2(bx.XLATE / name, work / name)
    for patch in program_patches:
        apply(patch, build, "xlate")
    out = build / "publish"
    shutil.rmtree(out, ignore_errors=True)
    cmd = [dotnet, "publish", str(work / "totk_xlate.csproj"), "-c", "Release", "-r", "win-x64" if bx.WIN else "linux-x64",
           "-p:PublishAot=true", "-p:InvariantGlobalization=true", f"-p:RyuSrc={src}", "-o", str(out)]
    if subprocess.run(cmd, env=env).returncode != 0:
        sys.exit("dotnet publish failed")
    exe = build / bx.EXE
    shutil.copy2(out / bx.EXE, exe)
    ident = identity()
    exe.with_suffix(".id").write_text(ident + "\n", encoding="ascii")
    print(f"translator identity {ident[:16]} -> {exe.with_suffix('.id')}")
    print(f"built {exe} ({exe.stat().st_size / 1e6:.1f} MB) in {time.time() - started:.0f} s")


if __name__ == "__main__":
    main()
