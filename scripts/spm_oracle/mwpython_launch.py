#!/Users/mhough/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none/bin/python3.12
"""Direct mwpython launch with -mlstartup options (bypasses matlab_runtime.cli). Usage: mw_direct.py script.py [args]"""
import os, sys, json, subprocess
BASE = "/Users/mhough/.local/share/uv/python/cpython-3.12.13-macos-aarch64-none"
MCR = "/Applications/MATLAB/MATLAB_Runtime/R2025b"
site = "/Users/mhough/Workspace/data/spm26_standalone/.venv-spm/lib/python3.12/site-packages"
env = dict(os.environ)
env["PYTHONHOME"] = BASE
env["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join([".", f"{MCR}/runtime/maca64", f"{MCR}/bin/maca64", f"{MCR}/sys/os/maca64", f"{BASE}/lib", "/usr/local/lib", "/usr/lib"])
env["PYTHONPATH"] = ":".join([site] + [p for p in sys.path if p])
env.setdefault("MCR_CACHE_ROOT", "/Users/mhough/Workspace/data/spm26_standalone/mcr_cache")
argv = json.dumps(sys.argv[1:])
code = "import sys, runpy, json; sys.argv = json.loads(%r); runpy.run_path(sys.argv[0], run_name='__main__')" % argv
cmd = [f"{MCR}/bin/maca64/mwpython.app/Contents/MacOS/mwpython", "-mlstartup", "-nojvm,-nodisplay", "-c", code, "-mwpythonver", "3.12"]
p = subprocess.run(cmd, env=env)
print("mwpython returned", p.returncode, file=sys.stderr)
sys.exit(p.returncode)
