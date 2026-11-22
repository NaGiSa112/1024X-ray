"""Workarounds for two Windows file-permission quirks of this machine.

1. `tempfile.mkdtemp()` creates temporary directories with mode 0o700 (see
   Lib/tempfile.py).  On this machine such a directory ends up with a DACL the
   running process cannot write into -- and cannot even read back -- so every
   pip / ultralytics / torch call that touches a temporary directory dies with
   `PermissionError [Errno 13] Permission denied`.
   Creating the same directory with mode 0o777 inherits the workspace
   permissions and works, so `tempfile.mkdtemp` is re-implemented below with
   0o777.  That is the only difference from the stdlib version.

2. Python cannot create files in %APPDATA%\\Ultralytics on this machine
   (PowerShell can, Python gets PermissionError -- same class of problem), and
   ultralytics writes its settings and its Arial.ttf font cache there on first
   import.  So YOLO_CONFIG_DIR defaults to the project directory, which works.

Harmless elsewhere: CPython ignores the mode argument on Windows anyway, and on
POSIX umask decides the real permissions; YOLO_CONFIG_DIR is only a default and
an explicitly exported value still wins.  Set DSH_NO_TEMPFILE_SHIM=1 to disable
this patch.
"""

import os as _os
import sys as _sys

# --- 2. keep ultralytics' config/font cache inside the project -------------
if _sys.platform == "win32":
    _os.environ.setdefault("YOLO_CONFIG_DIR", r"E:\X光\.ultralytics")

# --- 1. tempfile.mkdtemp with 0o777 ---------------------------------------
if _os.environ.get("DSH_NO_TEMPFILE_SHIM") != "1" and _sys.platform == "win32":
    import tempfile as _tf

    def _mkdtemp_0777(suffix=None, prefix=None, dir=None):
        """Same as tempfile.mkdtemp but creates the directory with 0o777."""
        prefix, suffix, dir, output_type = _tf._sanitize_params(prefix, suffix, dir)

        names = _tf._get_candidate_names()
        if output_type is bytes:
            names = map(_os.fsencode, names)

        for _ in range(_tf.TMP_MAX):
            name = next(names)
            path = _os.path.join(dir, prefix + name + suffix)
            _sys.audit("tempfile.mkdtemp", path)
            try:
                _os.mkdir(path, 0o777)   # <- the only change: 0o700 -> 0o777
            except FileExistsError:
                continue
            except PermissionError:      # pragma: no cover - not expected
                if _os.name == "nt" and _os.path.isdir(dir) and _os.access(dir, _os.W_OK):
                    continue
                raise
            return path
        raise FileExistsError(_tf._errno.EEXIST, "No usable temporary directory name found")

    _tf.mkdtemp = _mkdtemp_0777
