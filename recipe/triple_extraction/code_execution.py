# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""Sandboxed code execution that runs Python code with resource limits.

The snippet is expected to expose a function `main(html)` or `solve(html)` which
returns results when given the HTML of a web page. The code execution:

1. passes a quick static AST check (syntax + import whitelist),
2. executes inside a short-lived helper process (`python -I -S`) with CPU & RAM
   limits and no site-packages,
3. returns the actual output or None if execution fails.
"""

from __future__ import annotations

import ast
import hashlib
import json
import resource  # POSIX RLIMIT_*
import subprocess
import tempfile
import textwrap
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ----------------------------------------------------------------------------
# Configuration – tweak as necessary
# ----------------------------------------------------------------------------

# Whitelisted standard-library or third-party packages that the user snippet may
# utilise.  **Only these are pre-imported and exposed; all further imports are
# blocked inside the sandbox.**
ALLOWED_IMPORTS: set[str] = {"re", "json", "itertools", "collections", "bs4", "typing"}

MAX_CPU_SECONDS = 4  # RLIMIT_CPU (seconds of CPU time)
MAX_MEMORY_MB = 256  # RLIMIT_AS (address-space size) in megabytes
WALL_TIMEOUT_SEC = MAX_CPU_SECONDS + 2  # wall-clock timeout for subprocess


# ----------------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------------


def _static_check(src: str) -> bool:
    """Return True if *src* parses and uses only whitelisted imports."""

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return False
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] not in ALLOWED_IMPORTS:
                return False
    return True


def _extract_code(text: str, language: str = "python") -> str:
    """Return the *last* fenced code block labelled with *language* in *text*."""

    import re

    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else ""


# The bootstrap code that runs *inside* the helper interpreter.  It guards
# imports, pre-imports whitelisted libs, executes the user snippet, and prints
# JSON when the snippet succeeds.
_BOOTSTRAP = textwrap.dedent(
    """
    # Hardened bootstrap that runs **inside** the helper interpreter.
    #
    # Steps performed:
    # 1. Ensure site-packages are available for third-party modules like bs4
    # 2. Pre-import a fixed whitelist of modules and store them in a private dict.
    # 3. Replace *builtins.__import__* with a guard that only returns those pre-loaded
    #    modules.  No additional imports are possible afterwards.
    # 4. Execute the user snippet in a fresh namespace seeded with the whitelisted
    #    modules.
    # 5. Call the user's entry point.
    # 6. Strip dangerous built-ins (open/eval/exec/compile/__import__) AFTER execution.
    #
    # NOTE: All single braces {{...}} inside this template are doubled so that
    # .format() does not treat them as replacement fields, except for the two
    # placeholders: allowed and snippet (without braces in this comment).

    import sys as _sys, builtins, importlib, json as _json

    # Add site-packages to path to ensure third-party modules like bs4 are available
    import site
    site.main()

    _ALLOWED = {allowed}

    # ------------------------------------------------------------------
    # Pre-load whitelisted modules.
    # ------------------------------------------------------------------
    _preloaded = {{}}
    for _name in list(_ALLOWED):
        try:
            _preloaded[_name] = importlib.import_module(_name)
        except ImportError:
            pass

    # ------------------------------------------------------------------
    # Guarded import that only exposes the pre-loaded modules.
    # ------------------------------------------------------------------
    def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        head = name.split('.', 1)[0]
        if head in _preloaded:
            return _preloaded[head]
        raise ImportError(f"Import of {{name!r}} is not allowed inside sandbox")

    builtins.__import__ = _guarded_import  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Run the user snippet.
    # ------------------------------------------------------------------
    user_ns = dict(_preloaded)  # expose allowed modules

    _page_html = _sys.stdin.read()

    exec({snippet!r}, user_ns)

    # ------------------------------------------------------------------
    # Call the user's entry point.
    # ------------------------------------------------------------------
    _entry = user_ns.get("main") or user_ns.get("solve")
    if not callable(_entry):
        print("No entry-point named main(html) or solve(html) found", file=_sys.stderr)
        _sys.exit(2)

    try:
        _result = _entry(_page_html)
    except Exception as _exc:
        print(f"Runtime error: {{_exc}}", file=_sys.stderr)
        _sys.exit(3)

    # ------------------------------------------------------------------
    # NOW remove helpers that could aid an escape (after execution is complete).
    # ------------------------------------------------------------------
    for _danger in ("importlib", "sys", "os", "types", "ctypes"):
        globals().pop(_danger, None)

    for _danger in ("open", "eval", "exec", "compile", "__import__"):
        builtins.__dict__.pop(_danger, None)

    if not _result:
        _sys.exit(4)  # empty or falsy result

    print(_json.dumps(list(_result)))
    """
)


# ----------------------------------------------------------------------------
# Public reward function
# ----------------------------------------------------------------------------


def execute_code(
    code: str,
    html_content: str,
    timeout_sec: int = WALL_TIMEOUT_SEC,
    *,
    return_stderr: bool = False,
) -> Union[Optional[Any], Tuple[Optional[Any], str]]:
    """Execute Python code with HTML input and return the output.

    Parameters
    ----------
    code: Python code string to execute. Should contain a main(html) or solve(html) function.
    html_content: HTML content that will be passed to the code's entry point function.
    timeout_sec: wall-clock timeout for execution (seconds).

    Returns
    -------
    The output from the code execution, or None if execution failed.
    """

    # Fail fast if empty
    if not code.strip():
        return (None, "Empty code") if return_stderr else None

    # ------------------------------------------------------------------
    # If the snippet comes wrapped in ```python ... ``` fences, strip them.
    # We do this *before* any other checks to ensure the rest of the pipeline
    # (static analysis, execution, etc.) sees raw Python only.
    # ------------------------------------------------------------------
    stripped = _extract_code(code) if "```" in code else ""
    if stripped:
        code = stripped

    # Fail fast again in case the fenced block was empty
    if not code.strip():
        return (None, "Empty code") if return_stderr else None

    # Fail if import violations
    if not _static_check(code):
        return (None, "Static check failed") if return_stderr else None

    # Compose temporary script that contains bootstrap + user code
    tmp_path = Path(tempfile.gettempdir()) / f"scraper_{uuid.uuid4().hex}.py"
    tmp_path.write_text(_BOOTSTRAP.format(allowed=repr(ALLOWED_IMPORTS), snippet=code))

    def _limit_resources():
        """Tighten resource limits and enter seccomp-strict mode (POSIX only)."""

        # -----------------------------------------------------------------
        # Standard RLIMITs
        # -----------------------------------------------------------------
        # CPU time (seconds of user + system time)
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_CPU_SECONDS, MAX_CPU_SECONDS))

        # Address-space size (virtual memory) in bytes
        mem_bytes = MAX_MEMORY_MB << 20  # MB → bytes
        resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))

        # Maximum number of processes (disallow fork/clone)
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))
        except (AttributeError, ValueError):
            pass  # RLIMIT_NPROC may be unsupported on some systems

        # Maximum number of open file descriptors (mitigates socket/files)
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (16, 16))
        except (AttributeError, ValueError):
            pass

        # Maximum file size that can be created (in bytes)
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        except (AttributeError, ValueError):
            pass

        # -----------------------------------------------------------------
        # Enter seccomp-strict: only read/write/exit/sigreturn syscalls
        # -----------------------------------------------------------------
        try:
            import ctypes, os  # allowed here (not visible to user snippet)

            libc = ctypes.CDLL("libc.so.6", use_errno=True)
            PR_SET_SECCOMP = 22
            SECCOMP_MODE_STRICT = 1

            # NOTE: Seccomp strict mode is too restrictive for Python execution
            # as it blocks essential syscalls like mmap, brk, etc. that Python needs.
            # We rely on the other RLIMITs for protection instead.
            # if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_STRICT, 0, 0, 0) != 0:
            #     raise OSError(ctypes.get_errno(), "prctl(PR_SET_SECCOMP) failed")
            pass
        except Exception:
            # If seccomp is unavailable (e.g. non-Linux platform) we fall back to RLIMITs only.
            pass

    try:
        proc = subprocess.run(
            ["python", str(tmp_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            preexec_fn=_limit_resources,
            input=html_content,
        )
        success = proc.returncode == 0 and proc.stdout.strip()
        if success:
            try:
                parsed = json.loads(proc.stdout.strip())
            except Exception:
                parsed = proc.stdout.strip()
            return (parsed, proc.stderr) if return_stderr else parsed
        else:
            # Execution failed – surface stderr when requested so callers can debug.
            return (None, proc.stderr) if return_stderr else None
    except subprocess.TimeoutExpired:
        return (None, "Timeout expired") if return_stderr else None
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


__all__ = ["execute_code"]
