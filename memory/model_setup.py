"""Fetch the free local embedding model, and prove it is the right bytes.

The weights are not in the repository. 23MB of binary does not belong in
version control, so a fresh clone has to download them - but "download and hope"
is how a machine ends up running a model nobody can name, on weights that
silently differ from the ones every threshold was calibrated against. So the
expected size and SHA-256 of each file is recorded here, and a mismatch is a
loud failure rather than a subtly different retrieval score.

Deliberately an explicit command, never an implicit startup step: quietly
spending someone's bandwidth and disk on a 23MB download the first time they run
a text editor is not a kindness.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import urllib.request

#: Where the weights live on Hugging Face. `main` may move, which is precisely
#: why the hashes below exist.
REPOSITORY = "Xenova/all-MiniLM-L6-v2"
BASE_URL = f"https://huggingface.co/{REPOSITORY}/resolve/main"

#: local filename -> (path in the repository, size in bytes, sha256)
#:
#: The ONNX graph is published in an `onnx/` subfolder while the tokenizer sits
#: at the top level, so the remote path and the local name are kept separate -
#: flattening both to one field is what produced a 404 here once already.
#:
#: Sizes are checked before the hash so a truncated download reports itself,
#: and the hash is checked after so a wrong-yet-plausible file cannot pass.
FILES = {
    "config.json": (
        "config.json",
        650,
        "7135149f7cffa1a573466c6e4d8423ed73b62fd2332c575bf738a0d033f70df7",
    ),
    "tokenizer.json": (
        "tokenizer.json",
        711661,
        "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    ),
    "model_quantized.onnx": (
        "onnx/model_quantized.onnx",
        22972370,
        "afdb6f1a0e45b715d0bb9b11772f032c399babd23bfc31fed1c170afc848bdb1",
    ),
}

#: Total download, for the message shown before anything is fetched.
TOTAL_BYTES = sum(size for _remote, size, _digest in FILES.values())

LICENSE = "Apache-2.0"


def model_dir(root=None):
    """Where the weights are expected, honouring `RETAIN_LOCAL_MODEL_DIR`."""
    if root is not None:
        return os.path.join(root, "all-MiniLM-L6-v2")
    configured = (os.environ.get("RETAIN_LOCAL_MODEL_DIR") or "").strip()
    if configured:
        return configured
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models",
        "all-MiniLM-L6-v2",
    )


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(directory):
    """Return the list of problems with `directory`; empty means it is good.

    A missing file is a problem, not an error, so callers can report
    "not installed" and "installed but wrong" differently.
    """
    problems = []
    for name, (_remote, size, expected) in FILES.items():
        path = os.path.join(directory, name)
        if not os.exists(path):
            problems.append(f"{name}: missing")
            continue
        actual_size = os.path.getsize(path)
        if actual_size != size:
            problems.append(
                f"{name}: {actual_size} bytes, expected {size} "
                f"(truncated or wrong file)"
            )
            continue
        actual = _sha256(path)
        if actual != expected:
            problems.append(
                f"{name}: sha256 {actual[:16]}..., expected {expected[:16]}... "
                f"(the upstream file has changed since these thresholds were "
                f"calibrated)"
            )
    return problems


def is_installed(directory=None):
    return not verify(directory or model_dir())


def _download(remote, destination):
    """Fetch one file, atomically.

    The bytes land in a temporary file in the destination directory and are
    renamed into place only once complete, so an interrupted download cannot
    leave a half-written model that later looks installed.
    """
    url = f"{BASE_URL}/{remote}"
    handle, temporary = tempfile.mkstemp(dir=os.path.dirname(destination),
                                         prefix=os.path.basename(destination) + ".")
    os.close(handle)
    try:
        with urllib.request.urlopen(url) as response, open(temporary, "wb") as out:
            shutil.copyfileobj(response, out, 1024 * 256)
        os.replace(temporary, destination)
    except BaseException:
        # Includes KeyboardInterrupt: a half-downloaded file left behind would
        # be picked up as installed on the next run.
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def install(directory=None, log=None):
    """Download and verify the model. Returns True if it is ready to use.

    `log` receives progress lines; the default writes to stdout.
    """
    directory = directory or model_dir()
    say = log or (lambda line: print(line))

    problems = verify(directory)
    if not problems:
        say(f"already installed: {directory}")
        return True

    broken = {problem.split(":", 1)[0] for problem in problems}
    if broken:
        say(f"fetching {len(broken)} of {len(FILES)} file(s) into {directory}")

    os.makedirs(directory, exist_ok=True)
    for name, (remote, _size, _digest) in FILES.items():
        # Only fetch what is actually absent or unusable. Re-downloading 23MB
        # of perfectly good weights because a 650-byte config was clobbered
        # would be a poor look on a slow connection.
        if name not in broken:
            say(f"  {name}: already present")
            continue
        _download(remote, os.path.join(directory, name))
        say(f"  {name}: downloaded")

    problems = verify(directory)
    if problems:
        for problem in problems:
            say(f"FAILED {problem}")
        raise RuntimeError(
            "the downloaded model did not match the expected files; "
            "the embedding thresholds would not apply to it"
        )

    say(f"verified {len(FILES)} files against recorded sha256")
    return True


def runtime_available():
    """Whether the code needed to *run* the model is importable.

    Kept separate from the weights: a machine can have the files and still lack
    onnxruntime, and the two problems need different fixes.
    """
    missing = []
    for module in ("onnxruntime", "tokenizers"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    return not missing, missing


RUNTIME_HINT = (
    "pip install onnxruntime tokenizers"
)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in {"-h", "--help"}:
        print(__doc__.strip())
        print(f"\nusage: {sys.executable} -m memory.model_setup [--check]")
        return 0

    directory = model_dir()
    ok, missing = runtime_available()

    if "--check" in argv:
        problems = verify(directory)
        if problems:
            print(f"not ready ({directory}):")
            for problem in problems:
                print(f"  {problem}")
            return 1
        print(f"ready: {directory}")
        if not ok:
            print(f"but the runtime is missing {', '.join(missing)}: {RUNTIME_HINT}")
            return 1
        return 0

    say = lambda line: print(line)  # noqa: E731 - local shorthand
    if not ok:
        say(f"runtime not installed: missing {', '.join(missing)}")
        say(f"  fix: {RUNTIME_HINT}")

    problems = verify(directory)
    if problems:
        say(
            f"{REPOSITORY} ({LICENSE}): {TOTAL_BYTES / 1e6:.0f}MB total, "
            f"fetching only what is missing or unusable into {directory}"
        )

    try:
        install(directory, log=say)
    except Exception as exc:  # noqa: BLE001 - report, do not traceback
        say(f"download failed: {exc}")
        return 1

    if not ok:
        say("weights are in place, but semantic search stays on the hashing "
            f"embedder until the runtime is installed: {RUNTIME_HINT}")
    else:
        say("done. semantic search is now local, private, and free.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
