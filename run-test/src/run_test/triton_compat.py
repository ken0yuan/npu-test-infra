"""Version-scoped Triton-Ascend launcher compatibility for PyTorch main."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import stat
import tempfile


def patch_cpp_standard(torch_header, build_file):
    """Match the launcher compiler to ATen's explicit C++20 requirement."""
    if 'C++20 or later' not in Path(torch_header).read_text(encoding='utf-8'):
        return {'state': 'not_needed'}
    path = Path(build_file)
    before = path.read_bytes()
    old, new = b'"-std=c++17"', b'"-std=c++20"'
    if before.count(old) == 0 and before.count(new) == 1:
        return {'state': 'already_compatible', 'sha256': hashlib.sha256(before).hexdigest()}
    if before.count(old) != 1 or new in before:
        raise RuntimeError('Unrecognized Triton-Ascend 3.6.0 compiler flags; refusing to patch')
    digest = hashlib.sha256(before).hexdigest()
    backup = path.with_name(path.name + '.cxx17-' + digest[:12] + '.bak')
    if backup.exists():
        if backup.read_bytes() != before:
            raise RuntimeError('Triton compiler backup differs; refusing to overwrite')
    else:
        with backup.open('xb') as stream:
            stream.write(before)
    after = before.replace(old, new)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(after)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.chmod(temporary, stat.S_IMODE(path.stat().st_mode))
        if path.read_bytes() != before:
            raise RuntimeError('Triton compiler changed during compatibility check')
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return {'state': 'patched', 'before_sha256': digest,
            'sha256': hashlib.sha256(after).hexdigest(), 'backup': str(backup)}


def ensure_compatibility():
    # Inspect installed metadata and headers without importing torch or
    # registering an accelerator backend in this maintenance subprocess.
    try:
        ascend = importlib.metadata.distribution('triton-ascend')
        torch = importlib.metadata.distribution('torch')
    except importlib.metadata.PackageNotFoundError:
        return {'state': 'not_installed'}
    if ascend.version != '3.6.0':
        return {'state': 'other_version', 'version': ascend.version}
    return patch_cpp_standard(torch.locate_file('torch/include/ATen/ATen.h'),
                              ascend.locate_file('triton/backends/ascend/utils.py'))


if __name__ == '__main__':
    print(json.dumps(ensure_compatibility(), sort_keys=True))
