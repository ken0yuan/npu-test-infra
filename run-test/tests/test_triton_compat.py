import pytest

from run_test.triton_compat import patch_cpp_standard


def test_cpp20_launcher_patch_is_reversible_and_idempotent(tmp_path):
    header = tmp_path / 'ATen.h'
    header.write_text('#error C++20 or later compatible compiler is required to use ATen.\n')
    builder = tmp_path / 'utils.py'
    original = b'cc_cmd += ["-std=c++17", "-shared"]\n'
    builder.write_bytes(original)
    result = patch_cpp_standard(header, builder)
    assert result['state'] == 'patched'
    assert builder.read_bytes() == original.replace(b'c++17', b'c++20')
    assert next(tmp_path.glob('utils.py.cxx17-*.bak')).read_bytes() == original
    assert patch_cpp_standard(header, builder)['state'] == 'already_compatible'


def test_cpp17_torch_needs_no_patch(tmp_path):
    header = tmp_path / 'ATen.h';header.write_text('#error C++17 or later is required\n')
    builder = tmp_path / 'utils.py';builder.write_bytes(b'"-std=c++17"\n')
    assert patch_cpp_standard(header, builder)['state'] == 'not_needed'
    assert builder.read_bytes() == b'"-std=c++17"\n'
    assert not list(tmp_path.glob('*.bak'))


@pytest.mark.parametrize('content', [b'flags_from_vendor()', b'"-std=c++17", "-std=c++17"'])
def test_unrecognized_build_logic_is_preserved(tmp_path, content):
    header = tmp_path / 'ATen.h';header.write_text('C++20 or later')
    builder = tmp_path / 'utils.py';builder.write_bytes(content)
    with pytest.raises(RuntimeError, match='refusing to patch'):
        patch_cpp_standard(header, builder)
    assert builder.read_bytes() == content
