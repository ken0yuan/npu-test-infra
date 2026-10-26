"""Check positive and negative test/API attribution for the bundled NPU sample."""
import argparse
import json
from pathlib import Path
import sqlite3

p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, required=True)
args = p.parse_args()
db = sqlite3.connect((args.out.resolve() / 'relations.sqlite').as_uri() + '?mode=ro', uri=True)
try:
    rows = {(n, name) for n, name in db.execute("SELECT nodeid,name FROM relations JOIN apis USING(api_id) WHERE phase='call' AND name IN ('torch.add','torch.Tensor.fill_')")}
    prefix = 'pytorch/examples/test_ascend_smoke.py::'
    expected = {(prefix + 'test_npu_add[size2]', 'torch.add'), (prefix + 'test_npu_add[size3]', 'torch.add'), (prefix + 'test_npu_fill', 'torch.Tensor.fill_')}
    # fill_ may legitimately occur inside zeros; don't reject transitive helper calls.
    missing = expected - rows
    unexpected_add = {(node, api) for node, api in rows if api == 'torch.add' and node == prefix + 'test_npu_fill'}
    gaps = list(db.execute('SELECT nodeid,phase,reason FROM gaps'))
    result = {'missing_expected_relations': sorted(missing), 'unexpected_add_in_fill': sorted(unexpected_add),
              'gaps': gaps, 'note': 'No claim of exhaustive coverage beyond this sample'}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if missing or unexpected_add or gaps: raise SystemExit(1)
finally:
    db.close()
