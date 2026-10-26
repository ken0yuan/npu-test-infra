"""Check six expected relations and ownership after the migration smoke run."""
import json
from pathlib import Path
import sqlite3
import sys

root = Path(sys.argv[1]).resolve()
prefix = 'pytorch/examples/migration/test_cpu_mapping.py::'
expected = {
    (prefix + 'test_fill[one]', 'python:torch.Tensor.fill_'),
    (prefix + 'test_fill[two]', 'python:torch.Tensor.fill_'),
    (prefix + 'test_native_profile', 'python:torch.add'),
    (prefix + 'test_native_profile', 'python:torch.mul'),
    (prefix + 'test_thread', 'python:torch.sub'),
    (prefix + 'test_child', 'python:torch.neg'),
}
with sqlite3.connect((root / 'relations.sqlite').as_uri() + '?mode=ro', uri=True) as db:
    observed = set(db.execute("SELECT nodeid,api_id FROM relations WHERE phase='call'"))
    names = {api for node, api in expected}
    relevant = {(node, api) for node, api in observed if api in names}
    missing = sorted(expected - relevant)
    extra = sorted(relevant - expected)
    gaps = list(db.execute('SELECT nodeid,phase,reason FROM gaps'))
    processes = list(db.execute('SELECT nodeid,state FROM processes'))
result = {'expected_edges': len(expected), 'missing': missing, 'unexpected_owners': extra,
          'gaps': gaps, 'processes': processes, 'scope': 'CPU migration smoke only; not proof of full coverage'}
print(json.dumps(result, ensure_ascii=False, indent=2))
assert not missing and not extra and not gaps, 'Migration check needs investigation; keep the full evidence directory'
assert any(node == prefix + 'test_child' and state == 'acknowledged_complete' for node, state in processes), 'Child completion receipt missing'
