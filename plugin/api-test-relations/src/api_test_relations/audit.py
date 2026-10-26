"""Read-only invariants. A clean result does not assert zero omissions."""
import sqlite3
from .cli import read, digest


def audit(root):
    manifest = read(root / 'manifest.json')
    path = root / ('relations-published.sqlite' if (root / 'storage.json').exists() else 'relations.sqlite')
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        db.execute('BEGIN')
        scalar = lambda sql: db.execute(sql).fetchone()[0]
        issues = {
            'relations_without_api': scalar('SELECT count(*) FROM relations r LEFT JOIN apis a USING(api_id) WHERE a.api_id IS NULL'),
            'relations_without_test': scalar('SELECT count(*) FROM relations r LEFT JOIN tests t USING(job,nodeid) WHERE t.nodeid IS NULL'),
            'invalid_test_paths': scalar("SELECT count(*) FROM tests WHERE file NOT LIKE 'pytorch/%' OR nodeid NOT LIKE 'pytorch/%'"),
        }
        gaps = dict(db.execute('SELECT reason,count(*) FROM gaps GROUP BY reason'))
        return {'structural_issues': issues, 'catalog_count': scalar('SELECT count(*) FROM apis'),
                'catalog_matches_manifest': scalar('SELECT count(*) FROM apis') == manifest['catalog_count'],
                'catalog_digest_matches': digest(manifest['catalog']) == manifest['catalog_sha256'],
                'jobs': dict(db.execute('SELECT state,count(*) FROM jobs GROUP BY state')),
                'gap_count': sum(gaps.values()), 'gap_examples': list(gaps.items())[:20],
                'unobserved_does_not_mean_uncalled': True, 'proves_zero_omissions': False}
    finally:
        db.close()
