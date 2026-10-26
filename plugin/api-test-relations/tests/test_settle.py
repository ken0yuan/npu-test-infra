import json
import os
import sqlite3
import pytest
from api_test_relations.cli import save
from api_test_relations.settle import settle


def fixture(root):
    shards=root/'shards';shards.mkdir(parents=True)
    save(root/'manifest.json',{'catalog':str(root/'api-catalog.json'),'capture_pid':2147483000})
    save(root/'api-catalog.json',{'apis':[{'api_id':'a','qualified_name':'torch.add','language':'python','kind':'function'}]})
    save(root/'progress.json',{'state':'capture_incomplete','gaps':2})
    node='pytorch/test/test_one.py::test_one[param]'
    db=sqlite3.connect(shards/'2147483000.sqlite')
    db.executescript('CREATE TABLE meta(key TEXT,value TEXT); CREATE TABLE tests(nodeid TEXT,file TEXT); CREATE TABLE relations(nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,site TEXT); CREATE TABLE gaps(nodeid TEXT,phase TEXT,reason TEXT);')
    db.execute('INSERT INTO meta VALUES (?,?)',('session_finished','true'))
    db.execute('INSERT INTO tests VALUES (?,?)',(node,'pytorch/test/test_one.py'))
    db.execute('INSERT INTO gaps VALUES (?,?,?)',(node,'call','raw_gap_must_remain'))
    db.commit();db.close()
    (shards/'launches').mkdir();(shards/'receipts').mkdir()
    launch={'token':'late','nodeid':node,'phase':'call','pid':2147483001,'kind':'python'}
    save(shards/'launches/late.json',launch)
    save(shards/'receipts/late.json',{**launch,'state':'finished'})
    # Arrives after the initial pytest summary; new relation must be merged.
    db=sqlite3.connect(shards/'2147483001-late.sqlite')
    db.executescript('CREATE TABLE meta(key TEXT,value TEXT); CREATE TABLE tests(nodeid TEXT,file TEXT); CREATE TABLE relations(nodeid TEXT,phase TEXT,api_id TEXT,aliases TEXT,site TEXT);')
    db.execute('INSERT INTO meta VALUES (?,?)',('role','child'))
    db.execute('INSERT INTO tests VALUES (?,?)',(node,'pytorch/test/test_one.py'))
    db.execute('INSERT INTO relations VALUES (?,?,?,?,?)',(node,'call','a','["a"]','["pytorch/test/test_one.py", 9]'))
    db.commit();db.close()
    return root


def test_post_exit_rebuild_adds_late_relations_without_erasing_raw_gaps(tmp_path):
    source=fixture(tmp_path/'run');before={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    output=tmp_path/'settled';result=settle(source,output)
    assert result['relations']==1 and result['gaps']==1
    with sqlite3.connect(output/'relations.sqlite') as db:
        assert db.execute('SELECT state FROM processes').fetchone()==('acknowledged_complete',)
        assert db.execute('SELECT reason FROM gaps').fetchone()==('raw_gap_must_remain',)
    assert before=={str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
    assert (output/'settlement.json').exists()


def test_settle_refuses_live_producer_and_overwrite(tmp_path):
    source=fixture(tmp_path/'run')
    save(source/'manifest.json',{'capture_pid':os.getpid()})
    with pytest.raises(RuntimeError,match='must exit'):settle(source,tmp_path/'new')
    assert not (tmp_path/'new').exists()
    with pytest.raises(ValueError,match='outside'):settle(source,source/'new')
    other=tmp_path/'other';other.mkdir()
    with pytest.raises(ValueError,match='must not exist'):settle(source,other)


def test_empty_child_database_is_a_declared_gap(tmp_path):
    source=fixture(tmp_path/'run')
    sqlite3.connect(source/'shards/2147483002-empty.sqlite').close()
    result=settle(source,tmp_path/'new')
    assert result['gaps']==2
    with sqlite3.connect(tmp_path/'new/relations.sqlite') as db:
        assert ('shard_without_metadata:2147483002-empty.sqlite',) in db.execute('SELECT reason FROM gaps').fetchall()
