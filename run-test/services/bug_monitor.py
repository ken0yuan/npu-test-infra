"""Rebuilt terminal-task monitor. Evidence stays local; no external publishing."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'run-test/src'))
from run_test.monitoring import main
if __name__=='__main__':
    raise SystemExit(main(['bugs',*sys.argv[1:]]))
