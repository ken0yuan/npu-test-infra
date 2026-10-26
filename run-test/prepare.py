#!/usr/bin/env python3
"""Generate host-specific settings without importing credentials or starting services."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

def main():
    if sys.platform != "linux" or sys.version_info < (3,10):
        raise SystemExit("请在 Linux 虚拟机内使用 Python 3.10 或更新版本执行")
    parser=argparse.ArgumentParser()
    parser.add_argument("--obsutil", help="Linux ARM64 obsutil 的绝对路径")
    parser.add_argument("--test-python", help="专用测试环境的 Python 解释器路径")
    parser.add_argument("--device",choices=["cpu","accelerator"],default="cpu")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    settings=root/"settings.json"
    if settings.exists():
        print("保留已有 settings.json；如需调整请编辑该文件")
    else:
        cfg=json.loads((root/"settings.example.json").read_text())
        cfg["data_root"]=str(root/"data/runtime")
        cfg["obs"]["downloader"]["bin"]=str(Path(args.obsutil).expanduser().resolve()) if args.obsutil else shutil.which("obsutil") or "obsutil"
        cfg["executor"]["test_python_env"]=str(Path(args.test_python).expanduser().resolve()) if args.test_python else None
        cfg["executor"]["allowed_devices"]=[args.device]
        settings.write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n")
    env=dict(os.environ,PYTHONPATH=str(root/"run-test/src"))
    subprocess.run([sys.executable,"-m","run_test.cli","--config",str(settings),"init"],cwd=root,env=env,check=True)
    print("初始化完成；未启动服务。宿主架构:",platform.machine())
    print("配置:",settings)
    print("接下来: bash verify.sh；配置 obsutil 后 bash run.sh obs-scan")

if __name__=="__main__":main()
