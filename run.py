#!/usr/bin/env python3
"""统一入口。

用法:
    python3 run.py catch patterns/base_break.yaml
    python3 run.py catch patterns/consolidation.yaml --set min_days=90
    python3 run.py catch patterns/zfzs.yaml --data-dir /path/to/csvs --out /tmp/x.csv
    python3 run.py analyze analyzers/base_break.yaml        # 任务8实现后可用
"""

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _load_yaml(path: str) -> dict:
    fp = Path(path)
    if not fp.is_absolute():
        fp = ROOT / fp
    if not fp.exists():
        print(f"配置不存在: {fp}")
        sys.exit(1)
    with open(fp, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["catch", "analyze"])
    ap.add_argument("config", help="YAML 配置路径")
    ap.add_argument("--data-dir", help="覆盖数据目录")
    ap.add_argument("--set", action="append", default=[],
                    metavar="k=v", help="覆盖 detector 参数，可多次")
    ap.add_argument("--out", help="覆盖输出路径")
    args = ap.parse_args()

    config = _load_yaml(args.config)

    overrides = {}
    for kv in args.set:
        if "=" not in kv:
            print(f"--set 格式应为 k=v，收到: {kv}")
            sys.exit(1)
        k, v = kv.split("=", 1)
        try:
            import json
            overrides[k] = json.loads(v)
        except (ValueError, TypeError):
            overrides[k] = v

    if args.command == "catch":
        from core.engine import run_pattern
        run_pattern(
            config,
            data_dir=args.data_dir,
            overrides=overrides or None,
            out_path=Path(args.out) if args.out else None,
        )
    elif args.command == "analyze":
        from analyze.runner import run_analyzer
        run_analyzer(
            config,
            data_dir=args.data_dir,
            overrides=overrides or None,
            out_path=Path(args.out) if args.out else None,
        )


if __name__ == "__main__":
    main()
