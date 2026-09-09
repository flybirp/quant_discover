"""exit 引擎入口（run.py exit <yaml>）。

exit_rules YAML 契约:
    name: r1_ma_break
    input: output/catch_xxx.csv          # 契约明细（含 code/base_date/base_price）
    max_horizon: 120                     # 路径截断（交易日）
    cost_roundtrip: 0.30                 # 往返成本%（附录六）
    passthrough: [base_low, group]       # 透传列（StateExit ref / 分组）
    baseline: hold                       # 对照规则集名（默认 hold=持有到期）
    rulesets:
      - name: ma20_x2
        rules:
          - {type: ma_break, ma: 20, confirm: 2}
输出: 每规则集全样本 + fit/validate 分段汇总 + 退出原因分布；
      明细 CSV 落 output/exit_<name>.csv。
"""
from pathlib import Path

import pandas as pd
import yaml

from exit.replay import load_paths
from exit.rules import build_rules
from exit.metrics import evaluate, summarize, reason_table

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"


def run_exit(config: dict, data_dir: Path | None = None,
             out_path: Path | None = None) -> Path:
    name = config["name"]
    input_fp = Path(config["input"])
    if not input_fp.is_absolute():
        input_fp = ROOT / input_fp
    if not input_fp.exists():
        raise SystemExit(f"输入不存在: {input_fp}")

    max_horizon = int(config.get("max_horizon", 120))
    cost = float(config.get("cost_roundtrip", 0.30))
    passthrough = tuple(config.get("passthrough", []))
    rulesets = config.get("rulesets") or []
    reenter_ma = config.get("reenter_ma")

    # 收集所有规则集用到的均线周期与 ref 列
    ma_periods = {20, 60}
    for rs in rulesets:
        for r in rs.get("rules", []):
            if r.get("type") == "ma_break":
                ma_periods.add(int(r.get("ma", 20)))
    ma_periods.discard(0)

    print("=" * 88)
    print(f"  [exit] {name}")
    print(f"  输入: {input_fp.name} | horizon<={max_horizon}d | "
          f"成本 {cost:.2f}% 往返 | 规则集 {len(rulesets) + 1} 个（含 baseline）")
    print("=" * 88)

    paths = load_paths(input_fp, data_dir=data_dir,
                       ma_periods=tuple(sorted(ma_periods)),
                       max_horizon=max_horizon,
                       passthrough=passthrough)
    print(f"  路径: {len(paths)} 条")

    all_detail = []
    # baseline: 持有到期（空规则集）
    from exit.rules import Rule
    base_detail = evaluate(paths, [], max_horizon, cost)
    base_detail["ruleset"] = "baseline_hold"
    all_detail.append(base_detail)
    print(f"\n  [baseline: 持有到期 {max_horizon}d]")
    print(summarize(base_detail).to_string(index=False))

    from exit.metrics import flying_stats
    for rs in rulesets:
        rules = build_rules(rs.get("rules", []))
        detail = evaluate(paths, rules, max_horizon, cost,
                          reenter_ma=reenter_ma)
        detail["ruleset"] = rs["name"]
        all_detail.append(detail)
        print(f"\n  [{rs['name']}]  rules: "
              f"{', '.join(r.name for r in rules)}")
        print(summarize(detail).to_string(index=False))
        fs = flying_stats(detail)
        line = (f"  卖飞: +5% {fs['卖飞率(+5%)']}% | +10% "
                f"{fs['卖飞率(+10%)']}% | 错失中位 {fs['错失中位']}% "
                f"| P75 {fs['错失P75']}% | 卖对 {fs['卖对率']}%")
        if "reenter" in detail.columns and detail["reenter"].notna().any():
            line += f" | 再入场 {detail['reenter'].mean() * 100:.1f}%"
        print(line)
        rt = reason_table(detail)
        if len(rt) > 1:
            print(rt.to_string())

    out = pd.concat(all_detail, ignore_index=True)
    out_fp = out_path or (OUTPUT_DIR / f"exit_{name}.csv")
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_fp, index=False, encoding="utf-8-sig")
    print(f"\n  明细已保存: {out_fp}")
    return out_fp


# 供 run.py 统一从 YAML 加载调用
def main_from_yaml(fp: str):
    with open(fp, encoding="utf-8") as f:
        return run_exit(yaml.safe_load(f))
