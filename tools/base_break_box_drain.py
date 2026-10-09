"""放量下漏检验：箱体前置 × 缩量前置 × 跌破日量能（discover_33）。

用户定义：长周期箱体震荡 + 中期阴跌 + 一段缩量横盘 → 突然放量跌漏
（跌破日量 > 2×20 日均量）。本项目已有 04 篇（跌破 + 量比）与 32 篇
（量比倒 U：1.5~2.1 甜点、>2.1 巨量最差），本脚本检验**两个形态前置**
（is_box / is_drain）是否改变该结论。

用法:
    python3 run.py catch patterns/base_break_box_drain.yaml
    python3 run.py analyze analyzers/base_break_box_drain.yaml
    python3 tools/base_break_box_drain.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
AN = ROOT / "output" / "an_base_break_box_drain.csv"


def _stat(g: pd.DataFrame, h: str = "ret_20d") -> dict:
    if len(g) < 20:
        return {"n": len(g)}
    dd = "maxdn_20d" if h == "ret_20d" else "maxdn_60d"
    return {
        "n": len(g),
        "E": round(g[h].mean(), 2),
        "中位": round(g[h].median(), 2),
        "胜率": round((g[h] > 0).mean() * 100, 1),
        "涨≥10%": round((g[h] >= 10).mean() * 100, 1) if h == "ret_20d" else None,
        "MaxDD": round(g[dd].mean(), 2) if dd in g else None,
    }


def _show(title: str, rows: list) -> None:
    print(f"\n=== {title} ===")
    print(pd.DataFrame(rows).to_string(index=False))


def main() -> int:
    if not AN.exists():
        print(f"缺少 {AN}", file=sys.stderr)
        return 1
    d = pd.read_csv(AN, encoding="utf-8-sig")
    print(f"样本 n={len(d):,}")
    print(f"is_box 占比 {d['is_box'].mean() * 100:.1f}%  "
          f"is_drain 占比 {d['is_drain'].mean() * 100:.1f}%  "
          f"两者交集 {(d['is_box'] & d['is_drain']).mean() * 100:.1f}%")

    quads = {
        "① 箱体+缩量（用户定义）": d["is_box"] & d["is_drain"],
        "② 箱体+非缩量": d["is_box"] & ~d["is_drain"],
        "③ 非箱体+缩量": ~d["is_box"] & d["is_drain"],
        "④ 非箱体+非缩量": ~d["is_box"] & ~d["is_drain"],
    }
    _show("四象限（20 日，跌破日收盘基准）",
          [{"组": k, **_stat(d[m])} for k, m in quads.items()])
    _show("四象限（4 周，04 篇原口径）",
          [{"组": k, **_stat(d[m], "ret_4w")} for k, m in quads.items()])

    # 甜点深度 C 组内
    c = d[d["group"] == "C"]
    if len(c) >= 100:
        cq = {
            "① 箱体+缩量": c[c["is_box"] & c["is_drain"]],
            "② 箱体+非缩量": c[c["is_box"] & ~c["is_drain"]],
            "③ 非箱体+缩量": c[~c["is_box"] & c["is_drain"]],
            "④ 非箱体+非缩量": c[~c["is_box"] & ~c["is_drain"]],
        }
        _show("C 组（跌破 10~15%，04 篇甜点）内四象限（20 日）",
              [{"组": k, **_stat(g)} for k, g in cq.items()])

    # 量比分档 × 前置
    d["vr_bin"] = pd.cut(d["volume_ratio"], [0, 0.8, 1.46, 2.13, 100],
                         labels=["缩量<0.8", "平量0.8-1.46", "温和1.46-2.13", "巨量>2.13"])
    rows = []
    for vb, g in d.groupby("vr_bin", observed=True):
        gq = {
            "① 箱体+缩量": g[g["is_box"] & g["is_drain"]],
            "② 箱体+非缩量": g[g["is_box"] & ~g["is_drain"]],
            "③ 非箱体+缩量": g[~g["is_box"] & g["is_drain"]],
            "④ 非箱体+非缩量": g[~g["is_box"] & ~g["is_drain"]],
        }
        for k, gg in gq.items():
            s = _stat(gg)
            if s.get("n", 0) >= 30:
                rows.append({"量比档": vb, "组": k, **s})
    _show("量比分档 × 四象限（20 日）", rows)

    # 用户的完整定义：箱体 + 缩量 + 放量>2
    full = d["is_box"] & d["is_drain"] & (d["volume_ratio"] > 2)
    sweet = d["is_box"] & d["is_drain"] & d["volume_ratio"].between(1.46, 2.13)
    _show("完整定义对照（20 日）", [
        {"组": "箱体+缩量+放量>2×（用户定义）", **_stat(d[full])},
        {"组": "箱体+缩量+温和放量1.46~2.13", **_stat(d[sweet])},
        {"组": "全部箱体+缩量（不看量）", **_stat(d[d["is_box"] & d["is_drain"]])},
        {"组": "全部（基线）", **_stat(d)},
    ])

    # 分年
    d["yr"] = d["base_date"].astype(str).str[:4]
    print("\n=== 分年：20 日胜率（%）===")
    piv = pd.DataFrame({
        "箱体+缩量": d[d["is_box"] & d["is_drain"]].groupby("yr")["ret_20d"]
        .apply(lambda s: round((s > 0).mean() * 100, 1) if len(s) >= 20 else None),
        "基线全体": d.groupby("yr")["ret_20d"].apply(lambda s: round((s > 0).mean() * 100, 1)),
        "箱体+缩量n": d[d["is_box"] & d["is_drain"]].groupby("yr").size(),
    })
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
