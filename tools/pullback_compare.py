"""新旧回踩口径对比（discover_09 对比节）。

三层样本:
  L1 旧 pullback_zf      （单线 zf 带, 无下界/无方向, 前 7 次）
  L2 新 pullback_zone    关闭启动确认（双线带 + 方向 + 前 3 次, 无启动过滤）
  L3 新 pullback_zone    开启启动确认（= 现状口径）

口径见 docs/appendix.md "09 / 09b"。用法见 discover_09 对比节复现命令。
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "output"
OLD = OUT / "an_pullback_zf.csv"
NOF = OUT / "an_pullback_zone_nofilter.csv"
NEW = OUT / "an_pullback_zone.csv"


def _load(p: Path, label: str) -> pd.DataFrame:
    if not p.exists():
        print(f"缺少 {p.name}（{label}），请先跑对应 analyze", file=sys.stderr)
        sys.exit(1)
    d = pd.read_csv(p, encoding="utf-8-sig")
    print(f"{label}: n={len(d):,}")
    return d


def _stat(g: pd.DataFrame, h: int) -> str:
    r = f"ret_{h}d"
    if r not in g.columns or len(g) == 0:
        return "   (无该 horizon)"
    nh = f"new_high_{h}d"
    nhs = f"/新高{round(g[nh].mean() * 100, 1)}%" if nh in g.columns else ""
    dd = f"maxdn_{h}d"
    dds = f"/DD{round(g[dd].mean(), 2)}%" if dd in g.columns else ""
    return (f"E{round(g[r].mean(), 2):+}% 胜{round((g[r] > 0).mean() * 100, 1)}%"
            f"{nhs}{dds}")


def main() -> int:
    old = _load(OLD, "L1 旧 pullback_zf")
    nof = _load(NOF, "L2 新口径·无启动过滤")
    new = _load(NEW, "L3 新口径·有启动过滤")

    old3 = old[old["pullback_no"] <= 3] if "pullback_no" in old.columns else old
    nzf = nof[nof["zone"] == "zf"]
    zzf = new[new["zone"] == "zf"]

    print("\n=== 表1 三套样本的基础收益（20 日）===")
    rows = [
        ("L1 旧 zf 全部(前7次)", old),
        ("L1 旧 zf 前3次", old3),
        ("L2 新口径无过滤·全部", nof),
        ("L2 新口径无过滤·zone=zf", nzf),
        ("L3 新口径有过滤·全部", new),
        ("L3 新口径有过滤·zone=zf", zzf),
    ]
    for name, g in rows:
        print(f"{name:26} n={len(g):>7,}  {_stat(g, 20)}")
    print("\n-- 60 日（旧版 analyze 无 60 日窗口，仅新版两层可比）--")
    for name, g in [("L2 无过滤·zone=zf", nzf), ("L3 有过滤·zone=zf", zzf)]:
        print(f"{name:26} n={len(g):>7,}  {_stat(g, 60)}")

    print("\n=== 表2 启动确认过滤的净贡献（唯一变量：require_breakout）===")
    keep = len(new) / len(nof) * 100
    print(f"样本保留率 {keep:.1f}%（{len(nof):,} → {len(new):,}）")
    for h in (20, 60):
        a, b = nof, new
        ra, rb = f"ret_{h}d", f"ret_{h}d"
        print(f"  {h}日: E {a[ra].mean():+.2f}% → {b[rb].mean():+.2f}% "
              f"(Δ{b[rb].mean() - a[ra].mean():+.2f}pct) | "
              f"胜率 {(a[ra] > 0).mean() * 100:.1f}% → {(b[rb] > 0).mean() * 100:.1f}% "
              f"(Δ{((b[rb] > 0).mean() - (a[ra] > 0).mean()) * 100:+.1f}pct) | "
              f"创新高 {a[f'new_high_{h}d'].mean() * 100:.1f}% → "
              f"{b[f'new_high_{h}d'].mean() * 100:.1f}% "
              f"(Δ{(b[f'new_high_{h}d'].mean() - a[f'new_high_{h}d'].mean()) * 100:+.1f}pct)")

    print("\n=== 表3a 启动确认的干净归因（L3 ⊆ L2，集合差 = 纯过滤效果）===")
    kn = set(zip(new["code"].astype(str), new["base_date"].astype(str)))
    keep_no = list(zip(nof["code"].astype(str), nof["base_date"].astype(str)))
    has_bk = [k in kn for k in keep_no]
    g_keep = nof[has_bk]                       # 波段有启动确认
    g_drop = nof[[not x for x in has_bk]]      # 波段无启动确认（被过滤剔除）
    print(f"  有启动确认(保留) n={len(g_keep):>7,}  {_stat(g_keep, 20)}")
    print(f"  无启动确认(剔除) n={len(g_drop):>7,}  {_stat(g_drop, 20)}")
    print(f"  {_stat(g_keep, 60)}")
    print(f"  {_stat(g_drop, 60)}")
    print("  ↑ 两组唯一差异 = 金叉前是否有放量阳线；差值即启动确认的真实效果")

    print("\n=== 表3b 事件进出（旧 zf 前3次 vs 新版有过滤；⚠ 混杂口径变化，仅供参考）===")
    ko = set(zip(old3["code"].astype(str), old3["base_date"].astype(str)))
    both, only_old, only_new = ko & kn, ko - kn, kn - ko
    print(f"共有 {len(both):,} | 旧独有(被新版剔除) {len(only_old):,} | "
          f"新独有(旧版未捕获) {len(only_new):,}")

    def _sub(d: pd.DataFrame, keys: set) -> pd.DataFrame:
        m = list(zip(d["code"].astype(str), d["base_date"].astype(str)))
        return d[[k in keys for k in m]]

    for name, keys in [("共有", both), ("旧独有/被剔除", only_old), ("新独有", only_new)]:
        g = _sub(old3 if name != "新独有" else new, keys)
        if len(g):
            print(f"  {name:14} n={len(g):>7,}  {_stat(g, 20)}")

    print("\n=== 表4 分年 20 日创新高率（%）===")
    comp = []
    for yr in sorted(set(old3["base_date"].astype(str).str[:4])):
        o = old3[old3["base_date"].astype(str).str[:4] == yr]
        n = zzf[zzf["base_date"].astype(str).str[:4] == yr]
        comp.append({
            "年": yr,
            "旧zf前3次": round(o["new_high_20d"].mean() * 100, 1) if len(o) else None,
            "旧n": len(o),
            "新zone=zf": round(n["new_high_20d"].mean() * 100, 1) if len(n) else None,
            "新n": len(n),
        })
    cdf = pd.DataFrame(comp)
    print(cdf.to_string(index=False))
    diff = cdf.dropna(subset=["旧zf前3次", "新zone=zf"])
    win = int((diff["新zone=zf"] > diff["旧zf前3次"]).sum())
    print(f"\n新口径优于旧口径的年份: {win}/{len(diff)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
