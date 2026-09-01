"""扫描引擎：数据加载、全市场遍历、进度输出、结果落盘。

与 core/primitives.py 的分工：
  - engine 只做编排，不理解任何形态语义；
  - primitives 注册表里的段检测器吃 (df, code, params) 吐形态记录 list[dict]。

统一契约：每个 pattern YAML 可声明 contract，把形态自有列映射为
标准列 base_date / base_price，供 analyze 零语义消费。
"""

from pathlib import Path

import pandas as pd

from core.primitives import get_primitive

DEFAULT_DATA_DIR = Path.home() / "Documents" / "mainland_data_2014"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def load_data(filepath: Path, min_days: int = 200) -> pd.DataFrame | None:
    """加载个股日线数据（与旧版各脚本行为一致）。"""
    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
        if len(df) < min_days:
            return None
        return df
    except Exception:
        return None


def _apply_dedup(df: pd.DataFrame, dedup: dict) -> pd.DataFrame:
    """按 keys 去重；keep_earliest 指定取该列值最早（最小）的行。"""
    keys = dedup["keys"]
    keep_col = dedup.get("keep_earliest")
    if keep_col and keep_col in df.columns:
        df = df.sort_values(keep_col, kind="stable")
    df = df.drop_duplicates(subset=keys, keep="first")
    return df.reset_index(drop=True)


def run_pattern(config: dict, data_dir: Path | None = None,
                overrides: dict | None = None, out_path: Path | None = None) -> Path:
    """按 pattern 配置扫描全市场并落盘。返回输出文件路径。"""
    name = config["name"]
    det = config["detector"]
    func = get_primitive(det["type"])
    params = dict(det.get("params") or {})
    if overrides:
        for k, v in overrides.items():
            if k not in params:
                raise ValueError(
                    f"未知参数: {k}（pattern={name} 已定义: {sorted(params)}）")
            params[k] = v

    data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    min_days = int(config.get("min_data_days", 200))

    files = sorted(data_dir.glob("*.csv"))
    total = len(files)
    print("=" * 72)
    print(f"  [catch] {name}  (detector={det['type']})")
    print(f"  params: {params}")
    print(f"  数据: {data_dir} | {total} 只 | 最少 {min_days} 天")
    print("=" * 72)

    results: list[dict] = []
    scanned = skipped = 0
    for f in files:
        scanned += 1
        df = load_data(f, min_days)
        if df is None:
            skipped += 1
            continue
        results.extend(func(df, f.stem, params))
        if scanned % 500 == 0:
            print(f"  进度: {scanned}/{total} | 已发现 {len(results)} 个形态")

    print(f"  完成: {scanned}/{total} 扫描 | {skipped} 跳过 | 共 {len(results)} 个形态")

    if not results:
        print("  未发现任何形态。")
        out_df = pd.DataFrame(columns=["code"])
    else:
        out_df = pd.DataFrame(results)
        dedup = config.get("dedup")
        if dedup:
            before = len(out_df)
            out_df = _apply_dedup(out_df, dedup)
            print(f"  去重: {before} -> {len(out_df)}")

    # 统一契约列：base_date / base_price（值来自形态自有列）
    contract = config.get("contract")
    if contract and len(out_df):
        if "date" in contract:
            out_df["base_date"] = out_df[contract["date"]]
        if "price" in contract:
            out_df["base_price"] = out_df[contract["price"]]

    sort_cfg = config.get("sort")
    if sort_cfg and len(out_df) and sort_cfg.get("by") in out_df.columns:
        out_df = out_df.sort_values(sort_cfg["by"],
                                    ascending=sort_cfg.get("ascending", False))
        out_df = out_df.reset_index(drop=True)

    out_path = out_path or (OUTPUT_DIR / config.get(
        "output", f"catch_{name}.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {out_path}  ({len(out_df)} 行)")
    return out_path
