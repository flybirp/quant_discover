#!/usr/bin/env python3
"""R2 开工前置检查：数据集退市股覆盖验证（holding_or_sell_upgrade R2 节）。

用已知退市股抽查数据目录：若缺失，discover_28 的大顶结论必须
标注幸存者偏差上限（大顶后一去不回的极端案例大量走向退市）。

已知退市样本（退市年份）：乐视网 300104(2020)、康得新 002450(2021)、
暴风集团 300431(2020)、雏鹰农牧 002477(2020)、长生生物 002680(2019)、
保千里 600074(2019)、印纪传媒 002143(2019)、欧浦智网 002711(2020)、
神雾环保 300156(2020)、华泽钴镍 000693(2019)、退市西水 600291(2022)、
济民可信? 600696(2021 ST岩石未退,跳过)。
"""
import sys
from pathlib import Path

DATA = Path("/Users/flybirp/Documents/mainland_data_2014")

DELISTED = {
    "300104": "乐视网(2020退)",
    "002450": "康得新(2021退)",
    "300431": "暴风集团(2020退)",
    "002477": "雏鹰农牧(2020退)",
    "002680": "长生生物(2019退)",
    "600074": "保千里(2019退)",
    "002143": "印纪传媒(2019退)",
    "002711": "欧浦智网(2020退)",
    "300156": "神雾环保(2020退)",
    "000693": "华泽钴镍(2019退)",
    "600291": "退市西水(2022退)",
}


def main():
    covered = missing = 0
    for code, name in DELISTED.items():
        fp = DATA / f"{code}.csv"
        if fp.exists():
            covered += 1
            import pandas as pd
            df = pd.read_csv(fp, encoding="utf-8-sig", usecols=["date"])
            print(f"  ✅ {code} {name}: 数据至 {df['date'].max()}")
        else:
            missing += 1
            print(f"  ❌ {code} {name}: 数据目录缺失")
    print(f"\n覆盖 {covered}/{len(DELISTED)}（缺失 {missing}）")
    if missing > len(DELISTED) * 0.3:
        print("⚠ 覆盖率 <70%：discover_28 必须标注幸存者偏差上限")
        sys.exit(1)
    print("通过：退市股基本覆盖，R2 可开工（缺失个股在文中注明）")


if __name__ == "__main__":
    main()
