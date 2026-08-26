# quant_discover

A 股量化形态发现项目。基于全市场个股日线数据，用「技术指标 + 状态机」扫描特定的技术形态与波段模式，并对命中样本做统计分析，提炼可复用的交易规律。

项目定位是**探索与发现（discover）**，而非完整交易系统：先用脚本从数据里把"长什么样"的形态捞出来，再用统计验证"这种形态后续表现如何"，把结论沉淀到 `discoveris_doc/` 文档中。

---

## 数据源

脚本默认从以下目录读取数据（**不在本仓库内**，需自行准备）：

| 数据 | 路径 | 说明 |
|------|------|------|
| 个股日线 | `~/Documents/mainland_data_2014/` | 每只股票一个 CSV，文件名为股票代码（如 `600000.csv`） |
| 指数日线 | `~/Documents/mainland_index_data_2014/` | 沪深300、中证1000 等指数（用于环境对照分析） |

个股 CSV 至少包含 `date`、`open`、`high`、`low`、`close`、`volume` 列；`consolidation_scanner.py` 还会用到可选的 `turnover_rate`、`pe_ttm`、`pb`、`total_mv` 列。

---

## 项目结构

```
quant_discover/
├── indicator.py                                # 技术指标库（核心依赖）
├── wave_catcher.py                             # 波段识别器（3 种模式）
├── find_main_waves.py                          # 主升浪扫描器（2 种模式）
├── consolidation_scanner.py                    # 横盘震荡区域扫描器
├── find_flat_zone_after_a_main_dropping.py     # 加速下跌后横盘平台扫描器
├── analyze_flat_zone_returns.py                # 横盘平台买点收益分析
├── discoveris_doc/                             # 量化发现文档（15 篇）
│   ├── discover.md                             # 核心发现总纲
│   ├── discover_回踩买点体系.md                # 回踩买点完整体系
│   ├── discover_绝地逢生.md                    # 强支撑触底反转画像
│   ├── discover_下跌横盘.md                    # 下跌后横盘平台
│   ├── 金叉前特征_discover.md                  # 选股：金叉前特征
│   ├── 买点特征_discover.md                    # 回踩买点当天特征
│   ├── zfzs金叉_discover.md                    # 金叉后回踩 zhixing_slow 支撑
│   ├── zs回踩创新高_discover.md                # 回踩 zhixing_slow 后创新高
│   ├── zf回踩_discover.md                      # 回踩 zhixing_fast
│   ├── k线击败反转信号.md                       # 缩量击败反转（威科夫努力比）
│   ├── 顶底分型_discover.md                    # 底分型反转
│   ├── 底部反转on_zf.md                        # 底部放量突破
│   ├── 横盘震荡_discover.md                    # 横盘震荡后的突破方向
│   ├── 恒生电子横盘案例.md                     # 案例复盘
│   └── 中信建投横盘案例.md                     # 案例复盘
└── *.csv                                       # 各脚本的输出结果（历史运行）
```

---

## 核心代码模块

### 1. `indicator.py` — 技术指标库

被其余脚本 `import` 的公共依赖，提供：

- **基础指标**：`sma` / `ema` / `macd` / `rsi` / `kdj` / `bollinger` / `atr` / `volume_ma` / `bbi` / `ma_cross`。
- **知行量化指标**：
  - `zhixing_fast` — 双重平滑 10 日 EMA：`EMA(EMA(close,10),10)`。
  - `zhixing_slow` — 四条均线均值：`(MA14 + MA28 + MA57 + MA114) / 4`。
  - `zz_short` / `zz_long` — 3 日 / 21 日价格位置百分位。
- **量能异动**：`volume_rank_pct`、`price_position_pct`、`is_double_volume_signal`（倍量柱）、`is_volume_top5`、`pocket_pivot_volume`（口袋支点，威科夫 SOS 量化）。
- **形态**：`double_bottom_level`（双底支撑位）。
- **一站式入口**：`compute_all_indicators(df)`，给 DataFrame 附加全部指标列。

### 2. `wave_catcher.py` — 波段识别器

用「指标信号」实时锁定上涨波段，与 `find_main_waves.py`（事后归纳）互补。

| 模式 | 入场 | 离场 |
|------|------|------|
| `zfzs`（默认） | `zhixing_fast` 金叉 `zhixing_slow` | MA20 死叉 MA60 |
| `volume` | 长期下跌 + 放量 > 2 倍均量 + 站上 `zhixing_fast` | 跌破入场前最低价（止损）或涨 100%（止盈） |
| `oversold` | 加速下跌（近 20 天跌速 > 前 40 天 × 1.5 倍）后反弹 ≥ 30% | 仅记录形态数据点 |

```bash
python3 wave_catcher.py                           # 默认 zfzs 模式
python3 wave_catcher.py --mode volume --lookback 90 --decline 30
python3 wave_catcher.py --mode oversold
```

### 3. `find_main_waves.py` — 主升浪扫描器

| 模式 | 逻辑 |
|------|------|
| `drawdown`（默认） | 回撤状态机：持续上涨过程中任意局部高点回撤 ≤ 50%，直到突破阈值波段结束 |
| `trendline` | 沿趋势线（`ma60` 或 `zhixing_slow`）运行，无连续 2 日跌破，涨幅 > 100%，持续 > 200 天 |

```bash
python3 find_main_waves.py                        # 默认 drawdown，回撤 ≤ 50%
python3 find_main_waves.py --max-dd 30            # 自定义回撤阈值
python3 find_main_waves.py --mode trendline --indicator ma60
```

### 4. `consolidation_scanner.py` — 横盘震荡区域扫描器

扫描约 2 年（400–600 交易日）的横盘震荡区间，提取区间内的量价 / 均线 / 估值 / 市值特征，并按后续突破方向（上 / 下 / 继续震荡）分组对比，输出区分度排序。

```bash
python3 consolidation_scanner.py
python3 consolidation_scanner.py --range-pct 50 --window-min 400 --window-max 600
```

### 5. `find_flat_zone_after_a_main_dropping.py` — 加速下跌后横盘平台扫描器

采用项目已论证的「加速下跌（末期杀跌 / capitulation）」逻辑，寻找：

1. **加速下跌段**：近 20 天平均跌速 > 前 40 天平均跌速 × 1.5 倍（均为负值）。
2. **横盘段**：加速下跌结束后，紧接着出现持续 ≥ 15 日、波幅 ≤ 5% 的横盘区间。

输出 `flat_zone_after_drop.csv`（含横盘均价，作为后续买点）。

```bash
python3 find_flat_zone_after_a_main_dropping.py
python3 find_flat_zone_after_a_main_dropping.py --lookback-decline 40 \
    --lookback-accel 20 --accel-threshold 1.5 --min-flat-days 15 --flat-range 5
```

### 6. `analyze_flat_zone_returns.py` — 横盘平台买点收益分析

读取 `flat_zone_after_drop.csv`，以横盘区间均价为买点，统计横盘结束后 30 / 90 / 200 个交易日的收益分布（均值、中位数、胜率、分位数）。

```bash
python3 analyze_flat_zone_returns.py
python3 analyze_flat_zone_returns.py --input flat_zone_after_drop.csv --horizons 30,90,200
```

---

## 发现文档（`discoveris_doc/`）

记录经过全市场数据验证的量化发现，供其他项目借鉴。核心结论包括：

- **`discover.md`**（总纲）：
  - **回撤-涨幅反向关系**：主升浪波段内最大回撤越小，最终涨幅越大；回撤 < 5% 的"安静上涨"几乎必翻倍（96.9% 涨超 100%），而回撤 45–50% 的波段涨幅中位仅 112%。
  - **超大波段启动画像**：启动前半年"先阴跌缩量 → 末期杀跌放量 → 恐慌中启动"，启动日是"放量长下影阴线"。
  - **大盘环境**：超大波段启动于**中证 1000 的恐慌性下跌**中，沪深 300 无区分度。
- **`discover_回踩买点体系.md`**：选股 → 等待 → 买入 → 持有的完整顺势交易框架（金叉前选股 + 回踩 `zhixing_slow` 买入 + 持有）。
- **`discover_绝地逢生.md`**：强支撑触底反转画像，按信号强度汇总（缩量击败、底分型、恐慌底、`zhixing_slow` 回踩、横盘突破、底部放量突破）。
- **`discover_下跌横盘.md`**：论证"下跌 + 窄幅横盘"不是独立买点（中位收益为负），但"加速下跌"优于"缓慢阴跌"，价值在右尾（需叠加过滤）。
- 其余文档为上述体系的单项专题与案例复盘，详见各文件。

---

## 输出文件

| 文件 | 生成脚本 | 说明 |
|------|----------|------|
| `main_waves_dd50.csv` | `find_main_waves.py`（drawdown） | 回撤 ≤ 50% 的主升浪波段清单 |
| `main_waves_ma60.csv` | `find_main_waves.py`（trendline，ma60） | 沿 MA60 趋势线波段 |
| `main_waves_zhixing_slow.csv` | `find_main_waves.py`（trendline，zhixing_slow） | 沿 zhixing_slow 趋势线波段 |
| `wave_catcher_volume.csv` | `wave_catcher.py`（volume） | 放量突破波段 |
| `wave_catcher_oversold.csv` | `wave_catcher.py`（oversold） | 超跌反弹形态 |
| `wave_catcher_results.csv` | `wave_catcher.py`（zfzs 历史运行，旧命名） | zhixing 金叉波段（当前代码输出名为 `wave_catcher_zfzs.csv`） |
| `consolidation_zones.csv` | `consolidation_scanner.py` | 横盘震荡区域及特征 |
| `flat_zone_after_drop.csv` | `find_flat_zone_after_a_main_dropping.py` | 加速下跌后横盘平台清单 |
| `flat_zone_returns.csv` | `analyze_flat_zone_returns.py` | 横盘平台买点后续收益明细 |

---

## 快速开始

```bash
# 1. 准备数据：将个股日线 CSV 放入 ~/Documents/mainland_data_2014/

# 2. 安装依赖
pip install pandas numpy

# 3. 运行扫描（任选其一）
python3 find_main_waves.py                          # 主升浪
python3 wave_catcher.py --mode zfzs                 # 波段识别
python3 consolidation_scanner.py                    # 横盘震荡
python3 find_flat_zone_after_a_main_dropping.py     # 加速下跌后横盘平台
python3 analyze_flat_zone_returns.py                # 平台收益分析
```

> 环境要求：Python 3.10+（代码使用了 `X | None` 类型注解），依赖仅 `pandas`、`numpy`。

---

## 免责声明

本项目仅用于量化研究学习与历史数据回测，所有结论均为统计规律，不构成任何投资建议。股市有风险，据此操作盈亏自负。
