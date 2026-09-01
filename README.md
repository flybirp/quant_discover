# quant_discover

A 股量化形态发现项目。基于全市场个股日线数据，用「段检测器 + YAML 配置」扫描特定技术形态，并以统一 analyze 引擎对命中样本做统计验证，提炼可复用的交易规律。

项目定位是**探索与发现（discover）**，而非完整交易系统：先从数据里把"长什么样"的形态捞出来，再统计验证"这种形态后续表现如何"，把结论沉淀到 `docs/` 文档中。

---

## 数据源

数据目录不在本仓库内，需自行准备：

| 数据 | 路径 | 说明 |
|------|------|------|
| 个股日线 | `~/Documents/mainland_data_2014/` | 每只股票一个 CSV（如 `600000.csv`） |
| 指数日线 | `~/Documents/mainland_index_data_2014/` | 沪深300、中证1000 等（环境对照） |

个股 CSV 至少含 `date/open/high/low/close/volume`；`long_consolidation` 形态可选使用 `turnover_rate/pe_ttm/pb/total_mv` 列。

依赖：Python 3.10+，`pandas`、`numpy`、`pyyaml`。

---

## 架构

```
quant_discover/
├── run.py                    # 统一入口: run.py catch / run.py analyze
├── indicator.py              # 技术指标库（MA/MACD/RSI/KDJ/BOLL/ATR/知行/量能/双底…）
├── core/                     # 波段捕捉引擎（catcher）
│   ├── engine.py             #   扫描引擎: 加载/遍历/进度/去重/契约列/落盘
│   ├── resample.py           #   真自然周（ISO 周）聚合
│   └── primitives.py         #   段检测器注册表（16 种检测器，5 类结构）
├── analyze/                  # 统计分析引擎
│   ├── stats.py              #   统计指标库（均值/中位/P15/P85/胜率/盈亏结构…）
│   └── runner.py             #   forward 收益/自然周/cross_above 事件/分组/稳健性
├── patterns/*.yaml           # 18 份形态配置（要捕哪种 wave 由配置决定）
├── analyzers/*.yaml          # 统计分析配置（买点契约/horizons/分组/稳健性）
├── topics/                   # 专题分析脚本（启动画像/横盘方向等跨形态统计）
├── tools/verify.py           # 快照对比（重构语义等价验证）
├── docs/                     # 发现文档（总纲 + 14 篇，全部可复现）
├── output/                   # 新引擎输出（catch_*.csv / an_*.csv）
├── _baseline/                # 旧版重跑基线（语义等价验证的对照）
├── _deprecated/              # 旧版脚本存档（已被 core+analyze 完全替代）
└── README.md
```

### 核心抽象

**catcher**（`run.py catch patterns/X.yaml`）：
- 引擎只做编排；形态语义封装在 `core/primitives.py` 的段检测器里，
  签名统一 `(df, code, params) -> list[dict]`，通过 `@register("name")` 注册；
- 检测器分五类结构：**信号状态机**（zfzs/volume/main_dd/main_trendline）、
  **贪婪区间扩展**（consolidation/long_consolidation/flat_zone）、
  **两窗口速度比较**（oversold）、**关键点+多段确认**（base_break/spike_top）、
  **事件型**（pullback_zfzs 回踩 / fractal_bottom 底分型 / effort_defeat 缩量击败）；
- YAML 声明：检测器类型 + 参数 + 最小数据天数 + 去重规则 + 排序 + `contract`
  （把形态自有列映射为统一契约列 `base_date` / `base_price`）。

**analyze**（`run.py analyze analyzers/X.yaml`）：
- 零语义：只消费契约三列 `code/base_date/base_price` + 任意透传分组/阈值列；
- horizon 支持 `days`（第 N 交易日收盘）与 `weeks`（真自然周：基准日所在
  ISO 周为第 0 周残周，第 N 周取该周最后交易日收盘，空周记 NaN 不顺延）；
- 事件原语 `cross_above: {ref_col}`：窗口内最高收盘 ≥ 阈值列的占比；
- 任意 `group_cols` 分组、分年度稳健性（可剔除指定年份）。

### 15 种 pattern 一览

| pattern | 形态 | 契约基准 |
|---------|------|---------|
| `zfzs` | zhixing 金叉波段（MA20/60 死叉离场） | 入场日收盘 |
| `volume` | 长期下跌后放量突破 zhixing_fast（止损/止盈） | 入场日收盘 |
| `oversold` | 加速下跌后触底反弹 ≥30% | 底部价 |
| `base_break` | 周线筑底首次跌破支撑（"漏了"），按深度 A/B/C/D × 量能分组 | 跌破日收盘 |
| `consolidation` | ≥60 天、均价 ±10% 窄幅横盘 | 横盘结束日收盘 |
| `spike_top` | 翻倍 → 单顶（真自然周）→ 腰斩 | 腰斩日收盘 |
| `main_dd50` | 主升浪（回撤 ≤50% 状态机切分） | 波段起点 |
| `main_trendline_zhixing_slow` / `_ma60` | 沿趋势线 >200 天、涨幅 >100% | 波段起点 |
| `flat_zone` | 加速下跌后 ≥15 天、波幅 ≤5% 横盘平台 | 横盘均价 |
| `long_consolidation` | 400–600 天宽幅震荡 + 21 个区间特征透传列 | 区间末日收盘 |

---

## 快速开始

```bash
# 1. 捕形态（全市场扫描）
python3 run.py catch patterns/base_break.yaml
python3 run.py catch patterns/consolidation.yaml --set min_days=90   # 临时改参数

# 2. 统计后续收益
python3 run.py analyze analyzers/base_break.yaml
python3 run.py analyze analyzers/flat_zone.yaml

# 3. 快照对比（重构/调参后的语义等价验证）
python3 tools/verify.py --all
python3 tools/verify.py --preset catch_zfzs
```

## 新增一个形态

1. 在 `core/primitives.py` 写一个段检测器并 `@register("my_pattern")`；
2. 新建 `patterns/my_pattern.yaml` 声明参数与 contract；
3. （可选）新建 `analyzers/my_pattern.yaml` 配置后续收益统计。

## 与旧版的差异（有意的行为变更）

- `base_break` 捕捉器不再内嵌 4 周收益（剥离到 analyze），CSV 新增 `base_price`；
- `spike_top` 周线单顶从"每 5 个交易日切片"改为**真自然周**聚合；
- `long_consolidation` 废除突破三分类（direction 等 4 列），改由 analyze 的
  连续指标（maxup/maxdn）承担；
- 周单位 horizon 一律为真自然周（旧版 `w*5` 交易日近似废弃）。

旧版脚本与历史输出存档于 `_deprecated/`、`_baseline/old_outputs/`。

---

## 输出文件

| 文件 | 生成命令 | 说明 |
|------|----------|------|
| `output/catch_*.csv` | `run.py catch` | 15 种 pattern 清单（含 base_date/base_price 契约列） |
| `output/an_*.csv` | `run.py analyze` | 形态后续收益明细 + `_summary.csv` 汇总 |
| `_baseline/*.csv` | `_baseline/run_baseline.sh` | 旧版重跑基线（对照用） |

---

## 免责声明

本项目仅用于量化研究学习与历史数据回测，所有结论均为统计规律，不构成任何投资建议。股市有风险，据此操作盈亏自负。
