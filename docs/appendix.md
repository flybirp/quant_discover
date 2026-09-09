# 附录：术语与代码口径对照表

> 目的：**确保复刻结论时零歧义**。本文档把 `docs/` 各篇中的每个用词映射到
> 精确的代码逻辑。与代码不一致处以代码为准（`core/primitives.py`、
> `analyze/runner.py`、`analyze/stats.py`）。

---

## 一、通用口径

### 数据与价格

| 用词 | 精确定义 |
|------|---------|
| 收盘价 close | 前复权日收盘价，来源 CSV `close` 列 |
| 股票池 | `~/Documents/mainland_data_2014/*.csv` 全部 5062 只（2014-01 ~ 2026-08），按文件名代码 |
| ⚠ 幸存者偏差（2026-09-10 查明） | **数据池为纯幸存者集合**：11 只已知退市股（乐视网/康得新/暴风/长生/保千里等）**0 覆盖**（`tools/delisting_precheck.py`）。含义：① 一切左尾/大跌结论实际比报告值**更极端**（最惨的退市股不在池内）；② 大顶/做空类研究（28 篇）的破坏力被系统性低估，结论必须标注偏差上限 |
| min_data_days | 每形态在 pattern YAML 声明；**日线长度不足该值的股票整只跳过**（复刻时股票池必须一致） |
| 交易日 N | 数组下标偏移（第 N 个**交易日**，非自然日） |

### 统计指标（`analyze/stats.py`）

| 用词 | 精确定义 |
|------|---------|
| 收益 | `(目标日收盘 / base_price - 1) × 100`，base_price 为契约基准价 |
| **E（期望）** | 算术平均（期望收益的样本估计） |
| **平均MaxDD（持仓感受）** | 持有期窗口内 `min(low)/base_price - 1` 的均值（窗口含基准日）——持有期内的平均最深浮亏，负值越浅持仓体验越好。明细列为 `maxdn_Nd/Nw`。注意与形态自身的 max_drawdown_pct（如 main_dd50 波段内回撤）区分：前者是**买入后**的前瞻浮亏，后者是形态历史属性 |
| 中位 median | 50 分位 |
| 胜率 win_rate | `ret > 0`（严格大于）的样本占比 %；"下跌率" = `ret < 0` 占比 |
| **RankIC** | 信号列与前瞻收益的 **Spearman 秩相关**（`pandas.corr(method='spearman')`），n≥30 才计算。解读惯例：\|IC\| < 0.02 无效 / 0.02~0.05 弱 / 0.05~0.10 中等 / > 0.10 强 |
| 盈亏比 plr | `mean(正收益) / |mean(负收益)|` |
| 大涨率 up20 / 大跌率 dn20 | `ret ≥ +20%` / `ret ≤ -20%` 占比 |

（P15/P85 仍由 `compute_stats` 计算并保留在 summary CSV 中，文档表格不再使用。）

### 收益基准与窗口（`analyze/runner.py`）

| 用词 | 精确定义 |
|------|---------|
| base_date / base_price | 契约三列之二，由 pattern YAML 的 `contract` 从形态自有列复制 |
| **day horizon** | `close[base_idx + N] / base_price - 1`（第 N 个交易日收盘） |
| **week horizon（真自然周）** | 基准日所在 ISO 周（周一为始）为第 0 周残周；第 N 周 = `Period('W') + N`；收益取**该周内最后一个交易日收盘**；整周无交易日记 NaN **不顺延** |
| **maxup / maxdn** | 窗口 = `[base_idx, 窗口末]` **含基准日**；`max(high)/base - 1` 与 `min(low)/base - 1`（用最高/最低价） |
| **事件 cross_above / cross_below** | 窗口同上（**含基准日**），但比较对象是**收盘价**：`max(close) ≥ ref_col` / `min(close) ≤ ref_col`。注意：事件用收盘、maxup 用最高价，两者口径不同 |

### 均线与量能（`indicator.py`）

| 用词 | 精确定义 |
|------|---------|
| MA(n) / sma | `rolling(n).mean()`，含当日 |
| vol_ma20 | `volume.rolling(20).mean()`，**含当日**（重要：绝大多数"量比"以此为分母，当日巨量会抬高自己的分母） |
| **量比（默认口径）** | `当日 volume / vol_ma20`，分母含当日 |
| zhixing_fast | `EMA(EMA(close,10),10)`，`ewm(span, adjust=False)` |
| zhixing_slow | `(MA14 + MA28 + MA57 + MA114) / 4` |
| 金叉 / 死叉 | `fast[i-1] ≤ slow[i-1] 且 fast[i] > slow[i]`（反之死叉） |

### 波段级输出列（zfzs / volume 等波段型 detector，`core/primitives.py`）

| 用词 | 精确定义 |
|------|---------|
| duration_days | `exit_idx − entry_idx`（**交易日下标差**，非自然日） |
| entry/exit_price | 对应日 close |
| peak_price | 波段内**最高收盘**（close 口径，非 high） |
| return_pct | `(exit_price/entry_price − 1) × 100` |
| **max_drawdown_pct（波段级）** | `(peak_price − min(close[peak_idx..exit_idx])) / peak_price × 100`——close 口径且**仅峰后段**；与 analyze 的 maxdn（low 口径、含基准日）**不同，勿混用** |
| holding=True | 数据末端仍未触发退出（未平仓波段），统计时须剔除或单列 |

---

## 二、逐形态定义（按文档编号）

### 01 zfzs 金叉（`patterns/zfzs.yaml`）

| 用词 | 代码逻辑 |
|------|---------|
| 金叉入场 | zhixing_fast 金叉 zhixing_slow（上述交叉定义） |
| 离场 | MA20 死叉 MA60 |
| 波段过滤 | `exit_idx - entry_idx ≥ 20` |
| base | 入场日收盘 |

### 02 volume 放量突破（`patterns/volume.yaml`）

| 用词 | 代码逻辑 |
|------|---------|
| 长期下跌 | 过去 60 天中 `close < zhixing_slow` 的天数占比 ≥ 70% **且** 距 60 天最高收盘跌幅 > 20% |
| 放量 | `volume > vol_ma20 × 2`（分母含当日） |
| 站上 zhixing_fast | `close > zhixing_fast` |
| 止损 / 止盈 | 收盘 < 入场前 60 天最低**低价（low）**；收盘 ≥ 入场价 × 2 |

### 03 oversold 超跌反弹（`patterns/oversold.yaml`）

| 用词 | 代码逻辑 |
|------|---------|
| 前期跌速 A | 窗口 `[i-60, i-20)` 内 `diff(close)/close[:-1]` 的均值（日均收益） |
| 近期跌速 B | 窗口 `[i-20, i)` 同上 |
| 加速下跌 | `A < -0.1%/日 且 B < A 且 B/A > 1.5` |
| 底部 | 加速窗口起点向后 30 天内 close 最低日 |
| 反弹确认 | 底部后 120 天内 `(close/底部价-1)×100 ≥ 30` |
| ⚠ 口径警告 | base = 底部价，但形态记录时收益已 ≥30%（前视），见表内说明 |

### 04 base_break 筑底跌破（`patterns/base_break.yaml`）

| 用词 | 代码逻辑 |
|------|---------|
| 周线 | **真自然周**：`date.dt.to_period('W')`（ISO 周，周一为始），非 5 日块 |
| 筑底段 | 从某周起逐周扩展，段内周 low 的 `(max-min)/min ≤ 5%`；至少 2 周 |
| 位置：下跌后首底 | 筑底段前 12 周 `(high最高 - base_low)/high最高 ≥ 20%` |
| 位置：震荡下沿 | 前 12 周 `(high最高-low最低)/low最低 ≥ 40%` 且 `(base_low-low最低)/low最低 ≤ 10%` |
| 跌破 | 筑底段结束后**首个** `low < base_low` 的交易日 |
| 分组 A/B/C/D | 跌破日 low 落入 `base_low × [0.95, 0.90, 0.85, 0.70]` 区间（浅→深） |
| **量比（本形态特殊）** | `跌破日 volume / 前 20 日均量`，**分母不含当日**（与默认口径不同！） |
| 放量/平量/缩量 | 量比 > 1.2 / 0.8~1.2 / < 0.8 |
| base_price | **跌破日收盘价**（不是 break_low） |
| recover_base | analyze 事件：4 周窗口内 `max(close) ≥ base_low`（收盘，含基准日） |

### 05 三横盘家族

**consolidation**（`patterns/consolidation.yaml`）：

| 用词 | 代码逻辑 |
|------|---------|
| 横盘区间 | 从起点逐日贪婪扩展：窗口内 `max(high) ≤ 窗口均价×(1+10%)` 且 `min(low) ≥ 均价×(1-10%)`，均价 = 窗口内 close 均值（增量更新），违反即停 |
| 记录条件 | 区间 ≥ 60 天；记录后跳过整个区间 |

**long_consolidation**（`patterns/long_consolidation.yaml`）：

| 用词 | 代码逻辑 |
|------|---------|
| 窗口 | 400~600 天，步长 20 天双层扫描，与已记录区间重叠则跳过 |
| 振幅 | `(max(high)-min(low))/median(close) ≤ 50%` |
| 无强趋势 | `|close末-close首|/close首 ≤ 30%` |
| 波动率收敛 | 后半段日收益 std ≤ 前半段 × 1.2 |
| 21 特征列 | 详见 `core/primitives.py::_extract_zone_features`：量能首尾比（三等分）、量能趋势（polyfit 斜率/均值）、ATR(14) 及收敛、布林带宽(20,2)、年化波动率(std×√250)、MA20/60/120 缠绕与多头排列比、知行金叉次数/缠绕、MACD DIF 零轴占比、PE/PB/市值中位（列缺失记 0） |
| 方向分类（专题） | `maxup_120d ≥ +30%` = up / `maxdn_120d ≤ -30%` = down / 其余 range |

**flat_zone**（`patterns/flat_zone.yaml`）：

| 用词 | 代码逻辑 |
|------|---------|
| 加速下跌 | 同 03 oversold（A/B 窗口、1.5 倍） |
| 横盘段 | 加速成立日起贪婪扩展：`(max(high)-min(low))/窗口均价 ≤ 5%`，≥ 15 天 |
| base | 横盘**均价**（非某日收盘） |

### 06 spike_top 尖顶腰斩（`patterns/spike_top.yaml`）

| 用词 | 代码逻辑 |
|------|---------|
| 局部低点 | close 同时为前后各 30 天窗口（rolling 31）最小值；相等平台只取首个 |
| 翻倍 | 顶点 close ≥ 低点 close × 2 |
| 见顶确认 | 翻倍后自最高收盘回落 ≥ 20% |
| 腰斩点 | 顶点后首个 `close ≤ 顶点/2` |
| 严格单边 | 顶点→腰斩途中，自 running_low 的反弹 ≤ 20% |
| **单顶（周线）** | **真自然周**聚合区间内每周 max(high)；局部周峰（严格高于前后周）中，除主峰外均 ≤ 主峰 × 0.9 |
| 去重 | 同 (code, peak_date, half_date) 保留 low_date 最早一条 |
| base | 腰斩日收盘 |

### 07 主升浪

**main_dd50**（回撤状态机）：

| 用词 | 代码逻辑 |
|------|---------|
| 波段 | running_peak 追踪历史最高收盘；`(peak-close)/peak > 50%` 即波段结束、从当点重启 |
| 波段起点 | 段内**最低收盘**日；持续 = 末日 - 低点日 + 1；涨幅 = 末日收盘/低点收盘 |
| 波段内回撤 | 段内 `(峰值-峰值后最低)/峰值` 的最大值 |
| 过滤 | 持续 ≥ 50 天且涨幅 ≥ 50% |

**main_trendline**：清洁段 = 沿趋势线（MA60 或 zhixing_slow）**连续 2 日收盘跌破**即终止；段 > 200 天；段内起点→此后最高收盘涨幅 > 100%。

**启动画像专题**（`topics/topic_startup_profile.py`）：窗口 = 起点前 120 个交易日 `[si-120, si)`；四段各 30 天，涨跌幅 `close[b]/close[a]`；末期杀跌占比 = `|最后30天涨跌幅| / |前120天总涨跌幅|`（分母 <0 时计入）。

### 09 pullback_zfzs 回踩（`patterns/pullback_zf.yaml` / `_zs`）

| 用词 | 代码逻辑 |
|------|---------|
| 回踩事件 | 金叉波段内，`dist = (low - line)/line×100` 从 `> +3%` 状态进入 `≤ +3%`（即 low 自上方进入支撑线 +3% 带内）；**跌破后未回到线上方 +3% 前不重复计数**（深跌破仍在带内） |
| 次数 | 每波段前 7 次 |
| prior_high | 回踩日前（不含当日）波段内最高收盘；创新高 = 20 日窗口 `max(close) ≥ prior_high` |
| wave_dd_so_far | 截至回踩日，波段内 `(峰值-当日)/峰值` |
| 前5天涨幅 | `close[i]/close[i-5]`（需 i-5 ≥ 金叉日） |

### 10 / 12 分型与击败

**fractal_bottom / fractal_top**（三 K）：

| 用词 | 代码逻辑 |
|------|---------|
| 底分型 | `high[i]<high[i±1] 且 low[i]<low[i±1]`（中间 K 完全低于两侧）；顶分型对称 |
| 确认日 | 第 3 根 K（i+1）；base = 确认日收盘 |
| 前20天跌幅 | `close[i+1]/close[i+1-20] - 1`（确认日口径） |
| 连跌天数 | 截至中间 K 的连续 `close[j]<close[j-1]` |
| **depth（底）** | `min((low[i-1]-low[i])/low[i], (low[i+1]-low[i])/low[i]) × 100`，**分母为中间 K 自身 low** |
| **depth（顶）** | `min((high[i]-high[i-1])/high[i-1], (high[i]-high[i+1])/high[i+1]) × 100`，**分母为两侧 high**（与底分型不对称，系实现差异，如实记录） |
| strength | `depth × suppress`（底：high_suppress；顶：low_suppress） |
| 不破低 | 事件 cross_below：20 日窗口 `min(close) ≤ fractal_low`（fractal_low = 中间 K 的 low） |

**effort_defeat**（缩量击败）：

| 用词 | 代码逻辑 |
|------|---------|
| 关键 K | `量比 ≥ 1.5`（vol_ma20 含当日）且为阴线或十字星（`close≤open 或 实体/振幅 < 0.3`） |
| 击败 K | 关键 K 后 1~5 天内，**阳线**且 `close > 关键K.high`；每关键 K 只记首次（去重口径，n=67,699；旧文档全组合 53.7 万不可比） |
| 努力比 | `击败K量比 / 关键K量比`（量比的量比） |
| 破位 | 10 日窗口 `min(close) ≤ 关键K.low` |

### 13 engulf 吞没（`patterns/engulf.yaml`）

| 用词 | 代码逻辑 |
|------|---------|
| 看跌吞没 | K1 阳线（close>open）+ 紧邻 K2 阴线；**完全吞没** = `K2.open ≥ K1.close 且 K2.close ≤ K1.open`（即红 1.1→1.2、绿 1.2→1.1） |
| engulf_pct | `(K1.open - K2.close)/K1.open × 100`，>0 = 收盘吞过红 K 开盘价 |
| 过滤 | K1 实体涨幅 ≥ 1%、前 20 天（以绿 K 日收盘计）涨幅 ≥ 5% |
| 量比 | 红绿各自 `vol/vol_ma20`（含当日）；分组用绿 K 量比 |

### 14 breakout_pullback 突破与回踩（`patterns/breakout_pullback.yaml`）

| 用词 | 代码逻辑 |
|------|---------|
| 突破日 | `volume ≥ vol_ma20×2`（含当日）且 `close > 前20日最高收盘`（prior_max，**收盘口径、不含当日**） |
| 回踩确认 | 突破后 10 个交易日内首个 `close ≤ prior_max × 1.02` |
| 缩量约束 | `mean(volume[t0+1..t1]) ≤ vol_ma20[t1] × 1.0`（vol_ma20 取 t1 处） |
| **放量（文档中"量比"）** | `volume / vol_ma20`，分母含当日 |
| 过度缩量 | 回调段均量 / vol_ma20 < 0.7 |
| 确认期最大回撤（细分） | `(close[t0] - min(low[t0+1..t0+10])) / close[t0]`（low 口径） |
| 回吐比例 | 确认期最大回撤 / 突破日相对平台涨幅 |
| **对照实验口径** | 不回踩组/回踩组：同突破日定义，按 10 日内是否 `close ≤ prior_max×1.02` 分组；突破日基准 = t0 收盘；确认日基准 = t0+10 收盘；混合期望实验为**事件化去重**（每突破事件后跳过 10 天） |

### 23 绝望期家族（`tools/despair_check.py` 单日口径 / `tools/despair_cycle.py` 周期口径）

| 用词 | 精确定义 |
|------|---------|
| 涨跌家数 | 逐股按 `close[i]/close[i-1]`（前复权）判定涨/跌/平，按日汇总全市场（min_data_days=180 过滤后） |
| 绝望日 | 当日**下跌家数占比 ≥ 80%**；有效截面（当日有涨跌数据的股票）≥ 1,500 只 |
| episode | 交易日历上**相邻**（连续交易日）的绝望日串；episode 末日 = 串最后一日；单日口径买入价 = 末日收盘 |
| 单日分组 | 绝望日当日个股涨跌幅：逆行者 >0% / 抗跌者 -1~0% / 承压 -5~-1% / 大跌 ≤-5%（单日带宽） |
| 绝望期确认 T0（初期） | zz1000 20 日累计收益 ≤ -10% 且过去 20 日（含 T0）绝望日 ≥ 2 个 |
| 低点 L | T0 起至企稳确认日之间的指数最低收盘日（收盘口径，非盘中 low） |
| 企稳确认 T1（后期） | 自 L 反弹（收盘）≥ 7% 且 T1 − L ≥ 10 个交易日；T1 后扫描指针跳至 T1+1，一年内无企稳则放弃该期 |
| 主跌段（分组基准） | 个股**自身日历**上 T0−20 交易日收盘 → L 日收盘的累计收益（T0−20 = 指数下跌起点近似）；分组带宽：逆行者 >0% / 抗跌者 -5~0% / 承压 -15~-5% / 大跌 ≤-15%（周期级带宽，与单日分组不同） |
| 反弹段 | 个股 L 日收盘 → T1 收盘收益（仅污染检查用，不入分组——刨除弹性股冒充抗跌者） |
| 企稳后收益 | T1 收盘买入（企稳日收盘成交），day horizon 口径；超额 = 个股收益 − zz1000 同期 |

---

## 三、相对强度（RPS）口径

| 用词 | 精确定义 |
|------|---------|
| rel_ret_120d | `(个股120日收益 − 中证1000同期收益)`，百分比；指数按日期 reindex + ffill 对齐 |
| RPS 截面分位 | **按 base_date 当日**全市场样本对 rel_ret_120d 降序排名百分位（0~99）——混合全期的排名会被年度效应污染，必须逐日截面 |
| enrich 机制 | pattern YAML 声明 `enrich: [rel_ret_120d]` + `enrich_index: zz1000`；engine 逐股计算并按 (code, 契约日期) 附加到输出，检测器无感知 |
| 指数数据 | `~/Documents/mainland_index_data_2014/`（cyb/hs300/kcb/sh/sz/zz1000） |

已接入 enrich 的 pattern：base_break、zfzs、spike_top、consolidation
（结论见 `discover_15`：RPS 对超跌类形态冗余，横盘形态的
高位过滤有效——RPS≥80 优 11.3pct，分年 10/10）。

## 四、各形态 RankIC 信号列映射（复刻必读）

RankIC = `spearman(信号列, 主 horizon 收益)`，按形态：

| 形态 | 信号列 | 主 horizon | RankIC | 解读 |
|------|--------|-----------|--------|------|
| base_break | penetration_pct（跌破深度） | ret_4w | **+0.135** | 强：越深越反弹 |
| spike_top | down_pct（尾段跌幅） | ret_90d | **-0.210** | 最强：跌越深反弹越强 |
| main_dd50 | max_drawdown_pct | ret_120d | **+0.205** | forward 口径回撤大→起点低→回归 |
| long_consolidation | vol_shrink_ratio | ret_120d | **+0.174** | 量能不缩=强 |
| flat_zone | decline_pct（前置跌幅） | ret_200d | -0.093 | 中等：跌深长线好 |
| fractal_bottom | depth（分型深度） | ret_20d | +0.091 | 中等：深分型可靠 |
| effort_defeat | effort_ratio（努力比） | ret_5d | -0.080 | 中等：量小击败可靠 |
| breakout_pullback | pullback_vr（回调段量比） | ret_20d | +0.079 | 中等：量足回调好 |
| fractal_top | depth | ret_20d | -0.056 | 中等负向 |
| consolidation | duration_days | ret_120d | -0.039 | 弱：横盘久反而弱 |
| engulf | engulf_pct（吞没深度） | ret_20d | -0.038 | 弱：形态强度无信息 |
| pullback_zf / _zs | dist_pct（距线） | ret_20d | +0.032 / +0.020 | 弱/无效 |
| oversold | decline_pct | ret_90d | -0.005 | 无效：深跌只分"反弹与否"，不分"反弹多少" |
| zfzs / volume | —（二元事件，无连续信号） | — | N/A | 用胜率相对基准增量（+4.6pct / -2.8pct） |

### 扩展系列（16~21）信号列

| 形态 | 信号列 | 主 horizon | RankIC | 备注 |
|------|--------|-----------|--------|------|
| double_bottom / double_top | depth_pct（形态深度） | ret_20d | -0.016 / **-0.071** | 双底无效；M 头深度是唯一有效维度 |
| gap_up / gap_down | gap_pct（缺口大小） | ret_20d | +0.041 / -0.016 | 缺口大小弱有效；核心分野在回补事件 |
| pv_divergence_top / _bottom | vol_ratio_peaks（峰对量比） | ret_20d | -0.022 / +0.008 | 萎缩"是否"有效、"多深"无效 |
| flag_breakout | flag_range_pct | ret_20d | -0.042 | 形态稀缺（n=1,939） |
| cup_handle | cup_depth_pct | ret_20d | **-0.096** | 杯越深越差（前高阻力区） |
| vol_squeeze_breakout | bb_width_min | ret_20d | — | 有效维度是挤压时长（40 天+） |
| candle_*（四种） | —（按背景分组） | — | ≈0 | 形状无信息（discover_20） |
| wyckoff_spring | spring_depth_pct | ret_20d | **+0.066** | spring 越深越强 |

**新增事件口径**：gap 回补（gap_filled_20d：收盘回到缺口对沿）、
双底破底（break_bottom_20d）、spring 再跌破（rebreak_20d——注意
支撑就在 base 下方 1~2%，该事件对浅回踩过敏，见 discover_21 口径警告）。

## 五、背景基准率（所有胜率/下跌率的对照基线）

胜率脱离基准无意义——形态的"有效率"= 相对**同背景随机基准**的增量。
以下为全市场 2014-01 ~ 2026-08 的 20 日收益基准（n 为交易日样本数）：

| 基准 | n | 20 日上涨率 | 20 日下跌率 | 中位 |
|------|-----|-----------|-----------|------|
| 全市场任意日 | 1,074 万 | 48.9% | 50.5% | -0.21% |
| 前 20 天涨 ≥5% | 338 万 | 46.6% | 53.0% | -0.93% |
| 前 20 天涨 ≥20% | 27.6 万 | 43.8% | 56.0% | -2.41% |
| 前 20 天涨 ≥40% | 6.6 万 | 38.1% | 61.8% | -5.94% |

含义：任何"前 20 天涨 X% 背景"的形态，其下跌率须先减去对应基准再谈
形态自身的信息量（例：吞没涨≥40% 组下跌率 63.6%，背景基准 61.8%，
形态增量仅 +1.8pct）。**"大涨背景"本身自带强看跌基准**，是形态统计中
最大的混淆源。

## 六、持有与卖出研究口径（discover_24~28，规划中）

> 规划见 `holding_or_sell_upgrade.md`；以下口径在研究开展前锁定，
> 产出文档（24~28 篇）必须逐字引用本表。

### R1 均线破位确认退出（discover_27）

| 用词 | 精确定义 |
|------|---------|
| 均线上穿（入场） | `close[i-1] ≤ MA[i-1] 且 close[i] > MA[i]`——与金叉（上表）**同构的交叉定义**，边界值（恰好等于）归"未站上" |
| N 日破位确认（退出） | 连续 N 日 `close < MA` 即于**第 N 日收盘**卖出 |
| 卖飞率 | 退出日后 20/60 日内 `max(close)/退出价 − 1 ≥ +5%`（或 +10%）的样本占比（close 口径，非 high） |
| 卖飞错失幅度 | 退出价 → 退出后 60 日 `max(close)` 的涨幅分布 |
| 卖对率 | 退出后 60 日内 close **持续**低于退出价的样本占比 |
| 再入场频率 | 退出后 20 日内重新触发"均线上穿"的样本占比（鞭打成本） |
| 交易成本 | 单边 0.15%（费用+滑点合并），规则净收益按往返 0.30% 扣减；年换手 = 样本平均年退出次数 |
| 个股波动率分层 | 分层变量 = 个股 60 日日收益 std（具体分位切点研究时锁定） |

### R2 大级别单顶（discover_28）

| 用词 | 精确定义 |
|------|---------|
| 大级别单顶 | 前置：顶点 close ≥ 250 日前 close × 2；后验：顶点后 250 日 close 不再创新高 且 自顶点回撤 ≥ 40%；单顶：顶点 ±5% 价格带在前后各 250 日内 **close 触及仅 1 次**（close 而非 high，排除影线） |
| ⚠ 与 06 单顶的区别 | 本定义是**日线价格带触及次数**口径；06 spike_top 的"单顶"是**周线 high 峰结构**口径（周 max(high) 主峰×0.9 判定），两者不可互引 |
| 假顶（对照组） | 满足前置涨幅条件、但之后 250 日内 close 创新高的"中途顶" |
| 顶点量比 | 默认量比口径（`volume/vol_ma20`，**分母含当日**；注意与 04 base_break 的不含当日口径不同） |
| 顶部前量能趋势 | 顶点前 N 日量峰位置 vs 价格峰位置（量峰先于价峰 = 顶背离泛化；具体 N 研究时锁定） |

### 通用（24~26 篇）

| 用词 | 精确定义 |
|------|---------|
| MFE / MAE | **直接采用** analyze 的 `maxup_Nd / maxdn_Nd`（含基准日、high/low 口径）——与经典 MFE/MAE 的差异：本项目口径**含基准日当天** |
| 持有期衰减曲线 | 同一形态的 E/胜率随 day horizon（20/40/60/90/120/180）的变化 |
| **止损扫描（静态近似）** | 对档位 s 与窗口 N：**触发** = `maxdn_Nd ≤ s`（low 口径、含基准日）；触发样本收益记 `s`（按止损价成交），未触发样本收益记 `ret_Nd`；**止损版期望** `E_stop = mean(触发 ? s : ret_Nd)`。指标：触发率、误伤率（触发样本中 `ret_Nd > 0` 占比）、误伤代价（触发赢家的原 ret 均值）。⚠ 静态近似：以窗口极值判定，忽略盘中触发时点与"先涨后跌"顺序；maxdn 含基准日意味着**基准日当天**跌穿 s 亦触发 |
| **止盈扫描（静态近似）** | 对档位 t 与窗口 N：**触发** = `maxup_Nd ≥ t`；触发样本收益记 `t`，未触发记 `ret_Nd`；`E_tp = mean(触发 ? t : ret_Nd)`。右尾代价 = 触发样本的原 `ret_Nd` 均值 − t（被砍掉的期望延续） |
| **重放成交假设（exit 引擎）** | 信号日收盘买入（base_price）、触发日收盘卖出；持有路径自 base_idx 起（含基准日） |
| **移动止盈（重放）** | peak = 持有期至今最高收盘；激活 = `peak/base − 1 ≥ X%`；触发 = `(peak − close)/peak ≥ Y%`；退出 = 触发日收盘。未激活前不触发 |
| **时间止损（重放）** | 第 N 交易日收盘 `close < base`（浮亏版）→ 第 N 日收盘退出 |
| **均线确认退出（重放）** | 连续 N 日 `close < MA` 后于第 N 日收盘退出（R1 节定义的逐日实现） |
| **状态退出（重放）** | 逐日 `close < 行透传阈值列 ref_col` → 当日收盘退出（如 base_low、spring 支撑） |
| **规则集** | 多规则 OR 组合，任一触发即退出；全部未触发持有至 max_horizon 期末收盘平仓 |
| **样本分段** | fit = 2014-2021、validate = 2022-2026（按 base_date 前 4 位）——规则回测必须双段报告 |
| 移动止盈峰值 | 持有期内**最高收盘**（close 口径，与波段级 peak_price 一致） |
| 持有期回撤深度 | `(close[t0] − min(low[t0..t]))/close[t0]`（low 口径，与 14 篇确认期回撤一致） |
| 单位时间收益 | 规则净收益（扣 0.30% 往返成本）/ 平均持仓交易日数 |

## 七、常见歧义点清单（复刻前必读）

1. **量比分母两种口径**：默认 `vol_ma20` 含当日；**仅 base_break 的量比**分母为前 20 日均量（不含当日）。
2. **事件 vs maxup 的价格口径**：事件（cross_above/below）用**收盘价**；maxup/maxdn 用**最高/最低价**。
3. **窗口含基准日**：maxup/maxdn/事件的窗口从 base_idx 起（含基准日当天）。
4. **周 = 真自然周**（ISO，周一起始）：base_break、spike_top 单顶、analyze 的 week horizon 均如此；项目内不存在 5 日块伪周线。
5. **前复权**：跨除权日价格直接可比，无缺口过滤。
6. **胜率的严格性**：`ret > 0` 才算胜，平局计入负。
7. **P15/P85 为 pandas 线性插值分位**，非最近邻。
8. **收益用收盘价**（不评估盘中可达性）；base_price 各形态不同（见逐节"base"行）。
9. **去重口径**：spike_top 按 (code, peak_date, half_date)；effort_defeat 每关键 K 一次；breakout_pullback 处理后跳至 t1+1。
10. **分年度稳健性**的年份取 `base_date` 前 4 位；剔除对照剔除的是整年样本。
11. **波段级 vs analyze 的回撤**：波段级 `max_drawdown_pct`（close 口径、仅峰后段）≠ analyze `maxdn_Nd`（low 口径、含基准日）——同一文档混用两者是最常见错误。
12. **两种"单顶"**：R2 大级别单顶（日线 ±5% 价格带 close 触及次数）≠ 06 spike_top 单顶（周线 high 峰结构，主峰×0.9 判定）——不可互引。
13. **绝望期主跌段起点** = 个股**自身日历**的 T0−20 交易日（近似指数下跌起点），非指数日历、也非 T0 本身（T0 已是"跌完 −10%"的确认日）。
14. **绝望期两组分组带宽不同**：单日口径（逆行 >0 / 抗跌 -1~0 / 承压 -5~-1 / 大跌 ≤-5）与周期口径（>0 / -5~0 / -15~-5 / ≤-15）——同名组在不同工具中带宽不同。
