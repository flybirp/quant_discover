# 发现：横盘家族（窄幅 / 超长期宽幅 / 下跌后平台）

> 生成: 2026-08-31 | patterns: `consolidation` / `long_consolidation` / `flat_zone`
> 数据: 全市场 5062 只个股日线（2014-01 ~ 2026-08），前复权

三个横盘形态统一对照——**同一个问题：横盘本身是不是买点？**

## 三形态定义

| 形态 | 定义 | 契约基准 |
|------|------|---------|
| `consolidation` 窄幅横盘 | ≥60 天，窗口均价 ±10% 内（贪婪扩展） | 横盘结束日收盘 |
| `long_consolidation` 超长期宽幅震荡 | 400~600 天，振幅 ≤50%、无强趋势、波动率收敛 | 区间末日收盘 |
| `flat_zone` 下跌后平台 | 加速下跌后 ≥15 天、波幅 ≤5% 横盘 | 横盘均价 |

## 后续收益对照

| 形态 | 持有期 | 样本 | E（期望） | 中位 | 胜率 |
| --- | --- | --- | --- | --- | --- |
| 窄幅横盘 | 60d | 31,947 | +3.77% | +0.22% | 50.4% |
|  | 90d | 31,152 | +5.49% | +0.96% | 52.1% |
|  | 120d | 30,579 | +7.41% | +0.79% | 51.3% |
| 超长期宽幅 | 120d | 4,222 | +1.25% | -2.11% | 45.0% |
|  | 250d | 3,841 | +5.31% | -1.73% | 47.3% |
| 下跌后平台 | 30d | 3,996 | -0.01% | -0.52% | 46.8% |
|  | 90d | 3,868 | +1.09% | -2.08% | 43.5% |
|  | 200d | 3,605 | +5.47% | -3.02% | 45.1% |

**RankIC**：窄幅横盘 duration_days × ret_120d = **-0.039**（横盘越久
反而越弱，弱负相关）；下跌后平台 decline_pct × ret_200d = **-0.093**
（前置跌得越深，长期收益越好，中等强度）；超长期震荡
vol_shrink_ratio × ret_120d = **+0.174**（量能不缩 = 强信号，
与 `discover_11` 量能区分度结论互证）。

## 结论

1. **横盘本身不是买点**。三个口径（窄幅 / 超长期 / 下跌后）胜率全部在
   43~52% 区间，中位数 ≤1%——均值微正完全由右尾贡献（少数大突破）。
   这推翻"横有多长竖有多高"作为**单独入场理由**的用法。
2. **均值 > 中位 ≈ 期权结构**：横盘的统计价值是右尾（向上突破的少数
   大行情），对应策略是"横盘 + 突破触发"而非"横盘即买"。旧文档
   `discover_下跌横盘.md` 的结论（下跌+横盘不是独立买点、价值在右尾）
   在新数据上**再次成立**。
3. 下跌后平台（flat_zone）是三者中最弱的：90 日中位 -2.1%、胜率 43.5%。
   加速下跌后的横盘若不叠加深跌破（见 `discover_04` C/D 组）或放量
   突破确认，承接并不成立。
4. `long_consolidation` 的 21 个特征列（量能收敛、ATR 收敛、均线缠绕等）
   已透传在 `output/catch_consol_scanner.csv`，可供后续"突破方向预测"
   专题做特征筛选（旧文档 `横盘震荡_discover.md` 的方向三分类已废除，
   改用 maxup/maxdn 连续指标）。

## 复现

```bash
python3 run.py catch patterns/consolidation.yaml && python3 run.py analyze analyzers/consolidation.yaml
python3 run.py catch patterns/long_consolidation.yaml && python3 run.py analyze analyzers/long_consolidation.yaml
python3 run.py catch patterns/flat_zone.yaml && python3 run.py analyze analyzers/flat_zone.yaml
```
