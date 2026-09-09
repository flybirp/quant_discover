"""退出规则引擎（与 core=catch、analyze=固定窗口 同级的第三个引擎）。

架构（工程纪律）：
    replay.py  唯一重放实现（契约明细 + 日线 → 持有期路径）
    rules.py   规则组件（参数由 exit_rules/*.yaml 注入，禁止硬编码）
    metrics.py 评估（净成本、分段报告）

口径全部见 docs/appendix.md 第六节。
"""
