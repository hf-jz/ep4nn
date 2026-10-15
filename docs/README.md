# ep4nn 文档索引

**英文论文与站点：** [hf-jz.github.io/ep4nn](https://hf-jz.github.io/ep4nn/)（源码 `docs/paper.md`，仓库 `github.com/hf-jz/ep4nn`）。

**一句话判语：** Cadence 的接线是真的、功能不是从接线里长出来的；它的扩展性是"稀疏图的扩展性"而不是"神经元数的扩展性"；
EP 是条件普遍的局部伴随法（有向真实接线上梯度语义失效）；ep4nn 把"何时增补结构"做成可证伪的判据，但目前卡在
**读出不动**这一步（不是容量不足）。详见各文件。

## 阅读顺序

1. `cadence-判定.md` —— 302 线虫那条到底是真是假（逐条核实 12 项主张）。
   （根目录 `REPORT.md` 是同内容的早期副本，以本目录为准。）
2. `果蝇生态审阅.md` —— 8 个果蝇脑项目对照：接线问题已解，动力学与映射是瓶颈。
3. `实测台账.md` —— 所有数字（tick 成本、缩放曲线、证书、控制负结果、增长 demo）。
4. `创新点.md` —— 六条创新点 + 明确不宣称的部分。
5. `方案.md` —— Phase 0–4 详细方案、关键决策、benchmark 与收据格式。
6. `ep4nn_discussion.md` —— 讨论原文（问题、推演、判语全过程）。
7. `paper.md` —— 论文正文（英文，Markdown 源）；`paper/ep4nn.tex` 是 LaTeX 正式版（编译产物 `ep4nn.pdf`）；`index.html` 是站点首页。
   论文正文另出 LaTeX 版：`paper/ep4nn.tex`（xelatex，`bash docs/paper/build.sh` 两条编译到 `docs/ep4nn.pdf`，12 页，0 overfull，无未解析引用），站点上 "Read the paper"/"PDF" 指向它。作者单位统一 Hefei Jiuzhai Big Data Technology Co., Ltd.。
   页面布局（配色 `--accent:#1d4e89`、`--maxw:1080px`、62px 吸顶导航、`.hero` 1.35fr/.65fr、`.result` 软色块、`.titem` 左边框、`.row` 290px/1fr、`.panel` 双栏、`pre` 复现块）逐行取自同一作者的 `hf-jz/eep` 的 `gep/` 页面，CSS 原样搬运。
   `assets/make_figures.py` 一条命令重画四张图，数据来自 `probes/` 与 `assets/prereq_scan.json`。
8. `receipts/` —— 运行收据（`ep4nn/receipt.py` 写出；六字段：dataset / run / dynamics / loop / premise / cost）。

## 现状一张表

| 层 | 状态 | 证据 | 缺口 |
|---|---|---|---|
| 前提层（工作点 + 读出响应） | **未通过（已量化）** | 60 工作点扫描最好 spread 0.0777、余弦仍 1.000（秩一）；`grow.py` demo 的 spread 0.019 | 换读出 / 换动力学（Shiu 2024 LIF + 异质细胞）/ 按协议选增益 |
| 增长层（需求→证据→生成→裁决→回滚） | 机制已验证，成效未验证 | 1200 tick：拒绝 1200 次、无假动作、自检过；早期无前提闸版本：30 次长出后回滚 | 一次 kept 增长（`before→after` 余弦下降可复核） |
| 数值层（结算与预算） | 有 2–4x 余量 | 输运 1.78x、迭代 2.5–4.2x、活动集证书 α=0.696 | 库内实现（探针层套壳不省墙钟） |
| 结构先验（距离/延迟） | 未开始 | Cadence 全库无位置、无延迟（grep 0 命中） | 距离核（不动契约）→ 延迟（换契约） |
| 演化层（内生价值尺度） | 未开始 | Cadence 已有 `genes`/`ArousalConfig.space()`/`evolve` | 世代实验与收据 |
| 文化层（共享经验库） | 未开始 | 每个脑独立，无教学/模仿通道 | 通道 + 同步偏见风险控制 |

## 代码与数据

- `ep4nn/grow.py` —— 增长机制（13 个基因的生成器 + 前提闸 + 成效裁决），`python ep4nn/grow.py` 自带断言自检。
- `probes/` —— `probe_settle_cost` `probe_accelerate` `probe_endtoend` `probe_transport_ab` `probe_certificate`
  `probe_profile` `probe_worm_graph` `probe_scaling` `probe_control_plant` `probe_rhythm`。
- `data/worm_connectome.json` —— 302 神经元（61 感觉 / 91 中间 / 118 运动 / 10 命令 / 22 咽部），3,137 个有向突触类，10,133 接触，密度 3.44%。

## 一条命令跑通全部

```bash
cd /Users/sheng/math/ep4nn
export PYTHONPATH=/Users/sheng/math/cadence/src
P=~/.hermes/cache/scratch/cad-venv/bin/python
$P ep4nn/grow.py && for f in probes/probe_*.py; do echo "== $f"; $P "$f" >/dev/null || echo FAIL; done && echo ALL-OK
```
