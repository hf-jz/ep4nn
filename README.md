# ep4nn

从 **302 神经元线虫接线**起步，随交互复杂度增长、**按证据**增补神经元与突触的研究系统。

一句话现状：**前提层与裁决层已跑通，增长机制已实现并拒绝在不该长的时候长（1200/1200）；尚无一次成功增长**，
原因不是增长层，而是上游——读出（AVB/PVC/AVA/AVD/AVE）在当前工作点下只动 0.019，答案本身不动，
所以系统按设计拒绝增长。诊断与下一步见 `docs/方案.md` 的 Phase 0。

## 目录

```
ep4nn/grow.py          增长机制本体（需求→证据→生成→裁决→回滚，含 __main__ 自检）
probes/                10 个实测探针（Cadence 数值、缩放、证书、控制负结果）
data/worm_connectome.json  302 神经元真实连接组（openworm/c302，sha256 见文件头注释）
docs/                  全部汇总：判定 / 生态 / 创新点 / 方案 / 实测台账 / 讨论原文
```

## 跑

```bash
cd /Users/sheng/math/ep4nn
PYTHONPATH=/Users/sheng/math/cadence/src ~/.hermes/cache/scratch/cad-venv/bin/python ep4nn/grow.py
# 环境与全部复现命令：docs/实测台账.md
```

## 文档

| 文件 | 内容 |
|---|---|
| `docs/README.md` | 索引与判语 |
| `docs/创新点.md` | 六条创新点（逐条标已验证/待验证）+ 明确不宣称的部分 |
| `docs/方案.md` | Phase 0–4 详细设计、8 条关键决策、benchmark 与收据格式 |
| `docs/实测台账.md` | 本会话全部实测数字与复现命令 |
| `docs/cadence-判定.md` | Cadence 与那篇公众号文章的逐条核实 |
| `docs/果蝇生态审阅.md` | 8 个果蝇脑项目的对照审阅 |
| `docs/ep4nn_discussion.md` | 讨论原文存档 |
