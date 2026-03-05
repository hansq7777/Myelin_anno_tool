# Myelin Line-Structure Segmentation — Merged TODO (ZH)

本文件是当前唯一维护版本。
目标是基于当前代码真实覆盖情况，给出“已实现评估 + 未实现实现清单（按效用优先）”。

## 评估范围
- 仅非深度学习/无监督方法。
- 场景为银染共聚焦反射髓鞘 Z-stack。
- 关注分割、连通、骨架、方向统计、Z 向衰减校正与断裂补全。

## 当前阶段任务目标（新增：预测质控与训练集扩充）
- [ ] G01: 建立统一审核台账（以 `zstack_annotation_tracker_2026-02-06.xlsx` 为主），确保每个 zstack 记录包含：原图路径、预测路径、匹配状态、审核状态、是否入下轮训练。
- [ ] G02: 定义样本分级规则并固化字段：`A=可直接入训练`、`B=轻量校正后入训练`、`C=暂不入库`。
- [ ] G03: 增加“人工干预成本”记录项（例如修改切片数、修改像素占比、耗时），用于判断样本是否值得纳入微调集。
- [ ] G04: 先完成一轮快速审核筛选，优先处理“预测完成度高、改动量低”的样本，形成首批微调数据池。
- [ ] G05: 输出下一轮训练清单（含 `train/val` 划分、质量等级、数据来源批次），并生成版本号（如 `finetune_pool_v1`）。
- [ ] G06: 建立闭环：训练后回填“该样本对新模型收益”的评估标签，持续优化筛选标准。

## 当前能力盘点（代码核验）

评分定义：
- `U`（效用）: 1-5，越高越关键。
- `D`（部署难度）: 1-5，越高越难。
- `P`（优先分）: `0.7*U + 0.3*(6-D)`。

| ID | 方法 | 当前状态 | 代码入口（已存在） | 主要缺口 | U | D | P |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M03 | Frangi | 已实现 | `morphology_tools.frangi_filter_slice` + script/menu | 需要参数基线和批处理模板 | 5 | 2 | 4.7 |
| M04 | Sato/Tubeness | 已实现 | `sato_filter_slice` + script/menu | 同上 | 5 | 2 | 4.7 |
| M05 | Meijering | 已实现 | `meijering_filter_slice` + script/menu | 作为对照策略参数需标准化 | 3 | 1 | 3.9 |
| M06 | Hessian | 已实现 | `hessian_filter_slice` + script/menu | 可加 3D 版本/参数建议 | 3 | 2 | 3.6 |
| M01 | 全局阈值（Abs/Norm） | 部分实现 | `threshold_absolute/normalized` | 缺 Otsu/百分位自动策略 | 4 | 1 | 4.5 |
| M02 | 局部自适应阈值 | 未实现 | 无 | 缺 local mean/gaussian threshold | 4 | 2 | 4.0 |
| M12 | 断裂连接（各向异性闭合） | 部分实现 | `close`（按孔洞填充） | 不是各向异性形态学闭合，不能稳定桥接 gap | 5 | 2 | 4.7 |
| M13 | 3D 连通域 | 未实现 | 当前仅 `label_components` 2D | 缺跨 Z 组件标记与体素统计 | 5 | 2 | 4.7 |
| M14 | 3D 骨架化 | 已实现（可选依赖） | `skeletonize_stack(algorithm=\"skeletonize_3d\")` | 缺拓扑统计与长度导出 | 5 | 2 | 4.7 |
| M08 | 结构张量方向分析 | 部分实现 | `structure_tensor_eigen_slice` | 仅特征值，无方向角/coherence/直方图 | 4 | 2 | 4.0 |
| M07 | Gabor 方向滤波 | 部分实现 | `gabor_filter_slice`/`cv_gabor_filter_slice` | 仅单方向；缺多角度 filter bank 和方向图 | 3 | 3 | 3.0 |
| M11 | 图模型连通补全 | 部分实现 | `shortest_path_slice` | 缺端点检测、方向约束、批量连接策略 | 4 | 4 | 3.4 |
| M09 | Radon 方向检测 | 未实现 | 无 | 缺 patch-based Radon 与方向统计 | 2 | 4 | 1.8 |
| M10 | Canny + 中心线细化 | 未实现 | 无 | 缺边缘入口与后处理链路 | 2 | 2 | 2.6 |

补充能力缺口（不在 M01-M14 但项目很关键）：
- Z 向衰减/层间不一致校正：未实现。
- 骨架拓扑指标：端点、分支点、长度、曲率、郎飞结候选检测：未实现。
- 3D 方向分布（体素尺度校正后）：未实现。

## 实现清单（仅缺口，按优先级）

### P0（先做，直接影响主线可用性）
- [ ] T01: 新增 Otsu 与百分位自动阈值（M01 缺口）。实现位置：`morphology_tools.py` + `script_helper.py` + `script_editor.py` + `main_controller.py` + `pipeline.py`。
- [ ] T02: 新增局部自适应阈值（M02）。实现位置同上；支持 `block_size`、`offset`、`mode` 参数。
- [ ] T03: 新增真实各向异性闭合（M12 缺口）。实现位置：`morphology_tools.py`；支持按 `(z,y,x)` 结构元和物理像素尺度输入。
- [ ] T04: 新增 3D 连通域标记与统计（M13）。实现位置：`morphology_tools.py` + `zstack_model.py` + `pipeline.py`；输出组件数/体素数/体积。
- [ ] T05: 新增骨架拓扑统计（M14 延伸）。实现位置：`morphology_tools.py` + `pipeline.py`；至少包含端点数、分支点数、总长度。

### P1（强烈建议，决定科研分析质量）
- [ ] T06: 结构张量输出方向角与 coherence（M08 缺口），并导出方向直方图。
- [ ] T07: Gabor 多角度 filter bank（M07 缺口），输出最大响应图和主方向图。
- [ ] T08: Z-stack 衰减校正（新增关键项），至少包含“逐层高分位归一”和“指数拟合校正”两种模式。
- [ ] T09: 图模型端点连接（M11 缺口），在最短路径上加入距离阈值+方向连续性约束。

### P2（增强项，放在主线稳定后）
- [ ] T10: Radon patch 方向检测（M09）。
- [ ] T11: Canny->细化路径（M10），作为噪声高样本备选流程。
- [ ] T12: 郎飞结候选检测（端点对匹配 + 共线 + 距离阈值）。
- [ ] T13: 3D 方向分布统计（方位角/倾角）并考虑体素尺度校正。

## 落地规范（新增功能统一要求）
- 每个新增算法都要同时接入：`script_helper.py`、`views/script_editor.py`、`controllers/main_controller.py`、`pipeline.py`。
- 每个新增算法都要给最小测试：`tests/` 下增加单元测试（正常输入、空输入、参数边界）。
- 每个新增算法都要有输出可视化：至少支持在现有 UI 中看到 mask 或方向图结果。

## 决策备注
- 第一轮不建议优先做 Radon/Canny；先把阈值、连通、骨架统计、衰减校正补齐，收益最高。
- 图模型连接（M11）在 P1 才值得做，否则容易早期过拟合规则、拖慢主线迭代。
