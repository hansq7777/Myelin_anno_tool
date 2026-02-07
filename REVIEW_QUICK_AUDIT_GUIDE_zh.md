# Myelin_anno_tool 快审与轻量校正使用手册（中文）

本手册对应当前 GUI 中的 `Review` 工作流，用于快速审核每组“原图 + 预测掩膜”，并做 A/B/C 分类与轻量修正。

## 1. 文件入口与启动

1. 进入项目目录：`C:\work\Myelin_anno_tool`
2. 安装依赖（首次）：`pip install -r requirements.txt`
3. 启动 GUI：`python -m zstack_anno`
4. 在 GUI 中点击：
   - 顶部按钮 `Open Review Tracker`
   - 或菜单 `Review -> Open Tracker...`
5. 选择你的跟踪表（`.xlsx`，例如 `zstack_annotation_tracker_2026-02-06.xlsx`）

说明：
- 程序会从表格中读取 `raw_path/raw_found_path` 与 `inference_found_path`。
- 只会加载“原图与预测路径都存在”的条目；缺失条目会跳过。
- 打开后默认切换到 `Unreviewed`，并自动加载第一条未审核样本开始工作（若无未审核则回退到 `All`）。

## 2. 界面布局与按钮功能

### 2.1 顶部导航区（切片级）

- `Prev` / `Next`：上一张/下一张 slice
- 中间滑条：在当前 zstack 内快速跳到指定 slice

### 2.2 Review 快审区（zstack 级）

- `Open Review Tracker`：打开 `.xlsx` 审核表
- `Prev Stack` / `Next Stack`：上一组/下一组 zstack
- `Filter`：筛选要处理的集合
  - `All`：全部
  - `Unreviewed`：未标记
  - `A` / `B` / `C`：只看对应类别
- `Grade`：当前样本类别（可直接切换）
- `Mark A` / `Mark B` / `Mark C`：一键打标并自动进入下一组
- `Save Corrected Mask`：保存你手动微调后的掩膜
- `Quick Auto Script`：对当前 slice 一键执行默认自动策略  
  `Seed -> Dilate -> Background Filter -> Intensity Grow -> Background Filter`
- 右侧状态条：显示当前位置、当前样本 ID、A/B/C/U 数量统计

### 2.3 画布与叠加显示

- 主画布默认显示原图，并叠加当前 mask（红色 overlay）
- 右下角 `Mask Visibility` 滑条调叠加透明度
- 可直接用现有工具做轻量修正（画笔、膨胀腐蚀、闭合、去小连通域等）

## 3. 推荐工作流（快审 + 轻改 + 入库决策）

1. 打开 tracker 后先将 `Filter` 设为 `Unreviewed`，只处理未审样本。
2. 对每组 zstack 快速浏览关键 slices（首、中、尾 + 问题层）。
3. 质量分级：
   - `A`：预测可直接入库（无需或几乎无需改动）
   - `B`：小幅修改后可入库
   - `C`：当前轮不入库（错分多、结构偏差明显、需大修）
4. 若为 `B`：
   - 用工具做轻量修正
   - 点 `Save Corrected Mask` 保存修正版
5. 完成后切换 `Filter=A` 或 `Filter=B` 做二次集中复核。
6. 导出时优先用 `A + B(已修正)` 作为下一轮训练候选。

## 4. 快捷键（高频）

- `Alt+,`：上一组 zstack
- `Alt+.`：下一组 zstack
- `Alt+1`：标记 A
- `Alt+2`：标记 B
- `Alt+3`：标记 C
- `Alt+Q`：运行 `Quick Auto Script`

## 5. 快捷键总表（UI 实际可用）

| 快捷键 | 功能 |
| --- | --- |
| `↑/←` | 上一张 slice |
| `↓/→` | 下一张 slice |
| `Alt+,` | 上一组 review stack |
| `Alt+.` | 下一组 review stack |
| `Alt+1` | 当前 stack 标记为 A |
| `Alt+2` | 当前 stack 标记为 B |
| `Alt+3` | 当前 stack 标记为 C |
| `Alt+Q` | 运行单张 `Quick Auto Script` |
| `Alt+W` | 运行 `Quick Auto Stack`（全栈/范围/关键切片） |
| `Alt+Shift+Q` | 回退到最近一次自动策略运行前快照 |
| `Alt+S` / `Ctrl+S` / `Meta+S` | Quick Save 掩膜 |
| `Alt+D` / `Meta+D` | 清除当前 slice 前景 |
| `D` | 膨胀当前掩膜 |
| `E` | 腐蚀当前掩膜 |
| `Z` | Undo |
| `X` | Redo |
| `P` | 画笔模式开关 |
| `[` / `]` | 缩小/增大画笔 |
| `H` | 切换手型拖拽 |

说明：
- `Meta` 对应 macOS 的 `Command` 键。
- 右键拖拽可删除触碰到拖拽框的连通域（无键盘快捷键）。

## 6. 自动策略进阶用法（新）

### 6.1 参数预设下拉

`Auto Preset` 提供 3 档：

- `Conservative`：保守，过分割风险低，漏检可能更高。
- `Balanced`：平衡，默认建议。
- `Aggressive`：激进，连通更充分，但过分割风险更高。

### 6.2 执行后质量门控（Quality Gate）

每次运行自动策略后会检查前景像素增幅是否异常。若超过阈值，会弹窗提示是否回退。

- 支持一键回退到自动策略执行前快照，不受“单动作 Undo”限制。
- 也可手动点 `Revert Auto Snapshot` 进行回退。

### 6.3 一键跑当前 stack（全栈/范围/关键切片）

点 `Quick Auto Stack` 后可选择：

- `All slices`：全栈执行
- `Slice range...`：输入起止范围
- `Key slices (first/middle/last)`：首/中/尾关键切片

执行结束后会停在最后处理的 slice，便于立即人工审核。

若批量执行中有切片触发质量门控，会给出汇总并可选择是否整体回退到运行前快照。

## 7. Excel 自动记录字段

程序会自动创建或更新以下列：

- `review_grade`：A/B/C
- `review_status`：
  - A -> `accept_direct`
  - B -> `accept_after_edit`
  - C -> `reject_or_later`
  - 未标记 -> `unreviewed`
- `review_note`：备注（预留）
- `review_updated_at`：最后标记时间
- `review_corrected_mask_path`：修正掩膜路径
- `review_corrected_saved_at`：修正掩膜保存时间

## 8. 修正掩膜保存位置

点击 `Save Corrected Mask` 后，文件写入：

`<tracker同目录>/review_corrected_masks/<GRADE>/`

文件名格式：

`<zstack_id>_review_mask.tif`

如果该文件已存在，会弹窗确认是否覆盖。

## 9. 常见问题

1. 打不开 tracker  
   先确认已安装 `openpyxl`（`requirements.txt` 已包含）。

2. 某些条目没加载  
   通常是表格路径为空，或文件实际不存在。程序会跳过并在状态栏提示。

3. Excel 无法写回  
   多数是表格正在 Excel 中占用；关闭后再标记/保存即可。

4. 保存后看不到修正结果  
   重新加载该样本时，程序会优先加载 `review_corrected_mask_path` 指向的修正版。
