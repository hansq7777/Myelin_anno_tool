# Review Tracker 人工测试流程（Linux/Windows 通用）

本文用于人工验证以下目标：

1. 可从 `raw` 与 `prediction` 文件夹自动构建 tracker（`xlsx/csv`）
2. UI 读取时随机抽取未完成样本
3. 已完成样本不会再参与随机抽取
4. 保存、重开后状态稳定

## 0. 环境准备

1. 安装依赖：
```bash
pip install -r requirements.txt
```
2. 启动：
```bash
python -m zstack_anno
```
Linux 也可：
```bash
./start_gui.sh
```

## 1. 准备一组最小测试数据

在临时目录准备：

- raw: `A01.ome.tif`, `A02.ome.tif`, `A03.ome.tif`
- pred: `A01.pred.ome.tif`, `A03.pred.ome.tif`, `B99.pred.ome.tif`

期望配对关系：

- matched: `A01`, `A03`
- raw_only: `A02`
- pred_only: `B99`

## 2. 构建 Tracker（首次）

1. 打开 GUI，点击 `Build Tracker`（或 `Review -> Build Tracker from Folders...`）
2. 选择 raw 文件夹
3. 选择 prediction 文件夹
4. 保存 tracker（先选 `.xlsx`）

验收：

1. 弹窗统计中 `matched/raw_only/pred_only` 数量正确
2. tracker 文件已生成
3. 点击 `Open Tracker` 后可成功加载
4. 只有 `matched` 且路径存在的样本进入可 review 列表

## 3. 随机未完成抽取验证

1. filter 设为 `Unreviewed`（默认通常就是）
2. 连续点 `Next Stack` 多次

验收：

1. 每次加载的是未完成项（状态栏或信息栏显示 `PENDING`）
2. 在可选项未耗尽前，尽量不重复
3. 所有未完成样本都被看过一轮后，允许再次随机轮转

## 4. 完成标记与排除验证

对当前样本：

1. 做少量编辑（可选）
2. 点击 `Save Corrected Mask`

验收：

1. 生成 `review_corrected_masks/<GRADE>/...` 文件
2. tracker 行写入：
   - `review_corrected_mask_path`
   - `review_corrected_saved_at`
   - `review_completed = 1`
   - `review_completed_at` 有时间戳
3. 再次随机 `Next Stack` 时，该样本不再被抽中（除非切到 `A/B/C` 过滤手动查看）

## 5. 重启稳定性验证

1. 关闭 GUI
2. 重新启动并 `Open Tracker`

验收：

1. 已完成样本状态仍保持
2. 随机抽取依旧只从未完成集合中抽
3. 数据不丢失、不重置

## 6. CSV 兼容验证

重复第 2-5 步，但 tracker 保存为 `.csv`。

验收：

1. `.csv` 能正常打开与保存
2. 完成标记字段同样生效
3. 随机未完成抽取逻辑一致

## 7. 兼容历史 Tracker 验证（回归）

用旧版 tracker（可能不含新字段）执行 `Open Tracker`。

验收：

1. 可正常加载
2. 缺失字段会自动补齐（如 `review_completed`, `review_completed_at`）
3. 不影响旧有 `review_grade/review_status/...` 字段内容

## 8. 导出回归验证

1. 对若干样本打 A/B/C（含有 corrected 与无 corrected 两种）
2. 执行 `Export Final Masks`

验收：

1. `review_final_masks/<GRADE>/...` 文件正确生成
2. tracker 中 `review_final_mask_path/review_final_mask_source/review_final_exported_at` 正确写入

## 常见失败点排查

1. tracker 无法保存：检查是否被 Excel 占用
2. Linux 下路径找不到：设置
```bash
export ZSTACK_WINDOWS_DRIVE_MAP="D=/data/confocal;E=/mnt/extra"
```
3. 某条样本不出现：通常是 raw/pred 任一路径缺失或文件不存在
