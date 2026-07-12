# Metasurface Data Pipeline

这是一个模型无关的真实超表面数据处理 pipeline。它负责把原始采集图片、AprilGrid 标定板参数、超表面 ROI、真实 `A_matrix` 响应矩阵整理成后端模型可直接读取的 `.npz/.json` 数据。

后端模型可以是 3DGS、NeSpoF 或后续新的重建算法；这些模型不应再依赖 NeSpoF 里的历史数据处理脚本。

默认输出目录：

```text
code/metasurface_data_pipeline/outputs/
```

`outputs/` 已被 `.gitignore` 忽略，不应提交到 git。

## 目录结构

```text
metasurface_data_pipeline/
  calibration/     AprilGrid JSON 读取、tag 检测、内参/外参估计、board prior
  config/          ROI、输出路径、txt 配置文件解析
  core/            光谱 basis、A_matrix 响应提取、OpenCV ray 几何
  observations/    base dataset、sparse/dense observation、board-constrained 数据
  cli/             命令行入口
  compat/          历史兼容层 wrapper，不建议新代码直接使用
  examples/        可编辑的 txt 配置模板
  skills/          项目维护规则
  tests/           单元测试与目录结构测试
```

新代码建议从分类后的包导入，例如：

```python
from metasurface_data_pipeline.config.roi import load_rois_json
from metasurface_data_pipeline.calibration.aprilgrid import load_aprilgrid_spec
from metasurface_data_pipeline.observations.dense import derive_dense_multi_roi_dataset
```

`compat/` 中保留了旧路径 wrapper。旧代码里的 `metasurface_data_pipeline.roi`、`metasurface_data_pipeline.dense_observations` 等导入仍可用，但后续不建议继续新增这类导入。

## 推荐运行流程

完整数据处理通常分三步：

1. 生成 base dataset：读取原始图片，检测 AprilGrid，计算相机内参和每帧位姿。
2. 选择超表面 ROI：保存 4 个 `160x160` ROI 坐标。
3. 生成 dense board-constrained observation：读取 ROI、真实 `A_matrix` 和 board prior，输出给 3DGS/NeSpoF 使用的数据。

运行时建议只写一个很短的命令，让脚本读取 txt 配置文件。

## txt 配置文件

所有主要 CLI 都支持 `--config`：

```powershell
cd G:\projects\metasurface_3d_reconstruction\code

G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m metasurface_data_pipeline.cli.build_base_dataset `
  --config metasurface_data_pipeline\examples\base_dataset_config.txt
```

dense observation 生成：

```powershell
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m metasurface_data_pipeline.cli.build_multi_roi_dense_board_constrained `
  --config metasurface_data_pipeline\examples\dense_board_constrained_config.txt
```

配置文件格式：

```text
# 注释必须单独一行
key = value
boolean-flag = true
disabled-flag = false
repeated-option = first_value
repeated-option = second_value
```

规则：

- 参数名和命令行参数一致，只是不写前面的 `--`。
- `dataset_dir` 和 `dataset-dir` 都可以，会统一成 `dataset-dir`。
- 路径有空格时要加引号，例如 `source-image-dir = "G:/my data/pawn"`。
- `intrinsic-candidate = pawn=G:/images,G:/grid.json` 会作为一个完整参数保留。
- `base-roi = 1746 1019` 这种多值参数可以用空格分开。
- 命令行中写在 `--config` 后面的参数会覆盖 txt 中的同名标量参数。

可参考：

```text
examples/base_dataset_config.txt
examples/dense_board_constrained_config.txt
```

## 脚本说明

### `cli/build_base_dataset.py`

用途：从原始图片和 AprilGrid JSON 生成基础真实数据集。

主要输入：

- `--cube-image-dir`：原始图片文件夹，支持 `.png/.jpg/.jpeg/.bmp/.tif/.tiff`。
- `--cube-aprilgrid-json`：当前物体拍摄时使用的 AprilGrid 参数。
- `--intrinsic-candidate`：用于估计相机内参的图片目录和对应 AprilGrid JSON。
- `--response-mat`：真实超表面响应矩阵，通常是 `A_matrix_normalized.mat`。
- `--output-dir`：输出目录。

主要输出：

```text
observations.npz
poses_bounds.npz
response_tables_center2x2.npz
calibration_report.json
metadata.json
```

说明：这个阶段会先计算内参，再对每张原始图像检测 AprilGrid 并求外参，最后裁剪固定 metasurface ROI。

### `cli/select_rois.py`

用途：选择 4 个 `160x160` 超表面 ROI，并保存为 JSON。

常用命令：

```powershell
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m metasurface_data_pipeline.cli.select_rois `
  --dataset-dir metasurface_data_pipeline\outputs\cube2_base `
  --output metasurface_data_pipeline\outputs\cube2_base\metasurface_rois.json `
  --overlay metasurface_data_pipeline\outputs\cube2_base\metasurface_rois_overlay.png
```

说明：如果 OpenCV GUI 不可用，会自动切到浏览器选择器。红框是原始 ROI，黄框是额外选择的 ROI。

### `cli/build_multi_roi_sparse.py`

用途：生成旧版 sparse observation。

特点：

- 主要使用 12 个 spectral subcell。
- 每个 subcell 取中心 `2x2` 像素。
- 用于兼容早期 observation-only 实验。

当前新实验一般优先使用 dense 入口。

### `cli/build_multi_roi_dense_board_constrained.py`

用途：生成当前主线使用的 dense all-16 board-constrained observation。

特点：

- 支持 4 个 `160x160` ROI。
- 16 个 subcell 全部参与。
- 默认每个 subcell 取中心 `3x3` 像素作为独立监督。
- 12 个 spectral subcell 用于 S0 光谱 basis。
- 4 个 polar subcell 使用 analyzer 信息。
- 加入 board prior 和 board/object 分解权重，供 3DGS board-constrained 训练使用。

主要输出：

```text
observations_dense_all16_center3x3_indexed.npz
observations_dense_all16_center3x3_board_constrained.npz
poses_bounds_multi_roi.npz
dense_dataset_report.json
board_dataset_report.json
dense_board_constrained_report.json
metasurface_rois.json
```

## 核心模块说明

- `calibration/aprilgrid.py`：读取 AprilGrid JSON，检测 tag，计算内参和外参。
- `calibration/board_prior.py`：读取 AprilGrid 图案图片，并在 board 平面坐标下采样灰度先验。
- `config/roi.py`：定义固定 `160x160` ROI、ROI JSON 读写、多个 ROI 的 union crop。
- `config/text_config.py`：把 txt 配置文件展开为 argparse 参数。
- `core/response.py`：从真实 `A_matrix` 中提取像素或 subcell 的响应向量。
- `core/basis.py`：构建 Gaussian 光谱 basis，并计算 response-basis 投影。
- `core/real_metasurface.py`：固定 MATLAB ROI、OpenCV ray 构建、中心像素采样、偏振 analyzer。
- `core/geometry.py`：ray-box 等几何辅助函数。
- `observations/base_dataset.py`：base dataset 生成逻辑。
- `observations/sparse.py`：多 ROI sparse observation。
- `observations/dense.py`：多 ROI dense all-subcell observation。
- `observations/board_constrained.py`：board prior、foreground/object 权重和 board-constrained 数据输出。
- `observations/spectral_pixel.py`：早期 spectral pixel observation 逻辑。

## AprilGrid JSON 要求

AprilGrid JSON 必须包含：

```text
dictionary
markers
```

每个 marker 必须包含：

```text
id
corners_m
```

其中 `corners_m` 必须是 `[4, 3]`，表示该 tag 四个角点在标定板世界坐标系下的位置，单位通常是米。

## 输出数据给 3DGS 使用

当前 3DGS board-constrained 训练主要读取：

```text
observations_dense_all16_center3x3_board_constrained.npz
poses_bounds_multi_roi.npz
```

因此后端模型只需要依赖 pipeline 输出文件，不需要 import pipeline 的内部实现。

## 测试

在 `code/` 目录下运行：

```powershell
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m unittest discover -s metasurface_data_pipeline\tests -v
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m compileall -q metasurface_data_pipeline
```

## Git 维护

不要提交生成数据：

```text
outputs/
__pycache__/
*.pyc
*.npz
*.npy
*.png
*.jpg
*.jpeg
*.bmp
*.tif
*.tiff
*.ply
*.pt
*.pth
```

提交代码前建议至少运行：

```powershell
git diff --check
G:\anaconda_Envs\nerfstudio_3dgs\python.exe -m unittest discover -s metasurface_data_pipeline\tests -v
```

## 维护约定

修改本文件夹下代码时，请同步检查 README 是否需要更新。尤其是以下内容发生变化时，必须更新 README：

- CLI 参数或运行命令变化。
- 输出文件名或 `.npz/.json` schema 变化。
- 目录结构变化。
- ROI、subcell、polar analyzer、A_matrix 读取逻辑变化。
- 3DGS/NeSpoF 读取的数据入口变化。

更详细的维护规则见：

```text
skills/readme-sync/SKILL.md
```
