# TennisVision

业余网球比赛视频分析工具。固定机位视频输入，输出：

- 球的飞行轨迹 + 每次落地的位置（场内坐标）
- **回合（rally）切分**：每局起止、击球次数、击球类型（正手 / 反手 / 发球）
- 可视化叠加：轨迹、右上角小地图、球员 bbox、击球标签、HUD
- 三个产物：`*_out.mp4`（标注）、`*_out_rally.mp4`（只剪辑回合）、`*_out_rallies.json`（回合元数据）

设计目标：**模块可替换、CPU 可运行、接口清晰**的工程化实现。借鉴 [CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck) 和 yastrebksv 系列（见 [docs/court_detection_survey.md](docs/court_detection_survey.md)）。

---

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 球场标定（一次，calib.json 可跨视频复用）
python scripts/calibrate.py blue-resnet \
    --video  input.mp4 \
    --weights weights/court_resnet.pth \
    --out    calib.json

# 3. 跑完整分析
python scripts/analyze.py \
    --video input.mp4 \
    --calib calib.json \
    --out   output.mp4
```

第一次运行会把 WASB 的 `.pth.tar` 权重自动导出为 ONNX，~10 秒一次性开销。默认路径开启球检测、轨迹、落点、回合检测、渲染；Pose + 击球分类默认关闭，加 `--config configs/remote_inpaint.yaml` 开启完整 V3 路线。

---

## 设计原则

- **关注点分离** —— 球场、球检测、轨迹、落点、回合、渲染彼此独立；任一模块算法可替换。
- **接口驱动** —— 同一任务多实现并存（如 `classical` / `wasb` 球检测），config 选型。
- **CPU 优先** —— 默认配置笔记本 CPU 可离线跑通，GPU 路径作为加速选项。
- **标定一次终生复用** —— 固定机位的 `calib.json` 缓存，跨视频免重跑。
- **薄 CLI** —— `scripts/` 仅做参数解析；业务逻辑全部在 `tennisvision/` 包内。
- **可观测** —— 每模块支持 `debug_dir`，落盘中间掩码 / 候选点 / 聚类可视化。

---

## 功能矩阵

每项功能在 `docs/issues/` 下有独立文档（设计、实现、权衡、已知问题）。

| # | 模块 | 默认实现 | 备选 | 权重 | Issue |
|---|---|---|---|---|---|
| 01 | 球场标定 | `blue-resnet`（HSV 蓝 → 透视矫正 → ResNet50 14 点回归）| `tcd` / `mark` / `blue` | `court_resnet.pth`（~95 MB） | [01-court-calibration](docs/issues/01-court-calibration.md) |
| 02 | 球检测 | `wasb`（HRNet 1.48 M, 9-ch 时序, MIT, BMVC 2023） | `classical`（HSV + MOG2） | `wasb_tennis_best.pth.tar` | [02-ball-detection](docs/issues/02-ball-detection.md) |
| 03 | 球轨迹 | Kalman 多目标 + champion 选择 | — | — | [03-ball-tracker](docs/issues/03-ball-tracker.md) |
| 04 | 轨迹补洞 | TrackNetV3 InpaintNet（可选，默认关） | — | `InpaintNet_best.pt` | [04-trajectory-inpainting](docs/issues/04-trajectory-inpainting.md) |
| 05 | 落点检测 | `catboost`（轨迹特征分类器） | `peak`（y 极值） | `bounce_catboost.cbm` | [05-bounce-detection](docs/issues/05-bounce-detection.md) |
| 06 | 球员姿态 | YOLO26n-pose + ByteTrack（可选） | — | `yolo26n-pose.pt` | [06-player-pose](docs/issues/06-player-pose.md) |
| 07 | 击球分类 | 30 帧滑窗 + Keras GRU（可选） | — | `tennis_rnn.h5` | [07-stroke-classifier](docs/issues/07-stroke-classifier.md) |
| 08 | 回合检测 | `OnlineRallyDetector` + merge + bounce-split 过滤 | online 单遍模式 | — | [08-rally-detection](docs/issues/08-rally-detection.md) |
| 09 | 渲染 | trail / minimap / HUD / player bbox / stroke label | — | — | [09-rendering](docs/issues/09-rendering.md) |
| 10 | 流水线编排 | 两遍（Pass 1a 球 → bounce → merge → Pass 1b pose → Pass 2 渲染） | `rally.online=true` 单遍 | — | [10-pipeline](docs/issues/10-pipeline.md) |

> 切换实现：写一份 `configs/your.yaml` 覆盖对应字段（如 `ball.detector: classical`、`action.enabled: true`、`rally.online: true`），运行时 `--config configs/your.yaml`。全部默认值见 [`tennisvision/config.py`](tennisvision/config.py)。

---

## 架构

### 数据流

```
video.mp4
   │
   ├── (首次) ──► court/detector ──► 14 keypoints ──► court/homography ──► H / H_inv ──► calib.json
   │                                                                                         │
   │                                                                                         ▼
   │                                                                            (后续视频直接加载 calib.json)
   │
   └── 每帧 ──► ball/wasb ──► candidates
                    │
                    ▼
           ball/tracker ──► champion track ──────────────► rally/OnlineRallyDetector
                    │                                             │
                    │                                             ▼ (gap<=N + 无 bounce)
                    │                                    rally/merge_close_rallies
                    │                                             │
                    │                                             ▼ (两侧各 ≥ 1 个 bounce)
                    │                                    rally/bounce-split filter
                    │                                             │
                    ▼                                             │
           bounce/catboost ──► bounces ─── (projected via H_inv) ─┘
                    │
                    ▼
           action/pose + stroke_classifier (per rally, Pass 1b)
                    │
                    ▼
           render/trail + minimap + hud + player + stroke_label ──► *_out.mp4
                                                                  ├──► *_out_rally.mp4
                                                                  └──► *_out_rallies.json
```

**Online 模式**（`rally.online=true`）把 `ball`、`action`、`OnlineRallyDetector` 全部塞到同一帧循环里；bounce 仍作为 batch 后处理（用于小地图可视化），但不再参与回合过滤或合并。适合实时推流。详见 [issue 08](docs/issues/08-rally-detection.md)。

### 目录结构

```
tennisvision/
├── README.md
├── requirements.txt                 # 基础依赖
├── requirements-ml.txt              # 扩展（torch / ultralytics / tf-keras / catboost / onnxruntime）
│
├── configs/
│   ├── local_run.yaml               # 本机跑的 override
│   ├── remote_inpaint.yaml          # 远程测试机：InpaintNet + action 全开
│   └── remote_inpaint_debug.yaml
│
├── weights/                         # 模型权重（gitignored）
│   ├── court_resnet.pth             # ResNet50 球场 14 点回归
│   ├── court_tcd.pt                 # TennisCourtDetector heatmap CNN（广播视角备选）
│   ├── wasb_tennis_best.pth.tar     # WASB HRNet 球检测（首次运行自动导出 .onnx）
│   ├── InpaintNet_best.pt           # TrackNetV3 InpaintNet 轨迹补洞（可选）
│   ├── bounce_catboost.cbm          # CatBoost 落点分类器
│   ├── yolo26n-pose.pt              # 球员检测 + 姿态（ultralytics）
│   └── tennis_rnn.h5                # 击球分类 GRU (Keras 2)
│
├── docs/
│   ├── court_detection_survey.md    # 球场检测开源调研
│   └── issues/                      # 功能级设计文档（上表的 Issue 列）
│
├── tennisvision/                    # 主包
│   ├── config.py                    # 所有超参默认值（DEFAULTS）
│   ├── court/                       # 球场几何 + 标定
│   ├── ball/                        # 球检测 + 轨迹 + 补洞
│   ├── bounce/                      # 落点
│   ├── action/                      # 球员 pose + 击球分类
│   ├── render/                      # 可视化
│   └── pipeline/                    # 编排
│       ├── analyze.py               # 两遍流水线主入口
│       └── rally.py                 # 回合检测状态机 + merge + 过滤 + 切片
│
├── scripts/                         # 薄 CLI
│   ├── calibrate.py                 # 球场标定
│   ├── analyze.py                   # 完整分析
│   ├── diagnose.py                  # 单帧调试
│   └── export_onnx.py               # 权重导出工具
│
├── tests/                           # 单元测试
└── samples/                         # 测试素材（gitignored）
```

---

## 使用

### 球场标定

```bash
# 推荐：业余蓝色硬地自动标定
python scripts/calibrate.py blue-resnet \
    --video input.mp4 \
    --weights weights/court_resnet.pth \
    --out calib.json --vis calib_vis.jpg

# 备选：手工红线图 | 广播视角 heatmap | 纯蓝色轮廓（无 ML）
python scripts/calibrate.py mark --image court_mark.jpg --out calib.json
python scripts/calibrate.py tcd  --video input.mp4 --weights weights/court_tcd.pt --out calib.json
python scripts/calibrate.py blue --video input.mp4 --out calib.json
```

所有方法均输出 `calib.json` + `calib_vis.jpg`（court overlay 供人工核验）。

| 方法 | 适用场景 | 权重 | Reproj err（样例） |
|---|---|---|---|
| `blue-resnet` | 业余蓝色硬地，低位摄像 | `court_resnet.pth` | ~0.8 m |
| `mark` | 任意场地，需手动画框 | 无 | 取决于标注精度 |
| `tcd` | 广播 / 高位摄像 | `court_tcd.pt` | ~1-2 m（低位失效）|
| `blue` | 蓝色硬地，无 ML 环境 | 无 | ~2-4 m |

`blue-resnet` 权重下载：`gdown 1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta -O weights/court_resnet.pth`

### 完整分析

```bash
python scripts/analyze.py \
    --video input.mp4 \
    --calib calib.json \
    --out   output.mp4
```

产物（以 `--out output.mp4` 为例）：

| 文件 | 内容 |
|---|---|
| `output.mp4` | 原视频叠加轨迹、小地图、落点、球员 bbox、HUD |
| `output_rally.mp4` | 仅剪辑出回合片段，回合之间插"Rally N" 黑底标题 |
| `output_rallies.json` | 每个回合的 `start_frame / end_frame / net_crossings / n_strokes` |

加 `--config configs/remote_inpaint.yaml` 开启 InpaintNet + pose + 击球分类，终端多出：

```
[pass 1b] pose + stroke for 4 rallies ...
[action] 42 stroke events: backhand=11, forehand=26, serve=5
[action]   player #1: backhand=6, forehand=14, serve=3
```

### 调试

```bash
python scripts/diagnose.py \
    --video input.mp4 \
    --frame 800 \
    --stage all                      # all | ball | court | bounce
```

落盘每一步中间产物到 `diagnostics/` 便于排错。

---

## 配置

所有默认参数集中在 [`tennisvision/config.py`](tennisvision/config.py) 的 `DEFAULTS` 字典（~200 行）。YAML override 只写要改的字段：

```yaml
# configs/custom.yaml —— 示例：开启 action 路线 + online 回合模式
action:
  enabled: true
rally:
  online: true
  pre_roll_seconds: 1.5
```

现有示例 config：

- [`configs/local_run.yaml`](configs/local_run.yaml) — 本机跑 sample 视频用
- [`configs/remote_inpaint.yaml`](configs/remote_inpaint.yaml) — 远程测试机完整路线（InpaintNet + action）
- [`configs/remote_inpaint_debug.yaml`](configs/remote_inpaint_debug.yaml) — 同上 + debug 落盘

---

## 路线图

### V0（已完成）
HSV + MOG2 + Kalman + 手动标定 + y-peak 落点 + mini-map + 两遍流水线包结构。

### V1（已完成）
Best-of-subsets homography（14 点容错）+ CatBoost 落点分类（从 5 → 26 bounces/tennis_raw.mp4）+ 四种标定路线（blue-resnet / tcd / mark / blue）+ 薄 CLI。

### V2（部分完成）
- [x] WASB 默认球检测（MIT，BMVC 2023）+ 首次运行自动 ONNX 导出
- [x] TrackNetV3 InpaintNet 轨迹补洞（可选）
- [x] 双阶段球检测：主尺度 + far-court crop（ball two-stage）
- [ ] CPU 5 fps（含 action）性能基准

### V3（部分完成）
- [x] YOLO26n-pose 球员检测 + 追踪 + 姿态一体（替代原 YOLOv8n + MoveNet TFLite 双模型）
- [x] 30 帧滑窗 GRU 击球分类（4 类：backhand / forehand / neutral / serve）
- [x] 回合检测：OnlineRallyDetector + 合并 + bounce-split 过滤
- [x] Online 单遍模式：ball + pose + rally 同帧
- [ ] 击球 ↔ 落点配对（正手打出 → 落点战术统计）
- [ ] 3D EKF（单目 3D 重建，参考 Tennis3DTracker）
- [ ] 热力图 / 统计报告

### 非目标
- 实时直播（定位离线后处理，不追求 > 10 fps）
- 职业级精度（单目近似，不对标 Hawk-Eye 硬件）
- 商业 SaaS 前后端

---

## 参考

### 直接借鉴

- [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) — 球检测（HRNet, MIT, BMVC 2023），复用源代码 + tennis 权重
- [CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck) — pipeline 分层、best-of-subsets homography、CatBoost bounce
- [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) — 14 点球场 + CatBoost bounce 参考实现
- [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) — 15 通道 heatmap 球场检测
- [antoinekeller/tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition) — 30 帧 GRU 击球分类（`tennis_rnn.h5`）
- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/) — YOLO26n-pose（R-ELAN 注意力，支持 CPU / CUDA）
- [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) — InpaintNet 轨迹补洞

### 候选 / 未采用

- [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) — 业界标准球追踪 CNN；被 WASB 在同基准上超过，故未采用
- [John-Boccio/Tennis3DTracker](https://github.com/John-Boccio/Tennis3DTracker) — 单目 3D EKF（V3 远期）

### 文档

- [docs/court_detection_survey.md](docs/court_detection_survey.md) — 球场检测完整调研（含各方案对比、权重下载、选型建议）
