# TennisVision

业余网球比赛视频分析工具。给定一段固定机位的比赛视频，输出：
- 球的飞行轨迹
- 每次落地的位置（打在场内哪里）
- 右上角小地图，用红点标出所有落点

目标是一个**轻量、可在 CPU 上运行、模块可替换**的工程化实现。本项目借鉴了 [AggieSportsAnalytics/CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck) 和 yastrebksv 系列工作（见 [docs/court_detection_survey.md](docs/court_detection_survey.md)）。

---

## 目录

- [设计原则](#设计原则)
- [目录结构](#目录结构)
- [模块详解](#模块详解)
- [数据流](#数据流)
- [技术栈](#技术栈)
- [使用方法](#使用方法)
- [配置](#配置)
- [路线图](#路线图)
- [参考](#参考)

---

## 设计原则

1. **关注点分离** —— 球场标定、球检测、轨迹跟踪、落点识别、渲染彼此独立；任一模块的算法可替换而不影响其他。
2. **接口驱动** —— 同一类任务（如球检测）有统一接口，多种实现（经典 CV、TrackNet、WASB）并存；config 选其一。
3. **配置集中** —— 所有超参在 `tennisvision/config.py`；运行参数在 `configs/*.yaml`。
4. **可观测** —— 每个模块支持 `debug_dir` 参数，落盘中间掩码 / 候选点 / 聚类可视化，便于排查。
5. **CPU 优先** —— 默认配置能在笔记本 CPU 上离线跑通；GPU 路径作为加速选项。
6. **标定一次，终生复用** —— 固定机位视频的球场标定结果缓存到 `calib.json`，不重复计算。
7. **薄 CLI** —— `scripts/` 下的入口只做参数解析 + 调用库函数，业务逻辑全部在 `tennisvision/` 包内。

---

## 目录结构

```
tennisvision/
├── README.md                        # 本文件
├── requirements.txt                 # 依赖
├── .gitignore                       # 忽略 weights/ *.mp4 等
│
├── configs/
│   └── default.yaml                 # 默认 pipeline 配置
│
├── weights/                         # 模型权重（gitignored）
│   ├── court_tcd.pt                 # TennisCourtDetector
│   ├── tracknet.pt                  # yastrebksv TrackNet（或 WASB）
│   └── bounce_catboost.cbm          # CatBoost 落点分类器
│
├── docs/
│   ├── court_detection_survey.md    # 开源项目调研
│   └── architecture.md              # 架构详细说明（待写）
│
├── tennisvision/                    # 主包
│   ├── __init__.py
│   ├── config.py                    # 超参常量
│   │
│   ├── court/                       # 球场几何 + 标定
│   │   ├── __init__.py
│   │   ├── reference.py             # 标准球场 14 关键点几何
│   │   ├── detector.py              # 关键点检测（CNN）
│   │   ├── homography.py            # H 估计（best-of-subsets）
│   │   └── calibration.py           # calib.json 读写与缓存
│   │
│   ├── ball/                        # 球检测 + 轨迹跟踪
│   │   ├── __init__.py
│   │   ├── classical.py             # HSV + MOG2（当前方案）
│   │   ├── tracknet.py              # TrackNet / WASB 推理
│   │   └── tracker.py               # Kalman 多目标 + champion 选择
│   │
│   ├── bounce/                      # 落点检测
│   │   ├── __init__.py
│   │   ├── peak.py                  # y 方向极大值（当前 baseline）
│   │   └── catboost.py              # CatBoost 轨迹分类器
│   │
│   ├── render/                      # 叠加渲染
│   │   ├── __init__.py
│   │   ├── minimap.py               # 右侧 mini-map
│   │   ├── trail.py                 # 球轨迹叠加
│   │   └── hud.py                   # 文字 HUD
│   │
│   └── pipeline/                    # 编排
│       ├── __init__.py
│       └── analyze.py               # 两遍流水线：跟踪 → 渲染
│
├── scripts/                         # CLI 入口（薄壳）
│   ├── calibrate.py                 # 球场标定
│   ├── analyze.py                   # 完整分析
│   └── diagnose.py                  # 调试工具
│
├── tests/                           # 单元测试
│   ├── test_reference.py
│   ├── test_homography.py
│   ├── test_tracker.py
│   └── test_bounce.py
│
└── samples/                         # 测试素材（或外链）
    └── README.md
```

---

## 模块详解

### `tennisvision/court/` —— 球场几何与标定

#### `reference.py` — 标准球场
- **职责**：定义 ITF 标准网球场尺寸与 14 个关键点的**真实世界坐标**（米）。提供若干预定义的 4 点子集 `court_conf`，供 homography 估计选择。
- **常量**：双打场 10.97 × 23.77 m，单打内缩 1.37 m，发球线距底线 6.40 m，网高（线）11.885 m。
- **关键点编号**：借鉴 yastrebksv/TennisCourtDetector 的 14 点方案（底线×2 端点、单打边线×2 端点、发球线 T 点、中线等）。
- **参考**：CourtCheck `backend/vision/court_reference.py`

#### `detector.py` — 关键点 CNN 检测
- **职责**：给定一帧图像，输出 14 个 `(x, y, confidence)` 关键点。
- **接口**：
  ```python
  class CourtDetector(Protocol):
      def detect(self, frame: np.ndarray) -> list[Keypoint]: ...
  ```
- **实现**：
  | 实现 | 算法 | 权重 | 精度 | CPU 推理 |
  |---|---|---|---|---|
  | `HeatmapDetector` | yastrebksv TennisCourtDetector（15 通道 heatmap，640×360） | Drive `1f-Co64...` | 中值误差 1.83 px | ~200 ms |
  | `ResNet50Regressor` | CourtCheck 风格：ResNet50 + Linear(28) 坐标回归 | 自训 or abdullahtarek 权重 | 较低 | ~100 ms |
  | `RedMarkExtractor` | 从用户手工红色标注图提取（HSV 红 + Hough） | 无 | 取决标注 | <50 ms |
- **默认**：`HeatmapDetector`；`RedMarkExtractor` 作为离线手动 fallback。

#### `homography.py` — H 估计（核心改进点）
- **职责**：从 14 个（可能部分缺失的）关键点估计单应矩阵 H（图像平面 → 球场平面）。
- **算法 —— best-of-subsets（借鉴 CourtCheck）**：
  1. `reference.court_conf` 预列出多组有效的 4 点 ID 组合；
  2. 对每组：如有缺失点则跳过，否则 `cv2.findHomography` + DLT 求候选 H；
  3. 把**未参与**的其他 ~10 个检测点用该 H 反投影到球场平面，与 reference 真值算 MSE；
  4. 选 MSE 最小的 H 作为最终结果。
- **等效 RANSAC 的鲁棒性**：即使球员遮挡了一半关键点，剩余点也能投票出正确 H。
- **API**：
  ```python
  def estimate_homography(keypoints, reference) -> Homography
  def project_image_to_court(H, pt_img) -> pt_court_m
  def project_court_to_image(H_inv, pt_court_m) -> pt_img
  ```

#### `calibration.py` — 缓存与序列化
- **职责**：把标定结果（H、14 点像素坐标、时间戳、camera_id）序列化到 `calib.json`；加载时校验 video hash。
- **缓存策略**：检测到的第一次成功标定结果写入；后续同机位视频直接复用。
- **参考**：CourtCheck `backend/vision/calibration.py`

---

### `tennisvision/ball/` —— 球检测与轨迹

#### `classical.py` — HSV + MOG2（当前方案）
- **职责**：在单帧上输出 0..N 个球候选 `(x, y)`。
- **算法**（现状）：
  1. MOG2 背景减除 → 运动掩码
  2. 大运动块（面积阈值内）的 bbox = 球员区域，排除内部候选
  3. HSV 黄绿过滤 AND 运动掩码
  4. 形态学开 + 膨胀
  5. 轮廓面积 + 圆度过滤 → 候选点
- **限制**：顶点慢速球易漏检；大面积风/树叶运动会误认为球员。
- **保留原因**：零依赖、CPU 几乎 0 成本，作为 fallback 和 demo。

#### `wasb.py` — WASB (HRNet) 深度学习球检测（**默认**）
- **架构**：HRNet 变体（1.48M 参数，~5.8 MB 权重），9 通道输入（连续 3 帧 RGB 堆叠），288×512 输入，输出 3 张热图对应 3 帧，MIT 协议
- **预处理**：letterbox 仿射变换（等比缩放 + 黑边填充，不失真）+ ImageNet mean/std 归一化
- **后处理**：sigmoid → 阈值 0.5 → connected components → 得分加权重心 → 逆仿射回原图坐标
- **时序门控**：新检测必须距上一帧 <= `max_disp` px（默认 300），拒绝抖动
- **接口**：
  ```python
  det = WASBBallDetector(WASBConfig(weights="weights/wasb_tennis_best.pth.tar"))
  for frame in video:
      det.push_frame(frame)
      xy = det.detect()            # (x, y) or None (first 2 frames empty)
  ```
- **CPU 推理**：~100–200ms/帧（CPU）；WASB 本身是跨运动 SOTA（BMVC 2023），tennis 权重来自官方 model zoo
- **源代码**：`hrnet.py` 复用自 nttcom/WASB-SBDT + Microsoft HRNet-Image-Classification（MIT），保留原版权头

#### `tracknet.py` — TrackNet / 其他备选（未实现）
对照组；yastrebksv/TrackNet + ONNX 也是可选项，但 WASB **MIT 协议** + 更小 + tennis 微调权重更适合公开发布

#### `tracker.py` — 多轨迹 Kalman + champion
- **职责**：把每帧候选点 association 成跨帧的 track；每帧选出唯一的"champion"（当前最可信的球轨迹）。
- **算法**（现状保留）：
  1. 每个 track 内置 4 态 Kalman（`[x, y, vx, vy]`，匀速模型）
  2. 每帧：所有 active track 调 `predict()`；贪婪匹配候选到 track（距离 < GATE=80 px）
  3. 未匹配的候选 → 新 track
  4. 连续观测 ≥ 3 且平均速度 ∈ [10, 150] px/帧 → 标记 validated
  5. `pick_champion()` = 最长的、最近仍在更新的 validated track
- **保留不变**：这部分在 TrackNet 引入后仍然是后处理关键（平滑、补帧、去噪）。

---

### `tennisvision/bounce/` —— 落点检测

#### `peak.py` — y 方向极大值（baseline）
- **算法**：在 champion track 的 y 历史里找局部极大值（左右 ±k 帧都比它小，峰高 ≥ threshold）。
- **问题**：击球顶点、Kalman 过度平滑、遮挡后重现都会误触发。

#### `catboost.py` — 轨迹特征分类器
- **职责**：给定一条 `(x, y, frame)` 序列，输出所有弹跳帧 + 置信度。
- **算法**（移植 yastrebksv / CourtCheck）：
  1. 三次样条插值补漏检 + 速度 >80 px/帧 外点剔除
  2. 特征：x/y + lag(1,2) 前后差 + 前向/后向差比值（速度反转 + 前后不对称）
  3. CatBoost 回归器（已有预训练 `bounce_catboost.cbm`）预测每帧弹跳概率
  4. 阈值 0.20 + 相邻帧 NMS
- **为什么比 y-peak 强**：y-peak 只看 y 单分量，真弹跳的特征是**速度双分量联合反转 + 前后不对称**（非弹性碰撞反弹变慢）。CatBoost 学的就是这种联合签名。

---

### `tennisvision/render/` —— 可视化叠加

#### `minimap.py`
- **职责**：根据 `court.reference` 生成标准比例的小球场模板；把累积落点投影上去。
- **布局**：宽 150 px，高 = 150 × (23.77 / 10.97) ≈ 325 px；放在帧右上角。
- **落点渲染**：新落点红色实心圆，随帧数渐变灰（20 秒内完全变灰）。

#### `trail.py`
- **职责**：画 champion track 最近 45 帧的轨迹（polyline + 渐变圆点），当前帧位置加大圈高亮。
- **占位扩展**：后续可加"当次击球的抛物线拟合"预测轨迹。

#### `hud.py`
- **职责**：左上角文字：`f=N/T cands=C tracks=K champ=#ID len=L bounces=B`。
- **打开 debug 模式**可叠加：player bbox、运动掩码、候选框。

---

### `tennisvision/pipeline/analyze.py` —— 编排

**两遍流水线**（借鉴 CourtCheck）：

**Pass 1 — 跟踪（不渲染）**
```
for frame in video:
    candidates  = ball_detector.detect(frame)
    tracker.update(candidates, frame_idx)
    player_bboxes = player_detector.detect(frame)   # 可选
trajectory = tracker.extract_champion_trajectory()
bounces    = bounce_detector.detect(trajectory)
```

**Pass 2 — 渲染**
```
for frame in video:
    vis = frame.copy()
    render.trail(vis, trajectory, frame_idx)
    render.minimap(vis, bounces, frame_idx, H_inv)
    render.hud(vis, stats)
    writer.write(vis)
```

这样设计的好处：
- 落点检测需要**完整轨迹**作为输入（CatBoost 要做样条插值），单遍流式做不到
- 渲染可以预览**未来**弹跳（"下个落点会在这里"）
- Pass 1 可以 headless 离线批处理，Pass 2 可延后或按需

---

## 数据流

```
 video.mp4
     │
     ├── (首次) ──► [court/detector] ──► 14 keypoints
     │                                       │
     │                                       ▼
     │                             [court/homography] ──► H, H_inv
     │                                       │
     │                                       ▼
     │                              [court/calibration]
     │                                       │
     │                                       ▼
     │                                  calib.json  ◄── (后续直接加载)
     │
     └── 每帧 ──► [ball/{classical|tracknet}] ──► candidates
                          │
                          ▼
                 [ball/tracker] ──► champion track
                          │
                          ▼
             [bounce/{peak|catboost}] ──► bounce frames
                          │       (projected via H)
                          ▼
                     bounces in court meters
                          │
                          ▼
           ┌───────────────────────────────┐
           │ [render/trail]                │
           │ [render/minimap] (H_inv)      │ ──► annotated.mp4
           │ [render/hud]                  │
           └───────────────────────────────┘
```

---

## 技术栈

### 必需依赖
| 组件 | 用途 | 版本 |
|---|---|---|
| Python | 3.8+ | — |
| opencv-python-headless | 视频 I/O、Kalman、Hough、形态学 | 4.6+ |
| numpy | 张量运算 | 1.19+ |

### 可选依赖（按模块）
| 组件 | 开启什么 |
|---|---|
| `torch` + `torchvision` | PyTorch TrackNet / 球场关键点 CNN |
| `onnxruntime` | TrackNet 推理（比纯 torch 快 5–10×）|
| `catboost` | 学习式落点检测 |
| `scipy` | 三次样条插值（bounce 预处理） |
| `ultralytics` | 未来加 YOLOv8 Pose 做球员姿态 |

`requirements.txt` 按"最小可运行集"+"扩展组"分两栏列出。

### GPU
完全可选。所有默认路径在 CPU 上可跑；GPU 加速点：
- 球场检测 CNN（一次性，非瓶颈）
- TrackNet 球检测（瓶颈，GPU 可达实时）

---

## 使用方法

### 安装
```bash
pip install -r requirements.txt
# 可选：下载预训练权重到 weights/（见 docs/court_detection_survey.md）
```

### 球场标定
```bash
python scripts/calibrate.py \
    --video input.mp4 \
    --out   calib.json \
    --method tcd             # tcd | resnet | mark
```
- `--method tcd`：调用 `court.detector.HeatmapDetector`（默认）
- `--method resnet`：`ResNet50Regressor`
- `--method mark --marked-image court_mark.jpg`：从手工红色标注图提取

会输出 `calib.json` 和 `calib_vis.jpg`（叠加投影回检查）。

### 完整分析
```bash
python scripts/analyze.py \
    --video input.mp4 \
    --calib calib.json \
    --out   output.mp4 \
    --ball-detector tracknet \
    --bounce-detector catboost
```

### 调试
```bash
python scripts/diagnose.py \
    --video input.mp4 \
    --frame 800 \
    --stage all              # all | ball | court | bounce
```
落盘每一步的中间图像到 `diagnostics/` 方便排错。

---

## 配置

`configs/default.yaml`（示意）：

```yaml
court:
  detector: tcd                  # tcd | resnet | mark
  weights: weights/court_tcd.pt
  input_size: [640, 360]
  cache_calib: true

ball:
  detector: tracknet             # classical | tracknet | wasb
  weights: weights/tracknet.pt
  input_size: [640, 360]
  buffer_frames: 3
  fallback_classical: true       # 当 TrackNet miss 时用经典 CV 补

tracker:
  gate_px: 80
  max_gap_frames: 5
  min_len: 3
  min_speed: 10
  max_speed: 150

bounce:
  detector: catboost             # peak | catboost
  weights: weights/bounce_catboost.cbm
  threshold: 0.20
  nms_window: 10

render:
  minimap_width_px: 150
  trail_length_frames: 45
  bounce_fade_frames: 600

debug:
  dir: null                       # 设为路径开启 debug 输出
```

每个模块接受的参数都可在此覆盖。

---

## 路线图

### V0 — 已完成
- HSV + MOG2 经典 CV 球检测
- Kalman 多轨迹 + champion 选择
- 手工红色标注图做标定（`RedMarkExtractor`）
- 4 点固定锚点 homography
- y-peak 落点检测
- 右上角 mini-map 渲染
- **已重构到 `tennisvision/` 包结构 + 两遍流水线**

### V1 — 部分完成
- [x] 实现 **best-of-subsets homography**（`court/homography.py:estimate_homography`）
- [x] 集成 **CatBoost 落点分类器**（`bounce/catboost.py`）—— tennis_raw.mp4 上从 5 个 bounce 提到 26 个
- [x] 重构到 `tennisvision/` 包结构 + 两遍流水线 Pass 1/Pass 2
- [x] `scripts/calibrate.py` 和 `scripts/analyze.py` 薄 CLI
- [x] 集成 **yastrebksv/TennisCourtDetector**（`court/tcd_model.py`）代码完成、权重已下载
- [ ] **已知限制**：TCD 模型在广播视角（TV 转播机位）上训练，对我们业余场地的**低位偏左机位** recall 很低（0-3 keypoints/帧）。下个版本需要 fine-tune 或改用另外的权重
- [ ] 单元测试

### V2 — 中期（引入深度学习球检测）
- [ ] 导出 TrackNet 到 **ONNX** + ONNXRuntime CPU 推理
- [ ] 实现 `ball/tracknet.py` 并与现有 tracker 集成
- [ ] 性能基准：目标 CPU 5–10 fps（离线批处理够用）
- [ ] （可选）换 **WASB-SBDT**（MIT 许可，tennis 权重）

### V3 — 长期（物理级准确）
- [ ] YOLOv8 Pose 检测球员姿态（击球判定）
- [ ] 3D EKF（参考 Tennis3DTracker）：真正的 3D 轨迹 + 物理 bounce（z=0 交越）
- [ ] 击球分类（正手/反手/发球）via 姿态时序分类
- [ ] 热力图、统计报告、多摄像机融合

### 非目标
- 实时直播分析（本项目定位**离线后处理**，不追求 > 10 fps）
- 职业比赛级精度（对照 Hawk-Eye 硬件多目视觉，本项目是单目近似）
- 商业 SaaS 前后端（只做 CV 核心）

---

## 参考

### 直接借鉴的项目
- **[CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck)** —— 整体 pipeline 分层、best-of-subsets homography、CatBoost 落点。
- **[yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject)** —— TrackNet + 14 点球场 + CatBoost bounce 的参考实现。
- **[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)** —— 15 通道 heatmap 球场检测模型，预训练权重开放。
- **[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)** —— 业界标准球追踪 CNN。

### 更前沿的替代
- **[nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)** —— MIT 许可的小球检测 SOTA（BMVC 2023）。
- **[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)** —— TrackNet + InpaintNet 轨迹补洞。
- **[John-Boccio/Tennis3DTracker](https://github.com/John-Boccio/Tennis3DTracker)** —— 单目 3D EKF 重建。

### 文档
- [docs/court_detection_survey.md](docs/court_detection_survey.md) —— 完整调研笔记（含各方案对比、权重下载链、选型建议）。
