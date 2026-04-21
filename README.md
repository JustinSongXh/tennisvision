# TennisVision

业余网球比赛视频分析工具。给定一段固定机位的比赛视频，输出：
- 球的飞行轨迹
- 每次落地的位置（打在场内哪里）
- 右上角小地图，用红点标出所有落点

目标是一个**轻量、可在 CPU 上运行、模块可替换**的工程化实现。本项目借鉴了 [AggieSportsAnalytics/CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck) 和 yastrebksv 系列工作（见 [docs/court_detection_survey.md](docs/court_detection_survey.md)）。

---

## 目录

- [设计原则](#设计原则)
- [方案矩阵](#方案矩阵)
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

## 方案矩阵

一张表回答"每个模块有哪些实现 / 哪个是默认 / 需要什么权重"。详细原理与对比在[模块详解](#模块详解)里展开。

| 模块 | 可选实现 | 默认 | 权重文件 |
|---|---|---|---|
| **球场标定** (`scripts/calibrate.py`) | `blue-resnet` — 蓝色区域透视变换 + ResNet50 回归（推荐）/ `mark` — 手画红线图 / `tcd` — TennisCourtDetector CNN（广播视角）/ `blue` — 蓝色轮廓（无 ML 备选） | `blue-resnet` | `blue-resnet` 需 `weights/court_resnet.pth`；`tcd` 需 `weights/court_tcd.pt`；`mark`/`blue` 无 |
| **球检测** (`ball.detector`) | `classical` — HSV + MOG2 / `wasb` — WASB HRNet | **`wasb`** | `wasb_tennis_best.pth.tar`（首次运行自动导出为 `.onnx`） |
| **球轨迹** | 多目标 Kalman + champion 选择 | 唯一 | 无 |
| **落点检测** (`bounce.detector`) | `peak` — y 方向极值 baseline / `catboost` — 轨迹特征分类器 | **`catboost`** | `bounce_catboost.cbm` |
| **轨迹补洞** (`ball.inpainter.enabled`) | TrackNetV3 InpaintNet —— 学习式填充 validated track 内部空洞（默认关闭） | 关闭 | `InpaintNet_best.pt` |
| **球员检测 + 姿态** (`action.enabled`) | YOLO26n-pose + ByteTrack —— 单次推理同时出 bbox、track ID、17 关节点 | 唯一 | `yolo26n-pose.pt` |
| **击球分类** (`action.enabled`) | 30 帧滑窗 + Keras GRU（默认关闭，需显式开启） | 关闭 | `tennis_rnn.h5`（需 `tf-keras` 加载） |
| **渲染** | trail / minimap / hud / player bbox + stroke label | 唯一 | 无 |

> **切换实现方式**：写一份 `configs/your.yaml` 覆盖对应字段，如 `ball.detector: classical` 或 `action.enabled: true`，运行时传 `--config configs/your.yaml`。完整字段见 [`tennisvision/config.py:DEFAULTS`](tennisvision/config.py)。

---

## 目录结构

```
tennisvision/
├── README.md                        # 本文件
├── requirements.txt                 # 依赖
├── .gitignore                       # 忽略 weights/ *.mp4 等
│
├── configs/
│   ├── default.yaml                 # 默认 pipeline 配置
│   ├── remote_inpaint.yaml          # 远程测试机：InpaintNet + action 全开
│   └── remote_inpaint_debug.yaml    # 同上，debug 模式
│
├── weights/                         # 模型权重（gitignored）
│   ├── court_resnet.pth             # ResNet50 球场 14 点回归（标定推荐，~95 MB）
│   ├── court_tcd.pt                 # TennisCourtDetector heatmap CNN（广播视角备选）
│   ├── wasb_tennis_best.pth.tar     # WASB HRNet 球检测（首次运行自动导出 .onnx）
│   ├── InpaintNet_best.pt           # TrackNetV3 InpaintNet 轨迹补洞（可选）
│   ├── bounce_catboost.cbm          # CatBoost 落点分类器
│   ├── yolo26n-pose.pt              # YOLO26n-pose 球员检测 + 姿态（ultralytics）
│   └── tennis_rnn.h5                # 击球分类 GRU (antoinekeller, Keras 2)
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
│   │   ├── inpainter.py             # TrackNetV3 InpaintNet 轨迹补洞（可选）
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
│   │   ├── hud.py                   # 文字 HUD
│   │   └── action.py                # 球员 bbox + 击球标签叠加
│   │
│   ├── action/                      # 球员姿态 + 击球分类（V3）
│   │   ├── __init__.py
│   │   ├── pose.py                  # YOLO26n-pose：检测 + ByteTrack + 姿态一体
│   │   └── stroke_classifier.py     # 多人 30 帧滑窗 GRU
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
│   ├── test_bounce.py
│   └── test_inpainter.py
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
  | 实现 | 算法 | 权重 | 适用场景 |
  |---|---|---|---|
  | `ResNet50CourtDetector` | abdullahtarek 风格：ResNet50 + Linear(28) 坐标回归，始终输出全 14 点 | `court_resnet.pth` | 业余蓝色硬地（推荐） |
  | `HeatmapDetector` | yastrebksv TennisCourtDetector（15 通道 heatmap，640×360） | `court_tcd.pt` | 广播 / 高位摄像 |
  | `RedMarkExtractor` | 从用户手工红色标注图提取（HSV 红 + Hough） | 无 | 手动 fallback |
- **blue-resnet pipeline**：`BlueContourDetector` 构建蓝色区域 mask → `_enclosing_trapezoid` 拟合外接梯形 → `cv2.warpPerspective` 矫正为正视图 → `ResNet50CourtDetector` 多帧回归 + median 聚合 → 逆透视投影回原图坐标。

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

#### `inpainter.py` — TrackNetV3 InpaintNet 轨迹补洞（可选，默认关）
- **来源**：移植自 [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)（InpaintNet + generate_inpaint_mask），原始权重在羽毛球数据集上训练，tennis 上经验性使用。
- **职责**：在 Pass 1 结束、落点检测开始前，对每条 validated Track 的**内部空洞**做学习式填充；返回填充后的 `(x, y, was_inpainted)` 序列。
- **设计选择 —— 逐 Track 而非跨 Track**：tracker 已决定哪些段属于同一轨迹；不在两条 Track 之间做桥接（跨 rally / 球出画面的合并是误检来源）。
- **`window_mode`**：
  - `single`（默认）—— 一次前向覆盖整条轨迹（全卷积，无接缝）
  - `nonoverlap` —— 非重叠滑窗（忠实于上游评估方式）
- **接口**：
  ```python
  inp = TrajectoryInpainter(InpaintNetConfig(...), frame_size=(W, H))
  res = inp.inpaint_tracks([track])   # InpaintResult: xs, ys, was_inpainted
  pts = result_to_trackpoints(res)
  ```
- **pipeline 位置**：`pipeline/analyze.py:_maybe_inpaint`，在 `_run_bounce_detector` 之前调用；`ball.inpainter.enabled=false` 时零开销。
- **统计日志**：启用时在终端打印 `coverage before/after + inpainted frames` 便于 A/B 对比。

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

#### `action.py` — 球员 bbox + 击球标签叠加
- **职责**：Pass 2 渲染层消费 `stroke_events`，为每个 track ID 叠加：
  1. **彩色 bbox**（`draw_players`）—— 颜色通过 track_id 做黄金比例 HSV 哈希，同一球员跨帧颜色稳定。
  2. **击球标签**（`draw_stroke_labels`）—— 在 bbox 上方显示最近一次击球类型（`backhand / forehand / serve`），线性淡出，TTL = 2 秒。
- **实现细节**：事件按 player_id 分组并按 frame 排序一次（`events_by_player_sorted`），Pass 2 每帧用 `bisect` 查找最近事件，O(log n)。

---

### `tennisvision/action/` —— 球员检测、姿态与击球分类（V3）

整个模块默认**关闭**（`action.enabled=false`），开启后在 Pass 1 里和球检测并行跑。所有重依赖（`tensorflow` / `tf-keras` / `ultralytics`）都是懒加载，只跑球不会付出这部分开销。

#### `pose.py` — YOLO26n-pose 检测 + 追踪 + 姿态一体
- **职责**：每帧一次 `.track()` 调用，同时返回 `{track_id: (bbox, Pose)}`——bbox 是 `(x0, y0, x1, y1)`，Pose 含 17 个 COCO 关键点 `(y_px, x_px, score)`。
- **模型**：Ultralytics YOLO26n-pose（7.6 MB），注意力架构（R-ELAN），支持 CPU 和 CUDA。
- **旁观者过滤**，分三层：
  1. `min_bbox_h`（默认 60 px）—— 丢弃远处 / 极小检测
  2. **on-court 过滤** —— bbox 底部中心经标定 H 投影到球场平面，只保留脚在 `[-margin, 场地+margin]` 米以内的人（默认 `court_margin_m=3.0`）
  3. `max_persons` —— 置信度排序取前 N（默认 4）
- **设备**：`pose_device: cpu` 或 `pose_device: cuda`，config 一行切换。
- **设计选择**：原方案用 YOLOv8n 检测 + MoveNet TFLite 姿态（两个模型，MoveNet CPU-only 是主瓶颈）。现在合并为一个 YOLO26n-pose，每帧推理次数从 `1 + N_players` 降为 `1`，且全程可走 GPU。

#### `stroke_classifier.py` — 滑窗 GRU 击球分类
- **职责**：消费 `YOLOPoseTracker.push_frame()` 的输出，维护每个球员的 pose 滑窗，定期喂模型，输出 `StrokeEvent(frame, label, confidence, cx, cy, player_id)`。
- **入口**：`MultiPlayerStrokeRecognizer` —— 以 track ID 为 key，每个 ID 一条独立滑窗；单打 / 双打同一份代码。
- **算法**（移植自 [antoinekeller/tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition)）：
  1. 每帧取 pose 的 13 个 COCO 关键点（去掉眼 / 耳）→ 26 维 `(y, x)` 特征
  2. **关键点相对 bbox 归一化**（而非相对整帧）—— 否则特征会编码"球员在场地哪里"，分类崩掉
  3. 30 帧滑窗（≈1 s @ 30 fps）→ Keras GRU → softmax 4 类（`backhand / forehand / neutral / serve`）
  4. 置信度阈值 `min_confidence=0.9` + `stride=5` 避免重复触发 + 可选抑制 `neutral`
- **track TTL**：track 短暂消失（<30 帧，例如被网挡）不重置滑窗，避免瞬时漏检打断识别。
- **Keras 兼容**：`tennis_rnn.h5` 是 Keras 2 保存的，代码优先 `import tf_keras`，不可用时 fallback 到 Keras 3。

#### 在 pipeline 中的位置
`pipeline/analyze.py:_build_stroke_recognizer` 构造 `YOLOPoseTracker` + `MultiPlayerStrokeRecognizer`；Pass 1 每帧调 `stroke_rec.push_frame(frame, frame_idx)`，事件追加到 `AnalyzeResult.stroke_events`，`last_detections`（本帧 bbox 字典）存入 `_FrameState.player_bboxes` 供 Pass 2 直接渲染。**Pass 2** 用 `render/action.py` 的 `draw_players` + `draw_stroke_labels` 叠加 bbox 和击球标签（2 s TTL 淡出）。

---

### `tennisvision/pipeline/analyze.py` —— 编排

**两遍流水线**（借鉴 CourtCheck）：

**Pass 1 — 跟踪（不渲染）**
```
for frame in video:
    candidates  = ball_detector.detect(frame)
    tracker.update(candidates, frame_idx)
    player_bboxes = player_detector.detect(frame)   # 可选，存入 frame_states
trajectory = tracker.extract_champion_trajectory()
# 可选：InpaintNet 填充 validated tracks 内部空洞
tracks     = inpainter.fill_gaps(tracks)            # ball.inpainter.enabled
bounces    = bounce_detector.detect(tracks)
```

**Pass 2 — 渲染**
```
for frame in video:
    vis = frame.copy()
    render.trail(vis, trajectory, frame_idx)
    render.players(vis, frame_states[frame_idx].player_bboxes)   # 可选
    render.stroke_labels(vis, stroke_events, frame_idx)          # 可选
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
| `ultralytics` | YOLO26n-pose 球员检测 + ByteTrack 跟踪 + 姿态（`action.enabled`）|
| `tensorflow` + `tf-keras` | 加载 `tennis_rnn.h5`（Keras 2 GRU）做击球分类 |

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
# 推荐：自动标定（业余 / 蓝色硬地球场）
python scripts/calibrate.py blue-resnet \
    --video  input.mp4 \
    --weights weights/court_resnet.pth \
    --out    calib.json \
    --vis    calib_vis.jpg

# 手动标注图备选
python scripts/calibrate.py mark \
    --image  court_mark.jpg \
    --out    calib.json

# 广播视角（TCD heatmap CNN）
python scripts/calibrate.py tcd \
    --video  input.mp4 \
    --weights weights/court_tcd.pt \
    --out    calib.json

# 无 ML 依赖备选（蓝色轮廓，精度较低）
python scripts/calibrate.py blue \
    --video  input.mp4 \
    --out    calib.json
```

所有方法均输出 `calib.json` + `calib_vis.jpg`（court overlay 供人工核验）。

| 方法 | 适用场景 | 权重 | Reprojection err（样例） |
|---|---|---|---|
| `blue-resnet` | 业余蓝色硬地，低位摄像 | `court_resnet.pth`（~95 MB） | ~0.8 m |
| `mark` | 任意场地，需手动画框 | 无 | 取决于标注精度 |
| `tcd` | 广播 / 高位摄像 | `court_tcd.pt` | ~1–2 m（低位失效）|
| `blue` | 蓝色硬地，无 ML 环境 | 无 | ~2–4 m |

`blue-resnet` 权重下载：`gdown 1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta -O weights/court_resnet.pth`

### 完整分析
```bash
python scripts/analyze.py \
    --video input.mp4 \
    --calib calib.json \
    --out   output.mp4 \
    --ball-detector tracknet \
    --bounce-detector catboost
```

默认只跑球。加上 `--config configs/remote_inpaint.yaml`（或自定义 yaml）可启用 InpaintNet + action 全路线，终端日志会多出：

```
[action] stroke classifier enabled (multi-player, window=30, min_conf=0.90)
[action] 42 stroke events: backhand=11, forehand=26, serve=5
[action]   player #1: backhand=6, forehand=14, serve=3
[action]   player #2: backhand=5, forehand=12, serve=2
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

inpainter:                     # TrackNetV3 InpaintNet 轨迹补洞（默认关）
  enabled: false               # true → 在 bounce 检测前填充 validated track 空洞
  weights: weights/InpaintNet_best.pt
  device: cpu
  window_mode: single          # single | nonoverlap
  max_gap_frames: 60           # 超过此值的空洞不填（跨 rally）
  th_h_frac: 0.05              # 视野外 y 阈值（帧高比例）

bounce:
  detector: catboost             # peak | catboost
  weights: weights/bounce_catboost.cbm
  threshold: 0.20
  nms_window: 10

render:
  minimap_width_px: 150
  trail_length_frames: 45
  bounce_fade_frames: 600

action:                          # 球员检测 + 姿态 + 击球分类（默认关）
  enabled: false                 # true → Pass 1 多一路 pose + 分类
  pose_weights: weights/yolo26n-pose.pt
  pose_device:  cpu              # cpu | cuda
  rnn_weights:  weights/tennis_rnn.h5
  window_frames:  30             # 滑窗长度（30 ≈ 1s @ 30fps）
  min_confidence: 0.9            # 低于此值不发事件
  stride:         5              # 每球员最多 stride 帧推理一次
  emit_neutral:   false
  player:                        # 球员过滤参数（检测权重已由 pose_weights 覆盖）
    conf:         0.3
    max_persons:  4              # 双打+教练可放 5–6；单打收紧到 2
    min_bbox_h:   60             # 丢弃远处 / 极小检测
    court_margin_m: 3.0          # on-court 过滤容差（米）

debug:
  dir: null                       # 设为路径开启 debug 输出
```

开启 InpaintNet + action 全路线的示例见 [`configs/remote_inpaint.yaml`](configs/remote_inpaint.yaml)（远程测试机用的配置，权重路径指向 `/root/personal/`）。

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

### V2 — 部分完成（深度学习球检测与轨迹）
- [x] 采用 **WASB-SBDT**（MIT 许可，BMVC 2023）作为默认 detector（`ball/wasb.py`）
- [x] 首次运行自动导出 ONNX + ONNXRuntime CPU 推理（`ball/wasb.py`，~100–200 ms/帧 CPU）
- [x] 集成 **TrackNetV3 的 InpaintNet**（`ball/inpainter.py`）—— 可选填充 validated track 内部空洞；逐 Track 运行避免跨 rally 误桥接；`ball.inpainter.enabled=false` 时零开销
- [ ] 性能基准：Pass 1 端到端目标 CPU 5 fps（含 action 路线）

### V3 — 长期（物理级准确）
- [x] **球员检测 + 姿态一体**（`action/pose.py`）：YOLO26n-pose + ByteTrack，单次推理出 bbox + track ID + 17 关节点，支持 CPU / CUDA；替换原 YOLOv8n + MoveNet TFLite 双模型方案
- [x] **击球分类**（`action/stroke_classifier.py`）：30 帧滑窗 + Keras GRU → backhand / forehand / neutral / serve，每个 track ID 一条独立滑窗
- [x] Pass 2 渲染层消费 `stroke_events`（`render/action.py`：ball player bbox + 击球标签叠加，2 s TTL 淡出）
- [ ] 击球与落点的配对（正手打出 → 落点位置 → 战术统计）
- [ ] 3D EKF（参考 Tennis3DTracker）：真正的 3D 轨迹 + 物理 bounce（z=0 交越）
- [ ] 热力图、统计报告、多摄像机融合

### 非目标
- 实时直播分析（本项目定位**离线后处理**，不追求 > 10 fps）
- 职业比赛级精度（对照 Hawk-Eye 硬件多目视觉，本项目是单目近似）
- 商业 SaaS 前后端（只做 CV 核心）

---

## 参考

### 直接借鉴的项目
- **[nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT)** —— **当前球检测**（HRNet，MIT 许可，BMVC 2023）；`ball/wasb.py` + `ball/hrnet.py` 复用其代码与 tennis 权重。
- **[CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck)** —— 整体 pipeline 分层、best-of-subsets homography、CatBoost 落点。
- **[yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject)** —— 14 点球场 + CatBoost bounce 参考实现。
- **[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)** —— 15 通道 heatmap 球场检测模型，预训练权重开放。
- **[antoinekeller/tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition)** —— 30 帧滑窗 GRU 击球分类（`tennis_rnn.h5`）。
- **[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)** —— YOLO26n-pose，球员检测 + 追踪 + 姿态，替换原 YOLOv8n + MoveNet 双模型方案。

### 候选替代 / 未采用
- **[yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)** —— 业界标准球追踪 CNN；更大（~11M 参数）、tennis-only，被 WASB 在同基准上超过，故未采用。
- **[qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3)** —— TrackNet + **InpaintNet**（学习式轨迹补洞）。TrackerNet 那端我们用 WASB 替代；**InpaintNet 仍是 V2 的增量目标**（见路线图）。
- **[John-Boccio/Tennis3DTracker](https://github.com/John-Boccio/Tennis3DTracker)** —— 单目 3D EKF 重建（V3 远期）。

### 文档
- [docs/court_detection_survey.md](docs/court_detection_survey.md) —— 完整调研笔记（含各方案对比、权重下载链、选型建议）。
