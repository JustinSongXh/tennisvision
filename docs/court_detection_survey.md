# 开源网球视觉项目调研：球场检测与球追踪

针对 **CourtCheck** (AggieSportsAnalytics) 及 5–6 个同类开源项目做的技术调研，目的是为本项目（HSV+MOG2+Hough 的轻量 CV pipeline）规划下一步升级方向。

---

## 1. CourtCheck 深入剖析

仓库：<https://github.com/AggieSportsAnalytics/CourtCheck>
架构：后端 FastAPI + 前端 Next.js，CV pipeline 在 `backend/` 下。**权重文件未开源**（`backend/weights/` 被 gitignore），但源码完整可读。

### 1.1 整体 pipeline（两遍流式）

- **初始化**：加载 5 个模型 —— `BallDetector`（TrackNet）、`PlayerTracker`（YOLOv8 Pose）、`BounceDetector`（CatBoost）、`PoseStrokeClassifier`（TCN，可选）、`SwingDetector`（基于规则）。球场标定：读 `court_calibration.json`，或用前 10 帧即时检测。
- **Pass 1（追踪，不渲染）**：逐帧球检测 + YOLOv8 Pose + 挥拍事件（姿态+球距离）+ 原始轨迹上的落点检测 + 击球帧（速度突变）。
- **Pass 2（渲染）**：叠球轨迹、球场轮廓、球员框、击球标签、左上角 mini-map。
- **后处理**：homography 投影做热力图（落点密度 / 球员位置 / 击球点）+ 高斯平滑 → ffmpeg H.264 封装 → Supabase 上传 → GPT-4o-mini 生成文字报告。

### 1.2 球场关键点检测 (`backend/models/court_line_detector.py`)

- **架构**：纯 `torchvision.models.resnet50(weights=ImageNet1K_V1)`，最后一层 FC 换成 `nn.Linear(fc.in_features, 14*2)` —— **28 维坐标回归头**，不是 heatmap，不是分割。
- **输入**：224×224，ImageNet mean/std 归一化。
- **输出**：14 个 (x, y) 点。按原图尺寸比例回缩。没有 subpixel 精化，没有 Hough，也没有 RANSAC。
- **权重**：`backend/weights/keypoints_model.pth`（未开源）。
- **CPU 可行性**：✅ 一次 224×224 ResNet50 前向 ~100ms，而且只在校准时跑一次（不是逐帧）。

14 个关键点的配置与 yastrebksv/TennisCourtDetector 一致：两端底线各 2 点、内外边线各 2 点、发球线上下各 2 点、中线 2 点（见 `backend/vision/court_reference.py`，定义了一个 1117×2408 的标准球场）。

### 1.3 球检测 (`backend/models/ball_tracker.py`)

- **架构**：编码-解码 CNN，**不是 YOLO**。9 通道输入（3 帧 RGB 堆叠）、360×640、编码器 64→128→256→512、解码器镜像、256 通道 heatmap 输出。**逐字逐句拷自 yastrebksv/TrackNet `model.py`**（`BallTrackerNet`）。
- **推理**：`infer_single(frame)` 内部维护 3 帧缓冲；满了才返回 `(x, y)`（2× 上采样回 1280×720）。
- **权重**：`backend/weights/tracknet_weights.pt`（猜测是 yastrebksv Drive checkpoint `1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl` 或其 finetune）。
- **时序过滤**：用上一帧预测筛异常值。
- **CPU 可行性**：⚠️ 勉强。~11M 参数、360×640 输入，CPU 上 300–800 ms/帧。官方部署在 Modal A10G GPU 上。

### 1.4 落点检测 (`backend/models/bounce_detector.py`)

**不是** y 方向极大值，是 **CatBoost 回归器学轨迹特征**：

- **特征**：x/y 位置 + lag(1,2) 前后差（速度代理）+ **前向/后向差比值**（速度反转代理，带 epsilon 平滑）。
- **预处理**：三次样条插值补漏检 + 速度 >80 px/帧 的外点剔除。
- **后处理**：阈值 0.20，对连续帧做 NMS，保最高置信度。
- **权重**：`backend/weights/bounce_detection_weights.cbm`（同样来自 yastrebksv TennisProject）。

**为什么打败纯 y-peak**：y-peak 会在每次击球顶点、网柱遮挡、Kalman 平滑过度等地方误触发。CatBoost 学的是**速度双分量联合反转**特征 —— 顶点只有 y 反转，真弹跳有 y 反转 **加上** y 速度前后不对称（反弹因非弹性碰撞会变慢）。

### 1.5 Homography (`backend/vision/homography.py`) —— 最有启发的部分

- 在 14 个检测点里，**预列出多组 4 点子集**（`court_ref.court_conf`）。
- 对每组：任意点缺失则跳过；否则 `cv2.findHomography(..., method=0)`（DLT）。
- **Best-of**：用每个候选 H 把**其他 ~10 个**检测点反投影，挑**反投影误差最小**的配置。
- 等效于无 RANSAC 的 RANSAC —— 在球员遮挡一半关键点时仍能优雅退化。
- 标定结果缓存到 `court_calibration.json`（按 `camera_id`），支持"检测一次，永久复用"。

> **这是本项目最该照搬的模式**。我们目前固定 4 个锚点，只要一个被遮挡就崩；best-of-subsets 天然鲁棒。

### 1.6 框架依赖

`requirements.txt`：torch、ultralytics、catboost、opencv、scipy、lap、modal。**没用 ONNX**。部署目标 Modal A10G GPU，但 court 模型和 bounce 分类器在 CPU 上没问题；瓶颈在 ball 模型。

---

## 2. 同类开源项目对比

| 项目 | 球场 | 球 | 落点 | 权重 | License |
|---|---|---|---|---|---|
| **yastrebksv/TennisProject** | 14-pt heatmap CNN | TrackNet (9-ch) | CatBoost 轨迹特征 | 全开放（Drive） | 无声明 |
| **yastrebksv/TennisCourtDetector** | 15-ch heatmap, 640×360, 中值误差 **1.83 px** | — | — | Drive `1f-Co64ehgq...` | 无声明 |
| **yastrebksv/TrackNet** | — | 9-ch 3-frame → heatmap | — | Drive `1XEYZ4myUN7...` | 无声明 |
| **ArtLabss/tennis-tracking** | 经典线检测（调参跨场地） | TrackNet (Keras) | sktime TimeSeriesForest | 全开放（含 `clf.pkl`） | Unlicense |
| **qaz812345/TrackNetV3** | — | TrackNet + **InpaintNet**（学习式轨迹补洞）+ mixup | — | Drive | 无声明 |
| **nttcom/WASB-SBDT** | — | 多尺度 heatmap 小球检测，5 种运动预训练 | — | 全开放，tennis Drive `14AeyIOCQ2UaQ...` | **MIT** |
| **John-Boccio/Tennis3DTracker** | 调用 yastrebksv TennisCourtDetector | 调用 yastrebksv TrackNet | **3D EKF + 物理** → z=0 时即落地 | 复用上游 | 无声明 |
| **abdullahtarek/tennis_analysis 等一批** | ResNet50 + Linear(28) 回归 | **YOLOv5 单帧** finetune | 基本没做 | Drive | 无声明 |

### 个别亮点

- **ArtLabss**：目前唯一主流项目还在用经典 CV 做球场检测。用 sktime 的 `TimeSeriesForestClassifier` 做落点（claim 98%/83% accuracy），和 CatBoost 是平级替代方案。License 是 Unlicense，可以放心抄。
- **TrackNetV3**：把 TrackNet 的 3 帧输入扩到 seq_len=8 的预测 + seq_len=16 的 **InpaintNet 专学遮挡补洞**。羽毛球基准 F1=98.56%。但论文（WASB）显示在 tennis 数据集上仍被 WASB 击败。
- **WASB (BMVC 2023)**：当前小球检测 SOTA + MIT 协议 + 跨 5 种运动的预训练。`MODEL_ZOO.md` 直接给每种球类 × 6 种模型的 Drive 下载链接。**如果我们要商用或公开，这是唯一可选。**
- **Tennis3DTracker**：最小可读的 "TrackNet + 球场检测 + EKF" 拼装示例，核心在 `Tennis3DTracker.py` / `Tennis3DTrackerHelper.py`。单目 3D 重建，落点通过 EKF 状态 z=0 穿越推断（比任何 2D 启发式都物理可靠）。
- **abdullahtarek 一系**（YouTube 教程克隆）：ResNet50 28-float 回归 + YOLOv5 球检测。重要作为对照 —— YOLOv5 单帧对**小而模糊**的球召回差，这也是业界普遍不选它做 tennis 的原因。

---

## 3. 综述：2024–2026 的主流做法

### 3.1 球场检测：关键点回归 CNN 已赢，但有两派

1. **每点一张 heatmap**（yastrebksv/TennisCourtDetector，15 通道，640×360，Gaussian peak 提取 → 中值误差 1.83 px）。**精度高**、支持 subpixel 精化、推理更重。
2. **直接坐标回归**（CourtCheck、abdullahtarek，ResNet50 + Linear(28)）。**简单快速**，精度略低、无法精化。

**没人在 2024+ 还用 blue-contour + Hough 做生产。** 经典方法只剩 ArtLabss 作为"跨场地鲁棒"的营销卖点，而且他们自己也承认脆弱。

### 3.2 Homography：`findHomography` + best-of-subsets

所有项目都用 `cv2.findHomography` + DLT。**关键巧思在 CourtCheck 和 yastrebksv/TennisProject 的 `homography.py`**：从 14 个关键点里遍历多组 4 点组合，每组算 H，然后用**其他 10 个点的反投影误差**打分，挑最优。等价于无需编 RANSAC 代码就能获得 RANSAC 的鲁棒性。

### 3.3 球检测：TrackNet 家族统治

- **前沿**：WASB ≥ TrackNetV3 > TrackNetV2 ≥ TrackNet（原版）。
- **社区生态**：多数项目仍用 yastrebksv/TrackNet（原版），因为预训练权重开放、架构简单易懂。
- **YOLO 单帧** 是反面教材 —— 对远场小球和运动模糊召回差。

### 3.4 落点检测：轨迹分类器 > 几何启发式

- **y-peak**（我们当前方案）在顶点、遮挡、Kalman 抖动处都会误触发。
- **CatBoost / TSF 学习轨迹特征**（速度双分量联合反转 + 前后速度不对称）是开源标准做法。
- **EKF + 物理**（Tennis3DTracker）是最优解，但要求良好 homography + 相机标定。

---

## 4. 对本项目的迁移建议（按 ROI 排序）

### Step 1：替换球场检测（高收益、低成本）

**Drop-in 推荐**：<https://github.com/yastrebksv/TennisCourtDetector>
- 权重：Drive `1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG`
- 输入 640×360，输出 15 通道 heatmap，1.83 px 中值误差
- **CPU 上 150–300 ms/帧**，但每个机位**只需跑一次并缓存**（视频开头 + 写入 `calib.json`），后续复用 —— 不是瓶颈
- Homography 直接照搬 CourtCheck `backend/vision/homography.py` 的 best-of-4-subsets 模式

这一步做完，彻底摆脱"用户手标红线"和"蓝色轮廓 + Hough"两条我们已经证明不靠谱的路线。

### Step 2：替换落点检测（高收益、极低成本）

从 yastrebksv/TennisProject 直接抄：
- `bounce_detector.py`
- `ctb_regr_bounce.cbm`（CatBoost 权重，几百 KB）
- 依赖：`catboost` + `scipy`（样条插值）

输入我们的 champion track `(x, y, frame)` 序列，输出弹跳帧索引 + 置信度。**只换这一块，预计误检率能降一个数量级。**

### Step 3：替换球检测（高收益、中等成本）

- **优先**：WASB (nttcom/WASB-SBDT) tennis 权重，MIT 许可，日后要分享/商用唯一能打的方案
- **次选**：yastrebksv/TrackNet 原版，导出 ONNX + ONNXRuntime（Intel 用 OpenVINO EP，Mac 用 CoreML EP），CPU 通常 5–10× 提速
- 保留我们现有的 Kalman tracker 做后处理，两者互补

**不要用 YOLO** —— 单帧没时序上下文，对小球召回差，而且丢了 TrackNet 天然的轨迹平滑，下游 bounce detector 也更难用。

### Step 4（可选）：Tennis3DTracker 式 3D EKF

做完 Step 1–3 之后，如果想要**真正的 3D 轨迹 + 物理级落点**，可以照 `Tennis3DTracker.py` 接 EKF。代价是相机外参标定 + EKF 参数调优，回报是彻底告别启发式 bounce detection。

---

## 关键源文件速查

动手实施时值得精读的文件：

- **Homography best-of-subsets**：`CourtCheck/backend/vision/homography.py` + `court_reference.py`
- **CatBoost bounce**：`CourtCheck/backend/models/bounce_detector.py`
- **TrackNet 规范架构**：`yastrebksv/TrackNet/model.py`
- **3D EKF 拼装范例**：`Tennis3DTracker/Tennis3DTracker.py` + `Tennis3DTrackerHelper.py`
- **14-pt 标准球场几何**：`CourtCheck/backend/vision/court_reference.py`
