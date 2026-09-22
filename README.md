# TennisVision

业余网球比赛视频分析工具。固定机位视频输入，输出：

- 球场几何标定（单应性矩阵，像素 ↔ 真实坐标映射）
- 球的飞行轨迹 + 每次落地位置（场内坐标）
- 发球检测（GRU 姿态分类器，区分一发/二发）
- 回合（rally）切分：起止时间、击球次数、击球类型（正手 / 反手 / 发球）
- 可视化叠加：轨迹、小地图、球员 bbox、击球标签、HUD
- 产物：`rally_cuts.mp4`（回合剪辑）、`rally_events.json`（回合元数据）、`ball_positions.json`（球轨迹）等

设计目标：**模块可替换、CPU 可运行、接口清晰**的工程化实现。

---

## 快速开始

### 方式一：全流程编排（推荐）

```bash
python scripts/run_full_pipeline.py \
    --video samples/sample2.mp4 \
    --weights-dir weights/
```

自动按序执行 5 个步骤，结果输出到 `results/<video_name>/`：

| Step | 脚本 | 输出 | 说明 |
|------|------|------|------|
| 1 | `calibrate.py blue-resnet` | `calib.json` + `court_overlay.jpg` | 球场标定（可跨视频复用） |
| 2 | `extract_keypoints.py` | `keypoints.json` | 球员检测 + 姿态（yolo11m + yolo26s-pose） |
| 3 | `extract_ball_positions.py` | `ball_positions.json` | 球检测 + Kalman 追踪（WASB HRNet） |
| 4 | `detect_serves.py` | `serve_events.json` | GRU v4 发球分类 + 多重过滤 |
| 5 | `detect_rallies.py` | `rally_events.json` + `rally_cuts.mp4` | 回合切分 + 视频剪辑 |

每步检测到已有输出会自动跳过，删除文件后重跑即可。

### 方式二：集成两遍流水线

```bash
python scripts/analyze.py \
    --video input.mp4 \
    --calib calib.json \
    --config configs/local_run.yaml \
    --out output.mp4
```

单进程两遍扫描，输出标注视频 + 回合剪辑 + JSON。适合需要渲染叠加（轨迹、小地图、HUD）的场景。

### 方式三：单步执行

```bash
# Step 1: 球场标定
python scripts/calibrate.py blue-resnet \
    --video VIDEO --out calib.json --vis court_overlay.jpg \
    --weights weights/court_resnet.pth

# Step 2: 球员姿态提取
python -u scripts/extract_keypoints.py \
    --video VIDEO --out keypoints.json [--max-frames N]

# Step 3: 球轨迹提取
python -u scripts/extract_ball_positions.py \
    --video VIDEO --out ball_positions.json \
    --weights weights/wasb_tennis_best.pth.tar [--device cuda]

# Step 4: 发球检测
python -u scripts/detect_serves.py \
    --keypoints keypoints.json --calib calib.json \
    --gru weights/stroke_gru_v4_best.pt --out serve_events.json \
    [--ball ball_positions.json]

# Step 5: 回合检测
python -u scripts/detect_rallies.py \
    --video VIDEO --results-dir results/<name>/
```

---

## Pipeline 架构

### 数据流

```
video.mp4
   |
   +-- Step 1: court/detector -----> 14 keypoints --> homography --> calib.json
   |                                                                     |
   +-- Step 2: yolo11m + yolo26s-pose -------> keypoints.json            |
   |                                                |                    |
   +-- Step 3: WASB HRNet + Kalman tracker -> ball_positions.json        |
   |                                                |                    |
   |   Step 4: GRU v4 serve classifier <-----------+---------+----------+
   |              + post-merge filtering                      |
   |              --> serve_events.json                       |
   |                       |                                  |
   |   Step 5: rally_fusion (net crossing + triple bounce + timeout)
   |              --> rally_events.json + rally_cuts.mp4
   |
   +-- (Optional) analyze.py: two-pass render --> annotated output.mp4
```

### 回合检测（rally_fusion）

基于发球事件切分回合，三个结束条件：

1. **Net crossing timeout** — 球超过 N 秒未过网（默认 5s），判断为死球
2. **Triple bounce** — 同一半场 2 秒内连续 3 次落地，兜底结束
3. **Next serve** — 下一个发球事件作为硬截止

过网检测使用 detected-only positions（排除 Kalman 预测），并限制在球场 x 范围内（含 2m margin），减少误判。

### 发球检测（detect_serves）

1. **触发**：手腕速度 > 阈值 → 激活 GRU 推理窗口
2. **分类**：30 帧滑窗 GRU v4（4 类：backhand / forehand / serve / background）
3. **合并**：同 slot 连续帧合并为单个事件
4. **过滤**：网前距离、边线、画面边缘、球 x 距离四项检查

---

## 目录结构

```
tennisvision/
├── README.md
├── CLAUDE.md                        # 开发约定
├── requirements.txt
│
├── configs/
│   ├── local_run.yaml               # 本机 CPU（含 action + inpainter）
│   ├── remote_inpaint.yaml          # 远程 GPU 完整路线
│   └── ...
│
├── weights/                         # 模型权重（gitignored）
│   ├── court_resnet.pth             # ResNet50 球场 14 点回归
│   ├── wasb_tennis_best.pth.tar     # WASB HRNet 球检测（自动导出 .onnx）
│   ├── stroke_gru_v4_best.pt        # GRU v4 发球/击球分类
│   ├── bounce_catboost.cbm          # CatBoost 落点分类器
│   ├── InpaintNet_best.pt           # TrackNetV3 轨迹补洞（可选）
│   ├── yolo26s-pose.pt              # 球员姿态（ultralytics, Step 2）
│   └── tennis_rnn.h5               # 击球分类 GRU（analyze.py 路线用）
│
├── scripts/                         # CLI 入口
│   ├── run_full_pipeline.py         # 全流程编排器（subprocess, 5 步）
│   ├── calibrate.py                 # 球场标定
│   ├── extract_keypoints.py         # 球员姿态提取
│   ├── extract_ball_positions.py    # 球轨迹提取
│   ├── detect_serves.py             # 发球检测
│   ├── detect_rallies.py            # 回合检测 + 剪辑
│   ├── analyze.py                   # 集成两遍流水线（含渲染）
│   └── export_onnx.py               # ONNX 导出工具
│
├── tennisvision/                    # 主包
│   ├── config.py                    # 超参默认值（DEFAULTS）
│   ├── court/                       # 球场几何 + 标定
│   │   ├── detector.py              # ResNet / TCD / 蓝色检测
│   │   ├── homography.py            # 单应性计算
│   │   ├── calibration.py           # calib.json 读写
│   │   └── reference.py             # 标准球场尺寸
│   ├── ball/                        # 球检测 + 轨迹
│   │   ├── wasb.py                  # WASB HRNet（含 two-stage far-crop）
│   │   ├── tracker.py               # Kalman 多目标追踪
│   │   ├── inpainter.py             # TrackNetV3 InpaintNet（可选）
│   │   └── classical.py             # HSV + MOG2（备选）
│   ├── bounce/                      # 落点检测
│   │   ├── catboost.py              # CatBoost 分类器
│   │   └── peak.py                  # y 极值法（备选）
│   ├── action/                      # 球员动作
│   │   ├── slot_mapper.py           # 4-slot 空间映射（NEAR_L/R, FAR_L/R）
│   │   ├── gru_classifier.py        # GRU v4 击球分类
│   │   ├── pose.py                  # YOLO-pose 封装
│   │   └── stroke_classifier.py     # analyze.py 用的分类器
│   ├── pipeline/                    # 编排
│   │   ├── analyze.py               # 两遍流水线主逻辑
│   │   ├── rally.py                 # OnlineRallyDetector + 验证 + 剪辑
│   │   └── rally_fusion.py          # 多信号回合检测（serve + ball + net crossing）
│   └── render/                      # 可视化
│       ├── trail.py                 # 球轨迹渲染
│       ├── minimap.py               # 小地图
│       ├── hud.py                   # HUD 信息
│       └── action.py                # 击球标签叠加
│
├── docs/                            # 设计文档
│   ├── workflow.md                  # 工作流（含 Kaggle）
│   ├── issues/                      # 功能级设计文档
│   └── ...
│
├── tests/                           # 单元测试
└── results/                         # 输出目录（gitignored）
```

---

## 配置

所有默认参数在 [`tennisvision/config.py`](tennisvision/config.py) 的 `DEFAULTS` 字典。YAML override 只写要改的字段：

```yaml
# configs/custom.yaml
ball:
  two_stage: true
  score_threshold: 0.30
action:
  enabled: true
rally:
  online_min_net_crossings: 3
```

现有配置：

- `configs/local_run.yaml` — 本机 CPU，含 action + inpainter
- `configs/remote_inpaint.yaml` — 远程 GPU，完整路线
- `configs/remote_inpaint_debug.yaml` — 同上 + debug 落盘

---

## 输出目录结构

```
results/
  sample2/
    calib.json              # 球场标定（H_img_to_real 矩阵）
    court_overlay.jpg       # 球场检测可视化
    keypoints.json          # 全帧球员检测 + 17 点姿态
    ball_positions.json     # 球轨迹（detected + predicted）
    serve_events.json       # 发球事件（含 all_events 全部 GRU 输出）
    rally_events.json       # 回合边界
    rally_cuts.mp4          # 回合剪辑视频
```

---

## 设计原则

- **关注点分离** — 球场、球检测、姿态、发球、回合、渲染彼此独立
- **CPU 优先** — 默认配置笔记本 CPU 可跑通，GPU 作为加速选项
- **标定复用** — 固定机位的 `calib.json` 可跨视频复用
- **幂等输出** — 每步检测到已有文件自动跳过，删除后重跑
- **薄 CLI** — `scripts/` 仅做参数解析，业务逻辑在 `tennisvision/` 包内

---

## 参考

- [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) — 球检测（HRNet, MIT, BMVC 2023）
- [CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck) — pipeline 分层、best-of-subsets homography、CatBoost bounce
- [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) — 14 点球场 + CatBoost bounce
- [antoinekeller/tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition) — 30 帧 GRU 击球分类
- [Ultralytics YOLO](https://docs.ultralytics.com/) — 球员检测 + 姿态
- [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) — InpaintNet 轨迹补洞
