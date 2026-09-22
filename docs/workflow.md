# TennisVision — Workflow

## Pipeline 流程

`run_full_pipeline.py` 编排 5 个独立脚本，按序执行：

```
video.mp4
  │
  ├─ Step 1: 场地识别 (calibrate.py)
  │    输入: 视频
  │    输出: calib.json (单应性矩阵) + court_overlay.jpg (可视化验证)
  │    模型: court_resnet.pth (ResNet50, 14 点回归)
  │    说明: 检测球场关键点，计算像素↔真实坐标映射。固定机位可跨视频复用。
  │
  ├─ Step 2: 球员姿态提取 (extract_keypoints.py)
  │    输入: 视频
  │    输出: keypoints.json (全帧球员检测 + 17 点 COCO 姿态)
  │    模型: yolo11m (人体检测+跟踪) + yolo26s-pose (姿态估计)
  │    说明: 两阶段检测，先全帧检测人体 bbox，再逐 bbox 提取关键点。
  │
  ├─ Step 3: 球轨迹提取 (extract_ball_positions.py)
  │    输入: 视频 + calib.json (用于 two-stage far-crop)
  │    输出: ball_positions.json (detected + Kalman predicted)
  │    模型: wasb_tennis_best.pth.tar (WASB HRNet, 自动导出 ONNX)
  │    说明: 双尺度检测 (全帧 + 远场裁剪)，Kalman 多目标追踪，输出检测位置和预测位置。
  │
  ├─ Step 4: 发球检测 (detect_serves.py)
  │    输入: keypoints.json + calib.json + ball_positions.json (可选)
  │    输出: serve_events.json (发球事件 + 全部 GRU 分类结果)
  │    模型: stroke_gru_v4_best.pt (GRU v4, 4 类分类)
  │    说明: 手腕速度触发 → 30 帧滑窗 GRU 分类 → 合并连续帧 →
  │          四项过滤 (网前距离/边线/画面边缘/球距离)
  │
  └─ Step 5: 回合检测 + 剪辑 (detect_rallies.py)
       输入: video + calib.json + ball_positions.json + serve_events.json
       输出: rally_events.json (回合边界) + rally_cuts.mp4 (回合剪辑视频)
       模型: 无 (纯算法)
       说明: 发球 dedup (10s 窗口) → 每个发球后检测回合结束:
             1. 过网超时 (5s 无过网 → 死球)
             2. 三连弹 (同半场 2s 内 3 次落地)
             3. 下一个发球 (硬截止)
```

### 依赖关系

```
Step 1 (场地) ──┬──────────────────────┐
                │                      │
Step 2 (姿态) ──┤                      │
                ├── Step 4 (发球) ──── Step 5 (回合)
Step 3 (球) ────┘                      │
                                       └── rally_cuts.mp4
```

- Step 1/2/3 互相独立，可并行
- Step 4 依赖 Step 1 + 2 (+ 可选 Step 3)
- Step 5 依赖 Step 1 + 3 + 4

---

## 运行方式

### 全流程

```bash
python scripts/run_full_pipeline.py \
    --video samples/sample2.mp4 \
    --weights-dir weights/
```

每步检测到已有输出会自动跳过，删除文件后重跑即可。

### 单步执行

```bash
# 指定步骤
python scripts/run_full_pipeline.py --video VIDEO --step 3

# 或直接调用脚本
python -u scripts/extract_ball_positions.py \
    --video VIDEO --out ball_positions.json \
    --weights weights/wasb_tennis_best.pth.tar --device cuda
```

### 混合执行（GPU 服务器提取 + 本地分析）

```bash
# 1. GPU 服务器上跑 Step 1-3（耗时步骤）
python scripts/run_full_pipeline.py --video VIDEO --step 1
python scripts/run_full_pipeline.py --video VIDEO --step 2
python scripts/run_full_pipeline.py --video VIDEO --step 3

# 2. 下载 results/<video_name>/ 到本地

# 3. 本地跑 Step 4-5（秒完，不需要 GPU）
python scripts/run_full_pipeline.py --video VIDEO --step 4
python scripts/run_full_pipeline.py --video VIDEO --step 5
```

---

## 输出目录

```
results/<video_name>/
  calib.json              # Step 1: 场地标定
  court_overlay.jpg       # Step 1: 场地可视化
  keypoints.json          # Step 2: 球员姿态
  ball_positions.json     # Step 3: 球轨迹
  serve_events.json       # Step 4: 发球事件
  rally_events.json       # Step 5: 回合边界
  rally_cuts.mp4          # Step 5: 回合剪辑
```

---

## 模型权重

| 文件 | 步骤 | 说明 |
|------|------|------|
| `court_resnet.pth` | Step 1 | ResNet50 球场 14 点回归 (~95MB) |
| `yolo26s-pose.pt` | Step 2 | YOLO26s-pose 球员姿态 (ultralytics) |
| `wasb_tennis_best.pth.tar` | Step 3 | WASB HRNet 球检测 (首次运行自动导出 .onnx) |
| `stroke_gru_v4_best.pt` | Step 4 | GRU v4 发球/击球分类 |
| `bounce_catboost.cbm` | Step 5 | CatBoost 落点分类器 |
| `InpaintNet_best.pt` | (可选) | TrackNetV3 轨迹补洞 |

所有权重放在 `weights/` 目录（gitignored），不随代码分发。
