# 06 — 球员检测 + 姿态

## 背景

V3 引入的模块：为每个球员输出 bbox、稳定的 track ID、17 关节点姿态。下游击球分类（[issue 07](07-stroke-classifier.md)）吃的是这个输出。默认**关闭**（`action.enabled=false`），开启后在 Pass 1b 的每个 rally 窗口内运行（或 online 模式下逐帧运行）。

## 当前实现

[tennisvision/action/pose.py](../../tennisvision/action/pose.py)

### 模型

Ultralytics YOLO26n-pose（7.6 MB）：

- R-ELAN 注意力架构
- 单次推理同时出 bbox + ByteTrack track ID + 17 个 COCO 关键点
- CPU / CUDA 都支持

### 接口

```python
tracker = YOLOPoseTracker(...)
results = tracker.track(frame)              # {track_id: (bbox, Pose)}
#   bbox = (x0, y0, x1, y1)
#   Pose has 17 COCO keypoints as (y_px, x_px, score)
```

### 旁观者过滤（三层）

1. `min_bbox_h`（默认 60 px）—— 丢弃远处 / 极小检测
2. **on-court 过滤** —— bbox 底部中心经标定 H 投影到球场平面，只保留脚在 `[-margin, 场地+margin]` 米以内的人（默认 `court_margin_m=3.0`）
3. `max_persons` —— 置信度排序取前 N（默认 4，双打 + 教练可放 5-6；单打收紧到 2）

### 设备切换

```yaml
action:
  pose_device: cuda      # cpu | cuda
```

CPU 上 YOLO26n-pose 约 80-120 ms/帧 @1080p；CUDA T4 级别 < 20 ms/帧。

## 设计决定

- **YOLO26n-pose 替代原 YOLOv8n + MoveNet 双模型**：原方案每帧推理次数 = `1 + N_players`（YOLOv8n 找人 + MoveNet 对每个人姿态），且 MoveNet 只有 TFLite CPU 版本，是整个 pipeline 瓶颈。合并后每帧推理 1 次，全程 GPU 可用
- **on-court 过滤放在 H 投影之后**：直接在像素空间卡 y 坐标在低位机位会漏掉远端半场的球员；投到球场米再判 in-bounds，几何意义明确
- **ByteTrack 跨帧 ID 稳定**：击球分类需要"同一球员的连续 30 帧"，若每帧 ID 乱跳，GRU 滑窗无法维护
- **ByteTrack ID 不跨 rally**：offline 模式下 Pass 1b 每个 rally 开头会 reset tracker，避免上一回合的 ID 污染本回合判断

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 的 `action` 段（player 部分）：

```python
"action": {
    "enabled": False,
    "pose_weights": "weights/yolo26n-pose.pt",
    "pose_device": "cpu",                # cpu | cuda
    "player": {
        "conf": 0.3,
        "max_persons": 4,
        "min_bbox_h": 60,
        "court_margin_m": 3.0,
    },
}
```

## 依赖

- `ultralytics` —— YOLO26n-pose + ByteTrack
- 懒加载：`action.enabled=false` 时不 import，不付 startup 成本

## 已知限制 / TODO

- [ ] 双打场景偶尔把球童 / 教练识别为球员（`max_persons` 调高即兜住，但偶发漏掉真球员）
- [ ] 球员重度遮挡（正反手挥拍瞬间身体挡住自己）姿态质量下降，下游击球分类对应一段 confidence 降低
- [ ] `court_margin_m` 对倾斜机位不够鲁棒（3 米在接近镜头的一侧可能过小）

## 参考

- [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- COCO 17 关键点定义
