# 03 — 球轨迹跟踪

## 背景

球检测每帧独立输出 0..N 个候选点，互相没有身份关联。下游（落点、回合）需要**跨帧的轨迹**，以及每帧一个唯一的"当前球"（champion）。Tracker 的职责就是把候选点关联成轨迹，并平滑、去噪、补帧。

## 当前实现

[tennisvision/ball/tracker.py](../../tennisvision/ball/tracker.py)

### 数据结构

- `TrackPoint(x, y, frame)` — 单点
- `Track(id, pts, validated, last_det_frame, ...)` — 一条轨迹，内置 4 态 Kalman `[x, y, vx, vy]`
- `MultiTrackManager` — 管理 active + retired tracks，每帧做关联 + champion 选择

### 算法

1. **每帧 predict**：所有 active track 调 Kalman `predict()` 得到本帧预期位置
2. **贪婪关联**：候选点 → track 距离最小优先匹配，距离 < `gate_px` 才算匹配
3. **未匹配候选 → 新 track**；未匹配 track → 累计 silence，超阈值 retire
4. **validation**：连续观测 ≥ 3 且平均速度 ∈ `[min_speed, max_speed]` px/帧 → `validated=True`
5. **`pick_champion(frame_idx)`**：选最长的、最近仍在更新的 validated track（优先当前帧有检测的）

### 为什么保留 tracker

WASB 引入后，单帧检测已经足够准。但：

- **短时遮挡 / 漏检补帧**：Kalman 能给出合理预测，让 trail 连续
- **假阳过滤**：孤立的 2-3 帧短 track 直接丢弃（不 validated）
- **bounce / rally 需要整条序列**：CatBoost bounce 要做三次样条插值；rally detector 要跨帧统计过网次数

### 比例化阈值

硬编码 80 px gate 在 720p 过紧、4K 过松。现在配置项支持 `gate_ratio`、`min_speed_ratio`、`max_speed_ratio`，运行时乘以 `frame_diagonal` 得到像素值，同一份 config 跨分辨率可用。

## 设计决定

- **多目标而非单目标**：训练/比赛现场经常能看到两只球同时在画面里（旁边场地的球、捡球），保留多 track 再用 champion 逻辑选最可信的一条，比硬性单目标更鲁棒
- **Kalman 匀速模型 vs 抛物线模型**：抛物线更准但维度 / 噪声设定麻烦，对短时预测（1-3 帧）增益有限，保留匀速模型
- **champion 选长而非选近**：刚出现的 track 一开始就做 champion 会导致每次球离场再出现都换一个 champion，轨迹抖。选长 track 能穿透瞬时漏检
- **retired_tracks 保留到最后**：Pass 1a 结束后所有曾 validated 的 track 都在 `retired_tracks` 字典里，交给 bounce / inpaint 做跨 rally 全局处理

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 的 `tracker` 段：

```python
"tracker": {
    "gate_ratio": 0.054,          # * frame diag → ~120 px @ 1080p
    "max_gap_frames": 5,
    "min_len": 3,
    "min_speed_ratio": 0.0023,    # * frame diag → ~5 px/f @ 1080p
    "max_speed_ratio": 0.091,     # * frame diag → ~200 px/f @ 1080p
    "render_tail": 45,            # Pass 2 trail 长度
}
```

## 已知限制 / TODO

- [ ] 贪婪匹配在近距两个 track 交叉时会瞬间错乱（应改 Hungarian）
- [ ] Kalman 过程噪声固定，快速变向（截击）时预测偏保守
- [ ] 球员自己手上 / 地上 的球常被识别为 track，靠 validation 剔除但 tail 上会闪烁

## 参考

无外部直接借鉴，参考了 OpenCV 的 KalmanFilter 文档。
