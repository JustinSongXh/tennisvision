# Experiment Log — Rally Detection Pipeline

Issue: #8 (回合检测), #6 (球员检测+姿态), #12 (音频击球检测)

## 最终验证通过的流程

```
视频输入 (30fps)
    │
    ├── Pass 1a: 球轨迹提取
    │     WASB ball detector + Kalman tracker
    │     → ball_positions (detected + predicted)
    │
    ├── Pass 1b: 球员检测 + 动作分类
    │     │
    │     ├── Stage 1: yolo11m 全图检测 (所有人 bbox + track ID)
    │     ├── Homography on-court 过滤 (去掉场外人)
    │     ├── 4 槽位空间映射 (NEAR_L/R, FAR_L/R → 固定 4 ID)
    │     ├── Stage 2: yolo26s-pose crop → 17 COCO keypoints (bbox 归一化)
    │     ├── 手腕速度触发 (帧间距离 > 0.15 → 活跃)
    │     ├── GRU v4 (活跃期间每帧推理, 30帧×34维 → 4类)
    │     └── Serve 特殊标记 (prob > 0.8, 合并连续帧)
    │     → 击球事件列表 (时间, 槽位, 类别, 置信度)
    │
    └── 回合检测状态机
          serve (底线附近) → 回合开始
          球轨迹过网 → 回合继续
          N秒没过网 → 回合结束
          → 回合列表 + 标注视频 + 剪辑视频
```

## Serve 检测最终结果 (sample_short.mp4)

| 真实 serve | 检出 | conf  | 槽位  |
|-----------|------|-------|-------|
| 9.2s      | ✅   | 0.995 | FAR_L |
| 31s       | ✅   | 0.996 | FAR_R |
| 39.4s     | ✅   | 0.955 | FAR_R |
| 58.6s     | ✅   | 0.996 | FAR_L |
| 80s       | ❌   | 0.636 | -     |
| 97s       | ✅   | 0.909 | FAR_L |
| 136s      | ✅   | 0.991 | FAR_R |
| 148.6s    | ✅   | 0.992 | FAR_R |

**7/8 检出, 1 误检 (14.7s NEAR_R, 可用底线过滤去掉)**

---

## 成功经验

### 1. 两阶段检测 (解决远端小目标问题)

**问题**: YOLO-pose 单阶段在全图上完全检测不到远端球员 (conf=0.01 也无候选)

**原因**: pose 模型需要 keypoint 特征, 远端人在 feature map 上太小

**方案**: yolo11m 全图检测 (能检到远端 conf=0.76) → crop → yolo26s-pose 提取 keypoints

**验证**: 远端人 crop 后 pose conf=0.79-0.90, 14-17/17 keypoints 可见

**关键发现**: 不是像素太少 (80px), 而是 pose 模型架构对小目标不友好。同样的 crop 单独跑 YOLO-pose 能检出 conf=0.89

### 2. 4 槽位空间映射 (解决 Track ID 碎裂)

**问题**: ByteTrack 产生 40-58 个 ID (应该只有 4 人), 发球挥拍瞬间 ID 断裂

**原因**: 远端检测 conf 波动 + 快速运动模糊 → ByteTrack 丢失重建 ID

**方案**: 用 homography 投影到球场坐标, 按象限分配固定 slot ID [NEAR_L, NEAR_R, FAR_L, FAR_R]

**效果**: 不管 ByteTrack 给什么 ID, 空间映射始终稳定。GRU 窗口数据不再因 ID 跳变而断裂

### 3. 智能切片训练 (解决 GRU 误分类)

**问题**: 初版 GRU (3类) 把走路/站着也判成击球, conf=0.96

**原因**: 训练只有 3 类 (backhand/forehand/serve), softmax 强制三选一, 无 "非动作" 选项

**方案**:
- 用 THETIS 视频的手腕速度定位击球核心帧 (peak ± 15帧)
- 核心帧保持原标签, 头尾帧标为 background (第4类)
- 训练样本: action 56% / background 44%, 均衡

**效果**: val_acc=81.4%, background recall=80%, 走路不再被判成击球

### 4. 手腕速度触发 + 密集采样 (解决算力和漏检)

**问题**: 每帧跑 GRU 太慢; 固定 stride 可能跳过击球瞬间

**方案**:
- 平时只算手腕帧间速度 (近零成本)
- 速度 > 0.15 → 活跃, 每帧跑 GRU
- 速度 < 0.15 连续 5 帧 → 活跃结束
- serve > 0.8 单独标记, 不受其他类 peak 压制

**效果**: GRU 只在活跃期间调用 (~6300次/4500帧), 覆盖所有击球瞬间

### 5. yolo11m 替代 yolo11n (解决远端 track 断裂)

**问题**: yolo11n 远端 conf=0.35-0.69 不稳定, 发球瞬间漏检导致 track 断裂

**方案**: 换 yolo11m (20M params vs 2.6M), 远端 conf 提升到 0.76+

**效果**: 80s 发球时 track 不再断裂, 97s 只有 3 帧短暂缺失 (vs 之前完全消失)

---

## 失败经验

### 1. 音频击球检测 — 搁置

**尝试**: 从视频音频轨提取击球声, librosa onset detection

**结果**:
- 纯规则: recall=97% 但 precision=47%, 假阳性太多
- MFCC + Random Forest: hit F1=0.10, 完全无效
- GBT classifier: 同样无效

**原因**: 手机麦克风高频响应差, 击球声和环境噪声在特征空间不可区分

**结论**: 需要专业麦克风或大量标注数据, 当前条件不可行

### 2. tennis_rnn.h5 (原始 GRU) — 一区 serve 偏差

**问题**: antoinekeller/tennis_shot_recognition 的预训练 GRU

**发现**: 只能检测二区 (deuce court) 的 serve, 一区 (ad court) serve 概率始终为 0

**排查**: 不是 pose 质量问题 (13-17 keypoints), 是模型训练数据偏差

**结论**: 弃用, 改用自己训练的 GRU

### 3. Roboflow YOLO 端到端模型 — 远端不可用

**尝试**: 在 Roboflow 数据集上 fine-tune yolo26s-pose, 一体化输出 bbox + class + keypoints

**结果**: 近端能分类, 远端 crop 太小检测不到

**结论**: 端到端模型不适合两阶段 pipeline, 弃用

### 4. ONNX 端到端模型 (ShadowMasterAJ) — 同上

**结果**: 远端 conf 太低 (0.2-0.47), 不可靠

### 5. 初版 GRU 盲切训练 — 分类全错

**问题**: 把 THETIS 77帧视频用 30帧窗口盲切, 所有切片打同一标签

**结果**: 模型把准备动作学成了击球, 走路判 serve conf=0.96

**原因**: 引拍和收拍阶段不是真正击球, 但被标注为击球类别

**修复**: 智能切片 + background 类 (见成功经验 #3)

### 6. MLP 单帧分类 — 精度不够

**方案**: sklearn MLP, bbox 归一化 17 keypoints, 98% test accuracy

**问题**:
- 单帧无时序信息, ready 和 serve 准备阶段混淆
- 特征顺序 bug (concatenate vs interleave) 浪费了 Kaggle GPU 时间

**结论**: 单帧分类不足以区分时序动作, 需要 GRU

### 7. 冷却期 (cooldown) — 吞掉真实事件

**问题**: 触发后 1.5 秒冷却期, 期间忽略所有信号

**结果**: serve 后紧接着的回球被冷却期挡住, 检出率从 5/7 降到 2/7

**修复**: 去掉冷却期, 改用密集采样 + peak 选取

---

## 关键数据和权重

### 权重文件 (weights/, gitignored)

| 文件 | 来源 | 用途 |
|------|------|------|
| action_mlp_bbox.pkl | 本地训练 (Roboflow 数据) | MLP 单帧分类 (弃用) |
| stroke_gru_v4_best.pt | 本地训练 (THETIS 智能切片) | **GRU 4类分类器 (在用)** |
| stroke_gru_best.pt | Kaggle 训练 (THETIS v2) | GRU 3类 (弃用) |
| stroke_gru_full_best.pt | Kaggle 训练 (THETIS v3 全类别) | GRU 3类 (弃用) |
| yolo26s-pose.pt | ultralytics 官方 | Stage 2 keypoint 提取 |
| tennis_action_best.pt | Kaggle fine-tune Roboflow | 端到端模型 (弃用) |

### 数据文件

| 文件 | 位置 | 内容 |
|------|------|------|
| keypoints_all_frames_v6.json | results/ | yolo11m 两阶段提取的全帧 keypoints |
| keypoints_all_frames.json | results/ | yolo11n 版本 (旧) |
| ball_positions.json | results/ | WASB 球轨迹 (detected + predicted) |
| thetis_keypoints.pkl | results/ | THETIS 7类 keypoint 序列 |
| thetis_keypoints_full.pkl | results/ | THETIS 12类 keypoint 序列 |
| keypoints_train/valid/test.csv | data/roboflow_keypoints/ | Roboflow 18点标注 |

### GRU v4 训练参数

```
数据: THETIS 1155 视频, 智能切片 (peak±15帧)
类别: backhand(1789) / forehand(1751) / serve(2836) / background(5099)
架构: GRU 2层 hidden=128, FC 128→64→4, dropout=0.3
输入: 30帧 × 34维 (17 keypoints × x,y 交替, bbox 归一化)
训练: Adam lr=1e-3, CrossEntropyLoss, 30 epochs
结果: val_acc=81.4%
```

### 推理关键参数

```
检测: yolo11m, conf=0.3, imgsz=1280
Pose: yolo26s-pose, conf=0.15, imgsz=640
Crop padding: 15%
手腕触发阈值: 0.15 (帧间归一化距离)
活跃结束: 连续 5 帧低于阈值
Serve 标记阈值: prob > 0.8
回合结束 timeout: 5 秒无过网
```

---

## 待解决问题

1. **80s serve 漏检** — domain gap, serve 最高 prob=0.636, THETIS 训练数据与实际视角差异
2. **14.7s 误检** — NEAR_R 被判 serve, 需加底线位置过滤
3. **GRU background 过强** — 81.4% val_acc, background 有时吞掉真实动作
4. **回合状态机未验证** — serve 检测可用后, 需接入球轨迹过网 timeout 做完整回合检测
5. **sample2.mp4 上传中** — 3.7GB 上传 Kaggle 网络不稳定
