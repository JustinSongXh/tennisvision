# 07 — 击球分类

## 背景

给定球员的连续姿态（17 关键点 × 30 帧），判断其动作类型。输出 4 类：`backhand`、`forehand`、`neutral`（非击球 / 等待）、`serve`。用于战术统计（谁打了多少正手反手）和下游的击球 ↔ 落点配对（V3 远期）。

默认**关闭**（需 `action.enabled=true`）。

## 当前实现

[tennisvision/action/stroke_classifier.py](../../tennisvision/action/stroke_classifier.py)

### 算法

移植自 [antoinekeller/tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition)：

1. 每帧取 pose 的 **13 个 COCO 关键点**（去掉眼 / 耳）→ 26 维 `(y, x)` 特征
2. **关键点相对 bbox 归一化**（而非相对整帧）—— 否则特征会编码"球员在场地哪里"，分类崩掉
3. 30 帧滑窗（≈ 1s @ 30fps）→ Keras GRU → softmax 4 类
4. 置信度阈值 `min_confidence=0.9` + `stride=5` 避免重复触发 + 可选抑制 `neutral`

### 多球员并行

入口 `MultiPlayerStrokeRecognizer` —— 以 track ID 为 key，每个 ID 一条独立滑窗；单打 / 双打同一份代码。

### Track TTL

track 短暂消失（< 30 帧，例如被网挡）不重置滑窗，避免瞬时漏检打断识别。

### Keras 兼容

`tennis_rnn.h5` 是 Keras 2 保存的：

- 优先 `import tf_keras`（TF 2 上 Keras 2 的兼容包）
- 不可用 fallback 到 Keras 3（部分层需要手动适配）

### 输出

```python
@dataclass
class StrokeEvent:
    frame: int
    label: str              # backhand | forehand | neutral | serve
    confidence: float
    cx: float               # bbox center x
    cy: float
    player_id: int          # ByteTrack ID from pose module
```

## 设计决定

- **相对 bbox 归一化而非整帧**：原论文（antoinekeller）的关键选择。若用整帧坐标，特征里混了"球员在场地哪里"的位置信息——同一个人在远端半场和近端半场做同样的正手会被学成不同类。相对 bbox 就只剩动作本身
- **GRU 而非 Transformer**：30 帧序列、4 类分类、~10K 训练样本；轻量 GRU 足够
- **30 帧滑窗 + stride=5**：1s 窗口覆盖完整正反手动作；stride 5 相当于每 0.17s 推一次，避免一次挥拍触发多个事件
- **置信度门限 0.9 高**：宁可漏（confidence 低不发事件）也不错（假阳会污染战术统计）。实测真击球的 confidence 通常 > 0.95

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 的 `action` 段（分类部分）：

```python
"action": {
    "enabled": False,
    "rnn_weights": "weights/tennis_rnn.h5",
    "window_frames": 30,
    "min_confidence": 0.9,
    "stride": 5,
    "emit_neutral": False,
}
```

## 与 rally 的关系

Pass 1b 跑完 pose + stroke，`stroke_events` 列表累积；之后 [`rally.post_filter_min_strokes`](../../tennisvision/pipeline/rally.py) 根据每个 rally 窗口内的 stroke 数过滤：真 rally 总有 ≥ 2 个非 neutral 击球，warm-up / pickup 通常没有。

详见 [issue 08](08-rally-detection.md)。

## 依赖

- `tensorflow` + `tf-keras`（加载 Keras 2 的 `.h5`）
- 懒加载

## 已知限制 / TODO

- [ ] 切削 / 截击 / 高压 等细分动作都归到 forehand / backhand 里（4 类粒度偏粗）
- [ ] 同场对手如果同时挥拍，bbox 重叠时 pose 会串，导致相邻两个事件打错球员
- [ ] 权重训练集是广播视角 / 单打为主，双打 / 低位机位的召回略低（未做定量评估）

## 参考

- [antoinekeller/tennis_shot_recognition](https://github.com/antoinekeller/tennis_shot_recognition) — 原始权重 + 算法
