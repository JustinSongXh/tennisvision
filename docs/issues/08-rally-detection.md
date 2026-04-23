# 08 — 回合检测

## 背景

一段业余比赛视频常包含：发球练习、捡球、教练指导、换边、真正的比分回合混杂在一起。分析工具的核心价值之一是**把真 rally 切出来**——每个回合的起止帧、击球次数、过网次数，以及剪辑成 highlight 视频。

过滤"不是回合的活动"是这里最难的部分：球员原地拍球、热身对抗、跨场球滚过来，全都会让球检测 / tracker 看起来像真打球。

## 当前实现

[tennisvision/pipeline/rally.py](../../tennisvision/pipeline/rally.py)

### 数据结构

```python
@dataclass
class Rally:
    idx: int
    start_frame: int
    end_frame: int
    n_events: int        # 段内有球帧数
    net_crossings: int   # 段内过网次数
    n_strokes: int       # 段内非 neutral 击球数（Pass 1b 后回填）
```

### 核心：`OnlineRallyDetector` 状态机

每帧喂一次 `observe(frame_idx, cand_xys, champion)`，维护：

| 字段 | 作用 |
|---|---|
| `_activity_start` | 当前活动段起点（首次见球） |
| `_crossings` | 段内累计过网次数 |
| `_last_side` | 上一帧球在哪半场（`y < net_y_px` ? -1 : 1） |
| `_last_crossing_frame` | 最近一次过网的帧 |
| `_silent` | 连续无球帧数 |
| `_activity_frames` | 段内有球的帧数 |

**打开段**：首次 `cand_xys` 非空 → 记 `_activity_start`。

**过网计数**：每有球帧取球 y（champion 优先，次选首候选），与 `_last_side` 比较，翻转即 `_crossings += 1`。

**两个关段条件**（任一触发即调 `_close_at`）：

1. **静默关段**：`_silent >= silence_thresh`（默认 3s）→ 球彻底不见
2. **过网静默关段**：`frame_idx - _last_crossing_frame >= crossing_silence_thresh`（默认 4s）→ 球还在但长时间不过网（捡球 / 拍球 / 发球前原地颠球）

**确认条件**（在 `_close_at`）：

- `_crossings >= min_net_crossings`（默认 3）—— 真 rally 至少来回 3 次
- `_activity_frames / span >= min_activity_density`（默认 0.30）—— 段内至少 30% 帧有球检出
- 不与上一 rally 尾部重叠

通过即 `Rally` 追加到 `self.rallies`。套 `pre_roll` / `post_roll`（默认 1s / 1s）。

### 后置过滤器（offline 模式）

Pass 1a 结束 + bounce 检测完成后依次跑：

**1. `merge_close_rallies`**（gap 短 + gap 内无 bounce → 合并）

 真实比赛场景里，长 rally 偶有 3-5s 的球跟丢（高吊出画、遮挡）。OnlineRallyDetector 的 silence 阈值会把这种跟丢误判为两个独立 rally。但真正的 between-point gap 一定有捡球 / 拍球 bounce；mid-rally 跟丢时球在空中，没有 bounce。以此区分：

```python
if gap <= max_gap_frames and no_bounce_in_gap:
    merge(prev_rally, this_rally)
```

**2. `bounce-split filter`**（两侧各 ≥ 1 个 bounce）

 真 rally 球必然在两侧半场各有落地（否则说明球没过网）。单侧活动（发球前拍球、热身一人颠球）被直接丢。bounce 基于 z=0 单应投影，两半场判定可靠（用 `ry < ref.NET_Y`）。

**3. `post_filter_min_strokes`**（Pass 1b 后）

 Pass 1b 跑完 pose + stroke，回填 `r.n_strokes`。非 neutral 击球 < `post_filter_min_strokes`（默认 2）丢弃——真 rally 至少 2 次挥拍，warm-up / pickup 几乎没有。

### 被删除的过滤器：`drop_same_side_dribble`

历史上曾尝试"连续 > N 同侧 bounce 视为 dribble"的过滤，但 bounce 检测器本身有假阳（场边围栏、地面滚动、阴影），会在真 rally 内部也触发 3+ 同侧的假象，误杀合法回合。该过滤已彻底删除；如未来有专门的"捡球视频"测试集可考虑重新加入。见 commit `57b34d1`。

### Offline 流水线位置

```
Pass 1a (ball + tracker + online_det.observe)
  → bounce detection
  → merge_close_rallies (gap 短 + 无 bounce)
  → bounce-split filter (两侧各有 bounce)
  → Pass 1b (pose + stroke, per rally)
  → backfill n_strokes
  → post_filter_min_strokes
  → Pass 2 render
```

### Online 模式（`rally.online=true`）

将 ball + pose + stroke 揉进同一帧循环；`OnlineRallyDetector.observe()` 一经确认 rally 即追加。Bounce 检测仍是 batch 后处理但**不**参与过滤（merge / bounce-split 都跳过）。适合实时推流。rally 终版以 OnlineRallyDetector 为准，**无 merge、无 bounce-split**——代价是对 mid-rally 跟丢和单侧 warm-up 鲁棒性下降。

### 时序特性

rally 被确认的**帧时刻**不是 `end_frame`，而是 `_close_at` 触发那一帧。默认 silence_thresh=3s + post_roll=1s → 实际确认延迟 ≈ **2s**。对实时 HUD 来说，"比赛实际结束"到"rally 框落定"会有 2s 延迟，这是必要的——否则球员短暂挡住球都会当成回合结束。

### 剪辑输出

- `select_clip_rallies` —— 对 highlight 剪辑再严一档（默认 `clip_min_net_crossings=3`, `clip_min_duration_seconds=2.0`）
- `write_rally_video` —— 从**已渲染**的输出视频切段，回合间插"Rally N"黑底标题
- `save_rally_json` —— 写完整 rally 列表

## 设计决定

- **过网计数用中点 `net_y_px` 粗略近似**：瞬时过网判断不可靠（球高弧线时中点在网后很多帧），但**长期过网次数**可靠——真 rally 至少 3 次越过中线
- **bounce 作为物理验证，不作为触发**：bounce 噪声大（假阳多），不能"见 bounce 就开启 rally"。但它是**z=0 平面的硬信号**，用来做事后过滤和合并时很有价值
- **merge 先于 bounce-split**：合并后的 rally 才去判断两侧 bounce；单独判断会把一个真 rally 的两段各自算作"某侧 bounces 不足"而全部丢弃
- **online 模式砍过滤而非改逻辑**：bounce 是全局信号，硬要做流式会增加复杂度；online 模式下接受鲁棒性下降

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 的 `rally` 段（核心）：

```python
"rally": {
    "enabled": True,
    "online": False,                                # true = 单遍流水线
    "online_silence_seconds": 3.0,                  # 球不见 N 秒关段
    "online_crossing_silence_seconds": 4.0,         # 球在但不过网 N 秒关段
    "online_min_net_crossings": 3,                  # 确认所需过网次数
    "online_min_activity_density": 0.30,            # 段内有球帧占比下限
    "pre_roll_seconds": 1.0,
    "post_roll_seconds": 1.0,
    "require_bounces_both_halves": True,
    "merge_gap_seconds": 5.0,                       # 短 gap + 无 bounce → 合并
    "post_filter_min_strokes": 2,
    # 剪辑视频的更严阈值
    "clip_min_net_crossings": 3,
    "clip_min_duration_seconds": 2.0,
}
```

所有时间都是秒，运行时乘 `fps`。

## 验证结果（sample_short.mp4，~150s，4502 帧）

| 阶段 | rally 数 |
|---|---|
| Pass 1a 原始 | 7 |
| merge_close_rallies 合并 2 对 | 5 |
| bounce-split 丢 1 段（单侧 bounce） | 4 |

和人工标注的"约 4 个真回合"一致。

## 已知限制 / TODO

- [ ] silence_thresh=3s 仍然偶尔把长高吊误切（merge 救回大部分但不全）
- [ ] online 模式下无 merge / bounce-split，单侧 warm-up 会进结果。需要专门的"流式 bounce-lite" 才能补足
- [ ] 剪辑视频之间的黑底"Rally N"标题时长固定 1s，可能过长 / 过短
- [ ] `write_rally_video` 逐帧 seek + decode，~30 fps 写入，能再提速（batch 写、硬件编码）

## 参考

无直接外部借鉴，全部为自研逻辑。最接近的已有工作是体育比赛的 **highlight detection**（广播视角）；本实现偏向业余单机位，用几何信号（过网 + bounce）而非场景 / 观众反应。
