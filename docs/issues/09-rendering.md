# 09 — 渲染

## 背景

Pass 2 把 Pass 1a/1b 算出的所有结果（轨迹、落点、球员 bbox、击球标签、回合信息）叠加到原视频上，生成可视化产物。这一层不做分析，只做绘制——所有决策在 Pass 1 完成。

## 当前实现

[tennisvision/render/](../../tennisvision/render/)

### 模块

| 文件 | 职责 |
|---|---|
| [`trail.py`](../../tennisvision/render/trail.py) | Champion track 最近 N 帧的轨迹（polyline + 渐变圆点 + 当前帧大圈高亮） |
| [`minimap.py`](../../tennisvision/render/minimap.py) | 右上角小球场，累积落点红 → 灰渐变，回合之间 reset |
| [`hud.py`](../../tennisvision/render/hud.py) | 左上角文字：`rally N  f=F/T  cands=C tracks=K champ=#ID bounces=B` |
| [`action.py`](../../tennisvision/render/action.py) | 球员彩色 bbox + 最近击球标签（2s TTL 淡出） |

### trail

- 默认 45 帧（1.5s @ 30fps）尾迹
- 当前帧位置加大圈高亮（`is_current_det=True` 表示本帧 tracker 给出了新检测，否则是 Kalman 预测）
- 颜色随帧数线性淡出
- 所有参数在 config `tracker.render_tail` 和 `render.*`

### minimap

- 宽 150 px，高按 `23.77 / 10.97` 比例 → 约 325 px
- 场地几何由 [`court/reference.py`](../../tennisvision/court/reference.py) 生成
- 新落点红色实心圆，20s 内线性变灰（`bounce_fade_frames`）
- **回合感知**：回合之间小地图清空，回合内只累计本回合的 bounces。实现靠 `frame_to_rally[frame_idx]` 决定当前帧归哪个 rally

### HUD

分两档：

- **回合启用时**：`rally N  f=F/T  cands=C tracks=K champ=#ID bounces=B`——反映"本回合"的统计
- **回合禁用时**：`f=F/T cands=C tracks=K champ=#ID bounces=B/TOTAL`——累积统计

### action 渲染

消费 Pass 1b 产出的 `stroke_events`：

1. **彩色 bbox**（`draw_players`）—— 颜色通过 track_id 做黄金比例 HSV 哈希，同一球员跨帧颜色稳定
2. **击球标签**（`draw_stroke_labels`）—— bbox 上方显示最近一次击球类型（`backhand / forehand / serve`，`neutral` 不显示），线性淡出 TTL = 2s

**实现细节**：事件按 `player_id` 分组并按 frame 排序一次（`events_by_player_sorted`），Pass 2 每帧用 `bisect` 查找最近事件，O(log n) per player per frame。

## 设计决定

- **Pass 1 / Pass 2 严格分离**：Pass 1 决策、Pass 2 只画。若未来要改 UI（不同叠加风格、视频倍速、双语字幕），不需要改 Pass 1。也便于把 Pass 2 做成 headless / 非 opencv 的替代渲染器
- **Minimap 的回合级 reset**：累积整段视频的落点会让小地图越来越乱，看不清当前回合的战术图样。以 rally 为单位 reset 才符合使用意图
- **Stroke label TTL 2s**：上游 GRU `stride=5` → 最频繁 0.17s 一次事件；TTL 2s 远大于 stride，每次挥拍的 label 稳定显示到下一次，不会闪
- **color by track_id**：双打场景需要区分 4 个人，纯色轮询容易撞色，黄金比例 HSV 哈希保证视觉可区分

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 的 `render` 段：

```python
"render": {
    "minimap_width_px": 150,
    "minimap_margin_px": 20,
    "trail_length_frames": 45,
    "bounce_fade_frames": 600,        # 20s @ 30fps
}
```

stroke label 参数在 [`pipeline/analyze.py`](../../tennisvision/pipeline/analyze.py)：`stroke_label_ttl = int(round(fps * 2.0))`。

## 已知限制 / TODO

- [ ] HUD 文字硬编码位置，在小分辨率（< 720p）上可能溢出
- [ ] Minimap 不显示球员脚的位置（有时配合战术分析有用）
- [ ] 击球标签和 bbox 重叠时读取困难，缺自适应避让
- [ ] 无 debug 渲染通路：调参时想看候选点 / 运动掩码还得回去看 debug_dir 里的图，不如直接叠在输出视频上

## 参考

无外部直接借鉴。风格上参考 TennisProject / CourtCheck 的 minimap。
