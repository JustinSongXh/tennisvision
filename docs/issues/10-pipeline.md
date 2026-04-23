# 10 — 流水线编排

## 背景

把所有模块（球场、球检测、轨迹、落点、球员 pose、击球分类、回合、渲染）组织成一个可运行的 pipeline。职责分层：每个模块独立，orchestrator 负责决定什么时候调用谁、数据怎么流、什么时候写文件。

## 当前实现

[tennisvision/pipeline/analyze.py](../../tennisvision/pipeline/analyze.py) + [tennisvision/pipeline/rally.py](../../tennisvision/pipeline/rally.py)

### 两种模式

**Offline（默认，`rally.online=false`）**——两遍流水线：

```
Pass 1a — ball detection + tracker + OnlineRallyDetector.observe
          │
          ▼ (per frame: fs.champion_trail, fs.n_cands, ...)
        frame_states[frame_idx]
          │
          ▼
        retired_tracks   (validated tracks alive or retired)
          │
          ▼
        Inpainter (optional, ball.inpainter.enabled)
          │
          ▼
        bounce detection (catboost) + project via H_inv
          │
          ▼
        rally: merge_close_rallies  (gap 短 + 无 bounce)
          │
          ▼
        rally: bounce-split filter  (两侧各 ≥ 1 bounce)
          │
          ▼
Pass 1b — pose + stroke classifier (per rally, re-open video + seek)
          │
          ▼
        backfill n_strokes + post_filter_min_strokes
          │
          ▼
Pass 2  — render (读 frame_states + bounces_court + stroke_events)
          │
          ▼
        write annotated.mp4 + rally_clip.mp4 + rallies.json
```

**Online（`rally.online=true`）**——单遍流水线：

```
Pass 1  — ball detection + tracker + pose + stroke + OnlineRallyDetector (inline)
          │
          ▼
        retired_tracks + stroke_events (实时累积)
          │
          ▼
        bounce detection (batch post-pass, 仅供小地图可视化)
          │
          ▼
        backfill n_strokes + post_filter_min_strokes
          │
          ▼
Pass 2  — render
```

Online 模式跳过 `merge_close_rallies` 和 `require_bounces_both_halves`——这两个过滤器依赖 bounce 结果，而 bounce 不可能 per-frame。代价是鲁棒性下降；收益是 per-frame 产出（stroke_events、player_bboxes、rally 确认事件都能实时消费）。

### `frame_states`

Pass 1 产出、Pass 2 消费的中间状态字典：

```python
@dataclass
class _FrameState:
    n_cands: int
    n_tracks: int
    champion_id: Optional[int]
    is_current_det: bool
    champion_trail: list              # [(x, y, frame), ...]
    player_bboxes: dict               # {track_id: bbox}
```

所有渲染决策都从这里读，Pass 2 本身不做任何推理。

### `_run_pose_for_rally`

Pass 1b 的核心函数：给定一个 rally 窗口，重开视频 seek 到 `rally.start_frame`，逐帧跑 pose + stroke 分类，返回 `(events, bboxes, n_done, wall_time)`。

每个 rally 开头调 `stroke_rec.reset()`——防 ByteTrack ID 和 GRU 滑窗跨回合污染。

### CLI 入口

[scripts/analyze.py](../../scripts/analyze.py) —— 薄 CLI：

```bash
python scripts/analyze.py \
    --video input.mp4 \
    --out   output.mp4 \
    --calib calib.json \
    --config configs/optional.yaml \
    --progress-every 100
```

仅做参数解析 + 调 `pipeline.run(...)`。所有业务逻辑在 `tennisvision/` 包内。

### 配置加载

[`tennisvision/config.py::load_config`](../../tennisvision/config.py) —— 读默认 `DEFAULTS`，若指定 YAML 则深合并。支持 `*_ratio` 键（如 `max_disp_ratio`），运行时乘以 `frame_diagonal` 得到像素值，跨分辨率一份 config 通用。

## 设计决定

- **两遍流水线**：bounce 检测需要完整 champion track，单遍做不到。Pass 1 离线收集状态，Pass 2 渲染可以看到"未来"落点的信息（例如 minimap 预览下一次落点）
- **Pass 1b 独立于 Pass 1a**：两者都是 per-frame，但 pose 只需要跑在确认的 rally 窗口（通常 < 50% 总帧数），分开做省计算。分开后的重要副作用：parallel_pose 曾经能跑（现已删除，见 [issue 08](08-rally-detection.md)）
- **Online 模式的存在**：有时需要 per-frame 输出（实时推流 / OBS 叠加），即便鲁棒性降。保留 offline 作为默认，online 作为可选
- **配置集中在 `config.py`**：YAML 只作覆盖层；阅读时默认值永远在 Python 里查（不跨文件翻译）
- **渲染层不做 seek**：Pass 2 顺序读视频，所有"过去 / 未来"状态都已在 `frame_states` 里。不 seek 意味着 Pass 2 可以流式写输出，甚至可以并发到 ffmpeg 管道

## Parallel pose（已删除）

曾经在 Pass 1a 确认 rally 当下就 `ThreadPoolExecutor.submit(pose_worker, ...)`，让 pose 和剩余 ball detection 并行。但 bounce-filter 移到 Pass 1b 之前后，被 filter 丢弃的 rally 的 pose 就白跑了，净收益为负。彻底删除，Pass 1b 改为纯顺序。详见 commit `57b34d1`。

## 配置

```python
# tennisvision/config.py 顶层
DEFAULTS = {
    "ball": {...},
    "tracker": {...},
    "court": {...},
    "bounce": {...},
    "action": {...},
    "rally": {...},
    "render": {...},
    "debug": {"dir": None},
}
```

## 已知限制 / TODO

- [ ] CPU 5 fps 性能基准（含 action 全路线）——目前 sample_short.mp4 ~6 fps
- [ ] Pass 2 的 `cv2.VideoWriter` 用 mp4v codec，质量 / 压缩率一般；应支持 ffmpeg pipe
- [ ] Online 模式下无 merge / bounce-split 降鲁棒；想要介乎两者之间的"semi-online"（延迟 2-3s 换准确）可能是下一步
- [ ] YAML config 的 schema 不做校验，拼错键名 silently 被忽略
- [ ] 没有原生多视频批处理入口（需要外部 for 循环）

## 参考

- [CourtCheck](https://github.com/AggieSportsAnalytics/CourtCheck) — 两遍流水线思路
