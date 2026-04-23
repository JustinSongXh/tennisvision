# 04 — 轨迹补洞（InpaintNet）

## 背景

即使 WASB + tracker 都工作正常，一条 validated track 内部仍然会有短时空洞（3-8 帧）：球进入阴影、挡在球员身后、速度快到单帧虚化等。下游 CatBoost 落点检测做三次样条插值时，若空洞太大插值会飘，导致误检弹跳。InpaintNet 的任务是**在 bounce 检测前学习式填充这些内部空洞**。

默认**关闭**——空洞短的场景（光线好、机位高）插值 + 启发式已经够用。

## 当前实现

[tennisvision/ball/inpainter.py](../../tennisvision/ball/inpainter.py)

### 来源

移植自 [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) 的 `InpaintNet` + `generate_inpaint_mask` 模块。原始权重在羽毛球数据集训练，tennis 上经验性使用。

### 关键决定：逐 Track，不跨 Track

tracker 已经决定了哪些帧属于同一条轨迹；不在两条 track 之间桥接。跨 rally / 球出画面后再出现的合并常常是误检来源（两条独立轨迹拼成一条假轨迹）。

### window_mode

- `single`（默认）—— 一次前向覆盖整条轨迹（全卷积，无接缝）
- `nonoverlap` —— 非重叠滑窗（忠实于上游评估方式）

### 接口

```python
inp = TrajectoryInpainter(InpaintNetConfig(...), frame_size=(W, H))
res = inp.inpaint_tracks([track])            # InpaintResult: xs, ys, was_inpainted
pts = result_to_trackpoints(res)
```

### Pipeline 位置

[`pipeline/analyze.py::_maybe_inpaint`](../../tennisvision/pipeline/analyze.py)，在 `_run_bounce_detector` 之前调用。`ball.inpainter.enabled=false` 时零开销。

## 设计决定

- **Pass 1a 后、bounce 前做**：InpaintNet 需要完整 track 序列，不能流式跑；放在 bounce 之前是因为 bounce 直接受插值质量影响
- **权重从羽毛球迁移**：tennis 专有权重尚未训练，羽毛球经验性 OK（二者都是小球 + 类似速度分布）
- **默认关**：InpaintNet 加载要 ~50 MB torch 内存、推理慢；场景好的时候不必要。开启通过 config override

## 配置

```python
"ball": {
    "inpainter": {
        "enabled": False,                    # 默认关
        "weights": "weights/InpaintNet_best.pt",
        "device": "cpu",
        "window_mode": "single",             # single | nonoverlap
        "max_gap_frames": 60,                # 超此不填（跨 rally）
        "th_h_frac": 0.05,                   # 视野外 y 阈值（帧高比例）
    },
}
```

## 统计日志

启用时终端打印：

```
[inpaint] 129 orig tracks -> series len 4502; coverage 2831 -> 3124 (+293 filled frames)
```

便于 A/B 对比是否带来 recall 提升。

## 已知限制 / TODO

- [ ] 羽毛球权重在 tennis 上的定量评估（缺 ground truth）
- [ ] 只补 validated track 内部，两条 track 之间的空洞不补——某些机位下仍有可桥接的合理 gap
- [ ] `device="cuda"` 未测试（默认 CPU）

## 参考

- [qaz812345/TrackNetV3](https://github.com/qaz812345/TrackNetV3) — 原始 InpaintNet + 评估代码
