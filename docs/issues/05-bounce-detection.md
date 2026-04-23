# 05 — 落点检测

## 背景

给定完整的球轨迹 `(x, y, frame)` 序列，找出球**触地**的帧和位置。这是下游小地图、战术统计的核心信号。看起来像"y 坐标找极大值"（图像坐标 y 轴向下），但真实情况远比这复杂：

- 击球时球拍让球减速 → 速度 y 分量也过零点
- Kalman 过度平滑把真弹跳抹掉
- 遮挡后球重新出现看起来也像极大值

## 当前实现

[tennisvision/bounce/](../../tennisvision/bounce/)

### 文件

- [`catboost.py`](../../tennisvision/bounce/catboost.py) — **默认**，CatBoost 轨迹特征分类器
- [`peak.py`](../../tennisvision/bounce/peak.py) — baseline，y 方向极大值

### CatBoost 分类器（默认）

移植自 yastrebksv / CourtCheck：

1. **预处理**：对 champion 轨迹做三次样条插值补漏检，速度 > 80 px/帧 外点剔除
2. **特征**：x / y + lag(1, 2) 前后差 + 前向 / 后向差比值（速度反转 + 前后不对称）
3. **模型**：CatBoost 回归器预测每帧弹跳概率（预训练权重 `bounce_catboost.cbm`）
4. **后处理**：阈值 0.20 + 相邻帧 NMS（`nms_window=10`）

### 为什么比 y-peak 强

y-peak 只看 y 单分量，真弹跳的特征是**速度双分量联合反转 + 前后不对称**（非弹性碰撞反弹变慢）。CatBoost 学的就是这种联合签名。

tennis_raw.mp4 上的对照：

| 方法 | recall |
|---|---|
| `peak` | 5 bounces |
| `catboost` | 26 bounces |

### 落点投影

[`pipeline/analyze.py::_project_bounces`](../../tennisvision/pipeline/analyze.py) 拿 H 把像素坐标投到球场米坐标，并裁掉 `(-1m, court_width+1m)` × `(-1m, court_length+1m)` 以外的假阳（通常是远端看台上的误检）。下游 rally / minimap 用的都是这组"on-court" 点。

## 设计决定

- **轨迹特征 > 单点几何**：弹跳是短时动力学事件，需要窗口特征才能识别。CatBoost 对这类表格型特征训练快 / 精度高
- **`threshold=0.20` 偏低 + NMS**：宁可多出几个候选让下游过滤，也不漏真弹跳；NMS 窗口 10 帧 ≈ 0.3s，足以吸收单一弹跳事件周围的多次激活
- **阈值按 court meter 裁剪**：不直接看 y 坐标来判定 on/off court，投影到球场平面后按米级尺度判定，抗倾斜机位

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 的 `bounce` 段：

```python
"bounce": {
    "detector": "catboost",                   # peak | catboost
    "weights": "weights/bounce_catboost.cbm",
    "threshold": 0.20,
    "nms_window": 10,
}
```

## 对回合过滤的价值

bounce 投影后落在 z=0 平面（球场平面），是比空中轨迹更强的**物理信号**：

- bounce 在哪半场明确无歧义（用 `ry < ref.NET_Y`）
- 无法造假（发球前的空中抛球不会产生 bounce）

rally 检测的两个过滤器都直接用这个：

- `require_bounces_both_halves` —— 两侧半场各至少 1 个 bounce
- `merge_close_rallies` —— gap 内无 bounce 才合并

详见 [issue 08](08-rally-detection.md)。

## 已知限制 / TODO

- [ ] CatBoost 权重在广播视角训练，业余低位机位 recall 较好但 precision 偏低（假阳要靠下游过滤）
- [ ] 球员挥拍击球瞬间的速度反转偶尔被误判为弹跳（在空中）
- [ ] 网带顶端的球 deflection 看起来像弹跳

## 参考

- [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) — CatBoost bounce 原始实现
- [CourtCheck `backend/vision/bounce.py`](https://github.com/AggieSportsAnalytics/CourtCheck)
