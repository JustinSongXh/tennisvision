# 01 — 球场标定

## 背景

给定一段固定机位的网球比赛视频，需要求出**图像平面 → 球场平面（米）**的单应矩阵 H。所有后续模块（落点坐标、小地图投影、球员 on-court 过滤）都依赖 H。固定机位意味着标定只需一次，可以缓存复用。

## 当前实现

[tennisvision/court/](../../tennisvision/court/)

### 文件

- [`reference.py`](../../tennisvision/court/reference.py) — ITF 标准球场 14 关键点真实世界坐标（米），以及多组 4 点子集 `court_conf`
- [`detector.py`](../../tennisvision/court/detector.py) — 关键点检测（多实现）
- [`homography.py`](../../tennisvision/court/homography.py) — 从关键点估计 H（best-of-subsets）
- [`calibration.py`](../../tennisvision/court/calibration.py) — `calib.json` 序列化与加载

### 关键点检测（四种实现）

| 实现 | 算法 | 权重 | 适用 |
|---|---|---|---|
| `ResNet50CourtDetector` | abdullahtarek 风格：ResNet50 + Linear(28) 坐标回归 | `court_resnet.pth` | **推荐**，业余蓝色硬地 |
| `HeatmapDetector` | yastrebksv/TennisCourtDetector（15 通道 heatmap，640×360） | `court_tcd.pt` | 广播视角 |
| `RedMarkExtractor` | 从用户手画红线图提取关键点（HSV 红 + Hough） | 无 | 手动 fallback |
| `BlueContourDetector` | 纯蓝色轮廓拟合梯形 | 无 | 无 ML 环境兜底 |

### blue-resnet 流水线（默认推荐）

1. `BlueContourDetector` 构建蓝色区域 mask
2. `_enclosing_trapezoid` 拟合外接梯形
3. `cv2.warpPerspective` 矫正为正视图
4. `ResNet50CourtDetector` 多帧回归 + median 聚合
5. 逆透视投影回原图坐标

这样 ResNet50 总是看到"标准视角"的场地，对低位 / 偏左 / 业余机位鲁棒。

### Homography 估计 —— best-of-subsets

直接用 4 点 DLT 会被任何一个错误检测点毁掉。借鉴 CourtCheck 的做法：

1. `reference.court_conf` 预列出 ~20 组有效的 4 点 ID 组合
2. 每组：如有缺失点跳过，否则 `cv2.findHomography` + DLT 求候选 H
3. **未参与**的其他 ~10 个检测点用该 H 反投影到球场平面，与 reference 真值算 MSE
4. 选 MSE 最小的 H

效果等同 RANSAC：球员遮挡一半关键点也能投票出正确 H。

### 缓存

`calib.json` 保存 H / 14 点像素坐标 / 时间戳 / camera_id。加载时校验 video hash；同机位视频直接复用。

## 设计决定

- **双阶段 blue-resnet 优于单阶段**：单阶段 ResNet50 直接回归原图 14 点，在低位 / 偏左机位上精度差。先蓝色 mask → 透视矫正再回归，把"归一化视角"的问题交给传统 CV，把"标准视角下找 14 点"留给 CNN，各司其职。
- **best-of-subsets 用 MSE 投票而非 RANSAC**：关键点数少（14 个）、遮挡模式结构化（通常是球员挡住底线端），穷举子集比随机采样更稳定。
- **默认用 blue-resnet 而非 TCD**：TCD 权重在广播视角训练，在业余低位机位上 recall 仅 0-3 keypoints / 帧，无法使用。

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 里的 `court` 段：

```python
"court": {
    "detector": "blue-resnet",          # blue-resnet | tcd | mark | blue
    "weights": "weights/court_resnet.pth",
    "input_size": [640, 360],
    "cache_calib": True,
}
```

Reprojection error（样例）：

| 方法 | err |
|---|---|
| `blue-resnet` | ~0.8 m |
| `mark` | 取决于标注精度 |
| `tcd` | ~1-2 m（低位失效） |
| `blue` | ~2-4 m |

## 已知限制 / TODO

- [ ] TCD 模型 fine-tune 到低位机位（或改用其他权重）
- [ ] 多机位自动切换（目前假设单机位）
- [ ] 标定结果的置信度估计（目前只输出 H，没有 uncertainty）

## 参考

- [CourtCheck `backend/vision/court_reference.py`](https://github.com/AggieSportsAnalytics/CourtCheck)
- [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
- [docs/court_detection_survey.md](../court_detection_survey.md) — 完整调研笔记
