# 02 — 球检测

## 背景

每帧在画面里找出球的候选位置 `(x, y)`。网球直径 6.7 cm，在 1080p 30m 场地视频里通常只有 5-15 像素，且高速移动时会拖影或被背景吞掉。这是整个 pipeline 最早失败的一环——球丢了，后面 tracker / bounce / rally 全部受影响。

## 当前实现

[tennisvision/ball/](../../tennisvision/ball/)

### 文件

- [`wasb.py`](../../tennisvision/ball/wasb.py) — **默认**，WASB (HRNet) 深度学习检测
- [`hrnet.py`](../../tennisvision/ball/hrnet.py) — HRNet 主干网络（复用 nttcom/WASB-SBDT + MS HRNet-Image-Classification，MIT）
- [`classical.py`](../../tennisvision/ball/classical.py) — HSV + MOG2 经典 CV（fallback / demo）

### WASB（默认）

- **架构**：HRNet 变体（1.48 M 参数，~5.8 MB 权重），9 通道输入（连续 3 帧 RGB 堆叠），288 × 512 输入，输出 3 张热图对应 3 帧
- **许可**：MIT（WASB 官方 model zoo 的 tennis 权重，BMVC 2023）
- **预处理**：letterbox 仿射变换（等比缩放 + 黑边填充，不失真）+ ImageNet mean / std 归一化
- **后处理**：sigmoid → 阈值 0.5 → connected components → 得分加权重心 → 逆仿射回原图坐标
- **时序门控**：新检测必须距上一帧 `<= max_disp` 像素（默认 300 px，实际按 frame diagonal 自适应），拒绝抖动假阳
- **CPU 推理**：~100-200 ms / 帧；首次运行把 `.pth.tar` 自动导出 ONNX，后续用 onnxruntime + CoreML EP（Apple Silicon）或 CPU EP
- **双阶段**：`TwoStageBallDetector`（`wasb.py:320`）—— 主尺度一次 + far-court crop 一次，拼接去重；专门捡回远端半场的小球

### Classical（备选）

- **算法**：MOG2 背景减除 → 运动掩码；HSV 黄绿过滤 AND 运动掩码；形态学开 + 膨胀；轮廓面积 + 圆度过滤
- **限制**：顶点慢速球易漏检；大面积风 / 树叶运动被误认为球员
- **保留原因**：零依赖、CPU 几乎 0 成本，作 fallback 和 demo

## 设计决定

- **选 WASB 而非 TrackNet**：同基准上 WASB 更准 + 参数量更小（1.48 M vs TrackNet ~11 M），MIT 协议（TrackNet 非商用）
- **首次运行自动 ONNX 导出**：PyTorch 推理 / ONNX 推理在 CoreML EP 上差 3-4×。一次性导出开销 ~10s，之后全程省
- **Two-stage 而非单尺度**：WASB 固定 288×512，远端球员 / 球会被降采样到亚像素，几乎不可能检出。先主尺度过一遍，再把远端半场 crop 出来按原分辨率重跑一次，拼接去重（`two_stage_dedup_px` 默认 10 px）。几乎 2× 耗时但 recall 显著提升
- **max_disp 按 frame diagonal 自适应**：固定 300 px 在 720p 太松、4K 太紧；改成 `max_disp_ratio * frame_diagonal`，同一份 config 跑 720p / 1080p / 4K 都对

## 配置

[`tennisvision/config.py`](../../tennisvision/config.py) 的 `ball` 段：

```python
"ball": {
    "detector": "wasb",                 # wasb | classical
    "weights": "weights/wasb_tennis_best.pth.tar",
    "max_disp_ratio": 0.136,            # * frame diag → ~300 px @ 1080p
    "two_stage_enabled": True,
    "two_stage_dedup_ratio": 0.0045,    # * frame diag → ~10 px @ 1080p
    "inpainter": {
        "enabled": False,               # 见 issue 04
    },
}
```

## 接口

```python
det = WASBBallDetector(WASBConfig(weights="weights/wasb_tennis_best.pth.tar"))
for frame in video:
    det.push_frame(frame)
    xy = det.detect()            # (x, y) or None (first 2 frames empty buffer)
```

`TwoStageBallDetector` 实现同样的 `push_frame` + `detect` 接口，pipeline 无感切换。

## 已知限制 / TODO

- [ ] ONNX 导出后 CoreML EP 在部分 M1 Mac 上卡顿，需 fallback CPU
- [ ] 球与网打到一起的瞬间经常丢球（3 帧时序窗口内全被网主导）
- [ ] 光线剧变（阴影切换）下 detector 会有一段恢复期

## 参考

- [nttcom/WASB-SBDT](https://github.com/nttcom/WASB-SBDT) — 源代码与 tennis 权重
- [Microsoft HRNet-Image-Classification](https://github.com/HRNet/HRNet-Image-Classification) — HRNet 主干
