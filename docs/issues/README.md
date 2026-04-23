# Feature Issues

One markdown per functional module. Matches the "功能矩阵" in the top-level [README](../../README.md).

| # | 功能 | 状态 | 文档 |
|---|---|---|---|
| 01 | 球场标定 | ✅ 已完成 | [01-court-calibration.md](01-court-calibration.md) |
| 02 | 球检测 | ✅ 已完成 | [02-ball-detection.md](02-ball-detection.md) |
| 03 | 球轨迹跟踪 | ✅ 已完成 | [03-ball-tracker.md](03-ball-tracker.md) |
| 04 | 轨迹补洞 | ✅ 可选 | [04-trajectory-inpainting.md](04-trajectory-inpainting.md) |
| 05 | 落点检测 | ✅ 已完成 | [05-bounce-detection.md](05-bounce-detection.md) |
| 06 | 球员检测 + 姿态 | ✅ 可选 | [06-player-pose.md](06-player-pose.md) |
| 07 | 击球分类 | ✅ 可选 | [07-stroke-classifier.md](07-stroke-classifier.md) |
| 08 | 回合检测 | ✅ 已完成 | [08-rally-detection.md](08-rally-detection.md) |
| 09 | 渲染 | ✅ 已完成 | [09-rendering.md](09-rendering.md) |
| 10 | 流水线编排 | ✅ 已完成 | [10-pipeline.md](10-pipeline.md) |

每份 issue 遵循同一模板：

- **背景** — 解决什么问题
- **当前实现** — 算法、关键文件、接口
- **设计决定** — 为什么这么做，权衡了什么
- **配置** — 可调参数与默认值
- **已知限制 / TODO** — 下一步
