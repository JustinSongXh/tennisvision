# TennisVision — Workflow

## 环境准备

### 本地环境
```bash
cd /Users/justinsong/WorkSpace/solo/tennisvision
source /Users/justinsong/WorkSpace/solo/venv/bin/activate
```

### Kaggle 环境
- 账号: pinesquirrel
- API token: `~/.kaggle/kaggle.json`
- GPU: T4 x2
- CLI: `/Users/justinsong/WorkSpace/solo/venv/bin/kaggle`

### Kaggle 资源

| 类型 | 名称 | 内容 | 版本 |
|------|------|------|------|
| Dataset | pinesquirrel/tennisvision-test | sample_short.mp4 | - |
| Dataset | pinesquirrel/tennisvision-sample2 | sample2.mp4 (crf28 压缩) | - |
| Model | pinesquirrel/tennisvision-weights | court_resnet + WASB + GRU 权重 | v2 |

### Kaggle Notebooks

| 名称 | 用途 |
|------|------|
| sample2-pipeline | 跑 sample2 全流程（主力） |
| sample2-test | 调试用 |

---

## Pipeline 架构

`run_full_pipeline.py` 是纯 subprocess 编排器，每个 step 调独立脚本：

| Step | 脚本 | 输出 |
|------|------|------|
| 1 | `calibrate.py blue-resnet` | calib.json + court_overlay.jpg |
| 2 | `extract_keypoints.py` | keypoints.json |
| 3 | `extract_ball_positions.py` | ball_positions.json |
| 4 | `detect_serves.py` | serve_events.json |
| 5 | `detect_rallies.py` | rally_events.json + rally_cuts.mp4 |

---

## 完整测试流程

### 方式一：Kaggle 上跑全流程（推荐，有 GPU）

通过 `sample2-pipeline` notebook 自动执行：
1. `git clone -b issue-08` 拉代码
2. `os.walk` 搜索权重和视频路径（Kaggle 挂载路径不固定）
3. `run_full_pipeline.py --weights-dir <found_path>` 跑全流程
4. 结果拷贝到 `/kaggle/working/sample2/`

**手动在 notebook 里执行：**

```python
import subprocess, sys, os

subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "-q"])
subprocess.check_call(["git", "clone", "-b", "issue-08",
    "https://github.com/JustinSongXh/tennisvision.git"])

# 搜索权重和视频
weights_dir = video_path = None
for root, dirs, files in os.walk("/kaggle/input/"):
    if "wasb_tennis_best.pth.tar" in files:
        weights_dir = root
    if "sample2.mp4" in files:
        video_path = os.path.join(root, "sample2.mp4")

subprocess.check_call([
    sys.executable, "-u", "scripts/run_full_pipeline.py",
    "--video", video_path,
    "--weights-dir", weights_dir,
], cwd="tennisvision")
```

**结果在 `results/sample2/` 下：**
- `calib.json` — 场地标定
- `court_overlay.jpg` — 场地可视化（验证用）
- `keypoints.json` — 所有帧 keypoints
- `ball_positions.json` — 球轨迹
- `serve_events.json` — 发球事件
- `rally_events.json` — 回合边界
- `rally_cuts.mp4` — 回合剪辑视频

### 方式二：本地跑（CPU，慢但不依赖 Kaggle）

```bash
python scripts/run_full_pipeline.py --video samples/sample_short.mp4
```

注意：Step 2 (keypoints) 和 Step 3 (ball) 在 CPU 上很慢。

### 方式三：混合（Kaggle 提取 + 本地分析）

```bash
# 1. Kaggle 上只跑 Step 1-3（耗时步骤）
python scripts/run_full_pipeline.py --video VIDEO --step 1
python scripts/run_full_pipeline.py --video VIDEO --step 2
python scripts/run_full_pipeline.py --video VIDEO --step 3

# 2. 下载 results/<video_name>/ 到本地

# 3. 本地跑 Step 4-5（秒完，不需要 GPU）
python scripts/run_full_pipeline.py --video VIDEO --step 4
python scripts/run_full_pipeline.py --video VIDEO --step 5
```

---

## 单步命令参考

```bash
# Step 1: 场地标定
python scripts/calibrate.py blue-resnet --video VIDEO --out OUT --vis VIS --weights weights/court_resnet.pth

# Step 2: Keypoints 提取
python -u scripts/extract_keypoints.py --video VIDEO --out OUT [--max-frames N] [--no-half]

# Step 3: 球轨迹
python -u scripts/extract_ball_positions.py --video VIDEO --out OUT --weights weights/wasb_tennis_best.pth.tar [--max-frames N]

# Step 4: Serve 检测
python -u scripts/detect_serves.py --keypoints KP --calib CALIB --gru weights/stroke_gru_v4_best.pt --out OUT

# Step 5: Rally 检测
python -u scripts/detect_rallies.py --video VIDEO --results-dir RESULTS_DIR

# 全跑（编排器）
python scripts/run_full_pipeline.py --video VIDEO [--weights-dir DIR] [--max-frames N] [--no-half] [--step N]
```

---

## Kaggle Kernel 管理

### 查看状态
```bash
kaggle kernels status pinesquirrel/<kernel-name>
```

### 下载结果
```bash
kaggle kernels output pinesquirrel/<kernel-name> -p /tmp/output
```

### 提交 kernel（Save & Run All 模式）
```bash
kaggle kernels push -p /path/to/kernel_dir
```

kernel 目录需要：
- 脚本文件（如 `sample2-pipeline.py`）
- `kernel-metadata.json`

### 上传模型权重新版本
```bash
kaggle models instances versions create \
    pinesquirrel/tennisvision-weights/pytorch/pytorch \
    -p /tmp/weights-dir \
    -n "version notes"
```

### 上传数据集
```bash
mkdir -p /tmp/upload
cp VIDEO /tmp/upload/
cat > /tmp/upload/dataset-metadata.json << EOF
{"title":"name","id":"pinesquirrel/name","licenses":[{"name":"CC0-1.0"}]}
EOF
kaggle datasets create -p /tmp/upload
```

---

## 输出目录结构

```
results/
  sample_short/           ← 每个视频一个子目录
    calib.json
    court_overlay.jpg
    keypoints.json
    ball_positions.json
    serve_events.json
    rally_events.json
    rally_cuts.mp4
  sample2/
    ...
```

---

## 注意事项

1. **Kaggle GPU 限额**: 30 小时/周，每周六 UTC 0:00 重置
2. **Kaggle 磁盘**: /kaggle/working 20GB, /kaggle/tmp 60GB
3. **大文件上传**: 超过 500MB 建议压缩后上传（`ffmpeg -crf 28`）
4. **FP16**: GPU 环境默认开启，`--no-half` 关闭。ultralytics 用 `quantize="fp16"` 而非已废弃的 `half=`
5. **已有结果自动跳过**: 重跑时检测到已有文件会跳过，删除文件后重新生成
6. **Kaggle 路径不固定**: 用 `os.walk("/kaggle/input/")` 搜索文件，不要硬编码挂载路径
7. **model_sources 格式**: kernel-metadata.json 中需完整路径含版本号，如 `pinesquirrel/tennisvision-weights/PyTorch/pytorch/2`
