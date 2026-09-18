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

### Kaggle 资源

| 类型 | 名称 | 内容 |
|------|------|------|
| Dataset | pinesquirrel/tennisvision-test | sample_short.mp4 |
| Dataset | pinesquirrel/tennisvision-sample2 | sample2.mp4 (crf28 压缩) |
| Model | pinesquirrel/tennisvision-weights | WASB + GRU 权重 |

---

## 完整测试流程

### 方式一：Kaggle 上跑全流程（推荐，有 GPU）

**1. 在 Kaggle notebook 里执行：**

```python
# Cell 1: 环境准备
!git clone -b issue-08 https://github.com/JustinSongXh/tennisvision.git
!pip install ultralytics -q

# 复制权重到 repo
!cp /kaggle/input/tennisvision-weights/pytorch/pytorch/* tennisvision/weights/

# Cell 2: 跑全流程
!cd tennisvision && python scripts/run_full_pipeline.py \
    --video /kaggle/input/tennisvision-sample2/sample2.mp4

# Cell 3: 跑测试版（只跑前 1 分钟验证）
!cd tennisvision && python scripts/run_full_pipeline.py \
    --video /kaggle/input/tennisvision-sample2/sample2.mp4 \
    --max-frames 1800
```

**2. 结果在 `tennisvision/results/sample2/` 下：**
- `calib.json` — 场地标定
- `court_overlay.jpg` — 场地可视化（验证用）
- `keypoints.json` — 所有帧 keypoints
- `ball_positions.json` — 球轨迹
- `serve_events.json` — 发球事件
- `rally_events.json` — 回合边界
- `rally_cuts.mp4` — 回合剪辑视频

**3. 下载结果到本地：**

从 Kaggle notebook 的 Output 标签下载，或用 Save & Run All 后 API 下载：
```bash
kaggle kernels output <username>/<kernel-name> -p /tmp/output
```

### 方式二：本地跑（CPU，慢但不依赖 Kaggle）

```bash
python scripts/run_full_pipeline.py --video samples/sample_short.mp4
```

注意：Step 2 (keypoints) 和 Step 3 (ball) 在 CPU 上很慢。

### 方式三：混合（Kaggle 提取 + 本地分析）

```bash
# 1. Kaggle 上只跑 Step 1-3（耗时步骤）
!cd tennisvision && python scripts/run_full_pipeline.py \
    --video /path/to/video.mp4 --step 1
!cd tennisvision && python scripts/run_full_pipeline.py \
    --video /path/to/video.mp4 --step 2
!cd tennisvision && python scripts/run_full_pipeline.py \
    --video /path/to/video.mp4 --step 3

# 2. 下载 results/<video_name>/ 到本地

# 3. 本地跑 Step 4-5（秒完，不需要 GPU）
python scripts/run_full_pipeline.py --video samples/video.mp4 --step 4
python scripts/run_full_pipeline.py --video samples/video.mp4 --step 5
```

---

## 单步命令参考

```bash
# Step 1: 场地标定
python scripts/run_full_pipeline.py --video VIDEO --step 1

# Step 2: Keypoints 提取
python scripts/run_full_pipeline.py --video VIDEO --step 2

# Step 3: 球轨迹
python scripts/run_full_pipeline.py --video VIDEO --step 3

# Step 4: Serve 检测
python scripts/run_full_pipeline.py --video VIDEO --step 4

# Step 5: Rally 检测
python scripts/run_full_pipeline.py --video VIDEO --step 5

# 全跑
python scripts/run_full_pipeline.py --video VIDEO

# 限制帧数（测试用）
python scripts/run_full_pipeline.py --video VIDEO --max-frames 1800

# 关闭 FP16
python scripts/run_full_pipeline.py --video VIDEO --no-half

# 使用已有 keypoints
python scripts/run_full_pipeline.py --video VIDEO --keypoints path/to/keypoints.json
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
- `pipeline.py`（或其他脚本）
- `kernel-metadata.json`

### 上传数据集
```bash
mkdir -p /tmp/upload
cp VIDEO /tmp/upload/
cat > /tmp/upload/dataset-metadata.json << EOF
{"title":"name","id":"pinesquirrel/name","licenses":[{"name":"CC0-1.0"}]}
EOF
kaggle datasets create -p /tmp/upload
```

### 上传模型权重
```bash
kaggle models init -p /tmp/model
# 编辑 model-metadata.json
kaggle models create -p /tmp/model
kaggle models instances init -p /tmp/instance
# 编辑 model-instance-metadata.json，放入权重文件
kaggle models instances create -p /tmp/instance
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
4. **FP16**: GPU 环境默认开启，`--no-half` 关闭
5. **已有结果自动跳过**: 重跑时检测到已有文件会跳过，删除文件后重新生成
