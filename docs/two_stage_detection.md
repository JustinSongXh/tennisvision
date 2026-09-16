# Two-Stage Player Detection + Pose Estimation

Issue: #06

## Problem

YOLO26-pose single-pass detection completely misses far-side players (~80-100px tall)
in fixed-camera amateur footage. Even at conf=0.01, imgsz=2560, zero far-side detections.

Root cause: pose models need sufficient keypoint signal to produce candidates. When the
person is small on the feature map after resize, the keypoint head drags down overall
confidence below any threshold.

Standard detection models (yolo11n/m) detect the same far-side players at conf=0.7+.

## Verified Solution

Two-stage pipeline:

1. **yolo11m** full-frame detection (classes=[0], conf=0.3, imgsz=1280)
   - Detects all persons including far-side (~80px)
   - Also produces false positives (adjacent courts, spectators)

2. **yolo26m-pose** on each bbox crop (padded 15%, imgsz=640)
   - Extracts 17 COCO keypoints per person
   - Far-side players: 14-17/17 visible keypoints, conf 0.79-0.90
   - Near-side players: 12-13/17 visible keypoints, conf 0.94+

3. **On-court filter** (existing homography calib) to reject off-court detections

## Test Results (frame 300, sample_short.mp4, doubles)

| Player | Det conf | Pose conf | Visible KP |
|--------|----------|-----------|------------|
| Near-L | 0.931 | 0.947 | 13/17 |
| Near-R | 0.897 | 0.942 | 12/17 |
| Far-L  | 0.765 | 0.901 | 17/17 |
| Far-R  | 0.510 | 0.791 | 14/17 |

yolo11m conf=0.3 produced 9 detections total; 5 off-court rejected by pose failure
(no person detected in crop). On-court filter will further clean these up.

## TODO

- [ ] Integrate into `tennisvision/action/pose.py` — replace single YOLO-pose pass
      with two-stage: yolo11 detect + yolo26-pose crop
- [ ] Add yolo11m.pt to weights/ (auto-download or manual)
- [ ] Wire on-court filtering after stage 1 to reduce stage 2 crops
- [ ] Benchmark speed: 2-stage vs 1-stage on full video
- [ ] Update Kaggle test script
- [ ] Consider: use yolo11n (lighter) instead of yolo11m for stage 1

## Related Research

- Audio hit detection for rally segmentation (complements visual detection)
- Multi-signal fusion (audio + ball trajectory + pose) for robust rally detection
- See full research notes in conversation history
