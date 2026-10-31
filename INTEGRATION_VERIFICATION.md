# Integration verification — 31 October 2026

## Executed in this build environment

| Check | Result |
|---|---|
| AeroDetect original baseline | 30 passed |
| AeroFusion original baseline | 19 passed |
| Updated AeroDetect | 32 passed, original 30 retained |
| Updated AeroFusion | 32 passed, original 19 retained |
| Isolated export environment `pip check` | No broken requirements found |
| Isolated pinned runtime `pip check` | No broken requirements found |
| YOLOv8n export and ONNX full graph checker | PASSED; FP32, opset 17, batch 1, 640 × 640 |
| CPU inference | PASSED on 10 original hash-checked VisDrone validation images |
| Direct AeroDetect versus AeroFusion adapter | PASSED for all ten images |
| PyTorch versus ONNX | All ten images passed; 132 final detections matched |
| Persistent IDs | PASSED on 30 synthetic moving frames in core tests, and three repeated identical real-image frames offline |
| Script static checks | Python compileall and Bash syntax check passed |
| Local verifier missing-ROS behavior | Executed; correctly stopped with exit 2 and explicit Jazzy blocker |

### Fresh ONNX CPU latency

Same ten-image manifest, one CPU thread, FP32, 640 × 640, batch one; five warm-ups per scope, five repeats per image, fifty measured samples per scope.

| Scope | Mean ms | Median ms | p95 ms | Sequential images/s | Failures |
|---|---:|---:|---:|---:|---:|
| Model call | 100.92 | 99.79 | 111.95 | 9.91 | 0 |
| End-to-end | 108.82 | 107.01 | 120.66 | 9.19 | 0 |

Model call includes validation and Python array handling. End-to-end includes file reads/decoding, preprocessing, inference, postprocessing, drawing and cached JPEG writes. Model initialization is excluded. This is not a real-time or hardware-flight guarantee. The earlier benchmark remains separate; no cross-environment speedup is inferred.

Host: Linux 6.18.57, AMD EPYC 9V74 virtual CPU, nine logical CPUs available, approximately 9.73 GiB RAM, Python 3.12.14. Runtime: NumPy 1.26.4, OpenCV headless 4.11.0.86, ONNX 1.23.2, ONNX Runtime 1.31.1, CPUExecutionProvider. Setuptools in the isolated runtime is 79.0.1. Complete package snapshots and model hashes accompany the evidence.

## Implemented, requiring local execution

- ROS `/camera/image_raw` input and separate `/detection/objects` and `/tracking/objects` outputs.
- A genuine rclpy/DDS verifier that checks all 30 synthetic frames, three repeated real-image frames, timestamps, sources, class labels, persistent IDs, and rejection of malformed input.
- Fresh colcon build with an isolated system-Python runtime and a generated-executable shebang check.
- Synthetic and ONNX launch checks, including final-frame processing and automatic termination.

**All four items above are NOT EXECUTED here.** This environment has no `/opt/ros/jazzy`, rclpy, ros2 or colcon and cannot operate the user's local WSL2 computer. Mocked ROS modules were not used to claim execution. The user's previously demonstrated 30-frame WSL run is user-reported prior evidence, not a new run of this upgrade.

## One remaining local command

Extract the bundle with `AeroDetect-Lite` and `aerofusion_ros2` as siblings. In WSL2 Ubuntu, from their parent folder:

```bash
bash aerofusion_ros2/scripts/verify_all.sh "$PWD/AeroDetect-Lite"
```

The script downloads hash-checked assets, creates separate export/runtime venvs, checks dependency consistency, exports ONNX, runs tests, benchmarks the CPU, verifies the offline adapter, builds colcon, tests actual DDS topics and launches both demos. It saves a unique evidence directory under `~/aerofusion_verification/`. It returns nonzero on an unmet prerequisite or failed check.

Prerequisites remain: Ubuntu 24.04 system Python 3.12 and venv support, ROS 2 Jazzy and colcon installed for that interpreter, network access, and enough disk space for CPU PyTorch dependencies and the validation archive. The original active workspace is not modified. The detector directory receives local downloaded assets and a refreshed export report.

## Limits and genuine blockers

- AeroFusion's attempted `yusufthecoder474/AeroFusion-ROS2` remote returned GitHub 404. The uploaded Stage 2 ZIP is the actual source used. A prepared commit patch is supplied; no remote AeroFusion PR is claimed.
- The available dataset is DET still images, not a chronological video sequence. Independent images reset association; repeated-still ID checks are software checks. Real-video tracking quality and MOTA/IDF1/HOTA remain NOT EVALUATED.
- COCO-pretrained YOLOv8n does not cover the complete VisDrone taxonomy. Precision, recall and mAP remain NOT EVALUATED.
- TensorRT, CUDA, Jetson and physical sensors were not verified or claimed.
- The runtime dependency set was verified in a clean environment here. Its interaction with the user's apt-installed ROS stack still needs the automated local check; a version pin alone is not proof of WSL compatibility.

## Evidence locations

`evidence/` in the combined delivery contains baseline and updated test logs, asset/export logs, raw parity and adapter reports, pip checks and environment snapshots. `AeroDetect-Lite/outputs/integration-20261031/` contains fresh raw timings, aggregate CSV, environment data and actual annotated images. `outputs/integration-parity/` contains the fresh PyTorch/ONNX comparison. Historical outputs remain in their original locations.
