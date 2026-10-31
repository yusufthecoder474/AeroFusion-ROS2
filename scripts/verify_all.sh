#!/usr/bin/env bash
# One command: isolated export + runtime envs, tests, benchmark, colcon and actual DDS.
set -Eeuo pipefail
package_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
detect_root="${1:-$(dirname "$package_root")/AeroDetect-Lite}"
verify_root="${AEROFUSION_VERIFY_ROOT:-$HOME/aerofusion_verification}"
mkdir -p "$verify_root"
run_root="${AEROFUSION_RUN_ROOT:-}"
if [[ -z "$run_root" ]]; then
    run_root="$(mktemp -d "$verify_root/run-XXXXXXXX")"
    set +e
    AEROFUSION_RUN_ROOT="$run_root" bash "${BASH_SOURCE[0]}" "$@" 2>&1 | tee "$run_root/verification.log"
    result=${PIPESTATUS[0]}
    exit "$result"
fi
trap 'status=$?; if ((status != 0)); then printf "FAILED (exit %s). Evidence: %s\n" "$status" "$run_root"; fi' EXIT
printf 'Verification output: %s\n' "$run_root"
[[ -f "$detect_root/src/core.py" ]] || { echo 'BLOCKED: pass the AeroDetect-Lite directory as argument 1'; exit 2; }
detect_root="$(cd "$detect_root" && pwd)"
[[ -f /opt/ros/jazzy/setup.bash ]] || { echo 'BLOCKED: ROS 2 Jazzy is absent'; exit 2; }
[[ -x /usr/bin/python3 ]] || { echo 'BLOCKED: Ubuntu system Python missing'; exit 2; }
/usr/bin/python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Jazzy requires matching system Python 3.12"'
set +u
source /opt/ros/jazzy/setup.bash
set -u
export PYTHONNOUSERSITE=1
# Full exporter is isolated from ROS/colcon and their setuptools constraint.
/usr/bin/python3 -m venv "$run_root/export-venv"
export_python="$run_root/export-venv/bin/python"
"$export_python" -m pip install 'setuptools>=68,<80'
"$export_python" -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
"$export_python" -m pip install -r "$detect_root/requirements.txt"
"$export_python" -m pip check
"$export_python" "$detect_root/scripts/prepare_assets.py" --root "$detect_root"
(cd "$detect_root" && "$export_python" scripts/export_model.py --weights yolov8n.pt --size 640) | tee "$run_root/export.log"
# system-site-packages exposes apt ROS and colcon. Pin build tooling only in this venv.
/usr/bin/python3 -m venv --system-site-packages "$run_root/ros-venv"
ros_python="$run_root/ros-venv/bin/python"
"$ros_python" -m pip install -r "$package_root/requirements-ros.txt"
"$ros_python" -m pip check
"$ros_python" -c 'import sys,rclpy,cv2,numpy,onnxruntime,colcon_core,setuptools; print(sys.executable, setuptools.__version__); assert int(setuptools.__version__.split(".")[0])<80'
"$ros_python" -m pytest -v "$detect_root/tests" -o "pythonpath=$detect_root" --junitxml="$run_root/detection-tests.xml"
(cd "$package_root" && "$ros_python" -m pytest -v test --junitxml="$run_root/fusion-tests.xml")
(cd "$detect_root" && "$ros_python" scripts/benchmark_onnx.py --dataset data/VisDrone2019-DET-val --output "$run_root/benchmark")
"$ros_python" "$package_root/scripts/verify_offline.py" --aerodetect "$detect_root" --model "$detect_root/yolov8n.onnx" --dataset "$detect_root/data/VisDrone2019-DET-val" --output "$run_root/offline.json"
workspace="$run_root/ros_ws"
mkdir -p "$workspace/src"
ln -s "$package_root" "$workspace/src/aerofusion_ros2"
touch "$run_root/export-venv/COLCON_IGNORE" "$run_root/ros-venv/COLCON_IGNORE"
# Invoking colcon with the chosen interpreter prevents /usr/bin/colcon shebang leakage.
(cd "$workspace" && "$ros_python" -c 'from colcon_core.command import main; raise SystemExit(main())' build --symlink-install --packages-select aerofusion_ros2)
set +u
source "$workspace/install/setup.bash"
set -u
executable="$workspace/install/aerofusion_ros2/lib/aerofusion_ros2/tracking_node"
[[ "$(head -n 1 "$executable")" == "#!$ros_python" ]] || { echo 'BLOCKED: generated node uses the wrong interpreter'; exit 3; }
# Isolate this local verification graph from usual ROS domain 0; allow explicit override.
export ROS_DOMAIN_ID="${AEROFUSION_TEST_DOMAIN:-87}"
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
"$ros_python" "$package_root/scripts/verify_ros_topics.py" --aerodetect "$detect_root" --model "$detect_root/yolov8n.onnx" --dataset "$detect_root/data/VisDrone2019-DET-val" --output "$run_root/ros-topics.json"
# Exercise the installed launch path as well; a failed/timeout launch fails verification.
timeout 45 "$ros_python" -c 'from ros2cli.cli import main; main()' launch aerofusion_ros2 tracking_demo.launch.py fps:=10.0 max_frames:=30 > "$run_root/launch-synthetic.log" 2>&1
timeout 60 "$ros_python" -c 'from ros2cli.cli import main; main()' launch aerofusion_ros2 tracking_onnx.launch.py image_dir:="$detect_root/data/VisDrone2019-DET-val/images" aerodetect_root:="$detect_root" onnx_path:="$detect_root/yolov8n.onnx" reset_each_frame:=true fps:=1.0 max_frames:=10 > "$run_root/launch-onnx.log" 2>&1
"$ros_python" - "$run_root" <<'PYVERIFY'
from pathlib import Path
import sys
root=Path(sys.argv[1])
for filename,expected in [('launch-synthetic.log','Frame 30:'),('launch-onnx.log','Frame 10:')]:
    text=(root/filename).read_text()
    assert expected in text, f'{filename}: final frame did not reach tracker'
    assert 'Frame rejected:' not in text and 'Traceback' not in text, filename
PYVERIFY
# Package snapshots accompany every successful verification.
"$ros_python" -m pip freeze > "$run_root/ros-packages.txt"
"$export_python" -m pip freeze > "$run_root/export-packages.txt"
printf 'PASSED: tests, model export, CPU benchmark, offline adapter and actual ROS topics. Evidence: %s\n' "$run_root"
