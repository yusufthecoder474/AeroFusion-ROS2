"""Stage-2 detectors: deterministic synthetic RGB marker, optional AeroDetect ONNX.

Synthetic red-marker detection is NOT neural-network object detection.
The optional ONNX class reuses the user's AeroDetect-Lite core + backend.
"""
from pathlib import Path
from .tracking import Detection


def validate_rgb_image(width, height, step, image_bytes):
    if width < 1 or height < 1 or width > 4096 or height > 4096:
        raise ValueError('Invalid image dimensions')
    if step < width * 3 or step > width * 3 + 65536 or len(image_bytes) != height * step:
        raise ValueError('Unexpected RGB8 image layout or payload length')


def detect_synthetic_red_box(width, height, step, image_bytes):
    """Find the red square emitted by the Stage-1 image publisher."""
    validate_rgb_image(width, height, step, image_bytes)
    minx, miny, maxx, maxy = width, height, -1, -1
    count = 0
    data = memoryview(image_bytes)
    for y in range(height):
        offset = y * step
        for x in range(width):
            p = offset + 3 * x
            if data[p] >= 200 and data[p + 1] <= 100 and data[p + 2] <= 100:
                minx = min(minx, x)
                miny = min(miny, y)
                maxx = max(maxx, x)
                maxy = max(maxy, y)
                count += 1
    if count < 16:
        return []
    area = (maxx - minx + 1) * (maxy - miny + 1)
    if count < area * 0.7:
        return []  # reject sparse red noise
    return [Detection((minx, miny, maxx + 1, maxy + 1), 'synthetic-red-box', 1.0)]


class AeroDetectONNX:
    """Adapts exactly the existing AeroDetect-Lite preprocessing/decode code.

    Requires the *user's real* AeroDetect-Lite repo and an exported yolov8n.onnx.
    CPU only, no downloading or fallback to fake predictions.
    """

    def __init__(self, repo_dir, model_path, size=640, threads=1, confidence=0.25):
        import hashlib
        import importlib
        import importlib.util
        import sys
        root = Path(repo_dir).expanduser().resolve()
        if not (root / 'src' / 'core.py').is_file() or not (root / 'src' / 'backends.py').is_file():
            raise FileNotFoundError('AeroDetect-Lite src/core.py or src/backends.py missing')
        if not Path(model_path).is_file():
            raise FileNotFoundError('Exported ONNX model missing: set onnx_path')
        if not 0 <= float(confidence) <= 1:
            raise ValueError('confidence must be in [0,1]')
        # An explicit package alias avoids accidentally importing another project's src.
        alias = '_aerodetect_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        if alias not in sys.modules:
            spec = importlib.util.spec_from_file_location(alias, root / 'src' / '__init__.py',
                                                        submodule_search_locations=[str(root / 'src')])
            if spec is None or spec.loader is None:
                raise ImportError('Cannot load AeroDetect package')
            package = importlib.util.module_from_spec(spec)
            sys.modules[alias] = package
            spec.loader.exec_module(package)
        core = importlib.import_module(alias + '.core')
        backends = importlib.import_module(alias + '.backends')
        self.preprocess = core.preprocess
        self.decode = core.decode
        self.targets = core.TARGETS
        self.backend = backends.OnnxBackend(str(Path(model_path).resolve()), int(size), int(threads))
        self.confidence = float(confidence)
        self.size = int(size)

    def detect(self, width, height, step, data, encoding='rgb8'):
        validate_rgb_image(width, height, step, data)
        if encoding not in ('rgb8', 'bgr8'):
            raise ValueError('ONNX mode requires rgb8 or bgr8')
        import numpy as np
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, step)[:, :width*3].reshape(height, width, 3)
        bgr_image = image[..., ::-1].copy() if encoding == 'rgb8' else image.copy()
        tensor, transform = self.preprocess(bgr_image, self.size)
        raw = self.backend.run(tensor)
        boxes = self.decode(raw, transform, confidence=self.confidence)
        return [Detection(tuple(float(x) for x in box[:4]), self.targets[int(box[5])], float(box[4])) for box in boxes]
