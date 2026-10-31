"""Real ONNX adapter + tracker checks without ROS; no transport claim."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--aerodetect',required=True,type=Path);p.add_argument('--model',required=True,type=Path);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from aerofusion_ros2.detectors import AeroDetectONNX
from aerofusion_ros2.pipeline import PerceptionPipeline
adapter=AeroDetectONNX(a.aerodetect,a.model)
core=sys.modules[adapter.preprocess.__module__]
manifest=json.loads((a.aerodetect/'config/subset.json').read_text())
records=[]
for n,item in enumerate(manifest['images']):
    path=a.dataset/'images'/item['filename']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Image hash mismatch')
    image=core.load_image(path);h,w=image.shape[:2]
    tensor,transform=core.preprocess(image,640);direct=core.decode(adapter.backend.run(tensor),transform)
    actual=adapter.detect(w,h,w*3,image.tobytes(),'bgr8')
    assert len(actual)==len(direct)
    for left,right in zip(actual,direct):
        np.testing.assert_allclose(left.bbox,right[:4],atol=1e-4);assert left.label==core.TARGETS[int(right[5])]
    independent=PerceptionPipeline('onnx',adapter,reset_each_frame=True)
    detection,tracking=independent.process(w,h,w*3,image.tobytes(),'bgr8','visdrone_still',n+1)
    assert len(tracking['tracks'])==len(actual)
    records.append(dict(filename=path.name,detections=len(actual),direct_adapter_agreement=True,tracking_scope='independent_image'))
# A repeated still is a controlled ID persistence check, NOT aerial-video tracking validation.
path=a.dataset/'images'/manifest['images'][0]['filename'];image=core.load_image(path);h,w=image.shape[:2]
pipeline=PerceptionPipeline('onnx',adapter);ids=[]
for n in range(3):
    _,result=pipeline.process(w,h,w*3,image.tobytes(),'bgr8','repeat_test',n+1)
    ids.append([t['track_id'] for t in result['tracks']])
assert ids[0] and ids[0]==ids[1]==ids[2]
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps(dict(status='PASSED',ros_transport='NOT EXECUTED',images=records,repeated_still_ids=ids,limitation='Repeated identical still only; no real video tracking metrics'),indent=2))
print(a.output)
