import json
import pytest
from aerofusion_ros2.pipeline import PerceptionPipeline
from aerofusion_ros2.frames import generate_rgb_frame

def frame(pipeline,n=0,stamp=None,frame_id='camera'):
    return pipeline.process(320,240,960,generate_rgb_frame(320,240,n),'rgb8',frame_id,stamp if stamp is not None else n+1)

def test_topics_preserve_metadata_and_ids():
    pipeline=PerceptionPipeline()
    for n in range(30):
        detection,tracking=frame(pipeline,n)
        assert tracking['tracks'][0]['track_id']==1
        assert detection['stamp_ns']==tracking['stamp_ns']==n+1
        assert detection['objects'][0]['bbox_xyxy']==tracking['tracks'][0]['bbox_xyxy']
        assert tracking['image_number']==n+1
        json.dumps(tracking,allow_nan=False)

def test_unrelated_stills_reset_association():
    pipeline=PerceptionPipeline(reset_each_frame=True)
    for n in range(3):
        _,result=frame(pipeline,n)
        assert result['association_reset'] and result['tracks'][0]['age_frames']==1
        assert result['tracking_scope']=='independent_image'

def test_timestamp_rewind_resets():
    p=PerceptionPipeline();frame(p,stamp=9);_,result=frame(p,stamp=8)
    assert result['association_reset'] and result['tracks'][0]['age_frames']==1

def test_camera_change_resets():
    p=PerceptionPipeline();frame(p);_,result=frame(p,1,frame_id='other')
    assert result['association_reset']

def test_rejection_does_not_advance_counter():
    p=PerceptionPipeline()
    with pytest.raises(ValueError):p.process(2,2,6,b'bad','rgb8','camera',1)
    assert p.received==0

def test_padded_rgb_synthetic():
    p=PerceptionPipeline();source=generate_rgb_frame(320,240,0)
    padded=b''.join(source[y*960:(y+1)*960]+b'\0'*8 for y in range(240))
    result,_=p.process(320,240,968,padded,'rgb8','camera',1)
    assert len(result['objects'])==1

@pytest.mark.parametrize('mode,detector',[('onnx',None),('bad',None)])
def test_no_onnx_fallback(mode,detector):
    with pytest.raises(ValueError):PerceptionPipeline(mode,detector)

def test_resolution_change_resets():
    p=PerceptionPipeline();frame(p)
    data=generate_rgb_frame(128,96,0)
    _,result=p.process(128,96,384,data,'rgb8','camera',2)
    assert result['association_reset'] and result['tracks'][0]['age_frames']==1
