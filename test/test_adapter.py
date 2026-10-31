"""A tiny constant ONNX fixture checks the adapter contract, never benchmark accuracy."""
from pathlib import Path
import sys
import numpy as np
import onnx
from onnx import helper,TensorProto,numpy_helper
import pytest
from aerofusion_ros2.detectors import AeroDetectONNX

@pytest.fixture
def detector(tmp_path):
    repo=Path(__file__).resolve().parents[2]/'AeroDetect-Lite'
    if not (repo/'src/core.py').is_file():pytest.skip('Sibling AeroDetect-Lite needed for adapter integration')
    values=np.zeros((1,84,1),np.float32);values[0,:4,0]=[320,320,320,160];values[0,6,0]=.9
    node=helper.make_node('Constant',[],['output'],value=numpy_helper.from_array(values))
    graph=helper.make_graph([node],'adapter_fixture',[helper.make_tensor_value_info('images',TensorProto.FLOAT,[1,3,640,640])],[helper.make_tensor_value_info('output',TensorProto.FLOAT,[1,84,1])])
    model=helper.make_model(graph,opset_imports=[helper.make_opsetid('',17)]);model.ir_version=9
    path=tmp_path/'fixture.onnx';onnx.save(model,path)
    return AeroDetectONNX(repo,path)

def test_actual_ort_adapter_scaling(detector):
    image=np.zeros((40,80,3),np.uint8);boxes=detector.detect(80,40,240,image.tobytes())
    assert boxes[0].label=='car';np.testing.assert_allclose(boxes[0].bbox,(20,10,60,30))

def test_rgb_bgr_and_padding(detector):
    image=np.zeros((40,80,3),np.uint8)
    packed=detector.detect(80,40,240,image.tobytes(),'bgr8')
    data=b''.join(row.tobytes()+b'\0'*4 for row in image)
    padded=detector.detect(80,40,244,data,'rgb8')
    assert packed==padded

def test_bad_encoding_rejected(detector):
    with pytest.raises(ValueError):detector.detect(80,40,240,bytes(9600),'mono8')

def test_import_alias_is_scoped(detector):
    assert detector.preprocess.__module__.startswith('_aerodetect_')
