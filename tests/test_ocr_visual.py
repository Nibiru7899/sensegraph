from PIL import Image
import torch
import pytest
from sensegraph.ocr_visual import encode_text,decode_path,crop_pixels,edit_distance,recognition_metrics


def test_ascii_ctc_repeats_sign_decimals_and_spaces():
    text='-11.00 C1'; encoded=encode_text(text); path=[]
    for token in encoded: path.extend([token,0])
    assert decode_path(path)==text
    assert decode_path([encoded[1],encoded[1]])=='1'
    with pytest.raises(ValueError): encode_text('値')


def test_crop_geometry_not_reference_text():
    image=Image.new('RGB',(100,100),'white')
    pixels,length,rotated=crop_pixels(image,[10,10,20,90])
    assert pixels.shape==(1,32,128) and pixels.dtype==torch.uint8 and rotated and 1<=length<=32
    a=crop_pixels(image,[10,10,90,30]); assert not a[2]
    with pytest.raises(ValueError): crop_pixels(image,[0,0,0,10])


def test_recognition_denominators_include_errors():
    assert edit_distance('123','13')==1 and edit_distance('','abc')==3
    samples=[{'text':'10.00','type':'tick_label','rotated_by_geometry':False},{'text':'C1','type':'tick_label','rotated_by_geometry':False}]
    r=recognition_metrics(samples,['10.00',''])
    assert r['exact']==.5 and r['cer']==2/7 and r['characters']==7
def test_predicted_region_inference_uses_pixels_and_accepts_empty_regions():
    from sensegraph.ocr_inference import recognize_regions

    class PixelRecognizer(torch.nn.Module):
        def forward(self,pixels):
            assert pixels.shape[1:]==(1,32,128)
            assert pixels.min()>=0 and pixels.max()<=1
            logits=torch.zeros(len(pixels),32,96)
            logits[:,:,34]=1  # ASCII 'A', collapsed once by CTC.
            return logits

    model=PixelRecognizer()
    image=Image.new('RGB',(100,40),'white')
    assert recognize_regions(model,image,[],device='cpu')==[]
    assert recognize_regions(model,image,[[0,0,80,30]],device='cpu')==['A']

