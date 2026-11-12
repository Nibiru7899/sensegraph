"""Recognition from observable image crops only; no annotations/transcriptions."""
from .ocr_visual import crop_pixels,transcribe


def recognize_regions(model,image,boxes,device='cpu'):
    inputs=[]
    for box in boxes:
        pixels,length,_=crop_pixels(image,box)
        inputs.append({'pixels':pixels,'length':length})
    return transcribe(model,inputs,device=device)
