"""Two real CPU checkpoint/CLI reruns with socket connection attempts rejected."""
from pathlib import Path
import sys,runpy,socket
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256


def main():
    d=ROOT/'experiments/visual_training/release'; attempts=[]
    def reject(*args,**kwargs):
        attempts.append('network connection attempted'); raise RuntimeError('Offline inference attempted network access')
    original=list(sys.argv)
    try:
        with patch.object(socket.socket,'connect',reject),patch.object(socket,'create_connection',reject):
            for name in ['offline_first','offline_second']:
                sys.argv=['infer_full_visual.py','--image','data/synthetic/v3/images/synthetic-01400.png','--output',str(d/f'{name}.json'),'--with-ocr']
                runpy.run_path(str(ROOT/'scripts/infer_full_visual.py'),run_name='__main__')
    finally: sys.argv=original
    assert read_json(d/'offline_first.json')==read_json(d/'offline_second.json') and not attempts
    assert sha256(d/'offline_first.png')==sha256(d/'offline_second.png')
    write_json(d/'offline_inference_audit.json',{'trained_weights_used':True,'device':'cpu','vision_and_ocr_json_exact':True,'overlay_png_exact':True,'socket_connect_attempts':len(attempts),'annotations_read':False,
        'scope':'Actual image/checkpoint CPU rerun and rejected socket connection methods; bounded cache/wiring evidence, not generalization or operating-system-wide network isolation'})
    print('Actual trained vision+OCR CPU reruns exact; zero intercepted socket connections.')


if __name__=='__main__': main()
