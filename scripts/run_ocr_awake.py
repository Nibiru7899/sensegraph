"""Temporary idle-sleep hold around the unchanged, frozen OCR trainer."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.suspend_accounting import WorkingState
from sensegraph.ocr_experiment import main

if __name__=='__main__':
    with WorkingState(): main()
