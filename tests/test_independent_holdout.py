import importlib.util
from pathlib import Path
import pytest
from sensegraph.io import write_json


def module():
    path=Path(__file__).resolve().parents[1]/'scripts/evaluate_visual_holdout.py'
    spec=importlib.util.spec_from_file_location('holdout_evaluation_test',path); loaded=importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded); return loaded


def test_selection_depends_on_validation_and_tie_prefers_corrected():
    m=module()
    assert m.select_winner({'original_v2':{'validation_ap50':.9},'corrected_v3':{'validation_ap50':.89}})=='original_v2'
    assert m.select_winner({'original_v2':{'validation_ap50':.9},'corrected_v3':{'validation_ap50':.9}})=='corrected_v3'


def test_holdout_refuses_repeated_attempt_before_model_loading(tmp_path):
    m=module(); m.ROOT=tmp_path
    write_json(tmp_path/'experiments/visual_training/independent_holdout/evaluation_attempt.json',{'started':True})
    with pytest.raises(RuntimeError,match='already attempted'): m.main()


def test_five_hour_guard_precedes_holdout_gpu_use(tmp_path):
    m=module(); m.ROOT=tmp_path
    write_json(tmp_path/'experiments/visual_training/v3/gpu_budget.json',{'historical_seconds':17900,'charged_seconds':50,'sessions':[{'actual_seconds':50}]})
    with pytest.raises(RuntimeError,match='Insufficient five-hour budget'): m.AuditSession().__enter__()
    assert not (tmp_path/'experiments/visual_training/independent_holdout/gpu_budget.json').exists()


def test_active_training_blocks_selection_record_itself(tmp_path):
    m=module(); m.ROOT=tmp_path
    write_json(tmp_path/'experiments/visual_training/v2/gpu_budget.json',{'sessions':[{'actual_seconds':None}]})
    with pytest.raises(RuntimeError,match='Training active'): m.main()
    assert not (tmp_path/'experiments/visual_training/independent_holdout/selection.json').exists()
