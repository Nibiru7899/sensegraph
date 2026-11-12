import pytest
from sensegraph.io import write_json,read_json,sha256
from sensegraph.quality_budget import QualitySession


def test_prior_active_session_blocks_new_gpu_work(tmp_path):
    prior=tmp_path/'old.json'; new=tmp_path/'new.json'
    write_json(prior,{'historical_seconds':100,'charged_seconds':20,'actual_seconds':None})
    with pytest.raises(RuntimeError,match='active'): QualitySession(new,prior,10,min_seconds=1).__enter__()
    assert not new.exists()


def test_closed_prior_is_inherited_not_reset(tmp_path):
    prior=tmp_path/'old.json'; new=tmp_path/'new.json'
    write_json(prior,{'historical_seconds':100,'charged_seconds':20,'actual_seconds':20}); before=sha256(prior)
    with QualitySession(new,prior,10,min_seconds=1):
        current=read_json(new); assert current['historical_seconds']==120 and current['charged_seconds']==10
    final=read_json(new)
    assert final['actual_seconds'] is not None and final['charged_seconds']==final['actual_seconds']
    assert sha256(prior)==before and final['historical_seconds']==120
