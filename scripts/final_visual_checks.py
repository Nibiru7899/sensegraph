"""Read-only final source/artifact checks; writes only its own evidence JSON."""
from pathlib import Path
import sys,subprocess
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from sensegraph.io import read_json,write_json,sha256
from sensegraph import full_visual_training as trainer
from sensegraph.corrected_visual_training import bind


def main():
    trainer.verify(ROOT); bind(); protected=trainer.verify(ROOT)
    base=ROOT/'experiments/visual_training'; release=base/'release'; snapshot=release/'source_and_release_hashes.json'
    files=read_json(snapshot)['files']; changed=[p for p,h in files.items() if not (ROOT/p).is_file() or sha256(ROOT/p)!=h]
    assert not changed,changed[:5]
    summary=read_json(release/'quality_summary.json'); assert not summary['gpu_session_in_flight'] and summary['cumulative_gpu_seconds_estimated']<=18000
    assert all(summary[k] for k in ['holdout_complete','ocr_complete','graph_transfer_complete','real_partial_complete','postselection_verification_complete'])
    assert summary['ready_for_reliable_real_chart_qa'] is False
    reconciliation=read_json(base/'v3/suspend_reconciliation.json')
    assert reconciliation['original_ledger_sha256']==sha256(base/'v3/gpu_budget.json') and reconciliation['power_events_sha256']==sha256(base/'v3/power_events.json')
    selection=read_json(base/'independent_holdout/selection.json')
    assert selection['script_sha256']==sha256(ROOT/'scripts/evaluate_visual_holdout.py')
    assert selection['candidates'][selection['selected']]['sha256']==sha256(release/'selected.pt')
    tests=ET.parse(release/'tests.xml'); assert not tests.findall('.//failure') and not tests.findall('.//error')
    test_count=len(tests.findall('.//testcase')); assert test_count==summary['tests']
    dependency=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True,check=True)
    download=read_json(ROOT/'experiments/download_ledger.json')['transferred_bytes']; assert download<=10000000000
    result={'historical_protected_files':protected['protected_files'],'changed_historical_files':0,'snapshot_files_checked':len(files),'changed_snapshot_files':changed,'snapshot_sha256':sha256(snapshot),
        'tests_passed':test_count,'pip_check':dependency.stdout.strip(),'all_quality_stages_complete':True,'GPU_session_in_flight':False,'charged_gpu_work_seconds':summary['cumulative_gpu_seconds_estimated'],
        'download_bytes':download,'original_suspension_ledger_and_power_evidence_unchanged':True,'locked_selection_script_and_checkpoint_unchanged':True,'GPU_execution_in_this_check':False,
        'ready_for_real_chart_QA':False,'limits':'Integrity/completeness checks do not turn poor OCR/real transfer into readiness or prove absence of all leakage/overfitting.'}
    write_json(base/'final_checks.json',result); print(result)


if __name__=='__main__': main()
