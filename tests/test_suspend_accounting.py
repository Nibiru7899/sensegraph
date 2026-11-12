from datetime import datetime
import pytest
from sensegraph.suspend_accounting import verified_suspend_correction


def event(provider,identity,record,stamp,fields):
    data=''.join(f'<Data Name="{k}">{v}</Data>' for k,v in fields.items())
    return {'Provider':provider,'Id':identity,'RecordId':record,'Xml':f'<Event xmlns="http://schemas.microsoft.com/win/2004/08/events/event"><System><TimeCreated SystemTime="{stamp}"/></System><EventData>{data}</EventData></Event>'}


def fixture():
    start=datetime.fromisoformat('2026-11-04T00:00:00+00:00').timestamp()
    raw={'charged_seconds':4000,'sessions':[{'start_unix':start,'actual_seconds':4000}]}
    fields={'OldTime':'2026-11-04T00:01:00Z','NewTime':'2026-11-04T01:01:00Z','TimeDeltaInMs':'3600000','Reason':'2','ProcessID':'4'}
    events=[event('Microsoft-Windows-Kernel-General',1,3,fields['NewTime'],fields),
            event('Microsoft-Windows-Kernel-Power',42,1,'2026-11-04T00:00:59Z',{}),
            event('Microsoft-Windows-Kernel-Power',107,2,'2026-11-04T00:01:00Z',{})]
    return raw,events


def test_subtract_only_evidenced_suspension_without_changing_original():
    raw,events=fixture(); result=verified_suspend_correction(raw,events)
    assert result['verified_excluded_seconds']==3600 and result['reconciled_session_seconds']==400
    assert raw['charged_seconds']==4000
    # An unrelated clock correction after session closure must not be subtracted.
    events.append(event('Microsoft-Windows-Kernel-General',1,4,'2026-11-04T02:00:00Z',{'OldTime':'2026-11-04T01:50:00Z','NewTime':'2026-11-04T02:00:00Z','TimeDeltaInMs':'600000','Reason':'2','ProcessID':'4'}))
    assert verified_suspend_correction(raw,events)['reconciled_session_seconds']==400


def test_clock_jump_alone_cannot_refund_training_budget():
    raw,events=fixture()
    with pytest.raises(RuntimeError,match='sleep/resume evidence'): verified_suspend_correction(raw,events[:1])
