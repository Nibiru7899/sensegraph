"""Windows working-state time and evidence-bound reconciliation; no GPU calls."""
import ctypes
from datetime import datetime
from xml.etree import ElementTree as ET


def working_seconds():
    value=ctypes.c_ulonglong()
    if not ctypes.windll.kernel32.QueryUnbiasedInterruptTime(ctypes.byref(value)): raise OSError('QueryUnbiasedInterruptTime failed')
    return value.value/10000000


class WorkingState:
    def __enter__(self):
        self.function=ctypes.windll.kernel32.SetThreadExecutionState
        self.function.argtypes=[ctypes.c_uint32]; self.function.restype=ctypes.c_uint32
        self.previous=self.function(0x80000001) # continuous + system required; no display/away mode.
        if not self.previous: raise OSError('Temporary system idle-sleep hold failed')
        return self
    def __exit__(self,*args):
        if not self.function(self.previous): raise OSError('Restoring execution state failed')


def verified_suspend_correction(raw,events):
    ns={'e':'http://schemas.microsoft.com/win/2004/08/events/event'}; parsed=[]
    for event in events:
        xml=ET.fromstring(event['Xml']); stamp=xml.find('e:System/e:TimeCreated',ns).attrib['SystemTime']
        fields={v.attrib['Name']:v.text for v in xml.findall('e:EventData/e:Data',ns)}
        parsed.append({**event,'stamp':datetime.fromisoformat(stamp.replace('Z','+00:00')).timestamp(),'fields':fields})
    assert len(raw['sessions'])==1 and raw['sessions'][0]['actual_seconds'] is not None
    start=raw['sessions'][0]['start_unix']; end=start+raw['sessions'][0]['actual_seconds']; matches=[]
    for event in parsed:
        if event['Provider']!='Microsoft-Windows-Kernel-General' or event['Id']!=1: continue
        fields=event['fields']
        if fields.get('Reason')!='2' or fields.get('ProcessID')!='4': continue
        old=datetime.fromisoformat(fields['OldTime'].replace('Z','+00:00')).timestamp()
        new=datetime.fromisoformat(fields['NewTime'].replace('Z','+00:00')).timestamp()
        delta=int(fields['TimeDeltaInMs'])/1000 # rounded down: retain extra fractions as charged.
        if not start<old<new<=end or delta<60: continue
        sleep=[p for p in parsed if p['Provider']=='Microsoft-Windows-Kernel-Power' and p['Id']==42 and abs(p['stamp']-old)<=5]
        wake=[p for p in parsed if p['Provider']=='Microsoft-Windows-Kernel-Power' and p['Id']==107 and abs(p['stamp']-old)<=5]
        if not sleep or not wake: raise RuntimeError('Clock change lacks adjacent sleep/resume evidence')
        matches.append({'clock_record':event['RecordId'],'sleep_record':sleep[0]['RecordId'],'wake_record':wake[0]['RecordId'],'excluded_seconds':delta,'old_unix':old,'new_unix':new})
    if len(matches)!=1: raise RuntimeError('Expected exactly one evidenced suspension inside this session')
    excluded=sum(v['excluded_seconds'] for v in matches); adjusted=raw['charged_seconds']-excluded
    if adjusted<0: raise RuntimeError('Invalid negative reconciled duration')
    return {'raw_wall_session_seconds':raw['charged_seconds'],'verified_excluded_seconds':excluded,'reconciled_session_seconds':adjusted,'evidence':matches,
            'limits':'Only evidenced hardware-clock correction during sleep is excluded; awake overhead and fractional rounding remain charged. A later hibernation after the session ended is not subtracted. Original ledger is preserved.'}
