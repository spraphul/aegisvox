"""Real F5-TTS complete-turn adapter; run in a separately provisioned model env.

CLI is deliberately one-shot for integration debugging. Use F5TurnRenderer in a
long-lived worker to keep its single model resident across turns/conversations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

class F5TurnRenderer:
    name = 'F5TTS_v1_Base-complete-turn'
    def __init__(self, device='cuda', nfe_step=16):
        from f5_tts.api import F5TTS
        self.model=F5TTS(model='F5TTS_v1_Base',device=device)
        self.nfe_step=nfe_step

    def render(self,request,reference,output):
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        from math import gcd
        if request['channels']!=1 or request['sample_width']!=2:
            raise ValueError('F5 example requires mono PCM16 input.')
        native=output.with_suffix('.native.wav')
        seed=int.from_bytes(hashlib.sha256(request['turn_id'].encode()).digest()[:4],'big')%(2**31-1)
        self.model.infer(ref_file=str(reference),ref_text=request['source_transcript'],
                         gen_text=request['target_transcript'],nfe_step=self.nfe_step,
                         file_wave=str(native),seed=seed,show_info=lambda *a,**kw:None,progress=None)
        samples,rate=sf.read(native,always_2d=False)
        if samples.ndim!=1 or not np.isfinite(samples).all(): raise ValueError('Invalid model output.')
        native_rate=rate;target_rate=request['sample_rate']
        if rate!=target_rate:
            factor=gcd(rate,target_rate)
            samples=resample_poly(samples,target_rate//factor,rate//factor)
        sf.write(output,samples,target_rate,subtype='PCM_16')
        return {'model':self.name,'native_sample_rate':native_rate,'output_sample_rate':target_rate,
                'resampled':native_rate!=target_rate,'nfe_step':self.nfe_step,'seed':seed}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--request',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--device',default='cuda')
    a=ap.parse_args();request=json.loads(a.request.read_text())
    F5TurnRenderer(a.device).render(request,Path(request['reference_audio']),a.output)
