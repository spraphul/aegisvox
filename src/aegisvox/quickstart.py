"""Offline mechanics demonstration using tones, never simulated clinical results."""
from pathlib import Path
import json
import math
import struct
from .complete_turns import PCM, read_pcm, write_pcm, render_conversation


def tone(freq: float, seconds: float, rate: int = 16000) -> bytes:
    n = round(seconds * rate)
    return b''.join(struct.pack('<h',round(4500*math.sin(2*math.pi*freq*i/rate)*min(1,i/160,(n-i)/160))) for i in range(n))


class ToneRenderer:
    name = 'mechanics-only-tone-fixture'
    def render(self, request, reference, output):
        base = read_pcm(reference)
        write_pcm(output, PCM(base.rate, 1, 2, tone(660, base.frames/base.rate+0.1)))
        return {'fixture': True, 'speech': False}


def demo(output: Path):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    gaps = b'\0\0' * 3200
    audio = gaps + tone(220,1) + gaps + tone(330,1) + gaps + tone(440,1) + gaps
    write_pcm(output/'converted_fixture.wav', PCM(16000,1,2,audio))
    manifest={'schema_version':'aegisvox.complete_turns.v1','converted_audio':'converted_fixture.wav',
              'fixture_only':True,'turns':[
                  {'turn_id':'01','speaker_id':'A','start_sec':0.2,'end_sec':1.2},
                  {'turn_id':'02','speaker_id':'B','start_sec':1.4,'end_sec':2.4,
                   'source_transcript':'Tone fixture before edit.','target_transcript':'Tone fixture after edit.'},
                  {'turn_id':'03','speaker_id':'A','start_sec':2.6,'end_sec':3.6}]}
    path=output/'turns.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
    result=render_conversation(path,output/'result',ToneRenderer())
    result.update({'demonstration':'PCM routing and exact insertion only; no speech or model inference.',
                   'output_directory':str(output)})
    return result
