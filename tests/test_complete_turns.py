from pathlib import Path
import json
import subprocess
import sys
import pytest
from aegisvox.complete_turns import (PCM, read_pcm, write_pcm, validate_turns,
    render_conversation, CommandTurnRenderer, TurnContractError)
from aegisvox.quickstart import demo, tone, ToneRenderer
from aegisvox.__main__ import main


def make_case(tmp_path):
    base=PCM(16000,1,2,b'\0\0'*1600+tone(220,1)+b'\0\0'*3200+tone(330,1)+b'\0\0'*1600)
    write_pcm(tmp_path/'converted.wav',base)
    manifest={'schema_version':'aegisvox.complete_turns.v1','converted_audio':'converted.wav','turns':[
        {'turn_id':'one','speaker_id':'A','start_sec':.1,'end_sec':1.1,
         'source_transcript':'Full original turn.','target_transcript':'Full revised turn.'},
        {'turn_id':'two','speaker_id':'B','start_sec':1.3,'end_sec':2.3}]}
    path=tmp_path/'turns.json';path.write_text(json.dumps(manifest))
    return path,base,manifest


def test_complete_reference_and_exact_unchanged_context(tmp_path):
    path,base,_=make_case(tmp_path)
    class Renderer(ToneRenderer):
        def render(self,request,reference,output):
            assert read_pcm(reference).frames==16000
            assert request['target_transcript']=='Full revised turn.'
            return super().render(request,reference,output)
    root=tmp_path/'output';report=render_conversation(path,root,Renderer())
    final=read_pcm(root/'final.wav')
    assert report['edits'][0]['sample_equivalent']
    assert final.data[:1600*2]==base.data[:1600*2]
    end=report['edits'][0]['output_interval_frames'][1]
    assert final.data[end*2:]==base.data[17600*2:]
    assert report['output_frames']==base.frames+1600
    assert report['semantic_validation']=='not_performed'


@pytest.mark.parametrize('bad',[
    {'start_sec':-1}, {'end_sec':99}, {'end_sec':0}, {'start_sec':float('nan')},
    {'end_sec':float('inf')}, {'target_transcript':''}, {'source_transcript':None},
    {'speaker_id':''}, {'start_sec':'bad'},
])
def test_bad_turns_fail_before_creating_output(tmp_path,bad):
    path,_,data=make_case(tmp_path);data['turns'][0].update(bad);path.write_text(json.dumps(data))
    with pytest.raises(TurnContractError):render_conversation(path,tmp_path/'out',ToneRenderer())
    assert not (tmp_path/'out').exists()


def test_overlapping_turns_rejected(tmp_path):
    path,_,data=make_case(tmp_path);data['turns'][1]['start_sec']=1
    path.write_text(json.dumps(data))
    with pytest.raises(TurnContractError,match='non-overlapping'):
        render_conversation(path,tmp_path/'out',ToneRenderer())


def test_duplicate_ids_and_source_requirement(tmp_path):
    _,base,data=make_case(tmp_path)
    data['turns'][1]['turn_id']='one'
    with pytest.raises(TurnContractError,match='unique'):validate_turns(data,base)
    data['turns'][1]['turn_id']='two';data['turns'][0].pop('source_transcript')
    with pytest.raises(TurnContractError,match='complete source'):validate_turns(data,base)


@pytest.mark.parametrize('failure',['raise','silence','wrong_format','missing'])
def test_failure_never_becomes_source_copy(tmp_path,failure):
    path,_,_=make_case(tmp_path)
    class Broken:
        name='broken-fixture'
        def render(self,request,reference,output):
            if failure=='raise':raise RuntimeError('fixture failure')
            if failure=='silence':write_pcm(output,PCM(16000,1,2,b'\0\0'*100))
            if failure=='wrong_format':write_pcm(output,PCM(8000,1,2,tone(100,1,8000)))
            return {}
    with pytest.raises(Exception):render_conversation(path,tmp_path/'out',Broken())
    assert not (tmp_path/'out/final.wav').exists()
    assert json.loads((tmp_path/'out/reconstruction.json').read_text())['status']=='failed'


def test_no_edits_preserves_pcm_and_refuses_overwrite(tmp_path):
    path,base,data=make_case(tmp_path)
    data['turns'][0].pop('target_transcript');path.write_text(json.dumps(data))
    render_conversation(path,tmp_path/'out',ToneRenderer())
    assert read_pcm(tmp_path/'out/final.wav').data==base.data
    with pytest.raises(FileExistsError):render_conversation(path,tmp_path/'out',ToneRenderer())


def test_command_adapter_argv_and_full_request(tmp_path):
    path,_,_=make_case(tmp_path)
    worker=tmp_path/'worker.py'
    worker.write_text('import json,shutil,sys\nr=json.load(open(sys.argv[1]))\nassert r["target_transcript"]=="Full revised turn."\nshutil.copyfile(r["reference_audio"],sys.argv[2])\n')
    renderer=CommandTurnRenderer([sys.executable,str(worker),'{request}','{output}'])
    result=render_conversation(path,tmp_path/'out',renderer)
    assert result['status']=='assembled' # structural contract only, not word correctness
    assert result['edits'][0]['sample_equivalent']


def test_no_hidden_shell_and_placeholder_validation():
    with pytest.raises(TurnContractError):CommandTurnRenderer(['echo','no-arguments'])
    with pytest.raises(TurnContractError):CommandTurnRenderer(['{request}','{output}'],timeout_sec=0)


def test_demo_and_cli(tmp_path,capsys):
    assert main(['demo','--output',str(tmp_path/'demo')])==0
    payload=json.loads(capsys.readouterr().out)
    assert 'no speech' in payload['demonstration']
    assert payload['privacy_or_clinical_guarantee'] is False
    assert main(['demo','--output',str(tmp_path/'demo')])==1


def test_enqueue_does_not_start_models(tmp_path,capsys):
    wav=tmp_path/'authorized.wav';write_pcm(wav,PCM(16000,1,2,tone(200,1)))
    assert main(['enqueue',str(wav),'--root',str(tmp_path/'artifacts'),'--voice-pool','fixture'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['model_inference_started'] is False
    assert (tmp_path/'artifacts/metadata.sqlite3').exists()
