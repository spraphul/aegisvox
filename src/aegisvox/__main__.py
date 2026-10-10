"""Public CLI. Model inference is never substituted by demo output."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from . import __version__
from .cli import print_json, print_json_error


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(prog='aegisvox',description='Factorized clinical speech transformation and stage-resolved evaluation.')
    parser.add_argument('--version',action='version',version=__version__)
    sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('demo',help='Offline tone-based reconstruction demonstration, not speech synthesis.')
    p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('render-turns',help='Render complete converted turns and insert their exact PCM.')
    p.add_argument('manifest',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--renderer',type=Path,required=True,help='JSON: name, argv, timeout_sec.')
    p=sub.add_parser('serve',help='Start the control-plane API, not the models.')
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8080)
    p=sub.add_parser('enqueue',help='Register authorized input audio and create the first service job.')
    p.add_argument('audio',type=Path);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--voice-pool',required=True);p.add_argument('--language',default=None)
    p=sub.add_parser('worker',help='Claim and execute one service job using a configured command.')
    p.add_argument('--root',type=Path,required=True);p.add_argument('--service',required=True)
    p=sub.add_parser('service-job',help='Execute one claimed job from AEGISVOX_JOB_JSON.')
    try:
        args=parser.parse_args(argv)
        if args.action=='demo':
            from .quickstart import demo
            result=demo(args.output)
        elif args.action=='render-turns':
            from .complete_turns import CommandTurnRenderer, render_conversation
            cfg=json.loads(args.renderer.read_text())
            backend=CommandTurnRenderer(**cfg)
            result=render_conversation(args.manifest,args.output,backend)
        elif args.action=='serve':
            if args.host not in ('127.0.0.1', 'localhost', '::1') and os.environ.get('AEGISVOX_API_AUTH_REQUIRED', '').lower() not in ('1','true','yes'):
                raise ValueError('Non-loopback serving requires API authentication.')
            import uvicorn
            uvicorn.run('aegisvox.api:create_app',factory=True,host=args.host,port=args.port)
            return 0
        elif args.action in ('enqueue','worker'):
            from .storage import LocalArtifactStore
            from .metadata import SQLiteMetadataStore
            store=LocalArtifactStore(args.root)
            metadata=SQLiteMetadataStore(args.root/'metadata.sqlite3')
            if args.action=='enqueue':
                from .services.orchestrator import Orchestrator
                from .events import InMemoryEventBus
                from .schema import RunRequest
                if not args.audio.is_file(): raise ValueError('Input audio does not exist.')
                request=RunRequest(input_audio_uri=args.audio.resolve().as_uri(),
                    voice_pool_id=args.voice_pool,language_hint=args.language)
                run=Orchestrator(store,InMemoryEventBus(),metadata_store=metadata).create_run(request)
                result={'run_id':run.run_id,'status':run.status.value,'model_inference_started':False}
            else:
                from .worker import CommandWorker
                result=CommandWorker(metadata,store).run_once(service=args.service)
        else:
            from .service_job import run_service_job
            from .storage import LocalArtifactStore
            job=json.loads(Path(os.environ['AEGISVOX_JOB_JSON']).read_text())
            result=run_service_job(job,artifact_store=LocalArtifactStore(os.environ['AEGISVOX_ARTIFACT_ROOT'])).to_worker_payload()
        print_json(result)
        return 1 if isinstance(result,dict) and result.get('status')=='failed' else 0
    except (Exception,) as exc:
        # Do not print transcripts, command output or paths embedded in exceptions.
        print_json_error(type(exc).__name__,reason='Check configuration and protected runtime artifacts; see docs/guides/troubleshooting.md.')
        return 1


if __name__=='__main__':
    raise SystemExit(main())
