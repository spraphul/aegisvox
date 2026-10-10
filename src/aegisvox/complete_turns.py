"""Complete-turn rendering and exact insertion, independent of semantic scoring.

Input is a *voice-converted* conversation with exclusive speaker turns. This
module does not perform diarization, identity conversion, or PHI detection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
import hashlib
import json
import math
import subprocess
import wave


class TurnContractError(ValueError):
    """Invalid audio or turn contract. Never replace a failed edit with source."""


@dataclass(frozen=True)
class PCM:
    rate: int
    channels: int
    width: int
    data: bytes

    @property
    def stride(self) -> int:
        return self.channels * self.width

    @property
    def frames(self) -> int:
        return len(self.data) // self.stride

    def slice(self, start: int, end: int) -> PCM:
        return PCM(self.rate, self.channels, self.width, self.data[start*self.stride:end*self.stride])


def read_pcm(path: Path) -> PCM:
    with wave.open(str(path), 'rb') as f:
        if f.getcomptype() != 'NONE':
            raise TurnContractError('Uncompressed PCM WAV is required.')
        result = PCM(f.getframerate(), f.getnchannels(), f.getsampwidth(), f.readframes(f.getnframes()))
        if result.frames != f.getnframes() or result.frames == 0:
            raise TurnContractError('Audio is empty or truncated.')
        return result


def write_pcm(path: Path, pcm: PCM) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), 'wb') as f:
        f.setframerate(pcm.rate); f.setnchannels(pcm.channels); f.setsampwidth(pcm.width)
        f.writeframes(pcm.data)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class TurnRenderer(Protocol):
    name: str
    def render(self, request: dict[str, Any], reference: Path, output: Path) -> dict[str, Any]: ...


@dataclass
class CommandTurnRenderer:
    """Trusted operator command, argv list, no shell or transcript interpolation."""
    argv: list[str]
    timeout_sec: int = 600
    name: str = 'command'

    def __post_init__(self) -> None:
        if not self.argv or not all(isinstance(x, str) for x in self.argv):
            raise TurnContractError('Renderer command must be a nonempty argv list.')
        if not all(any(token in x for x in self.argv) for token in ('{request}', '{output}')):
            raise TurnContractError('Renderer argv must contain {request} and {output}.')
        if self.timeout_sec <= 0:
            raise TurnContractError('Renderer timeout must be positive.')

    def render(self, request: dict[str, Any], reference: Path, output: Path) -> dict[str, Any]:
        request_path = output.parent / 'request.json'
        request_path.write_text(json.dumps({**request, 'reference_audio': str(reference.resolve()),
                                           'output_audio': str(output.resolve())}, indent=2)+'\n')
        argv = [x.replace('{request}', str(request_path.resolve())).replace('{output}', str(output.resolve()))
                for x in self.argv]
        # Neither raw subprocess output nor transcripts enter console logs.
        result = subprocess.run(argv, timeout=self.timeout_sec, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL, check=False)
        if result.returncode:
            raise TurnContractError(f'Renderer exited with status {result.returncode}. Inspect it in a protected environment.')
        return {'renderer': self.name, 'execution': 'command'}


def validate_turns(manifest: dict[str, Any], base: PCM) -> list[dict[str, Any]]:
    if manifest.get('schema_version') != 'aegisvox.complete_turns.v1':
        raise TurnContractError('Expected schema_version aegisvox.complete_turns.v1.')
    if not isinstance(manifest.get('turns'), list) or not manifest['turns']:
        raise TurnContractError('A nonempty turn list is required.')
    result, ids, previous_end = [], set(), 0
    for row in manifest['turns']:
        if not isinstance(row, dict): raise TurnContractError('Each turn must be an object.')
        tid, speaker = row.get('turn_id'), row.get('speaker_id')
        if not isinstance(tid, str) or not tid or tid in ids or not isinstance(speaker, str) or not speaker:
            raise TurnContractError('Each turn needs a unique turn_id and a speaker_id.')
        ids.add(tid)
        try:
            a, b = float(row['start_sec']), float(row['end_sec'])
        except (KeyError, TypeError, ValueError):
            raise TurnContractError('Turn start_sec and end_sec must be numbers.') from None
        if not all(math.isfinite(x) for x in (a,b)) or a < 0 or b <= a or b > base.frames/base.rate:
            raise TurnContractError('Turn boundaries must be finite and inside the converted audio.')
        start, end = round(a*base.rate), round(b*base.rate)
        if start < previous_end or end <= start:
            raise TurnContractError('Turns must be ordered, non-overlapping, and at least one sample long.')
        previous_end = end
        for k in ('source_transcript', 'target_transcript'):
            if k in row and (not isinstance(row[k], str) or not row[k].strip()):
                raise TurnContractError(f'{k} must be nonempty when provided.')
        if 'target_transcript' in row and not row.get('source_transcript'):
            raise TurnContractError('An edited turn requires its complete source transcript.')
        result.append({**row, 'start_frame': start, 'end_frame': end})
    return result


def render_conversation(manifest_path: Path, output_dir: Path, renderer: TurnRenderer) -> dict[str, Any]:
    """Render selected whole turns; copy every other sample of converted audio.

    The destination must not exist. A failed renderer leaves a failure record,
    but no final.wav. This is artifact assembly, not a privacy/clinical verdict.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict): raise TurnContractError('Manifest must be a JSON object.')
    raw_path = manifest.get('converted_audio')
    if not isinstance(raw_path, str) or not raw_path: raise TurnContractError('converted_audio is required.')
    base_path = (manifest_path.parent / raw_path).resolve()
    base = read_pcm(base_path)
    turns = validate_turns(manifest, base)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        'schema_version': 'aegisvox.reconstruction.v1', 'status': 'running',
        'input_sha256': digest(base_path.read_bytes()), 'input_pcm_sha256': digest(base.data),
        'sample_rate': base.rate, 'input_frames': base.frames,
        'renderer': renderer.name, 'edits': [],
        'semantic_validation': 'not_performed', 'fallback': 'disabled',
        'privacy_or_clinical_guarantee': False,
    }
    try:
        cursor, frames, pieces = 0, 0, []
        for index, turn in enumerate(turns):
            if 'target_transcript' not in turn: continue
            start, end = turn['start_frame'], turn['end_frame']
            context = base.slice(cursor, start)
            pieces.append(context.data); frames += context.frames
            folder = output_dir / 'turns' / f'{index:04d}'
            reference, edited = folder / 'reference.wav', folder / 'edited.wav'
            write_pcm(reference, base.slice(start, end))
            request = {k: v for k,v in turn.items() if k not in ('start_frame','end_frame')}
            request.update({'sample_rate': base.rate, 'channels': base.channels, 'sample_width': base.width})
            metadata = renderer.render(request, reference, edited)
            speech = read_pcm(edited)
            if (speech.rate,speech.channels,speech.width) != (base.rate,base.channels,base.width):
                raise TurnContractError('Renderer PCM format must match converted audio; resample explicitly in the adapter.')
            if not any(speech.data): raise TurnContractError('Renderer returned digital silence.')
            entry = {'turn_id': turn['turn_id'], 'speaker_id': turn['speaker_id'],
                     'source_interval_frames': [start,end], 'output_interval_frames': [frames, frames+speech.frames],
                     'reference_sha256': digest(reference.read_bytes()), 'edited_file_sha256': digest(edited.read_bytes()),
                     'inserted_pcm_sha256': digest(speech.data), 'edited_file': str(edited.relative_to(output_dir)),
                     'renderer_metadata': metadata}
            pieces.append(speech.data); frames += speech.frames; cursor = end
            report['edits'].append(entry)
        pieces.append(base.slice(cursor,base.frames).data)
        final = PCM(base.rate,base.channels,base.width,b''.join(pieces))
        temp = output_dir / 'final.partial.wav'
        write_pcm(temp, final)
        check = read_pcm(temp)
        for item in report['edits']:
            a,b = item['output_interval_frames']
            item['sample_equivalent'] = digest(check.slice(a,b).data) == item['inserted_pcm_sha256']
            if not item['sample_equivalent']: raise TurnContractError('Inserted samples do not match the generated turn.')
        temp.rename(output_dir / 'final.wav')
        report.update({'status': 'assembled', 'output_frames': final.frames,
                       'output_sha256': digest((output_dir/'final.wav').read_bytes()),
                       'duration_ratio': final.frames/base.frames})
    except Exception as exc:
        report.update({'status': 'failed', 'error_type': type(exc).__name__})
        raise
    finally:
        (output_dir/'reconstruction.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
