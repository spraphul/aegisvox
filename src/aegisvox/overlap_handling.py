"""Overlap handling service contracts.

MVP mode is intentionally conservative: detect overlap regions, optionally
extract the overlapped audio spans, and write an audit report. Production
`separate_convert_remix` requires a real separation/conversion backend and fails
loudly unless configured. No synthetic stems or fake converted overlap audio are
created.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import shutil
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .schema import AegisVoxContractError, OverlapMode, validate_artifact_uri
from .segmentation import extract_wav_segment
from .storage import LocalArtifactStore


class OverlapHandlingError(RuntimeError):
    """Raised when overlap handling cannot complete."""


class OverlapHandlingSetupError(RuntimeError):
    """Raised when production overlap backends are missing or unusable."""


@dataclass(frozen=True)
class OverlapRegion:
    overlap_id: str
    source_segment_ids: list[str]
    speaker_ids: list[str]
    start_sec: float
    end_sec: float
    padded_start_sec: float
    padded_end_sec: float
    source_audio_uri: str | None = None
    overlap_audio_uri: str | None = None
    status: str = "detected"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.overlap_id:
            raise AegisVoxContractError("overlap_id is required")
        if self.end_sec <= self.start_sec:
            raise AegisVoxContractError("overlap end_sec must be greater than start_sec")
        if self.padded_end_sec <= self.padded_start_sec:
            raise AegisVoxContractError("overlap padded_end_sec must be greater than padded_start_sec")
        if self.source_audio_uri is not None:
            validate_artifact_uri(self.source_audio_uri)
        if self.overlap_audio_uri is not None:
            validate_artifact_uri(self.overlap_audio_uri)

    @property
    def duration_sec(self) -> float:
        return round(self.end_sec - self.start_sec, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overlap_id": self.overlap_id,
            "source_segment_ids": list(self.source_segment_ids),
            "speaker_ids": list(self.speaker_ids),
            "start_sec": round(self.start_sec, 3),
            "end_sec": round(self.end_sec, 3),
            "padded_start_sec": round(self.padded_start_sec, 3),
            "padded_end_sec": round(self.padded_end_sec, 3),
            "duration_sec": self.duration_sec,
            "source_audio_uri": self.source_audio_uri,
            "overlap_audio_uri": self.overlap_audio_uri,
            "status": self.status,
            "metadata": dict(self.metadata),
        }


def _request_json(url: str, *, payload: dict[str, Any] | None = None, timeout: int = 20) -> dict[str, Any]:
    data = None
    headers: dict[str, str] = {}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator-configured local/internal endpoint
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise OverlapHandlingSetupError(f"audio-separator persistent service request failed: {exc}") from exc
    if not isinstance(decoded, dict):
        raise OverlapHandlingSetupError("audio-separator persistent service response must be a JSON object")
    return decoded


def _wav_duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate())


def _pad_wav_to_min_duration(source: Path, output: Path, *, min_duration_sec: float) -> tuple[Path, float, bool]:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(reader.getnframes())
        duration = params.nframes / float(params.framerate)
    if duration >= min_duration_sec:
        return source, duration, False
    output.parent.mkdir(parents=True, exist_ok=True)
    bytes_per_frame = params.nchannels * params.sampwidth
    target_frames = int(round(min_duration_sec * params.framerate))
    pad_frames = max(0, target_frames - params.nframes)
    with wave.open(str(output), "wb") as writer:
        writer.setparams(params._replace(nframes=0))
        writer.writeframes(frames)
        writer.writeframes(b"\x00" * pad_frames * bytes_per_frame)
    return output, duration, True


def _trim_wav_to_duration(source: Path, output: Path, *, duration_sec: float) -> None:
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        keep_frames = min(reader.getnframes(), max(0, int(round(duration_sec * params.framerate))))
        frames = reader.readframes(keep_frames)
    with wave.open(str(output), "wb") as writer:
        writer.setparams(params._replace(nframes=0))
        writer.writeframes(frames)


def _load_segments(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = payload.get("segments") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise AegisVoxContractError("segments.cleaned.json must contain a segments list")
    return [dict(row) for row in rows]


def _row_interval(row: dict[str, Any]) -> tuple[float, float]:
    start = row.get("start_sec")
    end = row.get("end_sec")
    if start is None or end is None:
        raise AegisVoxContractError(f"segment {row.get('segment_id')} missing start_sec/end_sec")
    start_f = float(start)
    end_f = float(end)
    if end_f <= start_f:
        raise AegisVoxContractError(f"segment {row.get('segment_id')} has invalid interval")
    return start_f, end_f


def _row_padded_interval(row: dict[str, Any], start: float, end: float) -> tuple[float, float]:
    padded_start = float(row.get("padded_start_sec") if row.get("padded_start_sec") is not None else start)
    padded_end = float(row.get("padded_end_sec") if row.get("padded_end_sec") is not None else end)
    return padded_start, padded_end


def detect_overlap_regions(segments: list[dict[str, Any]]) -> list[OverlapRegion]:
    """Detect connected overlap regions.

    Regions come from two evidence sources: temporal overlap between different
    diarized speakers, and explicit `overlap` flags emitted by the segmentation
    cleanup stage. The latter matters when a diarization backend flags uncertain
    or overlapped speech without assigning a distinct second speaker.
    """

    pair_regions: list[dict[str, Any]] = []
    for i, left in enumerate(segments):
        left_speaker = str(left.get("speaker_id") or "")
        left_id = str(left.get("segment_id") or "")
        if not left_speaker or not left_id:
            raise AegisVoxContractError("each segment row must include segment_id and speaker_id")
        left_start, left_end = _row_interval(left)
        left_pad_start, left_pad_end = _row_padded_interval(left, left_start, left_end)
        for right in segments[i + 1 :]:
            right_speaker = str(right.get("speaker_id") or "")
            right_id = str(right.get("segment_id") or "")
            if not right_speaker or not right_id:
                raise AegisVoxContractError("each segment row must include segment_id and speaker_id")
            if right_speaker == left_speaker:
                continue
            right_start, right_end = _row_interval(right)
            start = max(left_start, right_start)
            end = min(left_end, right_end)
            if end <= start:
                continue
            right_pad_start, right_pad_end = _row_padded_interval(right, right_start, right_end)
            pair_regions.append(
                {
                    "start_sec": start,
                    "end_sec": end,
                    "padded_start_sec": max(0.0, min(left_pad_start, right_pad_start)),
                    "padded_end_sec": max(left_pad_end, right_pad_end),
                    "speaker_ids": {left_speaker, right_speaker},
                    "source_segment_ids": {left_id, right_id},
                    "source_audio_uri": left.get("source_audio_uri") or right.get("source_audio_uri"),
                    "detection_sources": {"temporal_cross_speaker"},
                }
            )
    for row in segments:
        flags = row.get("flags") if isinstance(row.get("flags"), list) else []
        explicit_overlap = bool(row.get("overlap")) or "overlap" in {str(flag) for flag in flags}
        if not explicit_overlap:
            continue
        segment_id = str(row.get("segment_id") or "")
        speaker_id = str(row.get("speaker_id") or "")
        if not segment_id or not speaker_id:
            raise AegisVoxContractError("each explicit overlap segment row must include segment_id and speaker_id")
        start, end = _row_interval(row)
        padded_start, padded_end = _row_padded_interval(row, start, end)
        pair_regions.append(
            {
                "start_sec": start,
                "end_sec": end,
                "padded_start_sec": padded_start,
                "padded_end_sec": padded_end,
                "speaker_ids": {speaker_id},
                "source_segment_ids": {segment_id},
                "source_audio_uri": row.get("source_audio_uri"),
                "detection_sources": {"explicit_overlap_flag"},
            }
        )
    if not pair_regions:
        return []
    pair_regions.sort(key=lambda row: (row["start_sec"], row["end_sec"]))
    merged: list[dict[str, Any]] = []
    for row in pair_regions:
        if not merged or row["start_sec"] > merged[-1]["end_sec"]:
            merged.append(dict(row))
            continue
        cur = merged[-1]
        cur["end_sec"] = max(float(cur["end_sec"]), float(row["end_sec"]))
        cur["padded_start_sec"] = min(float(cur["padded_start_sec"]), float(row["padded_start_sec"]))
        cur["padded_end_sec"] = max(float(cur["padded_end_sec"]), float(row["padded_end_sec"]))
        cur["speaker_ids"] = set(cur["speaker_ids"]) | set(row["speaker_ids"])
        cur["source_segment_ids"] = set(cur["source_segment_ids"]) | set(row["source_segment_ids"])
        cur["detection_sources"] = set(cur.get("detection_sources", set())) | set(row.get("detection_sources", set()))
        if cur.get("source_audio_uri") is None:
            cur["source_audio_uri"] = row.get("source_audio_uri")
    regions: list[OverlapRegion] = []
    for idx, row in enumerate(merged, start=1):
        regions.append(
            OverlapRegion(
                overlap_id=f"overlap_{idx:06d}",
                source_segment_ids=sorted(row["source_segment_ids"]),
                speaker_ids=sorted(row["speaker_ids"]),
                start_sec=float(row["start_sec"]),
                end_sec=float(row["end_sec"]),
                padded_start_sec=float(row["padded_start_sec"]),
                padded_end_sec=float(row["padded_end_sec"]),
                source_audio_uri=row.get("source_audio_uri"),
                metadata={"detection_sources": sorted(row.get("detection_sources", []))},
            )
        )
    return regions



def _replace_region(region: OverlapRegion, *, status: str | None = None, metadata: dict[str, Any] | None = None) -> OverlapRegion:
    return OverlapRegion(
        overlap_id=region.overlap_id,
        source_segment_ids=list(region.source_segment_ids),
        speaker_ids=list(region.speaker_ids),
        start_sec=region.start_sec,
        end_sec=region.end_sec,
        padded_start_sec=region.padded_start_sec,
        padded_end_sec=region.padded_end_sec,
        source_audio_uri=region.source_audio_uri,
        overlap_audio_uri=region.overlap_audio_uri,
        status=status or region.status,
        metadata=dict(metadata if metadata is not None else region.metadata),
    )

def overlap_payload(regions: list[OverlapRegion], *, mode: OverlapMode | str) -> dict[str, Any]:
    mode = OverlapMode(mode)
    return {
        "overlap_mode": mode.value,
        "num_overlap_regions": len(regions),
        "total_overlap_duration_sec": round(sum(region.duration_sec for region in regions), 4),
        "regions": [region.to_dict() for region in regions],
    }


class OverlapSeparationBackend(Protocol):
    backend_name: str
    backend_version: str

    def preflight(self) -> dict[str, Any]: ...

    def separate(self, *, overlap_wav: Path, output_dir: Path, region: OverlapRegion, options: dict[str, Any]) -> dict[str, Any]: ...


@dataclass
class HTTPAudioSeparatorBackend:
    """Persistent BS-RoFormer/audio-separator service backend.

    The model lives in the long-running service (usually ORD port 8790). This
    backend submits a shared filesystem path, polls the job, copies returned
    stems, and writes the same ``stems_manifest.json`` consumed by the rest of
    AegisVox.
    """

    endpoint: str | None = None
    backend_name: str = "audio-separator-http"
    backend_version: str = "persistent-http-v1"

    def __post_init__(self) -> None:
        if self.endpoint is None:
            self.endpoint = os.environ.get("AEGISVOX_SEPARATOR_SERVICE_URL") or os.environ.get("BS_ROFORMER_SERVICE_URL")

    def preflight(self) -> dict[str, Any]:
        if not self.endpoint:
            raise OverlapHandlingSetupError("audio-separator persistent service is not configured. Set AEGISVOX_SEPARATOR_SERVICE_URL.")
        health = _request_json(f"{self.endpoint.rstrip('/')}/health", timeout=10)
        if health.get("ok") is not True:
            raise OverlapHandlingSetupError(f"audio-separator health check failed: {health}")
        self.backend_version = f"persistent-http-v1;model={health.get('default_model') or health.get('model') or 'unknown'}"
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version, "endpoint": self.endpoint, "health": health}

    def separate(self, *, overlap_wav: Path, output_dir: Path, region: OverlapRegion, options: dict[str, Any]) -> dict[str, Any]:
        if not overlap_wav.exists():
            raise AegisVoxContractError(f"overlap/source WAV not found: {overlap_wav}")
        self.preflight()
        output_dir.mkdir(parents=True, exist_ok=True)
        endpoint = str(self.endpoint).rstrip("/")
        min_duration_sec = float(options.get("min_separator_duration_sec") or os.environ.get("AEGISVOX_SEPARATOR_MIN_DURATION_SEC") or 10.0)
        separator_source, original_duration_sec, padded_for_separator = _pad_wav_to_min_duration(
            overlap_wav,
            output_dir / "separator_input_padded.wav",
            min_duration_sec=min_duration_sec,
        )
        submitted = _request_json(f"{endpoint}/jobs/path", payload={"input_path": str(separator_source)}, timeout=20)
        job_id = submitted.get("job_id")
        if not job_id:
            raise OverlapHandlingSetupError(f"audio-separator did not return job_id: {submitted}")
        timeout_sec = float(options.get("timeout_sec") or os.environ.get("AEGISVOX_SEPARATOR_TIMEOUT_SEC") or 1800)
        poll_interval_sec = float(options.get("poll_interval_sec") or os.environ.get("AEGISVOX_SEPARATOR_POLL_INTERVAL_SEC") or 2.0)
        deadline = time.monotonic() + timeout_sec
        last: dict[str, Any] = {}
        while time.monotonic() < deadline:
            last = _request_json(f"{endpoint}/jobs/{job_id}", timeout=20)
            status = last.get("status")
            if status == "succeeded":
                break
            if status == "failed":
                raise OverlapHandlingSetupError(f"audio-separator job failed: {last.get('error') or last}")
            time.sleep(poll_interval_sec)
        else:
            raise TimeoutError(f"audio-separator job {job_id} did not complete within {timeout_sec}s; last={last}")
        outputs = dict(last.get("outputs") or {})
        missing = [name for name in ("vocal", "background") if not outputs.get(name)]
        if missing:
            raise OverlapHandlingSetupError(f"audio-separator job {job_id} missing outputs {missing}; payload={last}")
        stems: list[dict[str, Any]] = []
        for name in ("vocal", "background"):
            source = Path(str(outputs[name]))
            if not source.exists() or source.stat().st_size <= 44:
                raise OverlapHandlingSetupError(f"audio-separator output missing/empty for {name}: {source}")
            dest = output_dir / f"{name}.wav"
            if padded_for_separator:
                _trim_wav_to_duration(source, dest, duration_sec=original_duration_sec)
            else:
                shutil.copy2(source, dest)
            stems.append({"stem": name, "path": str(dest), "size_bytes": dest.stat().st_size})
        manifest = {
            "backend": self.backend_name,
            "backend_version": self.backend_version,
            "service_endpoint": endpoint,
            "job_id": job_id,
            "source": str(overlap_wav),
            "separator_source": str(separator_source),
            "original_duration_sec": round(original_duration_sec, 4),
            "padded_for_separator": padded_for_separator,
            "min_separator_duration_sec": min_duration_sec,
            "status": "completed",
            "stems": stems,
            "claim_boundary": "persistent vocal/background separation service; downstream leakage and remix gates still required",
        }
        manifest_path = output_dir / "stems_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return {"stems_manifest_path": str(manifest_path), "job_id": str(job_id), "service_response_status": last.get("status")}


@dataclass
class CommandSeparationBackend:
    """Command adapter for a real overlap speech-separation backend.

    Configure `OVERLAP_SEPARATION_COMMAND` with placeholders `{source}` and
    `{output_dir}`. Optional placeholders: `{overlap_id}`.
    The command must write a non-empty `stems_manifest.json` in `output_dir`.
    """

    command_template: str | None = None
    backend_name: str = "command-separation"
    backend_version: str = "adapter-v1"

    def __post_init__(self) -> None:
        if self.command_template is None:
            self.command_template = os.environ.get("OVERLAP_SEPARATION_COMMAND")

    def preflight(self) -> dict[str, Any]:
        if not self.command_template:
            raise OverlapHandlingSetupError(
                "overlap production mode requires OVERLAP_SEPARATION_COMMAND with {source} and {output_dir} placeholders"
            )
        if "{source}" not in self.command_template or "{output_dir}" not in self.command_template:
            raise OverlapHandlingSetupError("OVERLAP_SEPARATION_COMMAND must include {source} and {output_dir}")
        return {"ok": True, "backend": self.backend_name, "backend_version": self.backend_version}

    def separate(self, *, overlap_wav: Path, output_dir: Path, region: OverlapRegion, options: dict[str, Any]) -> dict[str, Any]:
        self.preflight()
        if not overlap_wav.exists():
            raise AegisVoxContractError(f"overlap WAV not found: {overlap_wav}")
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.command_template.format(
            source=shlex.quote(str(overlap_wav)),
            output_dir=shlex.quote(str(output_dir)),
            overlap_id=shlex.quote(region.overlap_id),
        )
        completed = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=int(options.get("timeout_sec", 900)))
        if completed.returncode != 0:
            raise OverlapHandlingSetupError(
                f"overlap separation command failed with exit {completed.returncode}: {completed.stderr[-1000:]}"
            )
        manifest_path = output_dir / "stems_manifest.json"
        if not manifest_path.exists() or manifest_path.stat().st_size == 0:
            raise OverlapHandlingSetupError("overlap separation command did not create non-empty stems_manifest.json")
        return {
            "stems_manifest_path": str(manifest_path),
            "command_stdout_tail": completed.stdout[-1000:],
            "command_stderr_tail": completed.stderr[-1000:],
        }


@dataclass
class OverlapHandlingService:
    artifact_store: LocalArtifactStore
    separation_backend: OverlapSeparationBackend | None = None

    def handle_to_artifact(
        self,
        *,
        segments_uri: str,
        output_overlap_uri: str,
        mode: OverlapMode | str = OverlapMode.SKIP_AND_FLAG,
        output_prefix_uri: str | None = None,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        validate_artifact_uri(segments_uri)
        validate_artifact_uri(output_overlap_uri)
        mode = OverlapMode(mode)
        options = options or {}
        rows = _load_segments(self.artifact_store.local_path(segments_uri))
        regions = detect_overlap_regions(rows)
        prefix = output_prefix_uri.rstrip("/") if output_prefix_uri else output_overlap_uri.rsplit("/", 1)[0]
        for region in regions:
            if region.source_audio_uri:
                overlap_audio_uri = f"{prefix}/overlap_audio/{region.overlap_id}.source.wav"
                extract_wav_segment(
                    region.source_audio_uri,
                    overlap_audio_uri,
                    region.padded_start_sec,
                    region.padded_end_sec,
                    store=self.artifact_store,
                )
                object.__setattr__(region, "overlap_audio_uri", overlap_audio_uri)
        if mode is OverlapMode.SKIP_AND_FLAG:
            regions = [
                _replace_region(region, status="skipped_by_policy", metadata={"policy": "skip_and_flag"})
                for region in regions
            ]
            payload = overlap_payload(regions, mode=mode)
            payload["status"] = "completed_skip_and_flag"
            self.artifact_store.write_json(output_overlap_uri, payload)
            return payload

        if mode is OverlapMode.SEPARATE_CONVERT_REMIX:
            backend = self.separation_backend
            if backend is None:
                backend_name = os.environ.get("AEGISVOX_SOURCE_SEPARATION_BACKEND", "command").strip().lower()
                if backend_name in {"http", "audio_separator_http", "persistent", "persistent_http", "service"} or os.environ.get("AEGISVOX_SEPARATOR_SERVICE_URL") or os.environ.get("BS_ROFORMER_SERVICE_URL"):
                    backend = HTTPAudioSeparatorBackend()
                else:
                    backend = CommandSeparationBackend()
            backend.preflight()
            if not regions:
                payload = overlap_payload([], mode=mode)
                payload["status"] = "no_overlaps"
                self.artifact_store.write_json(output_overlap_uri, payload)
                return payload
            processed = []
            for region in regions:
                if not region.overlap_audio_uri:
                    raise OverlapHandlingError(f"overlap region {region.overlap_id} has no extracted audio to separate")
                output_dir_uri = f"{prefix}/separated_stems/{region.overlap_id}"
                metadata = backend.separate(
                    overlap_wav=self.artifact_store.local_path(region.overlap_audio_uri),
                    output_dir=self.artifact_store.local_path(f"{output_dir_uri}/"),
                    region=region,
                    options=options,
                )
                processed.append(
                    _replace_region(
                        region,
                        status="separated_pending_conversion",
                        metadata={**region.metadata, "backend": backend.backend_name, **metadata},
                    )
                )
            payload = overlap_payload(processed, mode=mode)
            payload["status"] = "separated_pending_conversion"
            stem_manifests: list[dict[str, Any]] = []
            for region in processed:
                manifest_path = region.metadata.get("stems_manifest_path")
                if not manifest_path:
                    continue
                path = Path(str(manifest_path))
                if path.exists() and path.stat().st_size > 0:
                    try:
                        manifest = json.loads(path.read_text())
                    except json.JSONDecodeError as exc:
                        raise OverlapHandlingSetupError(f"invalid stems_manifest JSON for {region.overlap_id}: {path}") from exc
                    if isinstance(manifest, dict):
                        stem_manifests.append(manifest)
            payload["stem_manifests"] = stem_manifests
            payload["num_stem_manifests"] = len(stem_manifests)
            payload["num_stems"] = sum(len(m.get("stems") or []) for m in stem_manifests)
            payload["claim_boundary"] = "separation boundary only; independent stem conversion/remix must be validated before production claims"
            self.artifact_store.write_json(output_overlap_uri, payload)
            return payload

        raise AegisVoxContractError(f"unsupported overlap mode: {mode}")
