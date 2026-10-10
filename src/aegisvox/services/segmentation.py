"""Segmentation service wrapper."""

from __future__ import annotations

from dataclasses import dataclass

from aegisvox.schema import AegisVoxContractError
from aegisvox.segmentation import clean_diarization_turns, extract_wav_segment, load_diarization_turns, segments_payload
from aegisvox.storage import LocalArtifactStore


@dataclass
class SegmentationService:
    artifact_store: LocalArtifactStore

    def clean_to_artifact(
        self,
        diarization_uri: str,
        output_segments_uri: str,
        *,
        source_audio_uri: str | None = None,
        audio_duration_sec: float | None = None,
        output_segment_prefix_uri: str | None = None,
        expected_speaker_count: int | None = None,
        exclusive_speaker_timeline: bool = False,
    ) -> dict:
        diarization_path = self.artifact_store.local_path(diarization_uri)
        turns = load_diarization_turns(diarization_path)
        segments = clean_diarization_turns(
            turns,
            source_audio_uri=source_audio_uri,
            audio_duration_sec=audio_duration_sec,
            expected_speaker_count=expected_speaker_count,
            exclusive_speaker_timeline=exclusive_speaker_timeline,
        )
        payload = segments_payload(segments)
        if output_segment_prefix_uri is not None:
            if source_audio_uri is None:
                raise AegisVoxContractError("source_audio_uri is required when output_segment_prefix_uri is provided")
            prefix = output_segment_prefix_uri.rstrip("/")
            for row in payload["segments"]:
                segment_uri = f"{prefix}/{row['segment_id']}.source.wav"
                extract_wav_segment(
                    source_audio_uri,
                    segment_uri,
                    float(row["padded_start_sec"]),
                    float(row["padded_end_sec"]),
                    store=self.artifact_store,
                )
                row["segment_audio_uri"] = segment_uri
        self.artifact_store.write_json(output_segments_uri, payload)
        return payload
