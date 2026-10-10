"""Manifest URI helpers."""

from __future__ import annotations


def run_prefix(run_id: str, bucket: str = "aegisvox") -> str:
    return f"s3://{bucket}/runs/{run_id}"


def manifest_uri(run_id: str, bucket: str = "aegisvox") -> str:
    return f"{run_prefix(run_id, bucket=bucket)}/manifest.json"


def artifact_uri(run_id: str, name: str, bucket: str = "aegisvox") -> str:
    return f"{run_prefix(run_id, bucket=bucket)}/{name}"
