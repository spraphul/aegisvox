-- AegisVox metadata schema. SQLite-compatible Phase 1 form; Postgres migrations
-- should preserve these table/field names.
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  input_audio_uri TEXT NOT NULL,
  privacy_mode TEXT NOT NULL,
  voice_pool_id TEXT NOT NULL,
  language_hint TEXT,
  enable_lexical_deid INTEGER NOT NULL,
  enable_speech_editing INTEGER NOT NULL,
  overlap_mode TEXT NOT NULL,
  manifest_uri TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  failure_reason TEXT,
  request_json TEXT NOT NULL,
  service_versions_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  event_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  service TEXT NOT NULL,
  emitted_at TEXT NOT NULL,
  payload_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT NOT NULL,
  name TEXT NOT NULL,
  uri TEXT NOT NULL,
  artifact_type TEXT NOT NULL,
  created_at TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  PRIMARY KEY(run_id, name)
);

CREATE TABLE IF NOT EXISTS jobs (
  job_id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  service TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  input_artifacts_json TEXT NOT NULL DEFAULT '[]',
  output_artifacts_json TEXT NOT NULL DEFAULT '[]',
  error_json TEXT,
  worker_id TEXT,
  claimed_at TEXT,
  heartbeat_at TEXT,
  attempt INTEGER NOT NULL DEFAULT 0,
  timeout_seconds INTEGER
);
