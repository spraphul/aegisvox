# Deployment and service boundaries

AegisVox is a research framework, not a production-certified clinical service.
Run inside a protected environment with authorized recordings and reference voices.

## Control-plane API

```bash
pip install -e '.[api]'
export AEGISVOX_ARTIFACT_ROOT=/secure/data/aegisvox/artifacts
export AEGISVOX_METADATA_DB=/secure/data/aegisvox/metadata.sqlite3
aegisvox serve --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/docs` for the actual generated API schema. This starts the
orchestrator, **not** any speech model. The SQLite metadata store and local artifact
store are useful on a single host. The local store maps `s3://bucket/key` identifiers
onto files; it does not connect to cloud S3. A Postgres metadata adapter is also supplied.

```bash
aegisvox enqueue /secure/data/conversation.wav \
  --root /secure/data/aegisvox/artifacts --voice-pool my-authorized-pool --language en
```

Enqueuing does not start inference. Use the same metadata database for API and workers.
The CLI places it at `<root>/metadata.sqlite3`; set the API variable to that path if
combining these commands. Service jobs require real backend configuration and artifacts.

```bash
export AEGISVOX_WORKER_COMMAND='aegisvox service-job'
aegisvox worker --root /secure/data/aegisvox/artifacts --service diarization
```

This claims one job. Run a supervised loop or process manager for persistent workers.
The generic command worker starts one subprocess per job; put the heavy model in an
HTTP service, or use an in-process resident adapter, to avoid repeated model loading.
The queue handles lifecycle/metadata, not automatic GPU placement or model provisioning.

## Configuration and boundaries

- `.env.example` lists names only. Load values through your process manager or secret
  manager. The CLI does not automatically source arbitrary shell or `.env` files.
- Explicitly select all model revisions, languages, replacement policy, and frontend
  settings. Do not assume the legacy service engine defaults reproduce the paper.
- Model commands execute trusted code. Never accept executable templates from clients.
- Filesystem/URI APIs are trusted-operator interfaces. Do not expose them directly to
  untrusted users or the public internet. Enforce authentication, authorization, TLS,
  allowed paths, retention, rate limits, and request-size limits in your deployment.
- Set model caches and temporary directories on a capacity-managed volume; avoid root
  disk exhaustion. Stop workers gracefully; keep their configs and artifact manifests.
- Use encrypted storage and separate keys for data, model services, and text providers.
- Console credential redaction is not proof that transcripts or PHI are removed from logs.

## Experiment records

Record every backend revision, prompt version, seed, GPU environment, error, retry,
and exact waveform stage. Keep failed attempts in the population. A successful process
exit and a completed metadata status do not certify lexical removal or clinical fidelity.
