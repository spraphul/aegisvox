# Security and responsible use

This research framework does not certify anonymity, clinical safety, HIPAA compliance,
or readiness for public release of recordings. Use only authorized recordings and voice
references; respect consent, licensing, retention, and institutional requirements.

Treat manifests, transcripts, references, generated audio, and logs as sensitive.
The API and command adapters are trusted-operator tools. They accept filesystem paths
and can execute configured commands; do not expose them directly to untrusted clients.
Use authentication, authorization, path restrictions, TLS, and network isolation.

Never commit `.env` files, keys, weights, clinical recordings, or raw reviewer exports.
Do not attach sensitive data to GitHub issues. Use GitHub private vulnerability reporting
when available; otherwise ask for a private contact without disclosing the exploit or
sensitive data in public. There is no promised response-time SLA.
