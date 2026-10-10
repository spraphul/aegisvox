import pytest
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from aegisvox.api import create_app


def test_api_schema_and_missing_credentials(tmp_path):
    app=create_app(artifact_root=tmp_path/'artifacts',metadata_db_path=tmp_path/'meta.sqlite3',api_auth_required=True,api_keys=['test-only-not-a-secret'])
    with TestClient(app) as client:
        schema=client.get('/openapi.json')
        assert schema.status_code==200
        assert '/v1/runs' in schema.json()['paths']
        response=client.post('/v1/runs',json={'input_audio_uri':'file:///does-not-exist.wav'})
        assert response.status_code in (401,403)
