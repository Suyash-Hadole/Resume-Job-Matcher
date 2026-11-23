"""Tests for the external backend auth wrapper."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_backend_requires_shared_token_for_inference_routes():
    root = Path(__file__).resolve().parents[1]
    code = """
import json
import nlp_api
client = nlp_api.app.test_client()
print(json.dumps({
    'no_token': client.get('/api/jobs').status_code,
    'with_token': client.get('/api/jobs', headers={'X-NLP-API-Token': 'local-test-token'}).status_code,
    'health_public': client.get('/api/health').status_code,
}))
"""
    env = os.environ.copy()
    env["NLP_API_TOKEN"] = "local-test-token"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout.strip()) == {
        "no_token": 401,
        "with_token": 200,
        "health_public": 200,
    }
