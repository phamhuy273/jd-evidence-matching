from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "UP"
    assert data["service"] == "ai-engine"

def test_evaluate_async_flow():
    payload = {
        "job_id": "a0000000-0000-0000-0000-000000000001",
        "candidate_name": "Test Candidate",
        "github_repo_url": "https://github.com/test/repo",
        "commit_hash": "abcdef1234567890abcdef1234567890abcdef12"
    }
    response = client.post("/api/v1/matching/evaluate-async", json=payload)
    assert response.status_code == 202
    res_data = response.json()
    assert res_data["status"] == "ACCEPTED"
    assert "task_id" in res_data["data"]

    task_id = res_data["data"]["task_id"]

    # Poll status
    status_resp = client.get(f"/api/v1/matching/tasks/{task_id}/status")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["data"]["progress_status"] in ("PROCESSING", "COMPLETED")

def test_job_results_endpoint():
    job_id = "a0000000-0000-0000-0000-000000000001"
    response = client.get(f"/api/v1/matching/jobs/{job_id}/results")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "SUCCESS"
    assert "ranking" in data["data"]
