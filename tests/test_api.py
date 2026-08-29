"""
Integration Tests for FastAPI REST Endpoints
--------------------------------------------
Validates all HTTP API routes using FastAPI's TestClient:
- Health check
- Image upload and inspection pipeline
- Invalid input validation (bad format, empty files)
- Historical records retrieval and pagination
- Analytics aggregation
- CSV and JSON report exports
"""

import io
import pytest
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.database.connection import init_database

# Initialize DB for tests
init_database()
client = TestClient(app)


def create_test_image_file(filename: str = "sample_test.png", is_valid: bool = True) -> tuple:
    """Helper to construct a mock multipart file upload tuple."""
    if not is_valid:
        return (filename, io.BytesIO(b"NOT_A_VALID_IMAGE"), "image/png")

    # Valid synthetic pattern
    img = np.full((200, 200, 3), 150, dtype=np.uint8)
    cv2.circle(img, (100, 100), 30, (30, 30, 200), -1)
    _, enc = cv2.imencode(".png", img)
    return (filename, io.BytesIO(enc.tobytes()), "image/png")


def test_health_endpoint():
    """Verifies that the /api/health endpoint returns 200 and healthy status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_loaded" in data
    assert data["database_connected"] is True


def test_inspect_valid_image():
    """Verifies that uploading a valid image returns a complete 200 inspection report."""
    file_tuple = create_test_image_file("inspection_test.png", is_valid=True)
    response = client.post(
        "/api/inspect",
        files={"file": file_tuple}
    )

    assert response.status_code == 200
    data = response.json()
    assert "inspection_id" in data
    assert "quality_score" in data
    assert 0.0 <= data["quality_score"] <= 100.0
    assert data["quality_label"] in ["ACCEPTABLE", "DEGRADED", "DEFECTIVE"]
    assert "statistics" in data
    assert "sharpness" in data["statistics"]
    assert "defect_summary" in data
    assert "image_url" in data
    assert data["processing_time_ms"] > 0


def test_inspect_unsupported_file_extension():
    """Verifies that uploading an unsupported file type returns 400 Bad Request."""
    fake_file = ("script.exe", io.BytesIO(b"binary_junk"), "application/octet-stream")
    response = client.post(
        "/api/inspect",
        files={"file": fake_file}
    )
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_inspect_corrupted_image():
    """Verifies that uploading corrupted image bytes returns 422 Unprocessable Entity."""
    corrupt_file = ("corrupt.png", io.BytesIO(b"garbage_not_an_image"), "image/png")
    response = client.post(
        "/api/inspect",
        files={"file": corrupt_file}
    )
    assert response.status_code == 422


def test_history_and_detail_flow():
    """Tests retrieving history, fetching a specific detail, and deleting the record."""
    # 1. Upload an image first
    file_tuple = create_test_image_file("history_flow.png", is_valid=True)
    post_res = client.post("/api/inspect", files={"file": file_tuple})
    assert post_res.status_code == 200
    created_id = post_res.json()["inspection_id"]

    # 2. Get history
    hist_res = client.get("/api/history?limit=10")
    assert hist_res.status_code == 200
    hist_data = hist_res.json()
    assert hist_data["total"] >= 1
    ids = [item["id"] for item in hist_data["items"]]
    assert created_id in ids

    # 3. Get specific detail
    detail_res = client.get(f"/api/history/{created_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["inspection_id"] == created_id

    # 4. Delete record
    del_res = client.delete(f"/api/history/{created_id}")
    assert del_res.status_code == 200

    # 5. Verify 404 after deletion
    detail_res_after = client.get(f"/api/history/{created_id}")
    assert detail_res_after.status_code == 404


def test_analytics_endpoint():
    """Verifies that the /api/analytics endpoint returns computed KPIs."""
    response = client.get("/api/analytics")
    assert response.status_code == 200
    data = response.json()
    assert "total_inspections" in data
    assert "pass_rate_pct" in data
    assert "class_distribution" in data
    assert "defect_distribution" in data


def test_export_csv_and_json():
    """Verifies CSV and JSON export endpoints."""
    csv_res = client.get("/api/export/csv")
    assert csv_res.status_code == 200
    assert csv_res.headers["content-type"].startswith("text/csv")
    assert "Inspection ID" in csv_res.text

    json_res = client.get("/api/export/json")
    assert json_res.status_code == 200
    assert json_res.headers["content-type"].startswith("application/json")
    assert isinstance(json_res.json(), list)
