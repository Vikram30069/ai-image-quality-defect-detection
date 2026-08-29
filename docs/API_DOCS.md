# REST API Reference Documentation

Base URL: `http://127.0.0.1:8000/api`  
Interactive Swagger UI: `http://127.0.0.1:8000/docs`  
ReDoc Documentation: `http://127.0.0.1:8000/redoc`

---

## 1. Inspect Image

### `POST /api/inspect`
Inspects an uploaded image file, computes 7 quality features, evaluates machine learning class probabilities, localizes surface defects, and saves the inspection record.

* **Content-Type**: `multipart/form-data`
* **Request Body**:
  - `file`: Binary image file (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tiff`)

* **Curl Example**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/api/inspect" \
       -F "file=@sample_images/sample_scratched.png"
  ```

* **Response (200 OK)**:
  ```json
  {
    "inspection_id": 1,
    "filename": "sample_scratched.png",
    "quality_score": 58.4,
    "quality_label": "DEFECTIVE",
    "confidence": 0.88,
    "primary_issue": "Scratch Defect",
    "explanation": "Detected 2 surface anomalies, notably a SCRATCH (high severity, 420.0px²). Quality analysis identified Image exhibits good sharpness, balanced exposure, and low noise. Machine learning model classified image state as DEFECTIVE (Confidence: 85.0%).",
    "issues": [
      {
        "category": "surface_defect",
        "type": "scratch",
        "severity": "high",
        "confidence": 0.95,
        "description": "SCRATCH anomaly localized at (40, 180) with area 420.0px².",
        "bounding_box": {
          "x": 40,
          "y": 180,
          "width": 280,
          "height": 22
        }
      }
    ],
    "statistics": {
      "sharpness": 1250.4,
      "brightness": 118.2,
      "contrast": 48.5,
      "noise": 2.1,
      "entropy": 6.8,
      "saturation": 0.42,
      "edge_density": 0.045,
      "defect_count": 2,
      "defect_density_pct": 0.85
    },
    "sub_scores": {
      "sharpness": 100.0,
      "exposure": 92.3,
      "contrast": 97.0,
      "noise": 86.0,
      "entropy": 90.7
    },
    "defect_summary": {
      "total_defects": 2,
      "defect_density_pct": 0.85,
      "defect_list": [...]
    },
    "ml_result": {
      "predicted_label": "DEFECTIVE",
      "confidence": 0.85,
      "class_probabilities": {
        "ACCEPTABLE": 0.05,
        "DEGRADED": 0.10,
        "DEFECTIVE": 0.85
      },
      "model_version": "1.0.0",
      "is_fallback": false
    },
    "processing_time_ms": 142.5,
    "image_url": "/uploads/uuid_orig.png",
    "annotated_url": "/uploads/uuid_annot.png",
    "heatmap_url": "/uploads/uuid_heat.png",
    "created_at": "2026-08-29T10:15:30Z"
  }
  ```

---

## 2. History & Logs

### `GET /api/history`
Returns paginated historical inspection logs sorted by date descending.

* **Query Parameters**:
  - `limit` (int, default: 50): Number of items per page.
  - `offset` (int, default: 0): Pagination offset.
  - `label` (string, optional): Filter by `ACCEPTABLE`, `DEGRADED`, or `DEFECTIVE`.

### `GET /api/history/{id}`
Returns complete details, bounding boxes, and metrics for a specific inspection ID.

### `DELETE /api/history/{id}`
Deletes an inspection record and its cascaded defects and metrics.

---

## 3. Analytics & Health

### `GET /api/analytics`
Returns aggregate system metrics (Total Inspections, Pass Rate %, Average Quality Score, Defect Breakdown, Recent Score Trends).

### `GET /api/health`
Returns health status, active model version, and database connectivity.

---

## 4. Export Endpoints

### `GET /api/export/csv`
Downloads all inspection records and 7 quality metrics as an RFC 4180 compliant CSV file.

### `GET /api/export/json`
Downloads all historical inspection records as a formatted JSON document.
