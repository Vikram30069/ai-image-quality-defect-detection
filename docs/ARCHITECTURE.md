# System Architecture & Technical Design

## 1. High-Level Architectural Blueprint

The **AI-Powered Image Quality & Visual Defect Detection System** follows a decoupled, modular 3-tier architecture adhering to Clean Architecture principles:

```mermaid
graph TD
    User([User / Quality Inspector]) -->|Uploads Image / Adjusts View| UI[Frontend: Vanilla HTML5 + CSS3 + JS]
    
    subgraph Web & Presentation Layer
        UI -->|Fetch API: multipart/form-data| FastAPI[FastAPI REST API: main.py]
        UI --> CanvasViewer[HTML5 Canvas: Zoom, Pan, Defect Overlay, Heatmap]
        UI --> ChartJS[Chart.js: Radar Quality & Analytics Trends]
    end

    subgraph Core AI & Vision Processing Pipeline
        FastAPI --> Preproc[Preprocessor: Multi-Color Space & Downscale]
        Preproc --> Extractor[Feature Extractor: 7 Canonical Features]
        
        Extractor -->|Feature Vector| MLModel[ML Quality Classifier: Random Forest]
        Extractor -->|Statistical Metrics| QualityEngine[Quality Analyzer: Calibrated Rules]
        Preproc --> DefectEngine[Defect Detector: Morphological Anomaly & Contours]
        
        MLModel --> ReportGen[Decision & Report Engine]
        QualityEngine --> ReportGen
        DefectEngine --> ReportGen
    end

    subgraph Persistence Layer
        ReportGen --> Repo[Database Repository]
        Repo --> SQLite[(SQLite Database: inspection_system.db)]
        SQLite --> Table1[inspections]
        SQLite --> Table2[quality_metrics]
        SQLite --> Table3[defects]
        SQLite --> Table4[system_logs]
    end

    ReportGen -->|JSON Diagnostic Payload + Image URLs| UI
```

---

## 2. Component Responsibility Matrix

| Component Layer | Primary File | Responsibility |
| :--- | :--- | :--- |
| **Presentation (UI)** | `frontend/index.html`, `js/app.js` | Single-page interface, interactive dual-canvas zoom/pan, defect toggles, Chart.js radar & trend visualizers. |
| **API & Gateway** | `backend/app/main.py`, `routes/` | HTTP request routing, multipart file validation, CORS, static file serving, error handling. |
| **Preprocessing** | `backend/app/core/preprocessor.py` | Safe image decoding, dimension bounding, color space transformations (BGR, Grayscale, HSV, LAB). |
| **Feature Engineering** | `backend/app/core/feature_extractor.py` | Extracts 7 canonical interpretable features: Sharpness, Brightness, Contrast, Noise, Entropy, Saturation, Edge Density. |
| **AI / ML Classification**| `backend/app/ml/model_loader.py` | Scikit-Learn `RandomForestClassifier` inference on numerical feature vector (`ACCEPTABLE`, `DEGRADED`, `DEFECTIVE`). |
| **Defect Localization** | `backend/app/core/defect_detector.py` | Morphological Black-Hat/Top-Hat, adaptive thresholding, contour extraction, bounding boxes, false-color density heatmaps. |
| **Decision Engine** | `backend/app/core/report_generator.py` | Fuses ML class probabilities + CV heuristic scores + defect penalties into unified score (0-100) and plain-English explanation. |
| **Data Persistence** | `backend/app/database/` | SQLite + SQLAlchemy ORM for audit tracking, historical inspection queries, and analytics aggregation. |

---

## 3. End-to-End Image Processing Workflow

```mermaid
sequenceDiagram
    autonumber
    actor Client as User / Browser
    participant API as FastAPI Gateway
    participant Pre as Preprocessor
    participant Ext as Feature Extractor
    participant ML as Random Forest Model
    participant CV as Defect Detector
    participant Rep as Report Engine
    participant DB as SQLite Database

    Client->>API: POST /api/inspect (multipart file)
    API->>API: Validate extension & size limit
    API->>Pre: validate_and_decode(bytes)
    Pre-->>API: PreprocessedImage (BGR, Gray, HSV, LAB)
    
    par Quality Analysis
        API->>Ext: extract_features(prep)
        Ext-->>API: 7-feature dict & vector
        API->>ML: predict(vector)
        ML-->>API: MLPredictionResult (label, confidence, probas)
    and Defect Detection
        API->>CV: detect(prep)
        CV-->>API: DefectDetectionResult (bboxes, mask, heatmap)
    end

    API->>Rep: generate_report(features, quality, defects, ml)
    Rep-->>API: InspectionReport (score, label, explanation)
    API->>DB: save_inspection(report, paths)
    DB-->>API: db_record (inspection_id)
    API-->>Client: 200 OK (JSON Diagnostic Payload)
    Client->>Client: Render Canvas Bboxes, Heatmap & Radar Chart
```

---

## 4. Key Architectural Design Decisions

1. **Separation of Feature Extraction from Model Inference**:
   - `FeatureExtractor` is a pure function that runs identically during offline dataset generation (`ml/extract_features.py`) and live API inference. This eliminates **training/serving skew**.
2. **Hybrid Decision Fusion**:
   - Machine learning is trained on global image quality features.
   - Classical Computer Vision is used for localized anomaly segmentations.
   - The `ReportGenerator` arbitrates between both components, ensuring high explainability without black-box opacity.
3. **Vanilla Frontend**:
   - Built using standard HTML5 Canvas and native JavaScript Fetch API without heavy framework overhead, ensuring fast loading and simple interview walkthroughs.
