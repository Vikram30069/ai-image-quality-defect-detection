"""
FastAPI Application Entrypoint
------------------------------
Initializes the FastAPI server, configures CORS, mounts routers,
serves uploaded image artifacts and static frontend dashboard files.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse

from backend.app.config import settings
from backend.app.database.connection import init_database
from backend.app.routes import inspection, history, analytics, export


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown event management."""
    # Initialize SQLite database schema
    init_database()
    print(f"[{settings.PROJECT_NAME}] System initialized. Database connected.")
    yield
    print(f"[{settings.PROJECT_NAME}] System shutting down.")


# Create FastAPI instance
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Full-stack AI-powered image quality assessment and industrial defect inspection API.",
    lifespan=lifespan
)

# Configure Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(inspection.router, prefix=settings.API_V1_PREFIX)
app.include_router(history.router, prefix=settings.API_V1_PREFIX)
app.include_router(analytics.router, prefix=settings.API_V1_PREFIX)
app.include_router(export.router, prefix=settings.API_V1_PREFIX)

# Resolve Base and Frontend Directories Robustly
def find_directory(name: str) -> Path:
    candidates = [
        settings.BASE_DIR / name,
        Path.cwd() / name,
        Path(f"/app/{name}"),
        Path(__file__).resolve().parent.parent.parent / name
    ]
    for c in candidates:
        if c.exists():
            return c
    # Fallback and create if needed
    target = settings.BASE_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    return target


UPLOAD_DIR = find_directory("uploads")
SAMPLE_IMAGES_DIR = find_directory("sample_images")
DATASET_DIR = find_directory("dataset")
FRONTEND_DIR = find_directory("frontend")

# Mount Static Directories
if UPLOAD_DIR.exists():
    app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

if SAMPLE_IMAGES_DIR.exists():
    app.mount("/sample_images", StaticFiles(directory=str(SAMPLE_IMAGES_DIR)), name="sample_images")

if DATASET_DIR.exists():
    app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")

if (FRONTEND_DIR / "css").exists():
    app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")

if (FRONTEND_DIR / "js").exists():
    app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")

if (FRONTEND_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")


# Serve Frontend Web Dashboard
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
async def serve_frontend_dashboard():
    """Serves the Single Page Web Dashboard."""
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse(
        content=f"<html><body><h2>{settings.PROJECT_NAME}</h2><p>API is live. Access <a href='/docs'>/docs</a> for Swagger UI.</p></body></html>"
    )
