"""
FastAPI Application Entrypoint
------------------------------
Initializes the FastAPI server, configures CORS, mounts routers,
serves uploaded image artifacts and static frontend dashboard files.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

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

# Mount Uploads directory to serve original and annotated inspection images
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Mount sample_images and dataset directories
SAMPLE_IMAGES_DIR = settings.BASE_DIR / "sample_images"
if SAMPLE_IMAGES_DIR.exists():
    app.mount("/sample_images", StaticFiles(directory=str(SAMPLE_IMAGES_DIR)), name="sample_images")

DATASET_DIR = settings.BASE_DIR / "dataset"
if DATASET_DIR.exists():
    app.mount("/dataset", StaticFiles(directory=str(DATASET_DIR)), name="dataset")

# Frontend directory paths
FRONTEND_DIR = settings.BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
    # Mount frontend static folders
    if (FRONTEND_DIR / "css").exists():
        app.mount("/css", StaticFiles(directory=str(FRONTEND_DIR / "css")), name="css")
    if (FRONTEND_DIR / "js").exists():
        app.mount("/js", StaticFiles(directory=str(FRONTEND_DIR / "js")), name="js")
    if (FRONTEND_DIR / "assets").exists():
        app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR / "assets")), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_frontend_dashboard():
        """Serves the Single Page Web Dashboard."""
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"message": f"{settings.PROJECT_NAME} API is running. Access /docs for Swagger UI."}
