from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from src.routers.recommendation import router as rec_router

app = FastAPI(title="SHL Assessment RAG")
app.include_router(rec_router, prefix="/api")

# serve frontend (index.html at root)
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
