from fastapi import APIRouter
from pydantic import BaseModel
from data_ingestion.ingestion import run_ingestion_with_summary


router = APIRouter(tags=["ingestion"])


class IngestionRequest(BaseModel):
    raw_subdir: str | None = None


@router.post("/ingestion/run")  # summary = "load database")
def run_ingestion(req: IngestionRequest):
    summary = run_ingestion_with_summary()
    return {"ok": True, "summary": summary}
