from fastapi import FastAPI
from api.routes.health import router as health_router
from api.routes.ingestion import router as ingestion_router


app = FastAPI(title="Recommendation System API", version="0.1.0")
app.include_router(health_router, prefix="/api")
app.include_router(ingestion_router, prefix="/api")