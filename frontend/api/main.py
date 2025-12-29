from fastapi import FastAPI
from frontend.api.router import router

app = FastAPI(
    title="Recommendation System API",
    version="1.0",
    description="API de scoring pour système de recommandation"
)

app.include_router(router)


@app.get("/")
def health_check():
    return {"status": "API is running"}
