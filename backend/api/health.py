"""Health check endpoint."""
from fastapi import APIRouter
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "healthy"
    service: str = "algorithm-visualizer-ml-api"
    version: str = "1.0.0"


router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse()