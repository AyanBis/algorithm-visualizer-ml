"""API routes."""
from fastapi import APIRouter
from .health import router as health_router
from .datasets import router as datasets_router
from .models import router as models_router
from .experiments import router as experiments_router
from .comparison import router as comparison_router


api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_router.include_router(models_router, prefix="/models", tags=["models"])
api_router.include_router(experiments_router, prefix="/experiments", tags=["experiments"])
api_router.include_router(comparison_router, prefix="/comparison", tags=["comparison"])