"""FastAPI application entry point."""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.health import router as health_router
from backend.api.datasets import router as datasets_router
from backend.api.models import router as models_router
from backend.api.experiments import router as experiments_router
from backend.core.config import settings
from backend.core.experiment_manager import experiment_manager
from backend.core.event_emitter import event_emitter


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await experiment_manager.start_cleanup()
    yield
    # Shutdown
    await experiment_manager.stop_cleanup()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Algorithm Visualizer ML API",
        description="Backend API for ML algorithm visualization and experimentation",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routes
    app.include_router(health_router, prefix="/api", tags=["health"])
    app.include_router(datasets_router, prefix="/api/datasets", tags=["datasets"])
    app.include_router(models_router, prefix="/api/models", tags=["models"])
    app.include_router(experiments_router, prefix="/api/experiments", tags=["experiments"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
    )