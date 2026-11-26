from contextlib import asynccontextmanager

from fastapi import FastAPI

from docqa.api.routes import router
from docqa.api.services import Services, build_services

#Create_app initializes the FastAPI application with optional services. If no services are provided, it builds the real ones at startup. It also sets up the lifespan context manager to manage the application's state and includes the API router.
def create_app(services: Services | None = None) -> FastAPI:
    """Pass fake services in tests; otherwise the real ones are built at startup."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not hasattr(app.state, "services"):
            app.state.services = build_services()
        yield

    app = FastAPI(title="Doc Q&A", version="0.1.0", lifespan=lifespan)
    if services is not None:
        app.state.services = services
    app.include_router(router)
    return app


app = create_app()