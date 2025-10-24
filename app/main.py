from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.messaging.consumer import RabbitMQConsumerManager
from app.messaging.producer import producer
from app.services.jobs.daily_dept_painpoints import run_daily_department_painpoints_job

consumer_manager = RabbitMQConsumerManager()
scheduler: AsyncIOScheduler | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        await connect_to_mongo()
        await producer.connect()
        await consumer_manager.start()
        global scheduler
        scheduler = AsyncIOScheduler(timezone="UTC")
        cron_expression = settings.PAINPOINT_CRON_EXPRESSION
        trigger = CronTrigger.from_crontab(cron_expression, timezone="UTC")
        scheduler.add_job(
            run_daily_department_painpoints_job,
            trigger=trigger,
            id="daily_department_painpoints",
            replace_existing=True,
        )
        scheduler.start()
        logger.info("Scheduled pain point aggregation job with cron '%s'", cron_expression)
    except Exception as exc:
        logger.error("Error during startup: %s", exc, exc_info=True)
        raise

    yield

    try:
        await close_mongo_connection()
        await producer.close()
        await consumer_manager.stop()
        if scheduler:
            scheduler.shutdown(wait=False)
    except Exception as exc:
        logger.error("Error during shutdown: %s", exc, exc_info=True)


app = FastAPI(
    title=settings.SERVICE_NAME,
    version=settings.VERSION,
    description="FastAPI application with MongoDB integration",
    lifespan=lifespan,
    root_path="/cognitive-service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_user_info(request: Request, call_next):
    tenant_id = request.headers.get("tenant-id") or "default"
    user_id = request.headers.get("user-id") or "68c7b828f3f92a7f537b536d"

    # If enforcing auth later, uncomment the guard below.
    # if not tenant_id or not user_id:
    #     raise HTTPException(status_code=403, detail="Unauthorized")

    request.state.user = {"tenant_id": tenant_id, "user_id": user_id}
    response = await call_next(request)
    return response


app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})


@app.get("/")
async def root():
    return {"message": "Cognitive Service"}


@app.get("/health")
async def health():
    return {
        "status": "success",
        "message": "Server is running!",
        "timestamp": datetime.now().isoformat(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
