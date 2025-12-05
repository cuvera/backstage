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
# from scripts.test_meeting_prep import quick_test
from app.services.meeting_analysis_orchestrator import MeetingAnalysisOrchestrator
from app.core.openai_client import llm_client, httpx_client
from app.messaging.producers.email_notification_producer import send_email_notification

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

        app.state.llm_client = llm_client
        app.state.httpx_client = httpx_client

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


import json
@app.get("/")
async def root():
    # print("Quick test")
    # await quick_test()
    orc = MeetingAnalysisOrchestrator()
    
    # payload = {
    #     "_id": "69034a94bd25edb427558ce0",
    #     "tenantId": "689ddc0411e4209395942bee",
    #     "platform": "google",
    #     "bucket": "cuverademo",
    #     "fileUrl": "689ddc0411e4209395942bee/google/6isa8pcg77vet5m2qkgpltc0t5/Cuvera Bot-2025-10-30T11:29:54.532Z.wav"
    # }

    # payload = {
    #     "_id": "691300ec64082d8a17bf85bb",
    #     "tenantId": "689ddc0411e4209395942bee",
    #     "platform": "google",
    #     "bucket": "cuverademo",
    #     "fileUrl": "689ddc0411e4209395942bee/google/3leetlec1vouor0iis1h7j0ig1/Cuvera Bot-2025-11-11T09:36:47.739Z.wav"
    # }
    payload = {
        "_id": "691347f002dabd405f1fbc1d",
        "tenantId": "689ddc0411e4209395942bee",
        "platform": "google",
        "bucket": "cuverademo",
        "fileUrl": "689ddc0411e4209395942bee/google/7uf3qub9n3kllnl2jqvq2u99e3_20251111T142900Z/Cuvera Bot-2025-11-11T14:30:28.488Z.wav"
    }

    payload_dict = json.loads(json.dumps(payload))

    await orc.analyze_meeting(payload_dict)

    # send_email_notification(
    #     attendees=[{"name": "Ashwini", "email": "ashwini@rootent.com"}],
    #     organizer={"name": "Participant", "email": "gulshan@rootent.com"},
    #     title="Test Meeting",
    #     startTime="2025-12-05T10:00:00Z",
    #     endTime="2025-12-05T11:00:00Z",
    #     duration="1 Hour 13 Minutes",
    #     summary="Test meeting analysis",
    #     redirectUrl="https://demo.cuvera.ai/meeting/online/692abe3447856a0242fca4a9",
    #     noOfKeyPoints=3,
    #     noOfActionItems=2,
    #     tenant_id="689ddc0411e4209395942bee"
    # )

    # payload = {
    #     "_id": "68c7b828f3f92a7f537b536d",
    #     "tenantId": "689ddc0411e4209395942bee",
    #     "platform": "google",
    #     "bucket": "cuverademo",
    #     "fileUrl": "/689ddc0411e4209395942bee/google/6isa8pcg77vet5m2qkgpltc0t5/Cuvera Bot-2025-10-30T11:29:54.532Z.wav"
    # })

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