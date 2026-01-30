from __future__ import annotations

from contextlib import asynccontextmanager
from dotenv import load_dotenv
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.messaging.consumer import RabbitMQConsumerManager
from app.messaging.producer import producer

load_dotenv()

consumer_manager = RabbitMQConsumerManager()

@asynccontextmanager
async def lifespan(_: FastAPI):
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        await connect_to_mongo()
        await producer.connect()
        await consumer_manager.start()
    except Exception as exc:
        logger.error("Error during startup: %s", exc, exc_info=True)
        raise

    yield

    try:
        await close_mongo_connection()
        await producer.close()
        await consumer_manager.stop()
    except Exception as exc:
        logger.error("Error during shutdown: %s", exc, exc_info=True)

app = FastAPI(
    title=settings.SERVICE_NAME,
    description="FastAPI application with MongoDB integration",
    lifespan=lifespan,
    root_path="/backstage",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", summary="Service health")
async def health():
    return {"ok": True}

@app.get("/health")
async def health():
    return {
        "status": "success",
        "message": "Server is running!",
        "timestamp": datetime.now().isoformat(),
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})