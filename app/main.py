from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import settings
from app.core.logging import setup_logging
from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.messaging.consumer import RabbitMQConsumerManager
from app.messaging.producer import producer
# from scripts.test_meeting_prep import quick_test
from app.services.meeting_analysis_orchestrator import MeetingAnalysisOrchestrator

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

@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"message": exc.detail})

@app.get("/health", summary="Service health")
async def health():
    return {"ok": True}

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
    # payload = {
    #     "_id": "691300ec64082d8a17bf85bb",
    #     "tenantId": "689ddc0411e4209395942bee",
    #     "platform": "google",
    #     "bucket": "recordings",
    #     "fileUrl": "689ddc0411e4209395942bee/google/3leetlec1vouor0iis1h7j0ig1/Cuvera Bot-2025-11-11T09:36:47.739Z.wav"
    # }

    payload = {
        "id": "692abe3447856a0242fca4a9",
        "tenantId": "689ddc0411e4209395942bee",
        "type": "meeting",
        "mode": "online",
        "platform": "google",
        # "speakerTimeframes": [],
    #     "speakerTimeframes": [
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 25287,
    #         "end": 32430
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 35725,
    #         "end": 47830
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 48479,
    #         "end": 63531
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 52128,
    #         "end": 54431
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 58128,
    #         "end": 75228
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 73695,
    #         "end": 78330
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 73962,
    #         "end": 83995
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 78728,
    #         "end": 87329
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 81828,
    #         "end": 83712
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 92225,
    #         "end": 153727
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 158625,
    #         "end": 173142
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 178072,
    #         "end": 179625
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 185225,
    #         "end": 199840
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 197240,
    #         "end": 200106
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 220227,
    #         "end": 234539
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 262725,
    #         "end": 271835
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 271988,
    #         "end": 281036
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 276225,
    #         "end": 282738
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 288027,
    #         "end": 336935
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 290040,
    #         "end": 290836
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 298906,
    #         "end": 300836
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 331928,
    #         "end": 335434
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 338728,
    #         "end": 503527
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 339135,
    #         "end": 342333
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 357233,
    #         "end": 360835
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 501261,
    #         "end": 502294
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 505529,
    #         "end": 513428
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 509428,
    #         "end": 510227
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 510894,
    #         "end": 515529
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 514027,
    #         "end": 525926
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 527225,
    #         "end": 528028
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 527531,
    #         "end": 536427
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 535725,
    #         "end": 552126
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 551827,
    #         "end": 557627
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 553893,
    #         "end": 564327
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 564925,
    #         "end": 578425
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 577227,
    #         "end": 584475
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 582730,
    #         "end": 595341
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 588791,
    #         "end": 598843
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 596240,
    #         "end": 638039
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 615639,
    #         "end": 616791
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 639125,
    #         "end": 646741
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 646125,
    #         "end": 649638
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 653866,
    #         "end": 664492
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 656922,
    #         "end": 664200
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 666495,
    #         "end": 667197
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 670427,
    #         "end": 675639
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 670588,
    #         "end": 671539
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 674942,
    #         "end": 683633
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 683437,
    #         "end": 884828
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 891027,
    #         "end": 894528
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 894745,
    #         "end": 923630
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 902995,
    #         "end": 905912
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 924827,
    #         "end": 953327
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 955328,
    #         "end": 972428
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 961827,
    #         "end": 963229
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 972825,
    #         "end": 981433
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 980927,
    #         "end": 986227
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 985325,
    #         "end": 988625
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 993628,
    #         "end": 1005743
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 998010,
    #         "end": 998610
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 1005130,
    #         "end": 1023940
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1006075,
    #         "end": 1007280
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1011142,
    #         "end": 1014445
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1019428,
    #         "end": 1028243
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1024543,
    #         "end": 1050141
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1040356,
    #         "end": 1043525
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1051730,
    #         "end": 1067142
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 1053939,
    #         "end": 1055440
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1054655,
    #         "end": 1055449
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1068237,
    #         "end": 1078740
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1070639,
    #         "end": 1074440
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1079740,
    #         "end": 1090444
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1089637,
    #         "end": 1097539
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1093337,
    #         "end": 1095040
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1098427,
    #         "end": 1117640
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1107138,
    #         "end": 1113538
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1119825,
    #         "end": 1154036
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1137835,
    #         "end": 1142037
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1158027,
    #         "end": 1165939
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1158435,
    #         "end": 1161036
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1164835,
    #         "end": 1240633
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1229032,
    #         "end": 1231650
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1237332,
    #         "end": 1286034
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1246131,
    #         "end": 1246834
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1256830,
    #         "end": 1259137
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1264431,
    #         "end": 1264934
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1275233,
    #         "end": 1282135
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1286730,
    #         "end": 1296432
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1297625,
    #         "end": 1298428
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1304228,
    #         "end": 1304830
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1305328,
    #         "end": 1305832
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1305629,
    #         "end": 1323828
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1322845,
    #         "end": 1362328
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1325128,
    #         "end": 1328131
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1339227,
    #         "end": 1339929
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1347327,
    #         "end": 1348229
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1353326,
    #         "end": 1359732
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1364528,
    #         "end": 1387027
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1389492,
    #         "end": 1397562
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1396326,
    #         "end": 1398031
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1401725,
    #         "end": 1418426
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1420428,
    #         "end": 1431341
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1425642,
    #         "end": 1432042
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 1434912,
    #         "end": 1453676
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1435941,
    #         "end": 1437440
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1445924,
    #         "end": 1453426
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1456532,
    #         "end": 1504321
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1485789,
    #         "end": 1490259
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1517983,
    #         "end": 1519636
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1521387,
    #         "end": 1523056
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1523136,
    #         "end": 1557835
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1554680,
    #         "end": 1568139
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1575029,
    #         "end": 1611133
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1600137,
    #         "end": 1612736
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 1605917,
    #         "end": 1607954
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1610434,
    #         "end": 1611535
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1614528,
    #         "end": 1631433
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1625133,
    #         "end": 1627332
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1628633,
    #         "end": 1633936
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1630938,
    #         "end": 1639432
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1642027,
    #         "end": 1643337
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1642733,
    #         "end": 1646732
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1642836,
    #         "end": 1908805
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1653835,
    #         "end": 1657632
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1659931,
    #         "end": 1663433
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1667431,
    #         "end": 1667933
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1721625,
    #         "end": 1722131
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1774428,
    #         "end": 1774929
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 1800925,
    #         "end": 1802142
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1801425,
    #         "end": 1801930
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1830125,
    #         "end": 1835228
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1853207,
    #         "end": 1854125
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1906038,
    #         "end": 1909440
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1913338,
    #         "end": 1924040
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 1913905,
    #         "end": 2069332
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 1927838,
    #         "end": 1929641
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2012035,
    #         "end": 2013933
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2019015,
    #         "end": 2020437
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2060632,
    #         "end": 2061236
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2066733,
    #         "end": 2067237
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2071828,
    #         "end": 2073833
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2076927,
    #         "end": 2088933
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2086931,
    #         "end": 2103930
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2092931,
    #         "end": 2094836
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2095397,
    #         "end": 2096732
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2101025,
    #         "end": 2129133
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2103430,
    #         "end": 2106234
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2110230,
    #         "end": 2114733
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2121130,
    #         "end": 2139431
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2140329,
    #         "end": 2140831
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2141529,
    #         "end": 2150928
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2149820,
    #         "end": 2162828
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2157728,
    #         "end": 2158929
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2164428,
    #         "end": 2168627
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2169127,
    #         "end": 2188830
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2189624,
    #         "end": 2214426
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2194528,
    #         "end": 2195031
    #         },
    #         {
    #         "speakerName": "Gulshan Banpela",
    #         "start": 2206313,
    #         "end": 2206313
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2213926,
    #         "end": 2230628
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2232325,
    #         "end": 2233843
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2239425,
    #         "end": 2241628
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2244177,
    #         "end": 2245012
    #         },
    #         {
    #         "speakerName": "Gurusankar Kasivinayagam",
    #         "start": 2247650,
    #         "end": 2264943
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2250525,
    #         "end": 2251729
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2253041,
    #         "end": 2253543
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2265525,
    #         "end": 2284244
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2280807,
    #         "end": 2281776
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2285541,
    #         "end": 2286168
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2287427,
    #         "end": 2295039
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2290540,
    #         "end": 2292644
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2296125,
    #         "end": 2302440
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2303525,
    #         "end": 2314943
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2304356,
    #         "end": 2305246
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2308238,
    #         "end": 2332740
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2318539,
    #         "end": 2322049
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2325638,
    #         "end": 2329842
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2327241,
    #         "end": 2329141
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2333125,
    #         "end": 2348140
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2346737,
    #         "end": 2357739
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2352640,
    #         "end": 2354045
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2356441,
    #         "end": 2365138
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2363537,
    #         "end": 2395739
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2394235,
    #         "end": 2396836
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 2397835,
    #         "end": 2440735
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2398935,
    #         "end": 2400339
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2409225,
    #         "end": 2410743
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2421527,
    #         "end": 2423744
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2429433,
    #         "end": 2430438
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2441825,
    #         "end": 2463535
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 2464333,
    #         "end": 2467535
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2468927,
    #         "end": 2474538
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 2470349,
    #         "end": 2477436
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 2477632,
    #         "end": 2486136
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2477732,
    #         "end": 2478340
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2487232,
    #         "end": 2492034
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 2487634,
    #         "end": 2488836
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 2491130,
    #         "end": 2498339
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2494533,
    #         "end": 2496149
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 2497831,
    #         "end": 2515334
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2510536,
    #         "end": 2513533
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2519229,
    #         "end": 2546134
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 2546713,
    #         "end": 2563630
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2551729,
    #         "end": 2553940
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2566331,
    #         "end": 2584330
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2571828,
    #         "end": 2593828
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2588027,
    #         "end": 2590635
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2594625,
    #         "end": 2596929
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2600927,
    #         "end": 2603526
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2610053,
    #         "end": 2613129
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2612926,
    #         "end": 2635225
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2621426,
    #         "end": 2623031
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2636803,
    #         "end": 2673743
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2642827,
    #         "end": 2643333
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2647827,
    #         "end": 2648331
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2667926,
    #         "end": 2691444
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2692040,
    #         "end": 2739144
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2706340,
    #         "end": 2706844
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2723638,
    #         "end": 2725140
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2737425,
    #         "end": 2749546
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2751732,
    #         "end": 2761539
    #         },
    #         {
    #         "speakerName": "Harsha Perla",
    #         "start": 2761486,
    #         "end": 2791307
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2761838,
    #         "end": 2764346
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2762843,
    #         "end": 2765739
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2774127,
    #         "end": 2775338
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2790739,
    #         "end": 2791939
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2798538,
    #         "end": 2804935
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2806225,
    #         "end": 2817736
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2808836,
    #         "end": 2812435
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2820032,
    #         "end": 2821775
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2825735,
    #         "end": 2828436
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2845738,
    #         "end": 2848336
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2853333,
    #         "end": 2855346
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 2857928,
    #         "end": 2863333
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2864625,
    #         "end": 2868834
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2868683,
    #         "end": 2900935
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2882932,
    #         "end": 2885841
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2901825,
    #         "end": 2904941
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2904225,
    #         "end": 2919134
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2907816,
    #         "end": 2916734
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2912331,
    #         "end": 2914134
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2920031,
    #         "end": 2926635
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2922531,
    #         "end": 2927237
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2922931,
    #         "end": 2960836
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 2945536,
    #         "end": 2948342
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2959829,
    #         "end": 2979732
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2964627,
    #         "end": 2965838
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 2981126,
    #         "end": 3008734
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 2985925,
    #         "end": 2987037
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3002477,
    #         "end": 3024929
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3017627,
    #         "end": 3019933
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 3025426,
    #         "end": 3026053
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3026626,
    #         "end": 3043729
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3039126,
    #         "end": 3088543
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3047827,
    #         "end": 3048933
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3055525,
    #         "end": 3062034
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3071627,
    #         "end": 3074430
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3079226,
    #         "end": 3079929
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3083127,
    #         "end": 3105645
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3094140,
    #         "end": 3109041
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3109525,
    #         "end": 3125044
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3121844,
    #         "end": 3143547
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3129039,
    #         "end": 3131547
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3135239,
    #         "end": 3160241
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3164426,
    #         "end": 3167545
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3165142,
    #         "end": 3203438
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3195136,
    #         "end": 3213338
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3211236,
    #         "end": 3218938
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3218235,
    #         "end": 3231739
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3224041,
    #         "end": 3229242
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 3233728,
    #         "end": 3255938
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3242034,
    #         "end": 3242542
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3247038,
    #         "end": 3261821
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3255734,
    #         "end": 3262843
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 3257737,
    #         "end": 3261240
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 3259843,
    #         "end": 3318838
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3288350,
    #         "end": 3292135
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3296525,
    #         "end": 3297240
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3311025,
    #         "end": 3311834
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3312632,
    #         "end": 3313236
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3317125,
    #         "end": 3319634
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3317732,
    #         "end": 3324537
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 3320431,
    #         "end": 3420133
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3334031,
    #         "end": 3334836
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3344131,
    #         "end": 3346335
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3374064,
    #         "end": 3375364
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3375232,
    #         "end": 3378334
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3389029,
    #         "end": 3391333
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3401129,
    #         "end": 3403634
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3412527,
    #         "end": 3413733
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3420028,
    #         "end": 3484830
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 3424311,
    #         "end": 3424931
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3445831,
    #         "end": 3447613
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3457359,
    #         "end": 3458611
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3466529,
    #         "end": 3467827
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 3485525,
    #         "end": 3501458
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3487925,
    #         "end": 3541044
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3543040,
    #         "end": 3543741
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3544440,
    #         "end": 3559345
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3557338,
    #         "end": 3566741
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3566629,
    #         "end": 3583946
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 3567539,
    #         "end": 3569655
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3575625,
    #         "end": 3579242
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3583525,
    #         "end": 3597038
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3590239,
    #         "end": 3624341
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3631725,
    #         "end": 3635739
    #         },
    #         {
    #         "speakerName": "Gurusankar Kasivinayagam",
    #         "start": 3632670,
    #         "end": 3657523
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3656535,
    #         "end": 3663840
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 3665465,
    #         "end": 3679736
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3665825,
    #         "end": 3666437
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3672534,
    #         "end": 3673843
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3677234,
    #         "end": 3680038
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3683325,
    #         "end": 3685239
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 3686234,
    #         "end": 3705132
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3689933,
    #         "end": 3693839
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3702600,
    #         "end": 3703669
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3705833,
    #         "end": 3710635
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3712465,
    #         "end": 3762270
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3728232,
    #         "end": 3728739
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3749825,
    #         "end": 3754340
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3762730,
    #         "end": 3766637
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 3767725,
    #         "end": 3787734
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3789225,
    #         "end": 3795233
    #         },
    #         {
    #         "speakerName": "Poornima",
    #         "start": 3792113,
    #         "end": 3792117
    #         },
    #         {
    #         "speakerName": "Harsha Perla",
    #         "start": 3795463,
    #         "end": 3829032
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3826635,
    #         "end": 3828440
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3827228,
    #         "end": 3835031
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3833345,
    #         "end": 3836931
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 3838077,
    #         "end": 3877327
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3873426,
    #         "end": 3886229
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3885632,
    #         "end": 3897694
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3893926,
    #         "end": 3899231
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 3898426,
    #         "end": 3911360
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 3905242,
    #         "end": 3907877
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3907727,
    #         "end": 3914830
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 3916351,
    #         "end": 3933745
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 3935125,
    #         "end": 3937145
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 3937706,
    #         "end": 4009825
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4009871,
    #         "end": 4038843
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 4034303,
    #         "end": 4067843
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4042627,
    #         "end": 4043446
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4062735,
    #         "end": 4088538
    #         },
    #         {
    #         "speakerName": "Harsha Perla",
    #         "start": 4086637,
    #         "end": 4096099
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4094236,
    #         "end": 4118038
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 4094537,
    #         "end": 4103746
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 4110300,
    #         "end": 4133639
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4121432,
    #         "end": 4125845
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 4122332,
    #         "end": 4125943
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 4133634,
    #         "end": 4139036
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4134332,
    #         "end": 4176933
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 4142634,
    #         "end": 4146438
    #         },
    #         {
    #         "speakerName": "Vellesh Narayanan",
    #         "start": 4167281,
    #         "end": 4174682
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4181625,
    #         "end": 4184236
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4188925,
    #         "end": 4190232
    #         },
    #         {
    #         "speakerName": "Harsha Perla",
    #         "start": 4192481,
    #         "end": 4202767
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4203429,
    #         "end": 4207934
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 4206430,
    #         "end": 4217781
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4217229,
    #         "end": 4218632
    #         },
    #         {
    #         "speakerName": "Gurusankar Kasivinayagam",
    #         "start": 4220727,
    #         "end": 4241168
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4235227,
    #         "end": 4236237
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4240029,
    #         "end": 4242131
    #         },
    #         {
    #         "speakerName": "Sasikumar S",
    #         "start": 4245095,
    #         "end": 4262101
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4255028,
    #         "end": 4261834
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4270625,
    #         "end": 4271932
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 4273826,
    #         "end": 4276096
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4275027,
    #         "end": 4277730
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 4277427,
    #         "end": 4296025
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4294126,
    #         "end": 4321331
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 4305442,
    #         "end": 4307579
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 4318226,
    #         "end": 4318927
    #         },
    #         {
    #         "speakerName": "Abhilash",
    #         "start": 4321175,
    #         "end": 4321175
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 4322026,
    #         "end": 4336425
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4327341,
    #         "end": 4328131
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4336741,
    #         "end": 4343028
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 4342141,
    #         "end": 4344451
    #         },
    #         {
    #         "speakerName": "Ranjith Kadamboor",
    #         "start": 4344708,
    #         "end": 4352198
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4347625,
    #         "end": 4383142
    #         },
    #         {
    #         "speakerName": "Harsha Perla",
    #         "start": 4368807,
    #         "end": 4391994
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4387325,
    #         "end": 4406741
    #         },
    #         {
    #         "speakerName": "Parthiban Vardha",
    #         "start": 4406141,
    #         "end": 4419011
    #         },
    #         {
    #         "speakerName": "pallavika nesargikar",
    #         "start": 4412925,
    #         "end": 4418743
    #         },
    #         {
    #         "speakerName": "Aniverthy Amrutesh",
    #         "start": 4417239,
    #         "end": 4418706
    #         },
    #         {
    #         "speakerName": "Harsha Perla",
    #         "start": 4419004,
    #         "end": 4419478
    #         }
    #     ],
        "audioUrl": "https://376a0630fa9a905b0a95eb9c5a083e11.r2.cloudflarestorage.com/cuvera-dev/689ddc0411e4209395942bee/google/d1160425-24f3-4889-b69f-a94a5cb17f78/meeting.m4a?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=fd49a6f9e9493c79f72161dfad1fe37d%2F20260129%2Fauto%2Fs3%2Faws4_request&X-Amz-Date=20260129T092524Z&X-Amz-Expires=3600&X-Amz-Signature=a0f1fa2ea4f3f30b06bb847f14e83a311f419e654455d8f59fb4355d6bc975fb&X-Amz-SignedHeaders=host&x-amz-checksum-mode=ENABLED&x-id=GetObject"
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