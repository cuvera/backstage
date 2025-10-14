"""Run the daily department pain-point aggregation job once."""

from __future__ import annotations

import asyncio
import logging
import sys

from app.db.mongodb import connect_to_mongo, close_mongo_connection
from app.services.jobs.daily_dept_painpoints import run_daily_department_painpoints_job


async def main() -> None:
    await connect_to_mongo()
    try:
        await run_daily_department_painpoints_job()
    finally:
        await close_mongo_connection()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    asyncio.run(main())
