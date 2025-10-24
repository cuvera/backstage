from __future__ import annotations

import asyncio
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional

from pymongo.errors import CollectionInvalid

from app.db.mongodb import get_database
from app.services.agents.painpoint_canonical_agent import PainPointCanonicalAgent

logger = logging.getLogger(__name__)

COLLECTION_NAME = "daily_dept_painpoint_top10"

SEVERITY_PRIORITY = {"critical": 4, "high": 3, "medium": 2, "low": 1}


async def run_daily_department_painpoints_job() -> None:
    """
    Aggregate the last 24h of pain points into department-level top 10 reports.
    """
    db = await get_database()
    await _ensure_timeseries_collection(db)

    painpoints = db["pain_points"]
    output_collection = db[COLLECTION_NAME]

    window_to = datetime.now(timezone.utc)
    window_from = window_to - timedelta(hours=24)
    bucket_start = datetime(window_from.year, window_from.month, window_from.day, tzinfo=timezone.utc)
    date_bucket = bucket_start.date().isoformat()

    query = {
        "created_at": {
            "$gte": window_from.isoformat(),
            "$lt": window_to.isoformat(),
        }
    }

    logger.info(
        "[DailyPainPointJob] Computing window %s -> %s", window_from.isoformat(), window_to.isoformat()
    )

    cursor = painpoints.find(query)
    docs = await cursor.to_list(length=None)
    if not docs:
        logger.info("[DailyPainPointJob] No pain points found for window")
        return

    canonical_agent = PainPointCanonicalAgent()

    departments: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"total": 0, "groups": {}})

    loop = asyncio.get_running_loop()

    for doc in docs:
        department = (doc.get("department") or "").strip() or "Unknown"
        departments[department]["total"] += 1

        try:
            canonical = await loop.run_in_executor(None, canonical_agent.canonicalise, doc)
        except Exception as exc:
            logger.exception("[DailyPainPointJob] Canonicalisation failure, using fallback: %s", exc)
            canonical = PainPointCanonicalAgent._fallback(doc)  # type: ignore[attr-defined]

        key = canonical.get("canonical_key") or "uncategorized"
        groups = departments[department]["groups"]
        group = groups.get(key)
        if group is None:
            group = {
                "canonical_key": key,
                "count": 0,
                "title": canonical.get("title"),
                "category_counter": Counter(),
                "severity_counter": Counter(),
                "product_area": canonical.get("product_area"),
                "tags": [],
                "root_causes": [],
                "impacted_flows": [],
                "suggested_actions": [],
                "example": None,
                "representative_ids": [],
            }
            groups[key] = group

        group["count"] += 1
        if canonical.get("category"):
            group["category_counter"][canonical["category"]] += 1
        if canonical.get("severity"):
            group["severity_counter"][canonical["severity"]] += 1
        if canonical.get("product_area") and not group.get("product_area"):
            group["product_area"] = canonical["product_area"]

        _extend_unique(group["tags"], canonical.get("tags") or [])
        _extend_unique(group["root_causes"], canonical.get("root_causes") or [])
        _extend_unique(group["impacted_flows"], canonical.get("impacted_flows") or [])
        _extend_unique(group["suggested_actions"], canonical.get("suggested_actions") or [])

        if not group["example"]:
            raw_text = (doc.get("raw_text") or "").strip()
            group["example"] = raw_text[:500]

        rep_ids: List[str] = group["representative_ids"]
        if len(rep_ids) < 10:
            rep_ids.append(str(doc.get("_id")))

    now_iso = window_to.isoformat()
    for department, data in departments.items():
        groups = data["groups"]
        top_entries = sorted(groups.values(), key=lambda item: item["count"], reverse=True)[:10]
        top10_payload = []
        for entry in top_entries:
            top10_payload.append(
                {
                    "key": entry["canonical_key"],
                    "title": entry["title"],
                    "category": _select_mode(entry["category_counter"]),
                    "severity": _select_severity(entry["severity_counter"]),
                    "product_area": entry.get("product_area"),
                    "count": entry["count"],
                    "tags": entry["tags"][:10],
                    "root_causes": entry["root_causes"][:10],
                    "impacted_flows": entry["impacted_flows"][:10],
                    "suggested_actions": entry["suggested_actions"][:10],
                    "example": entry["example"],
                    "representative_ids": entry["representative_ids"][:10],
                }
            )

        document_id = f"{department}:{date_bucket}"
        document = {
            "_id": document_id,
            "bucket_start": bucket_start,
            "date_bucket": date_bucket,
            "meta": {"department": department},
            "window": {"from": window_from.isoformat(), "to": window_to.isoformat()},
            "total_count": data["total"],
            "top10": top10_payload,
            "meta2": {"version": 1},
            "created_at": now_iso,
            "updated_at": now_iso,
        }

        await output_collection.delete_one({"_id": document_id})
        await output_collection.insert_one(document)
        logger.info(
            "[DailyPainPointJob] Upserted %s entries for department '%s'", len(top10_payload), department
        )


async def _ensure_timeseries_collection(db) -> None:
    existing = await db.list_collection_names()
    if COLLECTION_NAME in existing:
        return
    try:
        await db.create_collection(
            COLLECTION_NAME,
            timeseries={"timeField": "bucket_start", "metaField": "meta", "granularity": "hours"},
        )
        logger.info("[DailyPainPointJob] Created time-series collection '%s'", COLLECTION_NAME)
    except CollectionInvalid:
        # Another process may have created it between list and create.
        logger.debug("[DailyPainPointJob] Time-series collection '%s' already exists", COLLECTION_NAME)


def _extend_unique(target: List[str], items: Iterable[str], limit: int = 10) -> None:
    existing = set(target)
    for item in items:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not cleaned or cleaned in existing:
            continue
        target.append(cleaned)
        existing.add(cleaned)
        if len(target) >= limit:
            break


def _select_mode(counter: Counter) -> Optional[str]:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def _select_severity(counter: Counter) -> Optional[str]:
    if not counter:
        return None
    best = None
    best_count = -1
    best_priority = -1
    for severity, count in counter.items():
        priority = SEVERITY_PRIORITY.get(severity, 0)
        if count > best_count or (count == best_count and priority > best_priority):
            best = severity
            best_count = count
            best_priority = priority
    return best
