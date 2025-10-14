"""Convenience wrapper so `python run_daily_painpoints_job.py` works from repo root."""

from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.run_daily_painpoints_job", run_name="__main__")
