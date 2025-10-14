"""Wrapper so `python dev_send_painpoint.py` behaves like running the script module."""

from __future__ import annotations

import runpy

if __name__ == "__main__":
    runpy.run_module("scripts.dev_send_painpoint", run_name="__main__")
