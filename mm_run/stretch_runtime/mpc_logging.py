"""Stable-shape logging helpers for MPC diagnostics."""

from __future__ import annotations

import json
from typing import Mapping


def append_mpc_diagnostics(logger, diagnostics: Mapping) -> None:
    """Append MPC diagnostics without treating variable records as arrays."""

    for key, value in diagnostics.items():
        log_key = f"mpc_{key}s"
        if key == "esdf_invalid_queries":
            records = [] if value is None else list(value)
            # A JSON string is a scalar for DataLogger, so its shape remains ()
            # whether this cycle has zero, one, or many invalid query records.
            logger.append(
                log_key,
                json.dumps(records, sort_keys=True, separators=(",", ":")),
            )
            logger.append("mpc_esdf_invalid_query_counts", len(records))
            continue
        logger.append(log_key, value)
