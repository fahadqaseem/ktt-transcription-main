from __future__ import annotations

import hashlib
import json

from ktt_ocr_eval.core.schemas import MethodConfig


def method_fingerprint(method: MethodConfig) -> str:
    """Stable short hash of a method's *semantic* config (adapter + config).

    Excludes ``name`` and ``nickname`` (identity / display only), so renaming or
    relabelling a method does not invalidate already-collected predictions, while
    any change to the adapter or its config does. This is the cache key that makes
    a collected prediction "still valid" on resume.
    """
    payload = {"adapter": method.adapter, "config": method.config}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
