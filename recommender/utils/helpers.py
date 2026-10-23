import math
import uuid
from typing import Optional

import numpy as np


def sanitize_numeric(v: Optional[float]) -> Optional[float]:
    """Return a real float or None for NaN/Inf or non-numeric inputs (tries casting)."""
    if v is None:
        return None
    # numpy scalars
    if isinstance(v, (np.generic,)):
        try:
            v = v.item()
        except Exception:
            return None
    # ints are acceptable
    if isinstance(v, int):
        return float(v)
    if isinstance(v, float):
        return float(v) if math.isfinite(v) else None
    # try to coerce strings like "0.5"
    try:
        f = float(v)  # type: ignore
        return f if math.isfinite(f) else None
    except Exception:
        return None


def coerce_uuid(val: Optional[uuid.UUID | str]) -> uuid.UUID:
    """Normalize a uuid or string to uuid.UUID."""
    if val is None:
        return uuid.uuid4()
    if isinstance(val, uuid.UUID):
        return val
    try:
        return uuid.UUID(str(val))
    except Exception:
        # fallback: generate deterministic UUID5 from string
        return uuid.uuid5(uuid.NAMESPACE_OID, str(val))
