# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the AFD plugin project
"""Class resolution for runtime import checks, including spawned interpreters."""

import importlib

from afd_plugin.validation import normalize_qualname


def resolve_class_from_qualname(qualname: str, *, role: str = "class") -> type:
    """Resolve a dotted or colon-separated class path."""

    normalized = normalize_qualname(qualname.strip())
    if not normalized or "." not in normalized:
        raise ValueError(
            f"{role} must be a dotted qualname, got {qualname!r}",
        )
    module_name, obj_name = normalized.rsplit(".", 1)
    module = importlib.import_module(module_name)
    obj = getattr(module, obj_name)
    if not isinstance(obj, type):
        raise TypeError(
            f"{role} resolved to {type(obj).__name__}, expected a class",
        )
    return obj
