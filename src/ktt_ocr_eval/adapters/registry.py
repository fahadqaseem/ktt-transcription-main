from __future__ import annotations

from ktt_ocr_eval.adapters.example_provider import ExampleProviderAdapter
from ktt_ocr_eval.core.schemas import MethodConfig

ADAPTERS = {
    "example_provider": ExampleProviderAdapter,
}


def create_adapter(method: MethodConfig):
    try:
        adapter_class = ADAPTERS[method.adapter]
    except KeyError as exc:
        supported = ", ".join(sorted(ADAPTERS))
        message = f"Unknown adapter {method.adapter!r}. Supported adapters: {supported}"
        raise ValueError(message) from exc
    return adapter_class(method)
