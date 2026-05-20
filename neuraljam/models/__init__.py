"""
neuraljam.models — Gestión de modelos de Magenta.

Soporta dos familias en RAM simultáneamente:
- melody_rnn (MelodyRNN, attention_rnn bundle)
- improv_rnn (ImprovRNN, chord_pitches_improv bundle)

Lazy imports (PEP 562): importar este paquete NO dispara TF.
"""

__all__ = [
    "LoadedModel",
    "load_generator",
    "load_all_models",
    "warmup",
    "download_bundle_if_missing",
]


def __getattr__(name):
    if name in __all__:
        from neuraljam.models.loader import (
            LoadedModel,
            load_generator,
            load_all_models,
            warmup,
            download_bundle_if_missing,
        )
        return {
            "LoadedModel": LoadedModel,
            "load_generator": load_generator,
            "load_all_models": load_all_models,
            "warmup": warmup,
            "download_bundle_if_missing": download_bundle_if_missing,
        }[name]
    raise AttributeError(f"module 'neuraljam.models' has no attribute {name!r}")
