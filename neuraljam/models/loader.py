"""
neuraljam/models/loader.py

Carga modelos de Magenta. Soporta dos familias:
- melody_rnn:  MelodyRNN, MelodyRnnSequenceGenerator
- improv_rnn:  ImprovRNN, ImprovRnnSequenceGenerator

API pública:
- load_generator(): carga el modelo del MODE activo (default)
- load_all_models(): carga TODOS los perfiles con bundle disponible
  (la opción que usa el entry point cuando queremos dos modelos en RAM)
- download_bundle_if_missing(): asegura que un .mag esté en disco
- warmup(): inicializa TF con una generación descartable

El spike confirmó: dos modelos en RAM = 459 MB. Holgado. No hay razón
para hot-swap.
"""

import logging
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from neuraljam import config


log = logging.getLogger(__name__)


# ===========================================================================
# Tipo de salida
# ===========================================================================

@dataclass
class LoadedModel:
    """Wrapper sobre un modelo cargado de Magenta."""
    generator: Any
    magenta_config: Any
    profile: dict
    family: str    # "melody_rnn" | "improv_rnn" | "performance_rnn"


# ===========================================================================
# Descarga
# ===========================================================================

def download_bundle_if_missing(profile: dict) -> Path:
    path: Path = profile["model_path"]
    url: str = profile["model_url"]

    if path.exists():
        size_mb = path.stat().st_size / (1024 * 1024)
        log.info(f"Bundle ya existe: {path.name} ({size_mb:.1f} MB)")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Descargando bundle desde {url} ...")
    t0 = time.time()
    urllib.request.urlretrieve(url, path)
    elapsed = time.time() - t0
    size_mb = path.stat().st_size / (1024 * 1024)

    if size_mb < 0.5:
        path.unlink()
        raise ValueError(
            f"Bundle descargado es sospechosamente chico ({size_mb:.2f} MB). "
            f"Posible URL rota. Archivo borrado."
        )

    log.info(f"Descargado: {size_mb:.1f} MB en {elapsed:.1f}s")
    return path


# ===========================================================================
# Carga (dispatch por familia)
# ===========================================================================

def _build_improv_generator(bundle_path: Path, profile: dict):
    from magenta.models.improv_rnn.improv_rnn_sequence_generator import (
        ImprovRnnSequenceGenerator,
    )
    from magenta.models.improv_rnn.improv_rnn_model import (
        ImprovRnnModel, default_configs,
    )
    from magenta.models.shared.sequence_generator_bundle import read_bundle_file

    bundle = read_bundle_file(str(bundle_path))
    config_id = bundle.generator_details.id
    if config_id != profile["model_config_id"]:
        raise RuntimeError(
            f"Bundle config_id mismatch. Esperaba {profile['model_config_id']!r}, "
            f"bundle dice {config_id!r}."
        )
    if config_id not in default_configs:
        raise RuntimeError(
            f"config_id {config_id!r} no está en default_configs de ImprovRNN. "
            f"Conocidos: {list(default_configs)}"
        )
    cfg = default_configs[config_id]
    gen = ImprovRnnSequenceGenerator(
        model=ImprovRnnModel(cfg),
        details=cfg.details,
        steps_per_quarter=cfg.steps_per_quarter,
        checkpoint=None,
        bundle=bundle,
    )
    return gen, cfg


def _build_melody_generator(bundle_path: Path, profile: dict):
    from magenta.models.melody_rnn.melody_rnn_sequence_generator import (
        MelodyRnnSequenceGenerator,
    )
    from magenta.models.melody_rnn.melody_rnn_model import (
        MelodyRnnModel, default_configs,
    )
    from magenta.models.shared.sequence_generator_bundle import read_bundle_file

    bundle = read_bundle_file(str(bundle_path))
    config_id = bundle.generator_details.id
    if config_id != profile["model_config_id"]:
        raise RuntimeError(
            f"Bundle config_id mismatch. Esperaba {profile['model_config_id']!r}, "
            f"bundle dice {config_id!r}."
        )
    if config_id not in default_configs:
        raise RuntimeError(
            f"config_id {config_id!r} no está en default_configs de MelodyRNN. "
            f"Conocidos: {list(default_configs)}"
        )
    cfg = default_configs[config_id]
    gen = MelodyRnnSequenceGenerator(
        model=MelodyRnnModel(cfg),
        details=cfg.details,
        steps_per_quarter=cfg.steps_per_quarter,
        checkpoint=None,
        bundle=bundle,
    )
    return gen, cfg


def _build_polyphony_generator(bundle_path: Path, profile: dict):
    from magenta.models.polyphony_rnn import polyphony_sequence_generator
    from magenta.models.polyphony_rnn.polyphony_model import (
        PolyphonyRnnModel, default_configs,
    )
    from magenta.models.shared.sequence_generator_bundle import read_bundle_file

    bundle = read_bundle_file(str(bundle_path))
    config_id = bundle.generator_details.id
    if config_id != profile["model_config_id"]:
        raise RuntimeError(
            f"Bundle config_id mismatch. Esperaba {profile['model_config_id']!r}, "
            f"bundle dice {config_id!r}."
        )
    if config_id not in default_configs:
        raise RuntimeError(
            f"config_id {config_id!r} no encontrado en default_configs de PolyphonyRNN. "
            f"Conocidos: {list(default_configs)}"
        )
    cfg = default_configs[config_id]
    gen = polyphony_sequence_generator.PolyphonyRnnSequenceGenerator(
        model=PolyphonyRnnModel(cfg),
        details=cfg.details,
        steps_per_quarter=cfg.steps_per_quarter,
        checkpoint=None,
        bundle=bundle,
    )
    return gen, cfg


def _build_performance_generator(bundle_path: Path, profile: dict):
    from magenta.models.performance_rnn import performance_sequence_generator
    from magenta.models.performance_rnn.performance_model import (
        PerformanceRnnModel, default_configs,
    )
    from magenta.models.shared.sequence_generator_bundle import read_bundle_file

    bundle = read_bundle_file(str(bundle_path))
    config_id = bundle.generator_details.id
    if config_id != profile["model_config_id"]:
        raise RuntimeError(
            f"Bundle config_id mismatch. Esperaba {profile['model_config_id']!r}, "
            f"bundle dice {config_id!r}."
        )
    if config_id not in default_configs:
        raise RuntimeError(
            f"config_id {config_id!r} no encontrado en default_configs de PerformanceRNN. "
            f"Conocidos: {list(default_configs)}"
        )
    cfg = default_configs[config_id]
    gen = performance_sequence_generator.PerformanceRnnSequenceGenerator(
        model=PerformanceRnnModel(cfg),
        details=cfg.details,
        steps_per_second=cfg.steps_per_second,
        num_velocity_bins=cfg.num_velocity_bins,
        checkpoint=None,
        bundle=bundle,
    )
    return gen, cfg


# Registry de builders por familia.
_BUILDERS = {
    "melody_rnn": _build_melody_generator,
    "improv_rnn": _build_improv_generator,
    "performance_rnn": _build_performance_generator,
    "polyphony_rnn": _build_polyphony_generator,
}


def _load_one(profile: dict, do_warmup: bool) -> LoadedModel:
    """Carga UN modelo dado su perfil. Helper interno."""
    family = profile["model_family"]
    if family not in _BUILDERS:
        raise NotImplementedError(
            f"Familia {family!r} todavía no implementada en el loader."
        )

    bundle_path = download_bundle_if_missing(profile)

    t0 = time.time()
    gen, mcfg = _BUILDERS[family](bundle_path, profile)
    log.info(f"  {family}: cargado en {time.time() - t0:.1f}s")

    loaded = LoadedModel(
        generator=gen,
        magenta_config=mcfg,
        profile=profile,
        family=family,
    )
    if do_warmup:
        warmup(loaded)
    return loaded


# ===========================================================================
# API pública
# ===========================================================================

def load_generator(do_warmup: bool = True) -> LoadedModel:
    """Carga UN modelo: el del MODE activo. Compat con código viejo."""
    profile = config.active_profile()
    log.info(f"Cargando modelo del modo '{config.MODE}'")

    # Trigger los imports de TF temprano (una sola vez).
    log.info("Importando TensorFlow y Magenta (~16-20s en cold)...")
    t0 = time.time()
    import note_seq  # noqa: F401
    log.info(f"Imports listos ({time.time() - t0:.1f}s)")

    return _load_one(profile, do_warmup)


def load_all_models(do_warmup: bool = True) -> Dict[str, LoadedModel]:
    """
    Carga TODOS los modelos implementados que están en config.PROFILES.

    Se saltea perfiles cuya familia todavía no está implementada (ej.
    'performance' por ahora). Esto permite ir agregando familias sin
    romper este flujo.

    Returns:
        dict por nombre de modo: {"melody": LoadedModel, "improv": LoadedModel}
    """
    log.info("Cargando TODOS los modelos disponibles...")

    # Imports de TF una sola vez al inicio.
    log.info("Importando TensorFlow y Magenta (~16-20s en cold)...")
    t0 = time.time()
    import note_seq  # noqa: F401
    log.info(f"Imports listos ({time.time() - t0:.1f}s)")

    loaded: Dict[str, LoadedModel] = {}
    for mode_name, profile in config.PROFILES.items():
        family = profile["model_family"]
        if family not in _BUILDERS:
            log.info(f"  Skipping '{mode_name}' (familia {family!r} no implementada)")
            continue

        log.info(f"Cargando modo '{mode_name}'...")
        try:
            loaded[mode_name] = _load_one(profile, do_warmup)
        except Exception:
            log.exception(f"Falló carga de '{mode_name}', skipping")

    log.info(f"Modelos cargados: {list(loaded.keys())}")
    return loaded


# ===========================================================================
# Warmup
# ===========================================================================

def warmup(loaded: LoadedModel) -> float:
    """Inicializa TF con una generación descartable."""
    from note_seq.protobuf import music_pb2, generator_pb2

    log.info(f"  Warmup {loaded.family}...")
    t0 = time.time()

    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=config.QPM_FALLBACK)

    # Solo agregamos chord annotation si el modelo lo requiere.
    if loaded.profile["needs_chords"]:
        ann = seq.text_annotations.add()
        ann.text = "C"
        ann.annotation_type = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
        ann.time = 0.0

    note = seq.notes.add()
    note.pitch = 60
    note.start_time = 0.0
    note.end_time = 0.5
    note.velocity = 80
    note.instrument = 0
    note.program = 0
    seq.total_time = 2.0

    options = generator_pb2.GeneratorOptions()
    options.args["temperature"].float_value = 1.0
    options.generate_sections.add(start_time=0.5 + 0.001, end_time=2.0)

    loaded.generator.generate(seq, options)

    elapsed = time.time() - t0
    log.info(f"  Warmup completado en {elapsed:.1f}s")
    return elapsed


# ===========================================================================
# Test manual: python -m neuraljam.models.loader
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    config.ensure_dirs()

    print(f"Cargando TODOS los modelos...")
    print()
    try:
        models = load_all_models(do_warmup=True)
    except Exception as e:
        print(f"FALLO: {e}")
        sys.exit(1)

    print()
    print(f"Modelos cargados: {len(models)}")
    for name, m in models.items():
        print(f"  {name:10s}  family={m.family:12s}  "
              f"chords={m.profile['needs_chords']}  "
              f"bars={m.profile['response_bars']}")
