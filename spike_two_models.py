"""
spike_two_models.py — Verifica si MelodyRNN + ImprovRNN caben juntos en RAM.

Carga los dos modelos secuencialmente, hace warmup de cada uno,
mide RAM en cada paso, y prueba una generación con cada uno para
confirmar que ambos funcionan con ambos cargados.

Si revienta por OOM (Out of Memory), el output va a indicar exactamente
en qué paso explotó. Si todo pasa, mostramos RAM final y dejamos el
camino libre para el refactor del sistema.

Uso:
    python spike_two_models.py

NO toca el sistema actual. Es un test descartable.
"""

import logging
import sys
import time
import urllib.request
from pathlib import Path

import psutil


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spike2")

# Silenciar TF
logging.getLogger("tensorflow").setLevel(logging.ERROR)

PROC = psutil.Process()


def ram_mb() -> float:
    return PROC.memory_info().rss / (1024 * 1024)


def log_ram(label: str) -> None:
    log.info(f"  RAM {label}: {ram_mb():.0f} MB")


def section(title: str) -> None:
    log.info("")
    log.info("=" * 60)
    log.info(title)
    log.info("=" * 60)


# ---------------------------------------------------------------------------
# Config de modelos a probar
# ---------------------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parent / "models_data"

MODELS = {
    "improv": {
        "path": MODELS_DIR / "chord_pitches_improv.mag",
        "url":  "http://download.magenta.tensorflow.org/models/chord_pitches_improv.mag",
        "config_id": "chord_pitches_improv",
    },
    "melody": {
        "path": MODELS_DIR / "attention_rnn.mag",
        "url":  "http://download.magenta.tensorflow.org/models/attention_rnn.mag",
        "config_id": "attention_rnn",
    },
}


def ensure_bundle(spec: dict) -> Path:
    path = spec["path"]
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"  Descargando {path.name}...")
    urllib.request.urlretrieve(spec["url"], path)
    log.info(f"  OK: {path.stat().st_size / 1e6:.1f} MB")
    return path


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def load_improv(bundle_path: Path):
    from magenta.models.improv_rnn.improv_rnn_sequence_generator import (
        ImprovRnnSequenceGenerator,
    )
    from magenta.models.improv_rnn.improv_rnn_model import (
        ImprovRnnModel, default_configs,
    )
    from magenta.models.shared.sequence_generator_bundle import read_bundle_file

    bundle = read_bundle_file(str(bundle_path))
    cfg = default_configs[bundle.generator_details.id]
    return ImprovRnnSequenceGenerator(
        model=ImprovRnnModel(cfg),
        details=cfg.details,
        steps_per_quarter=cfg.steps_per_quarter,
        checkpoint=None,
        bundle=bundle,
    ), cfg


def load_melody(bundle_path: Path):
    from magenta.models.melody_rnn.melody_rnn_sequence_generator import (
        MelodyRnnSequenceGenerator,
    )
    from magenta.models.melody_rnn.melody_rnn_model import (
        MelodyRnnModel, default_configs,
    )
    from magenta.models.shared.sequence_generator_bundle import read_bundle_file

    bundle = read_bundle_file(str(bundle_path))
    cfg = default_configs[bundle.generator_details.id]
    return MelodyRnnSequenceGenerator(
        model=MelodyRnnModel(cfg),
        details=cfg.details,
        steps_per_quarter=cfg.steps_per_quarter,
        checkpoint=None,
        bundle=bundle,
    ), cfg


# ---------------------------------------------------------------------------
# Generación de prueba
# ---------------------------------------------------------------------------

def test_improv(generator):
    from note_seq.protobuf import music_pb2, generator_pb2
    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=120)
    ann = seq.text_annotations.add()
    ann.text = "Dm7"
    ann.annotation_type = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
    ann.time = 0.0
    n = seq.notes.add()
    n.pitch = 60; n.start_time = 0.0; n.end_time = 0.5
    n.velocity = 80; n.instrument = 0; n.program = 0
    seq.total_time = 4.0

    options = generator_pb2.GeneratorOptions()
    options.args["temperature"].float_value = 1.0
    options.generate_sections.add(start_time=0.501, end_time=4.0)
    return generator.generate(seq, options)


def test_melody(generator):
    from note_seq.protobuf import music_pb2, generator_pb2
    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=120)
    n = seq.notes.add()
    n.pitch = 60; n.start_time = 0.0; n.end_time = 0.5
    n.velocity = 80; n.instrument = 0; n.program = 0
    seq.total_time = 4.0

    options = generator_pb2.GeneratorOptions()
    options.args["temperature"].float_value = 1.0
    options.generate_sections.add(start_time=0.501, end_time=4.0)
    return generator.generate(seq, options)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info(f"Python {sys.version.split()[0]}")
    log_ram("inicial")

    section("[FASE 0] Asegurar bundles en disco")
    improv_path = ensure_bundle(MODELS["improv"])
    melody_path = ensure_bundle(MODELS["melody"])

    section("[FASE 1] Importar TF + Magenta")
    t0 = time.time()
    import note_seq  # noqa
    log.info(f"  note_seq cargado ({time.time() - t0:.1f}s)")
    log_ram("post note_seq")

    section("[FASE 2] Cargar ImprovRNN")
    t0 = time.time()
    improv_gen, improv_cfg = load_improv(improv_path)
    log.info(f"  Cargado en {time.time() - t0:.1f}s")
    log_ram("post improv carga")

    log.info("  Warmup ImprovRNN...")
    t0 = time.time()
    test_improv(improv_gen)
    log.info(f"  Warmup en {time.time() - t0:.2f}s")
    log_ram("post improv warmup")

    section("[FASE 3] Cargar MelodyRNN (con ImprovRNN aún en memoria)")
    t0 = time.time()
    try:
        melody_gen, melody_cfg = load_melody(melody_path)
        log.info(f"  Cargado en {time.time() - t0:.1f}s")
    except MemoryError:
        log.error("  OOM: MelodyRNN no cabe junto a ImprovRNN.")
        log.error("  Plan B: hot-swap entre modelos (descartar uno para cargar el otro).")
        log_ram("al momento del OOM")
        sys.exit(2)
    log_ram("post melody carga")

    log.info("  Warmup MelodyRNN...")
    t0 = time.time()
    test_melody(melody_gen)
    log.info(f"  Warmup en {time.time() - t0:.2f}s")
    log_ram("post melody warmup")

    section("[FASE 4] Probar que AMBOS modelos siguen funcionando")
    log.info("  Generando con ImprovRNN (con MelodyRNN cargado)...")
    t0 = time.time()
    out_improv = test_improv(improv_gen)
    log.info(f"  ImprovRNN: {len(out_improv.notes)} notas en {time.time() - t0:.2f}s")

    log.info("  Generando con MelodyRNN (con ImprovRNN cargado)...")
    t0 = time.time()
    out_melody = test_melody(melody_gen)
    log.info(f"  MelodyRNN: {len(out_melody.notes)} notas en {time.time() - t0:.2f}s")

    section("[RESUMEN]")
    final_ram = ram_mb()
    log.info(f"  RAM final con ambos modelos cargados: {final_ram:.0f} MB")
    log.info("")
    if final_ram < 1500:
        log.info("  VEREDICTO: cabe holgado. Adelante con dos modelos simultáneos.")
    elif final_ram < 2500:
        log.info("  VEREDICTO: cabe pero ajustado. Cuidado con Studio One encima.")
        log.info("  Si la sesión real se vuelve lenta, considerar hot-swap.")
    else:
        log.warning("  VEREDICTO: RAM apretada. Hot-swap puede ser más seguro.")


if __name__ == "__main__":
    main()
