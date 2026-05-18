"""
spike_improv_rnn.py — Spike técnico aislado de ImprovRNN

Objetivo: responder con números reales antes de diseñar la arquitectura de
NeuralJam:
  - ¿La URL oficial del bundle funciona?
  - ¿Qué notación de acordes acepta el modelo? (triadas, séptimas, alteraciones,
    slash, inválidos)
  - ¿Cuánto tarda en CPU? (cold start, warm, distintas longitudes)
  - ¿Cuánta RAM ocupa?
  - ¿La calidad musical es usable para jazz?

Uso:
  1. Activar venv:    .\venv\Scripts\Activate.ps1
  2. (opcional)       pip install psutil
  3. Ejecutar:        python spike_improv_rnn.py

Salidas:
  - spike_log.txt       log completo con métricas
  - outputs/*.mid       midis generados, abrir en Studio One
"""

import logging
import os
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuración del spike (todo hardcoded a propósito; este script NO usa
# config.py de NeuralJam, justamente para validar antes de diseñarlo).
# ---------------------------------------------------------------------------

BUNDLE_URL = "http://download.magenta.tensorflow.org/models/chord_pitches_improv.mag"
BUNDLE_PATH = Path("chord_pitches_improv.mag")
OUTPUT_DIR = Path("outputs")
LOG_PATH = Path("spike_log.txt")

QPM = 120
# A 120 BPM y 4/4, un compás = 2 segundos
SECONDS_PER_BAR = 2.0
# El primer ocupa 1 compás: notas de negra (0.5s) → 4 notas
PRIMER_NOTE_DURATION = 0.5
PRIMER_VELOCITY = 80

# Casos de notación a probar en Fase 2
NOTATION_TESTS = [
    ("triads", "Dm G C C"),
    ("sevenths", "Dm7 G7 Cmaj7 Cmaj7"),
    ("complex", "Em7b5 A7b9 Dm6 Dm6"),
    ("slash", "C/E F/A G/B C"),
    ("invalid", "XYZ Foo Bar Baz"),
]
NOTATION_PRIMER = [60, 62, 64, 65]  # C D E F → 1 compás

# Escenarios musicales de Fase 4
MUSICAL_SCENARIOS = [
    {
        "name": "escenario_1_major",
        "primer": [60, 62, 64, 65],
        "chords": "Dm7 G7 Cmaj7 Cmaj7",
        "temperature": 0.8,
        "bars_total": 4,
    },
    {
        "name": "escenario_2_minor",
        "primer": [59, 62, 65, 69],  # Bdim7 arpegiado
        "chords": "Em7b5 A7 Dm7 Dm7",
        "temperature": 1.0,
        "bars_total": 4,
    },
    {
        "name": "escenario_3_high_temp",
        "primer": [60, 62, 64, 65],
        "chords": "Dm7 G7 Cmaj7 Cmaj7",
        "temperature": 1.2,
        "bars_total": 4,
    },
]

LATENCY_BAR_COUNTS = [4, 8, 16]
LATENCY_RUNS_PER_LENGTH = 3

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logging():
    """Logger dual: archivo detallado + consola limpia."""
    logger = logging.getLogger("spike")
    logger.setLevel(logging.INFO)

    # Limpiar handlers previos si se relanza en notebook
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fh = logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger


log = setup_logging()


def section(title):
    """Header visual para separar fases en el log."""
    log.info("")
    log.info("=" * 70)
    log.info(title)
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# RAM (opcional, requiere psutil)
# ---------------------------------------------------------------------------

try:
    import psutil

    _proc = psutil.Process(os.getpid())

    def ram_mb():
        return _proc.memory_info().rss / (1024 * 1024)
except ImportError:

    def ram_mb():
        return None


def log_ram(label):
    val = ram_mb()
    if val is None:
        log.info(f"  RAM {label}: (psutil no instalado)")
    else:
        log.info(f"  RAM {label}: {val:.0f} MB")


# ---------------------------------------------------------------------------
# Fase 0 — Download del bundle
# ---------------------------------------------------------------------------


def phase_0_download():
    section("[FASE 0] Download del bundle")
    log.info(f"  URL:    {BUNDLE_URL}")
    log.info(f"  Path:   {BUNDLE_PATH.resolve()}")

    if BUNDLE_PATH.exists():
        size_mb = BUNDLE_PATH.stat().st_size / (1024 * 1024)
        log.info(f"  Status: ya existe ({size_mb:.1f} MB), skip download")
        return

    log.info("  Status: descargando...")
    t0 = time.time()
    try:
        urllib.request.urlretrieve(BUNDLE_URL, BUNDLE_PATH)
    except Exception as e:
        log.error(f"  FALLO descarga: {e}")
        raise
    size_mb = BUNDLE_PATH.stat().st_size / (1024 * 1024)
    log.info(f"  OK: {size_mb:.1f} MB en {time.time() - t0:.1f}s")


# ---------------------------------------------------------------------------
# Fase 1 — Carga del modelo + warmup
# ---------------------------------------------------------------------------


def phase_1_load_and_warmup():
    section("[FASE 1] Carga del modelo + warmup")
    log_ram("antes de cargar")

    # Imports diferidos: TF + Magenta tardan en importar.
    # Si los ponemos arriba, el "Status: skip download" sale 8s después de
    # arrancar y parece colgado. Acá los importamos cuando ya logueamos.
    log.info("  Importando TF + Magenta (esto puede tardar)...")
    t0 = time.time()
    global note_seq, music_pb2, generator_pb2
    global ImprovRnnSequenceGenerator, ImprovRnnModel, default_configs
    global read_bundle_file

    import note_seq
    from magenta.models.improv_rnn.improv_rnn_model import (
        ImprovRnnModel,
        default_configs,
    )
    from magenta.models.improv_rnn.improv_rnn_sequence_generator import (
        ImprovRnnSequenceGenerator,
    )
    from magenta.models.shared.sequence_generator_bundle import (
        read_bundle_file,
    )
    from note_seq.protobuf import generator_pb2, music_pb2

    log.info(f"  Imports listos ({time.time() - t0:.1f}s)")

    # Cargar bundle
    t0 = time.time()
    bundle = read_bundle_file(str(BUNDLE_PATH))
    config_id = bundle.generator_details.id
    log.info(f"  Bundle config_id: {config_id}")
    if config_id not in default_configs:
        log.error(f"  config_id no está en default_configs: {list(default_configs)}")
        raise SystemExit(1)

    config = default_configs[config_id]
    generator = ImprovRnnSequenceGenerator(
        model=ImprovRnnModel(config),
        details=config.details,
        steps_per_quarter=config.steps_per_quarter,
        checkpoint=None,
        bundle=bundle,
    )
    log.info(f"  Bundle cargado en {time.time() - t0:.1f}s")
    log_ram("post carga")

    # Warmup descartable: una generación corta para inicializar TF.
    log.info("  Warmup pass (descartado)...")
    t0 = time.time()
    primer_seq = build_input_sequence(
        primer_pitches=[60],
        chord_progression="C C C C",
        bars_total=2,
    )
    _ = generate(generator, primer_seq, bars_total=2, temperature=1.0)
    log.info(f"  Warmup en {time.time() - t0:.1f}s (este número es cold-start real)")
    log_ram("post warmup")

    return generator


# ---------------------------------------------------------------------------
# Construcción del input sequence (primer + chord progression)
# ---------------------------------------------------------------------------


def build_input_sequence(primer_pitches, chord_progression, bars_total):
    """
    Construye un NoteSequence con:
      - el tempo
      - las notas del primer (negras a 120 BPM, una atrás de la otra)
      - los acordes como text_annotations tipo CHORD_SYMBOL, espaciados 1 por compás
    """
    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=QPM)

    # Acordes
    chords = chord_progression.split()
    for i, chord in enumerate(chords):
        ann = seq.text_annotations.add()
        ann.text = chord
        ann.annotation_type = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
        ann.time = i * SECONDS_PER_BAR

    # Primer
    current_time = 0.0
    for pitch in primer_pitches:
        note = seq.notes.add()
        note.pitch = pitch
        note.start_time = current_time
        note.end_time = current_time + PRIMER_NOTE_DURATION
        note.velocity = PRIMER_VELOCITY
        note.instrument = 0
        note.program = 0
        current_time += PRIMER_NOTE_DURATION

    seq.total_time = max(current_time, bars_total * SECONDS_PER_BAR)
    return seq


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------


def generate(generator, input_seq, bars_total, temperature):
    """
    Llama al modelo. El primer ocupa el primer compás; la generación cubre
    desde el final del primer hasta el final de bars_total compases.
    """
    primer_end = input_seq.notes[-1].end_time if input_seq.notes else 0.0
    # ImprovRNN exige start_time ESTRICTAMENTE mayor que el final del primer.
    # Sumamos 1ms: el modelo cuantiza internamente a steps_per_quarter, así
    # que este epsilon no cambia la generación pero satisface el check.
    generate_start = primer_end + 0.001
    total_end = bars_total * SECONDS_PER_BAR

    options = generator_pb2.GeneratorOptions()
    options.args["temperature"].float_value = float(temperature)
    options.generate_sections.add(start_time=generate_start, end_time=total_end)

    return generator.generate(input_seq, options)


def save_midi(seq, name):
    """Guarda el NoteSequence como .mid en outputs/. Devuelve el path."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"{name}.mid"
    note_seq.sequence_proto_to_midi_file(seq, str(path))
    return path


# ---------------------------------------------------------------------------
# Fase 2 — Test de notación armónica
# ---------------------------------------------------------------------------


def phase_2_notation_tests(generator):
    """
    Para cada caso de notación, intenta generar 2 compases con primer fijo.
    Si tira excepción → notación no soportada. Si sale OK → guardar midi y
    dejar para inspección.
    """
    section("[FASE 2] Test de notación armónica")
    log.info("  Primer fijo: C D E F (4 negras), temp=1.0, 2 compases")
    log.info("")

    results = {}
    for test_id, progression in NOTATION_TESTS:
        log.info(f"  [{test_id}] '{progression}'")
        try:
            primer_seq = build_input_sequence(
                primer_pitches=NOTATION_PRIMER,
                chord_progression=progression,
                bars_total=2,
            )
            out_seq = generate(generator, primer_seq, bars_total=2, temperature=1.0)
            path = save_midi(out_seq, f"notation_{test_id}")
            n_notes = len(out_seq.notes)
            log.info(f"    OK — {n_notes} notas generadas → {path}")
            results[test_id] = ("ok", n_notes, str(path))
        except Exception as e:
            log.info(f"    EXCEPCIÓN ({type(e).__name__}): {e}")
            results[test_id] = ("error", type(e).__name__, str(e))

    log.info("")
    log.info("  Resumen notación soportada:")
    for test_id, _ in NOTATION_TESTS:
        status = results[test_id][0]
        log.info(f"    {test_id:10s} → {status.upper()}")
    return results


# ---------------------------------------------------------------------------
# Fase 3 — Latencia
# ---------------------------------------------------------------------------


def phase_3_latency(generator):
    section("[FASE 3] Latencia (post-warmup)")
    log.info(f"  Corridas por longitud: {LATENCY_RUNS_PER_LENGTH}")
    log.info("  Progresión: 'Dm7 G7 Cmaj7 Cmaj7' (loopeada por el modelo)")
    log.info("")

    primer_seq = build_input_sequence(
        primer_pitches=[60, 62, 64, 65],
        chord_progression="Dm7 G7 Cmaj7 Cmaj7",
        bars_total=max(LATENCY_BAR_COUNTS),
    )

    ram_peak = 0
    results = {}

    for bars in LATENCY_BAR_COUNTS:
        runs = []
        for i in range(LATENCY_RUNS_PER_LENGTH):
            t0 = time.time()
            try:
                _ = generate(generator, primer_seq, bars_total=bars, temperature=1.0)
            except Exception as e:
                log.warning(f"    {bars} bars run {i + 1} FALLÓ: {e}")
                continue
            runs.append(time.time() - t0)
            r = ram_mb()
            if r and r > ram_peak:
                ram_peak = r

        if runs:
            mean = sum(runs) / len(runs)
            log.info(
                f"  {bars:2d} compases: "
                f"{[f'{x:.2f}s' for x in runs]} → mean {mean:.2f}s"
            )
            results[bars] = {"runs": runs, "mean": mean}
        else:
            results[bars] = {"runs": [], "mean": None}

    log.info("")
    if ram_peak:
        log.info(f"  RAM pico durante latencia: {ram_peak:.0f} MB")
    return results


# ---------------------------------------------------------------------------
# Fase 4 — Escenarios musicales
# ---------------------------------------------------------------------------


def phase_4_musical_scenarios(generator):
    section("[FASE 4] Escenarios musicales (para escuchar en Studio One)")

    results = []
    for sc in MUSICAL_SCENARIOS:
        log.info(f"  [{sc['name']}]")
        log.info(f"    primer:  {sc['primer']}")
        log.info(f"    chords:  '{sc['chords']}'")
        log.info(f"    temp:    {sc['temperature']}")

        try:
            primer_seq = build_input_sequence(
                primer_pitches=sc["primer"],
                chord_progression=sc["chords"],
                bars_total=sc["bars_total"],
            )
            t0 = time.time()
            out_seq = generate(
                generator,
                primer_seq,
                bars_total=sc["bars_total"],
                temperature=sc["temperature"],
            )
            elapsed = time.time() - t0
            path = save_midi(out_seq, sc["name"])
            log.info(f"    {len(out_seq.notes)} notas en {elapsed:.2f}s → {path}")
            log.info(
                f"    para escuchar con contexto: tocá '{sc['chords']}' "
                f"en el piano sobre el playback del .mid"
            )
            results.append((sc["name"], "ok", elapsed, str(path)))
        except Exception as e:
            log.error(f"    FALLÓ ({type(e).__name__}): {e}")
            results.append((sc["name"], "error", None, str(e)))
        log.info("")

    return results


# ---------------------------------------------------------------------------
# Fase 5 — Resumen ejecutivo
# ---------------------------------------------------------------------------


def phase_5_summary(notation_results, latency_results, scenario_results):
    section("[RESUMEN]")

    log.info("  Notación armónica:")
    for test_id, _ in NOTATION_TESTS:
        status = notation_results.get(test_id, ("?",))[0]
        log.info(f"    {test_id:10s} → {status}")

    log.info("")
    log.info("  Latencia (mean por longitud):")
    for bars in LATENCY_BAR_COUNTS:
        r = latency_results.get(bars, {})
        if r.get("mean") is not None:
            log.info(f"    {bars:2d} compases: {r['mean']:.2f}s")
        else:
            log.info(f"    {bars:2d} compases: sin datos")

    log.info("")
    log.info("  Escenarios musicales generados:")
    for name, status, elapsed, info in scenario_results:
        if status == "ok":
            log.info(f"    {name}: OK ({elapsed:.2f}s) — {info}")
        else:
            log.info(f"    {name}: FAIL — {info}")

    log.info("")
    log.info("  Próximos pasos sugeridos según resultados:")
    log.info("    1. Escuchar los 3 escenarios en Studio One.")
    log.info("    2. Comparar notation_triads vs notation_sevenths a oído —")
    log.info("       si suenan distintos, el modelo entiende séptimas.")
    log.info("    3. Con la latencia confirmada, decidir silencio vs downbeat.")
    log.info("    4. Recién entonces: diseñar config.py de NeuralJam.")
    log.info("")
    log.info(f"  Log completo: {LOG_PATH.resolve()}")
    log.info(f"  MIDIs:        {OUTPUT_DIR.resolve()}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    log.info("")
    log.info("#" * 70)
    log.info(f"# SPIKE IMPROV_RNN — {datetime.now().isoformat(timespec='seconds')}")
    log.info(f"# Python {sys.version.split()[0]} — cwd {Path.cwd()}")
    log.info("#" * 70)

    try:
        phase_0_download()
        generator = phase_1_load_and_warmup()
        notation = phase_2_notation_tests(generator)
        latency = phase_3_latency(generator)
        scenarios = phase_4_musical_scenarios(generator)
        phase_5_summary(notation, latency, scenarios)
    except KeyboardInterrupt:
        log.warning("\n  Interrumpido con Ctrl+C")
    except Exception as e:
        log.exception(f"FALLO no manejado: {e}")
        raise


if __name__ == "__main__":
    main()
