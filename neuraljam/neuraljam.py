"""
neuraljam.py — Entry point principal.

Orquesta dúo MIDI con dos modelos en RAM:
    - MelodyRNN (default): dialoga con tu frase, responde melódicamente.
    - ImprovRNN: aparece cuando terminás una frase con una nota grave
      (rango definido en config.SIGNAL_NOTE_MIN/MAX).

Flujo:
    Casio → PhraseDetector → (señal?) → engine.respond(phrase, model_key)
                                                ↓
                                  Player → MidiOutput → Studio One

Uso:
    python neuraljam.py                  # default melody
    python neuraljam.py --mode improv    # default improv (señal cambia a melody)
    python neuraljam.py --debug          # logging detallado

Ctrl+C para salir limpio (all-notes-off + close de puertos).
"""

import argparse
import logging
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeuralJam — AI jazz improvisation assistant",
    )
    parser.add_argument(
        "--mode",
        default="melody",
        choices=["melody", "improv", "performance"],
        help="Modelo default (sin señal). Señal cambia al otro. Default: melody.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Logging detallado (DEBUG level)",
    )
    return parser.parse_args()


def setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def main():
    args = parse_args()

    from neuraljam import config

    config.MODE = args.mode
    config.ensure_dirs()

    setup_logging(args.debug)
    log = logging.getLogger("neuraljam")
    log.info(f"NeuralJam arrancando, default mode='{config.MODE}'")

    # Imports diferidos para que --help no cargue TF.
    from neuraljam.generation import GenerationEngine
    from neuraljam.harmony import Progression
    from neuraljam.midi import MidiOutput, PhraseDetector
    from neuraljam.models import load_all_models
    from neuraljam.playback import Player

    # ---- Bootstrap ------------------------------------------------------

    log.info("Cargando TODOS los modelos disponibles (tarda ~20s en cold)...")
    models = load_all_models(do_warmup=True)

    if not models:
        log.error("No se cargó ningún modelo. Abortando.")
        sys.exit(1)

    # Validar que el default está disponible
    default_key = config.MODE
    if default_key not in models:
        log.warning(
            f"Modo default '{default_key}' no cargó. "
            f"Usando '{list(models.keys())[0]}' como default."
        )
        default_key = list(models.keys())[0]

    # Decidir el "otro" para la señal. Solo tiene sentido si hay >1 modelo.
    if "melody" in models and "improv" in models:
        other_key = "improv" if default_key == "melody" else "melody"
    else:
        other_key = default_key  # solo un modelo cargado, señal no hace nada
        log.warning(
            f"Solo un modelo cargado: la señal no va a cambiar nada. "
            f"Cargados: {list(models.keys())}"
        )

    log.info(f"Default: {default_key.upper()} | Con señal: {other_key.upper()}")

    progression = Progression.from_config()
    log.info(f"Progresión (solo se usa con improv): {progression!r}")

    engine = GenerationEngine(models, progression)
    detector = PhraseDetector()
    midi_out = MidiOutput()
    player = Player(midi_out)

    # ---- Loop principal -------------------------------------------------

    try:
        detector.start()
        midi_out.open()
        log.info(
            "Sistema listo. Tocá una frase y esperá la respuesta. Ctrl+C para salir."
        )
        log.info(
            f"Para invocar {other_key.upper()}: terminá tu frase con una nota "
            f"en MIDI {config.SIGNAL_NOTE_MIN}-{config.SIGNAL_NOTE_MAX}."
        )

        turn = 0
        last_key = None

        while True:
            phrase = detector.wait_for_phrase()
            if phrase is None:
                continue

            turn += 1
            # Decidir qué modelo usar
            key = other_key if phrase.has_signal else default_key

            # Log de modelo: switch si cambió, normal si no
            if key != last_key:
                log.info(f"[MODEL SWITCH] -> {key.upper()}RNN")
            else:
                log.info(f"[MODEL] {key.upper()}RNN")
            last_key = key

            total_dur = phrase.notes[-1].start_time + phrase.notes[-1].duration
            log.info(
                f"--- Turno {turn} --- {len(phrase.notes)} notas, {total_dur:.2f}s"
            )

            t0 = time.perf_counter()
            response = engine.respond(phrase.notes, model_key=key)
            gen_time = time.perf_counter() - t0

            if response is None:
                log.warning("Sin respuesta (None). Esperando próxima frase.")
                continue

            log.info(
                f"Generación: {gen_time:.2f}s | "
                f"Reproduciendo {response.total_time:.2f}s..."
            )
            player.play(response)
            log.info("Listo. Tu turno.\n")

    except KeyboardInterrupt:
        log.info("\nCtrl+C detectado, cerrando...")
    except Exception:
        log.exception("Error fatal en el loop principal")
        raise
    finally:
        try:
            detector.stop()
        except Exception:
            log.exception("Error parando detector (no fatal)")
        try:
            midi_out.close()
        except Exception:
            log.exception("Error cerrando MIDI out (no fatal)")
        log.info("Apagado limpio.")


if __name__ == "__main__":
    main()
