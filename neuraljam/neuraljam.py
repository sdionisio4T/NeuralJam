"""
neuraljam.py — Entry point principal.

Orquesta duo MIDI con tres modelos en RAM:
    - MelodyRNN      (tecla 1): respuesta melodica, cuantizada, default.
    - ImprovRNN      (tecla 2): chord-conditioned, armonico.
    - PerformanceRNN (tecla 3): polifonico, expresivo, alta resolucion.

Cambio de modelo en caliente: presiона 1, 2 o 3 en la terminal mientras
el sistema corre. El cambio es inmediato — aplica a la siguiente frase.

Flujo:
    Casio -> PhraseDetector -> engine.respond(phrase, model_key)
                                        |
                          Player -> MidiOutput -> Studio One

Uso:
    python neuraljam.py                     # arranca con melody
    python neuraljam.py --mode improv       # arranca con improv
    python neuraljam.py --debug             # logging detallado
    python neuraljam.py --clock "s1-Clock"  # sincroniza con Studio One

Ctrl+C para salir limpio (all-notes-off + close de puertos).
"""

import argparse
import logging
import sys
import threading
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
    parser.add_argument(
        "--preload",
        metavar="CARPETA",
        default=None,
        help=(
            "Carpeta con MIDIs externos para precargar en el banco de memoria. "
            "Ej: --preload midi_externos\\bill_evans"
        ),
    )
    parser.add_argument(
        "--clock",
        metavar="PUERTO",
        default=None,  # None = usa config.MIDI_CLOCK_PORT
        help=(
            "Puerto MIDI para recibir clock de Studio One. "
            f"Default: config.MIDI_CLOCK_PORT. Ej: --clock \"S1-Clock\""
        ),
    )
    parser.add_argument(
        "--sync-beat",
        action="store_true",
        default=False,
        help=(
            "Esperar al próximo compás antes de reproducir (requiere --clock). "
            "Desactivado por default — el tempo se sincroniza igual sin esto."
        ),
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
    preload_folder = args.preload
    # Si no se pasó --clock, usar el puerto del config como default
    clock_port = args.clock if args.clock is not None else config.MIDI_CLOCK_PORT
    sync_beat = args.sync_beat

    setup_logging(args.debug)
    log = logging.getLogger("neuraljam")
    log.info(f"NeuralJam arrancando, default mode='{config.MODE}'")

    # Imports diferidos para que --help no cargue TF.
    from note_seq.protobuf import music_pb2

    from neuraljam.generation import GenerationEngine
    from neuraljam.generation.humanize import humanize
    from neuraljam.harmony import Progression
    from neuraljam.memory.bank import MemoryBank
    from neuraljam.memory.saver import save_phrase
    from neuraljam.midi import MidiOutput, PhraseDetector
    from neuraljam.models import load_all_models
    from neuraljam.playback import Player
    from neuraljam.scheduler import Scheduler
    from neuraljam.analysis.groove import GrooveEngine
    from neuraljam.modes import MODES, next_mode
    from neuraljam.recording import SessionRecorder
    from neuraljam.subconscious.engine import SubconsciousEngine, phrase_to_seq

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

    # Estado de modelo compartido: se actualiza desde el hilo del teclado.
    model_state = {"current": default_key}

    log.info(
        f"Modelos cargados: {[k.upper() for k in models]} | "
        f"Activo al arrancar: {default_key.upper()}RNN"
    )

    progression = Progression.from_config()
    log.info(f"Progresión (solo se usa con improv): {progression!r}")

    # Lock compartido: garantiza que solo un thread llama a generate() a la vez
    model_lock = threading.Lock()

    engine = GenerationEngine(models, progression, model_lock=model_lock)
    detector = PhraseDetector()
    midi_out = MidiOutput()
    player = Player(midi_out)

    # MIDI Clock (Fase 7) — Studio One como master de tempo
    clock = None
    if clock_port:
        from neuraljam.midi.clock import MidiClock
        clock = MidiClock(clock_port)
        clock.start()
        log.info(
            f"MIDI Clock activo en '{clock_port}'. "
            "Presioná Play en Studio One para sincronizar."
        )

    # Listener de teclado: guardar [s], modelo [1/2/3], modo [m], baseline [b]
    save_flag = threading.Event()
    mode_state = {"current": "normal"}
    baseline_state = {"active": False, "prev_mode": "normal"}
    _start_save_listener(save_flag, log)
    _start_model_switcher(models, model_state, log)
    _start_mode_cycler(mode_state, log)
    _start_baseline_listener(baseline_state, mode_state, log)

    bank = MemoryBank(maxlen=8)
    if preload_folder:
        log.info(f"Precargando MIDIs desde: {preload_folder}")
        n = bank.preload(preload_folder, max_files=30, chunk_bars=4)
        if n == 0:
            log.warning("Preload: ningún MIDI cargado — verificá la carpeta.")

    recorder = SessionRecorder()
    groove = GrooveEngine()
    subconscious = SubconsciousEngine(bank, model_lock=model_lock)
    if "improv" in models:
        subconscious.set_improv_model(models["improv"])
    scheduler = Scheduler(response_probability=0.85, max_consecutive_silences=2)
    log.info(
        "MemoryBank, SubconsciousEngine y Scheduler inicializados  |  "
        "Modo: NORMAL  |  [m] para ciclar modos"
    )

    # MusicVAE (Fase 4) — contexto por interpolación en espacio latente.
    # MelodyRNN SIEMPRE responde; MusicVAE solo enriquece el context_seq.
    # Si no se puede cargar, el sistema sigue funcionando con banco directo.
    try:
        from neuraljam.models.music_vae_loader import load_music_vae
        music_vae = load_music_vae(
            config.MUSIC_VAE_CHECKPOINT_DIR,
            config.MUSIC_VAE_URL,
        )
        if music_vae:
            subconscious.set_music_vae(music_vae)
        else:
            log.warning(
                "MusicVAE no disponible — contexto desde banco directamente. "
                "El sistema funciona igual."
            )
    except Exception:
        log.warning("MusicVAE: error al intentar cargar (no fatal).", exc_info=True)

    # ---- Loop principal -------------------------------------------------

    try:
        detector.start()
        midi_out.open()
        log.info(
            "Sistema listo. Toca una frase y espera la respuesta. Ctrl+C para salir."
        )

        turn = 0
        last_key = None

        while True:
            phrase = detector.wait_for_phrase()
            if phrase is None:
                continue

            turn += 1
            # Modo activo (cicla con [m])
            current_mode = MODES[mode_state["current"]]

            # Modelo elegido por el usuario desde la terminal (teclas 1/2/3).
            # Si el modelo seleccionado no está cargado, cae al default.
            key = model_state["current"]
            if key not in models:
                log.warning(
                    f"Modelo '{key}' no esta cargado. Usando {default_key}."
                )
                key = default_key
                model_state["current"] = default_key

            # Log: avisa solo cuando cambia el modelo
            if key != last_key:
                log.info(f"[MODEL] {key.upper()}RNN  (cambio)")
            else:
                log.info(f"[MODEL] {key.upper()}RNN")
            last_key = key

            total_dur = phrase.notes[-1].start_time + phrase.notes[-1].duration
            log.info(
                f"--- Turno {turn} --- {len(phrase.notes)} notas, {total_dur:.2f}s"
            )

            # BPM en vivo (del clock) o fallback — se necesita antes del if/else
            live_qpm = (
                clock.qpm if (clock and clock.has_sync)
                else config.QPM_FALLBACK
            )

            # ---- Baseline ([b]) — modelo limpio, sin capas -----------------
            if baseline_state["active"]:
                log.info(f"[BASELINE] {key.upper()}RNN — sin scheduler, sin contexto")
                context = None
                temp = 1.0
                bars = 2

            # ---- Modo normal/imitación/libre/experimental ------------------
            else:
                # Scheduler: decidir si responder este turno
                decision = scheduler.should_respond(mode=current_mode)
                if decision == "silent":
                    log.info(
                        f"[SCHEDULER] Silencio intencional "
                        f"(turno {scheduler.turn_count}). Escuchando..."
                    )
                    user_ns = phrase_to_seq(phrase.notes, qpm=config.QPM_FALLBACK)
                    subconscious.trigger(user_ns, music_pb2.NoteSequence(), mode=current_mode)
                    continue

                context = subconscious.get_context() if current_mode.use_memory else None
                if context is not None:
                    log.debug(f"Contexto subconciente: {len(context.notes)} notas")

                phrase_dur = phrase.notes[-1].start_time + phrase.notes[-1].duration
                temp = scheduler.temperature(len(phrase.notes), phrase_dur, mode=current_mode)
                bars = scheduler.response_bars(phrase_dur, bpm=live_qpm, mode=current_mode)
                log.debug(
                    f"[{current_mode.display}] Temperatura: {temp:.3f} | Compases: {bars} | QPM: {live_qpm:.1f}"
                )

            t0 = time.perf_counter()
            response = engine.respond(
                phrase.notes,
                model_key=key,
                context_seq=context,
                temperature=temp,
                response_bars=bars,
                qpm_override=live_qpm,
            )
            gen_time = time.perf_counter() - t0

            if response is None:
                log.warning("Sin respuesta (None). Esperando próxima frase.")
                continue

            # Esperar al próximo downbeat solo si --sync-beat está activo
            if sync_beat and clock and clock.has_sync:
                clock.wait_for_downbeat(max_wait_bars=0.75)

            log.info(
                f"Generación: {gen_time:.2f}s | "
                f"Reproduciendo {response.total_time:.2f}s... "
                f"[temp={temp:.2f}, bars={bars}, qpm={live_qpm:.0f}]"
            )

            # Swing + humanize antes de reproducir
            response = humanize(
                response,
                swing=0.08,
                velocity_variance=12,
                qpm=live_qpm,
            )

            user_ns = phrase_to_seq(phrase.notes, qpm=config.QPM_FALLBACK)

            # Groove: analizar perfil rítmico de la frase del usuario
            groove_profile = groove.update(user_ns, response, qpm=live_qpm)
            log.debug(
                f"[GROOVE] {groove_profile}  "
                f"→ temp_delta={groove.temperature_delta():+.2f}  "
                f"bars_hint={groove.bars_hint()}"
            )

            # Subconciente en background solo fuera de baseline
            if not baseline_state["active"]:
                subconscious.trigger(user_ns, response, mode=current_mode)

            # Grabar turno completo (usuario + IA)
            recorder.add_turn(user_ns, response, qpm=live_qpm)

            player.play(response)

            # Guardar si el usuario presionó 's' durante la reproducción
            if save_flag.is_set():
                save_flag.clear()
                save_phrase(user_ns, tag="user")
                save_phrase(response, tag="ai")
                log.info(
                    "Frase guardada en saved_phrases/. "
                    "Subila a Drive para fine-tuning en Colab."
                )

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
        # Exportar sesión completa al salir
        if recorder.turn_count > 0:
            exported = recorder.export()
            if exported:
                log.info(f"Sesión guardada en: {exported}")
        log.info("Apagado limpio.")


def _start_model_switcher(models: dict, model_state: dict, log) -> None:
    """
    Teclas 1/2/3 para cambiar el modelo activo en caliente.

    El mapeo es dinamico: si solo hay 2 modelos cargados, [1] y [2].
    Si el modelo no esta cargado, la tecla no hace nada.

    model_state: dict con clave "current" (string del modelo activo).
    """
    available = list(models.keys())
    key_map = {str(i + 1): name for i, name in enumerate(available) if i < 9}

    def _make_switch(name):
        def _do():
            model_state["current"] = name
            log.info(f"\n>>> Modelo: {name.upper()}RNN <<<\n")
        return _do

    def _listen():
        try:
            import keyboard
            for key, name in key_map.items():
                keyboard.add_hotkey(key, _make_switch(name))
            hint = "   ".join(f"[{k}] {v.upper()}" for k, v in key_map.items())
            log.info(f"Modelos disponibles -> {hint}   [s] guardar frase")
            keyboard.wait()
        except ImportError:
            log.warning("Selector de modelo desactivado -- pip install keyboard")
        except Exception:
            log.warning("Selector de modelo no disponible.", exc_info=True)

    threading.Thread(target=_listen, name="ModelSwitcher", daemon=True).start()


def _start_baseline_listener(baseline_state: dict, mode_state: dict, log) -> None:
    """
    Tecla [b] para activar/desactivar baseline.

    Baseline activo: el modelo corre limpio — sin scheduler, sin contexto,
    sin subconciente. Temperatura 1.0, 2 compases fijos.
    Sirve para comparar si las capas realmente mejoran el resultado.
    [b] de nuevo vuelve al modo anterior.
    """
    def _toggle():
        if baseline_state["active"]:
            baseline_state["active"] = False
            prev = baseline_state["prev_mode"]
            mode_state["current"] = prev
            log.info(f"\n>>> BASELINE desactivado — volviendo a {prev.upper()} <<<\n")
        else:
            baseline_state["prev_mode"] = mode_state["current"]
            baseline_state["active"] = True
            log.info("\n>>> BASELINE activo — subconciente desactivado <<<\n")

    def _listen():
        try:
            import keyboard
            keyboard.add_hotkey("b", _toggle)
            keyboard.wait()
        except ImportError:
            log.warning("Baseline desactivado -- pip install keyboard")
        except Exception:
            log.warning("Listener de baseline no disponible.", exc_info=True)

    threading.Thread(target=_listen, name="BaselineListener", daemon=True).start()


def _start_mode_cycler(mode_state: dict, log) -> None:
    """
    Tecla [m] para ciclar entre modos: normal → imitación → libre → experimental.
    El modo actual se almacena en mode_state["current"].
    """
    from neuraljam.modes import next_mode, MODES

    def _cycle():
        current = mode_state["current"]
        nxt = next_mode(current)
        mode_state["current"] = nxt
        log.info(f"\n>>> Modo: {MODES[nxt].display} <<<\n")

    def _listen():
        try:
            import keyboard
            keyboard.add_hotkey("m", _cycle)
            keyboard.wait()
        except ImportError:
            log.warning("Selector de modo desactivado -- pip install keyboard")
        except Exception:
            log.warning("Selector de modo no disponible.", exc_info=True)

    threading.Thread(target=_listen, name="ModeCycler", daemon=True).start()


def _start_save_listener(save_flag: threading.Event, log) -> None:
    """
    Inicia un thread que escucha la tecla 's' y activa save_flag.
    Si la librería 'keyboard' no está instalada, lo indica y sigue sin guardar.
    """
    def _listen():
        try:
            import keyboard
            keyboard.add_hotkey("s", save_flag.set)
            log.info("Guardado activo: presioná 's' después de un turno para guardar la frase.")
            keyboard.wait()
        except ImportError:
            log.warning(
                "Guardado de frases desactivado. "
                "Para activarlo: pip install keyboard"
            )
        except Exception:
            log.warning("Listener de teclado no disponible (¿falta permisos?).")

    threading.Thread(target=_listen, name="KeyboardListener", daemon=True).start()


if __name__ == "__main__":
    main()
