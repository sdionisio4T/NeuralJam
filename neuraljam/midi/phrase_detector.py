"""
neuraljam/midi/phrase_detector.py

Captura frases del teclado MIDI con fidelidad ritmica completa.

Una frase es una secuencia de notas separadas por silencio (>= silence_timeout).
Cuando el silencio se detecta y hay suficientes notas (>= min_notes), la frase
se publica en la cola interna y el caller la recibe via wait_for_phrase().

El cambio de modelo ya no se hace por nota de senal MIDI — se hace desde
la terminal con teclas 1/2/3 (gestionado en neuraljam.py).

Uso:
    detector = PhraseDetector()
    detector.start()
    try:
        while True:
            phrase = detector.wait_for_phrase()
            if phrase is None:
                continue
            # procesar phrase.notes ...
    finally:
        detector.stop()

Verificacion standalone:
    python -m neuraljam.midi.phrase_detector
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue, Empty
from typing import List, Optional

import mido

from neuraljam import config
from neuraljam.midi.ports import resolve_input_port


log = logging.getLogger(__name__)


# ===========================================================================
# Tipos de salida
# ===========================================================================

@dataclass(frozen=True)
class NoteEvent:
    """Una nota capturada."""
    pitch: int
    start_time: float
    duration: float
    velocity: int


@dataclass(frozen=True)
class Phrase:
    """
    Una frase capturada del usuario.

    notes: List[NoteEvent], ordenado cronologicamente por start_time.
    """
    notes: List[NoteEvent] = field(default_factory=list)


# ===========================================================================
# Detector
# ===========================================================================

class PhraseDetector:
    """
    Detecta y captura frases del teclado MIDI.

    Parametros (todos opcionales; default desde config):
        port_name: nombre exacto/prefijo del puerto MIDI input
        silence_timeout: segundos de silencio para disparar fin de frase
        min_notes: minimo de notas para considerar la frase valida
    """

    def __init__(
        self,
        port_name: Optional[str] = None,
        silence_timeout: Optional[float] = None,
        min_notes: Optional[int] = None,
    ):
        self.port_name = port_name or config.MIDI_INPUT_NAME
        self.silence_timeout = (
            silence_timeout if silence_timeout is not None
            else config.SILENCE_TIMEOUT
        )
        self.min_notes = (
            min_notes if min_notes is not None
            else config.MIN_NOTES_TO_RESPOND
        )

        # --- Estado del listener ---
        self._port = None
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self._active: dict = {}
        self._completed: List[NoteEvent] = []
        self._phrase_start_abs: Optional[float] = None
        self._last_release_abs: Optional[float] = None

        self._phrase_queue: Queue = Queue()

    # =====================================================================
    # API publica
    # =====================================================================

    def start(self) -> None:
        if self._listener_thread is not None and self._listener_thread.is_alive():
            log.warning("PhraseDetector ya esta corriendo, ignoring start()")
            return

        try:
            real_port_name = resolve_input_port(self.port_name)
            self._port = mido.open_input(real_port_name)
        except RuntimeError:
            raise
        except (IOError, OSError) as e:
            available = mido.get_input_names()
            raise RuntimeError(
                f"No se pudo abrir puerto MIDI '{self.port_name}'. "
                f"Disponibles: {available}"
            ) from e

        self._stop_event.clear()
        self._listener_thread = threading.Thread(
            target=self._listener_loop,
            name="PhraseDetector-listener",
            daemon=True,
        )
        self._listener_thread.start()
        log.info(
            f"PhraseDetector escuchando '{self.port_name}' "
            f"(silencio={self.silence_timeout}s, min_notas={self.min_notes})"
        )

    def stop(self) -> None:
        if self._listener_thread is None:
            return
        self._stop_event.set()
        self._listener_thread.join(timeout=2.0)
        if self._listener_thread.is_alive():
            log.warning("Listener thread no termino en 2s, forzando cierre")
        self._listener_thread = None

        if self._port is not None:
            try:
                self._port.close()
            except Exception:
                log.exception("Error cerrando puerto MIDI (no fatal)")
            self._port = None

        log.info("PhraseDetector detenido")

    def wait_for_phrase(self, timeout: Optional[float] = None) -> Optional[Phrase]:
        """
        Bloquea hasta que haya una frase lista o timeout.

        Returns:
            Phrase o None si timeout.
        """
        try:
            return self._phrase_queue.get(timeout=timeout)
        except Empty:
            return None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # =====================================================================
    # Internals
    # =====================================================================

    def _listener_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                for msg in self._port.iter_pending():
                    self._handle_message(msg)
                self._check_phrase_end()
                time.sleep(0.001)
        except Exception:
            log.exception("Error fatal en listener loop, terminando thread")

    def _handle_message(self, msg) -> None:
        now = time.monotonic()
        is_note_on = msg.type == "note_on" and msg.velocity > 0
        is_note_off = (
            msg.type == "note_off"
            or (msg.type == "note_on" and msg.velocity == 0)
        )

        if is_note_on:
            if self._phrase_start_abs is None:
                self._phrase_start_abs = now
            self._active[msg.note] = (now, msg.velocity)

        elif is_note_off:
            if msg.note not in self._active:
                return
            start_abs, velocity = self._active.pop(msg.note)
            event = NoteEvent(
                pitch=msg.note,
                start_time=start_abs - self._phrase_start_abs,
                duration=now - start_abs,
                velocity=velocity,
            )
            self._completed.append(event)
            self._last_release_abs = now

    def _check_phrase_end(self) -> None:
        if not self._completed:
            return
        if self._active:
            return
        if self._last_release_abs is None:
            return

        now = time.monotonic()
        if now - self._last_release_abs < self.silence_timeout:
            return

        if len(self._completed) < self.min_notes:
            log.debug(
                f"Frase descartada: {len(self._completed)} notas "
                f"(min {self.min_notes})"
            )
            self._reset_phrase()
            return

        # Ordenar cronologicamente.
        notes = sorted(
            self._completed,
            key=lambda e: (e.start_time, e.pitch),
        )

        total_dur = notes[-1].start_time + notes[-1].duration
        log.info(
            f"Frase capturada: {len(notes)} notas, {total_dur:.2f}s"
        )
        self._phrase_queue.put(Phrase(notes=notes))
        self._reset_phrase()

    def _reset_phrase(self) -> None:
        self._completed.clear()
        self._phrase_start_abs = None
        self._last_release_abs = None


# ===========================================================================
# Test manual: python -m neuraljam.midi.phrase_detector
# ===========================================================================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    print("Puertos MIDI disponibles:")
    for name in mido.get_input_names():
        print(f"  {name!r}")
    print()

    try:
        detector = PhraseDetector()
        detector.start()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Listening en '{detector.port_name}'.")
    print()
    print("Toca frases. Ctrl+C para salir.\n")

    try:
        while True:
            phrase = detector.wait_for_phrase()
            if phrase is None:
                continue
            print(f"\n=== Frase ({len(phrase.notes)} notas) ===")
            for i, evt in enumerate(phrase.notes):
                print(f"  [{i:2d}] pitch={evt.pitch:3d}  "
                      f"start={evt.start_time:6.3f}s  "
                      f"dur={evt.duration:6.3f}s  "
                      f"vel={evt.velocity:3d}")
            if phrase.notes:
                total = phrase.notes[-1].start_time + phrase.notes[-1].duration
                print(f"  Duracion total: {total:.2f}s")
            print()
    except KeyboardInterrupt:
        print("\nDeteniendo...")
        detector.stop()
