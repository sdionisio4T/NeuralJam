"""
neuraljam/midi/phrase_detector.py

Captura frases del teclado MIDI con fidelidad rítmica completa, y
detecta si la frase termina con una nota en el rango de señal.

Nuevo en esta versión:
- El detector ahora devuelve una `Phrase` (dataclass) que combina:
    - notes: lista de NoteEvent (igual que antes)
    - has_signal: bool, True si la ÚLTIMA nota cayó en el rango
      definido por config.SIGNAL_NOTE_MIN/MAX

- Cuando has_signal=True, la nota grave se DESCARTA del campo notes.
  El primer que llega al modelo no la incluye. El stamp es solo control.

- Si la frase queda vacía después de descartar la señal (caso extremo:
  el usuario toca solo una nota grave), se descarta toda la frase como
  ruido.

Uso:
    detector = PhraseDetector()
    detector.start()
    try:
        while True:
            phrase = detector.wait_for_phrase()
            if phrase is None:
                continue
            if phrase.has_signal:
                # responder con ImprovRNN
            # ... procesar phrase.notes ...
    finally:
        detector.stop()

Verificación standalone:
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

    notes:      List[NoteEvent], ya filtrado (sin la nota de señal).
                Ordenado cronológicamente por start_time.
    has_signal: True si la última nota original cayó en el rango grave
                de señal. En ese caso, esa nota NO está en notes.
    """
    notes: List[NoteEvent] = field(default_factory=list)
    has_signal: bool = False


# ===========================================================================
# Detector
# ===========================================================================

class PhraseDetector:
    """
    Detecta y captura frases del teclado MIDI.

    Parámetros (todos opcionales; default desde config):
        port_name: nombre exacto/prefijo del puerto MIDI input
        silence_timeout: segundos de silencio para disparar fin de frase
        min_notes: mínimo de notas para considerar la frase válida
        signal_min/signal_max: rango (inclusive) de la nota de señal grave
    """

    def __init__(
        self,
        port_name: Optional[str] = None,
        silence_timeout: Optional[float] = None,
        min_notes: Optional[int] = None,
        signal_min: Optional[int] = None,
        signal_max: Optional[int] = None,
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
        self.signal_min = signal_min if signal_min is not None else config.SIGNAL_NOTE_MIN
        self.signal_max = signal_max if signal_max is not None else config.SIGNAL_NOTE_MAX

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
    # API pública
    # =====================================================================

    def start(self) -> None:
        if self._listener_thread is not None and self._listener_thread.is_alive():
            log.warning("PhraseDetector ya está corriendo, ignoring start()")
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
            f"(silencio={self.silence_timeout}s, min_notas={self.min_notes}, "
            f"signal=MIDI[{self.signal_min}-{self.signal_max}])"
        )

    def stop(self) -> None:
        if self._listener_thread is None:
            return
        self._stop_event.set()
        self._listener_thread.join(timeout=2.0)
        if self._listener_thread.is_alive():
            log.warning("Listener thread no terminó en 2s, forzando cierre")
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
            Phrase o None si timeout. La Phrase tiene notes ya filtrada
            (sin la nota de señal si has_signal=True).
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

        # Ordenar cronológicamente.
        ordered = sorted(
            self._completed,
            key=lambda e: (e.start_time, e.pitch),
        )

        # Detección de señal: la ÚLTIMA nota cae en el rango grave?
        last_note = ordered[-1]
        has_signal = self.signal_min <= last_note.pitch <= self.signal_max

        # Si hay señal, descartar esa nota del primer (decisión A2).
        if has_signal:
            notes = ordered[:-1]
            log.info(
                f"[SIGNAL] Última nota MIDI {last_note.pitch} está en rango grave. "
                f"Se descarta del primer. Próximo turno: ImprovRNN."
            )
        else:
            notes = ordered

        # Si después de quitar señal queda vacío o menos del mínimo, descartar.
        if len(notes) < self.min_notes:
            log.debug(
                f"Frase descartada tras filtrar señal: "
                f"{len(notes)} notas (min {self.min_notes})"
            )
            self._reset_phrase()
            return

        total_dur = notes[-1].start_time + notes[-1].duration
        log.info(
            f"Frase capturada: {len(notes)} notas, {total_dur:.2f}s, "
            f"signal={has_signal}"
        )
        self._phrase_queue.put(Phrase(notes=notes, has_signal=has_signal))
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
    print(f"Rango de señal: MIDI {detector.signal_min}-{detector.signal_max}")
    print()
    print("Probá:")
    print("  1) Tocá una frase normal. La frase llega con has_signal=False.")
    print("  2) Tocá una frase y terminá con una tecla MUY grave. La frase llega")
    print(f"     con has_signal=True y SIN la nota grave en notes.")
    print()
    print("Ctrl+C para salir.\n")

    try:
        while True:
            phrase = detector.wait_for_phrase()
            if phrase is None:
                continue
            tag = "🚨 SIGNAL" if phrase.has_signal else "        "
            print(f"\n=== {tag} Frase ({len(phrase.notes)} notas) ===")
            for i, evt in enumerate(phrase.notes):
                print(f"  [{i:2d}] pitch={evt.pitch:3d}  "
                      f"start={evt.start_time:6.3f}s  "
                      f"dur={evt.duration:6.3f}s  "
                      f"vel={evt.velocity:3d}")
            if phrase.notes:
                total = phrase.notes[-1].start_time + phrase.notes[-1].duration
                print(f"  Duración total: {total:.2f}s")
            print()
    except KeyboardInterrupt:
        print("\nDeteniendo...")
        detector.stop()
