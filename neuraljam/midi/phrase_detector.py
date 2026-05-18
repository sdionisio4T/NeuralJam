"""
neuraljam/midi/phrase_detector.py

Captura frases del teclado MIDI con fidelidad rítmica completa.

Por qué importa: el spike de ImprovRNN confirmó que el modelo responde
fuertemente al ritmo del primer (no solo a los pitches). Si capturamos
mal el timing, las velocities o las duraciones, perdemos la palanca
principal sobre la calidad del output. Por eso este módulo prioriza
fidelidad sobre simplicidad.

Diseño:
- Thread separado escucha eventos MIDI sin bloquear el resto del sistema.
- Mantiene estado de notas activas (presionadas) y completadas (release ya
  registrado).
- Fin de frase = todas las teclas levantadas + SILENCE_TIMEOUT segundos
  sin nuevos eventos.
- Output: lista de NoteEvent (pitch, start_time relativo, duration,
  velocity). El módulo generation/ se encarga de convertir a NoteSequence
  si su modelo lo necesita.

Uso típico:
    detector = PhraseDetector()
    detector.start()
    try:
        while True:
            phrase = detector.wait_for_phrase()
            # ... procesar phrase ...
    finally:
        detector.stop()

Verificación standalone (sin integrar al resto del sistema):
    python -m neuraljam.midi.phrase_detector
"""

import logging
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue
from typing import List, Optional

import mido

from neuraljam import config
from neuraljam.midi.ports import resolve_input_port

log = logging.getLogger(__name__)


# ===========================================================================
# Tipo de salida
# ===========================================================================


@dataclass(frozen=True)
class NoteEvent:
    """
    Una nota completa capturada del usuario.

    Inmutable (frozen=True): una vez emitida en una frase, no debe
    modificarse. Las transformaciones (transposición, cuantización, etc.)
    deben crear NoteEvent nuevos, no mutar los existentes.

    Atributos:
        pitch:      número MIDI (0-127)
        start_time: segundos desde el INICIO de la frase (no absoluto).
                    La primera nota de una frase tiene start_time = 0.0.
        duration:   segundos entre note-on y note-off.
        velocity:   velocity del note-on (0-127).
    """

    pitch: int
    start_time: float
    duration: float
    velocity: int


# ===========================================================================
# Detector
# ===========================================================================


class PhraseDetector:
    """
    Detecta y captura frases del teclado MIDI.

    Thread-safe en la API pública:
    - start()/stop() llamados desde el thread principal
    - wait_for_phrase() bloquea el caller hasta que haya frase
    - El listener corre en su propio thread y no comparte estado mutable
      con el caller (las frases viajan por una Queue)

    Parámetros (todos opcionales; default desde config):
        port_name: nombre exacto del puerto MIDI input
        silence_timeout: segundos de silencio para disparar fin de frase
        min_notes: mínimo de notas para considerar la frase válida
    """

    def __init__(
        self,
        port_name: Optional[str] = None,
        silence_timeout: Optional[float] = None,
        min_notes: Optional[int] = None,
    ):
        self.port_name = port_name or config.MIDI_INPUT_NAME
        self.silence_timeout = (
            silence_timeout if silence_timeout is not None else config.SILENCE_TIMEOUT
        )
        self.min_notes = (
            min_notes if min_notes is not None else config.MIN_NOTES_TO_RESPOND
        )

        # --- Estado del listener (acceso solo desde el thread listener) ---
        self._port = None
        self._listener_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Notas activas: pitch -> (start_abs_time, velocity)
        self._active: dict = {}
        # Notas completadas en la frase actual, en orden de aparición
        self._completed: List[NoteEvent] = []
        # Tiempo absoluto del primer note-on de la frase actual
        self._phrase_start_abs: Optional[float] = None
        # Tiempo absoluto del último note-off (para medir silencio)
        self._last_release_abs: Optional[float] = None

        # --- Salida thread-safe ---
        # Las frases completas se ponen acá; wait_for_phrase() las consume.
        self._phrase_queue: Queue = Queue()

    # =====================================================================
    # API pública
    # =====================================================================

    def start(self) -> None:
        """
        Abre el puerto MIDI y lanza el thread listener.

        Idempotente: si ya está corriendo, log warning y vuelve.
        Raises RuntimeError si el puerto no existe o no se puede abrir
        (con la lista de puertos disponibles en el mensaje).
        """
        if self._listener_thread is not None and self._listener_thread.is_alive():
            log.warning("PhraseDetector ya está corriendo, ignoring start()")
            return

        try:
            real_port_name = resolve_input_port(self.port_name)
            self._port = mido.open_input(real_port_name)
        except RuntimeError:
            # resolve_input_port ya armó el mensaje; lo dejamos propagar.
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
            daemon=True,  # daemon: muere si el main thread muere
        )
        self._listener_thread.start()
        log.info(
            f"PhraseDetector escuchando '{self.port_name}' "
            f"(silencio={self.silence_timeout}s, min_notas={self.min_notes})"
        )

    def stop(self) -> None:
        """
        Para el listener y cierra el puerto. Idempotente.

        Bloquea hasta 2 segundos esperando que el thread termine
        limpiamente. Si no termina (no debería pasar), libera el puerto
        igual.
        """
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

    def wait_for_phrase(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[List[NoteEvent]]:
        """
        Bloquea hasta que haya una frase lista o se cumpla el timeout.

        Args:
            timeout: segundos a esperar. None = bloquear indefinidamente.

        Returns:
            Lista de NoteEvent representando la frase, o None si pasó
            el timeout sin que apareciera una frase.
        """
        try:
            return self._phrase_queue.get(timeout=timeout)
        except Empty:
            return None

    def __enter__(self):
        """Permite usar el detector con `with`."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False

    # =====================================================================
    # Internals (privados del thread listener)
    # =====================================================================
    # Los métodos `_handle_message` y `_check_phrase_end` están separados
    # del listener loop para permitir testear la máquina de estados sin
    # abrir un puerto MIDI real (inyectando mensajes simulados).

    def _listener_loop(self) -> None:
        """Loop principal del thread listener."""
        try:
            while not self._stop_event.is_set():
                # iter_pending() devuelve solo mensajes ya en buffer,
                # NO bloquea. Si bloqueara, no detectaríamos fin de frase
                # cuando el usuario simplemente para de tocar.
                for msg in self._port.iter_pending():
                    self._handle_message(msg)
                self._check_phrase_end()
                # 1ms de pausa: suficiente para captura precisa, evita
                # quemar CPU al 100%.
                time.sleep(0.001)
        except Exception:
            log.exception("Error fatal en listener loop, terminando thread")

    def _handle_message(self, msg) -> None:
        """
        Procesa un mensaje MIDI individual y actualiza el estado.

        Algunos teclados envían note-on con velocity=0 en lugar de
        note-off real. Ambos casos se tratan como note-off.
        """
        now = time.monotonic()

        is_note_on = msg.type == "note_on" and msg.velocity > 0
        is_note_off = msg.type == "note_off" or (
            msg.type == "note_on" and msg.velocity == 0
        )

        if is_note_on:
            # Si es la primera nota de la frase, anclamos el tiempo cero.
            if self._phrase_start_abs is None:
                self._phrase_start_abs = now
            self._active[msg.note] = (now, msg.velocity)
            log.debug(f"note_on  pitch={msg.note} vel={msg.velocity}")

        elif is_note_off:
            if msg.note not in self._active:
                # Note-off sin note-on previo: artefacto, ignorar.
                log.debug(f"note_off huérfano pitch={msg.note}")
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
            log.debug(
                f"note_off pitch={event.pitch} "
                f"start={event.start_time:.3f} dur={event.duration:.3f}"
            )

    def _check_phrase_end(self) -> None:
        """
        Decide si la frase actual terminó. Si terminó y cumple los
        criterios, la pone en la cola y resetea el estado.

        Condiciones para considerar terminada:
        1. Hay al menos una nota completada (con note-off).
        2. No quedan teclas presionadas (active vacío).
        3. Pasaron silence_timeout segundos desde el último note-off.

        Si la frase tiene menos de min_notes, se descarta silenciosamente
        (no llega a la cola, no llega al engine).
        """
        if not self._completed:
            return
        if self._active:
            return
        if self._last_release_abs is None:
            return

        now = time.monotonic()
        if now - self._last_release_abs < self.silence_timeout:
            return

        # La frase terminó. Validar mínimo de notas.
        if len(self._completed) < self.min_notes:
            log.debug(
                f"Frase descartada: {len(self._completed)} notas (min {self.min_notes})"
            )
            self._reset_phrase()
            return

        # Ordenar cronológicamente. Razón: append() se hace al recibir
        # el note-off, pero los consumers esperan orden por start_time.
        # Tiebreak por pitch (orden musical estándar para notas que arrancan
        # exactamente al mismo tiempo, ej. acordes alineados).
        phrase = sorted(
            self._completed,
            key=lambda e: (e.start_time, e.pitch),
        )
        total_dur = phrase[-1].start_time + phrase[-1].duration
        log.info(f"Frase capturada: {len(phrase)} notas, {total_dur:.2f}s")
        self._phrase_queue.put(phrase)
        self._reset_phrase()

    def _reset_phrase(self) -> None:
        """
        Limpia el estado de la frase actual SIN tocar las notas activas.

        Las activas no se tocan porque podrían pertenecer a una frase
        que el usuario empezó justo cuando se descartaba la anterior.
        """
        self._completed.clear()
        self._phrase_start_abs = None
        self._last_release_abs = None


# ===========================================================================
# Test manual (correr: python -m neuraljam.midi.phrase_detector)
# ===========================================================================
# Conecta al teclado real e imprime cada frase capturada con timing
# detallado. Útil para verificar a ojo que la captura es fiel antes de
# integrar el detector al sistema completo.

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
    print(
        f"Tocá una frase, soltá todas las teclas, esperá {detector.silence_timeout}s."
    )
    print("Ctrl+C para salir.\n")

    try:
        while True:
            phrase = detector.wait_for_phrase()
            if phrase is None:
                continue
            print(f"\n=== Frase capturada ({len(phrase)} notas) ===")
            for i, evt in enumerate(phrase):
                print(
                    f"  [{i:2d}] pitch={evt.pitch:3d}  "
                    f"start={evt.start_time:6.3f}s  "
                    f"dur={evt.duration:6.3f}s  "
                    f"vel={evt.velocity:3d}"
                )
            total = phrase[-1].start_time + phrase[-1].duration
            print(f"  Duración total: {total:.2f}s\n")
    except KeyboardInterrupt:
        print("\nDeteniendo...")
        detector.stop()
