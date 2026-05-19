"""
neuraljam/midi/output.py

Envío MIDI al DAW vía loopMIDI.

Wrapper liviano sobre mido.open_output. No conoce de NoteSequences ni
de música — solo abre el puerto, envía mensajes individuales, y se
asegura de cerrar limpio.

El módulo de playback se encarga de convertir un NoteSequence en una
secuencia de mensajes MIDI con timing; este módulo solo los emite.
"""

import logging
from typing import Optional

import mido

from neuraljam import config
from neuraljam.midi.ports import resolve_output_port


log = logging.getLogger(__name__)


class MidiOutput:
    """
    Puerto MIDI de salida. Context manager: usar con `with`.

    Ejemplo:
        with MidiOutput() as out:
            out.note_on(60, velocity=80)
            time.sleep(0.5)
            out.note_off(60)

    panic() envía all-notes-off en los 16 canales. Llamarlo en cleanup
    si una respuesta fue interrumpida.
    """

    def __init__(self, port_name: Optional[str] = None):
        self.port_name = port_name or config.MIDI_OUTPUT_NAME
        self._port = None

    def open(self) -> None:
        if self._port is not None:
            log.warning("MidiOutput ya está abierto")
            return
        real_name = resolve_output_port(self.port_name)
        self._port = mido.open_output(real_name)
        log.info(f"MidiOutput abierto: {real_name!r}")

    def close(self) -> None:
        if self._port is None:
            return
        try:
            self.panic()
        except Exception:
            log.exception("Error en panic() al cerrar (no fatal)")
        try:
            self._port.close()
        except Exception:
            log.exception("Error cerrando puerto MIDI out (no fatal)")
        self._port = None
        log.info("MidiOutput cerrado")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    # ----------------------------------------------------------------- #
    # Envío
    # ----------------------------------------------------------------- #

    def note_on(self, pitch: int, velocity: int = 80, channel: int = 0) -> None:
        self._send(mido.Message("note_on", note=pitch, velocity=velocity, channel=channel))

    def note_off(self, pitch: int, channel: int = 0) -> None:
        self._send(mido.Message("note_off", note=pitch, velocity=0, channel=channel))

    def panic(self) -> None:
        """All-notes-off en los 16 canales. Por las dudas."""
        if self._port is None:
            return
        for ch in range(16):
            # CC 123 = All Notes Off (estándar MIDI)
            self._send(mido.Message("control_change", control=123, value=0, channel=ch))

    def _send(self, msg) -> None:
        if self._port is None:
            raise RuntimeError("MidiOutput no está abierto. Llamar open() o usar `with`.")
        self._port.send(msg)
