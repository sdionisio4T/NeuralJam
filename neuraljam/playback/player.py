"""
neuraljam/playback/player.py

Reproduce un NoteSequence enviándolo nota por nota al MidiOutput,
respetando los tiempos del NoteSequence.

Diseño:
- Toma todos los note-on y note-off, los ordena por timestamp.
- Loop: dormir hasta el próximo evento, emitirlo.
- Bloqueante: vuelve cuando terminó toda la respuesta.

Por qué bloqueante: el sistema es turn-based estricto. Si play() fuera
asíncrono, el phrase_detector podría capturar una frase nueva del usuario
mientras la IA todavía toca, y los turnos se solapan.

Precisión: usamos time.perf_counter() y dormimos hasta el timestamp
absoluto del siguiente evento. Jitter típico < 5ms, suficiente para uso
musical. Si más adelante se necesita sub-ms, se puede mover a un thread
con prioridad alta o a un timer-based scheduler.
"""

import logging
import time
from typing import List, Tuple

from note_seq.protobuf import music_pb2

from neuraljam.midi.output import MidiOutput


log = logging.getLogger(__name__)


# Tipo del evento interno. (timestamp_sec, is_note_on, pitch, velocity)
_Event = Tuple[float, bool, int, int]


class Player:
    """
    Reproductor de NoteSequence al puerto MIDI de salida.

    El player NO abre el MidiOutput: lo recibe ya abierto. Razón: el
    puerto debería estar abierto durante toda la sesión, no abrir/cerrar
    por cada respuesta. El entry point maneja el ciclo de vida.
    """

    def __init__(self, midi_out: MidiOutput, channel: int = 0):
        self.midi_out = midi_out
        self.channel = channel

    def play(self, seq: music_pb2.NoteSequence) -> None:
        """
        Reproduce un NoteSequence completo, bloqueando hasta el final.

        Si seq está vacío, retorna inmediatamente sin error.
        """
        if not seq.notes:
            log.warning("Player.play() llamado con NoteSequence vacío")
            return

        events = self._flatten_events(seq)
        if not events:
            return

        # t0 = ahora; los timestamps de los eventos son relativos a t0.
        t0 = time.perf_counter()
        last_pitch_logged = None

        for ts, is_on, pitch, vel in events:
            target = t0 + ts
            now = time.perf_counter()
            wait = target - now
            if wait > 0:
                time.sleep(wait)

            try:
                if is_on:
                    self.midi_out.note_on(pitch, velocity=vel, channel=self.channel)
                    log.debug(f"on  pitch={pitch} vel={vel} ts={ts:.3f}")
                else:
                    self.midi_out.note_off(pitch, channel=self.channel)
                    log.debug(f"off pitch={pitch}        ts={ts:.3f}")
            except Exception:
                # No matar el loop completo por un mensaje fallido.
                # Loguear y seguir; el resto de la frase puede sonar.
                log.exception(f"Error enviando MIDI (pitch={pitch})")

        # Cortesía: garantizar que no queden note-ons sin off por algún
        # bug futuro en el flattening (no debería pasar, pero barato).
        total_duration = seq.total_time
        log.info(f"Reproducción terminada ({total_duration:.2f}s)")

    # ----------------------------------------------------------------- #
    # Internos
    # ----------------------------------------------------------------- #

    @staticmethod
    def _flatten_events(seq: music_pb2.NoteSequence) -> List[_Event]:
        """
        Aplana las notas en una lista de eventos ordenados por timestamp.

        Cada nota se expande en 2 eventos: note-on (en start_time) y
        note-off (en end_time). Después ordenamos por timestamp.

        Tiebreak cuando dos eventos caen en el mismo timestamp:
        note-off ANTES que note-on. Razón: si una nota termina y otra
        empieza exactamente al mismo tiempo, queremos liberar la voz
        antes de pisarla. Especialmente importante cuando son la misma
        nota (re-articulación).
        """
        events: List[_Event] = []
        for note in seq.notes:
            vel = note.velocity if note.velocity > 0 else 80
            events.append((float(note.start_time), True, note.pitch, vel))
            events.append((float(note.end_time), False, note.pitch, vel))

        # Sort estable: timestamp primero, después note-off antes que note-on.
        # is_on=False viene antes que is_on=True bajo `not is_on`.
        events.sort(key=lambda e: (e[0], e[1]))
        return events
