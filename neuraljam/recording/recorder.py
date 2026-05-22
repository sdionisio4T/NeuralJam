"""
neuraljam/recording/recorder.py

Acumula todos los turnos de la sesión y los exporta como MIDI de dos pistas.

Track 0: usuario   (canal MIDI 0)
Track 1: IA        (canal MIDI 1)

Las frases se colocan secuencialmente — primero la frase del usuario,
inmediatamente después la respuesta de la IA, luego el siguiente turno.
No captura los silencios reales entre turnos, pero la conversación
musical queda intacta y legible en cualquier DAW.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from note_seq.protobuf import music_pb2

log = logging.getLogger(__name__)

_TICKS_PER_BEAT = 480


class SessionRecorder:
    """
    Graba la sesión completa turno a turno.

    Uso:
        recorder = SessionRecorder()
        recorder.add_turn(user_ns, ai_ns, qpm=120.0)   # una vez por turno
        recorder.export()                                # al cerrar
    """

    def __init__(self) -> None:
        self._turns: List[Tuple[
            music_pb2.NoteSequence,
            music_pb2.NoteSequence,
            float,
        ]] = []

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def add_turn(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: music_pb2.NoteSequence,
        qpm: float = 120.0,
    ) -> None:
        """Agrega un turno (usuario + respuesta IA) a la grabación."""
        self._turns.append((user_seq, ai_seq, qpm))

    def export(self, path: Optional[Path] = None) -> Optional[Path]:
        """
        Exporta la sesión como MIDI tipo 1 (dos pistas).
        Devuelve el path creado, o None si la sesión está vacía o falla.
        """
        if not self._turns:
            log.info("SessionRecorder: sesión vacía, nada que exportar.")
            return None

        if path is None:
            now = datetime.now().strftime("%Y-%m-%d_%H%M")
            path = Path("sessions") / f"{now}.mid"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import mido

            # Usar el QPM del último turno como tempo del archivo
            qpm = self._turns[-1][2]
            tempo = mido.bpm2tempo(qpm)

            mid = mido.MidiFile(type=1, ticks_per_beat=_TICKS_PER_BEAT)

            user_track = mido.MidiTrack()
            ai_track = mido.MidiTrack()
            mid.tracks.append(user_track)
            mid.tracks.append(ai_track)

            user_track.append(
                mido.MetaMessage("set_tempo", tempo=tempo, time=0)
            )
            user_track.append(
                mido.MetaMessage("track_name", name="Usuario", time=0)
            )
            ai_track.append(
                mido.MetaMessage("track_name", name="NeuralJam", time=0)
            )

            user_events: List[Tuple[int, int, int]] = []  # (tick, pitch, vel)
            ai_events: List[Tuple[int, int, int]] = []

            cursor = 0.0  # segundos acumulados en la línea de tiempo

            for user_seq, ai_seq, turn_qpm in self._turns:
                user_dur = _seq_duration(user_seq)
                ai_dur = _seq_duration(ai_seq)

                for n in user_seq.notes:
                    on_tick = _secs_to_ticks(cursor + n.start_time, turn_qpm)
                    off_tick = _secs_to_ticks(cursor + n.end_time, turn_qpm)
                    vel = n.velocity if n.velocity > 0 else 80
                    user_events.append((on_tick, n.pitch, vel))
                    user_events.append((off_tick, n.pitch, 0))

                cursor += user_dur

                for n in ai_seq.notes:
                    on_tick = _secs_to_ticks(cursor + n.start_time, turn_qpm)
                    off_tick = _secs_to_ticks(cursor + n.end_time, turn_qpm)
                    vel = n.velocity if n.velocity > 0 else 80
                    ai_events.append((on_tick, n.pitch, vel))
                    ai_events.append((off_tick, n.pitch, 0))

                cursor += ai_dur

            _write_track(user_events, user_track, channel=0)
            _write_track(ai_events, ai_track, channel=1)

            mid.save(str(path))
            log.info(
                f"Sesión grabada: {path}  "
                f"({len(self._turns)} turnos, {cursor:.1f}s)"
            )
            return path

        except Exception:
            log.exception("SessionRecorder: error al exportar (no fatal)")
            return None

    @property
    def turn_count(self) -> int:
        return len(self._turns)


# ===========================================================================
# Helpers
# ===========================================================================

def _seq_duration(seq: music_pb2.NoteSequence) -> float:
    if seq.total_time > 0:
        return seq.total_time
    if seq.notes:
        return max(n.end_time for n in seq.notes)
    return 0.0


def _secs_to_ticks(secs: float, qpm: float) -> int:
    beats = secs * (qpm / 60.0)
    return max(0, int(beats * _TICKS_PER_BEAT))


def _write_track(
    events: List[Tuple[int, int, int]],
    track,
    channel: int,
) -> None:
    """Convierte lista de (tick, pitch, vel) a mensajes delta-time."""
    import mido
    sorted_ev = sorted(events, key=lambda e: e[0])
    prev_tick = 0
    for abs_tick, pitch, vel in sorted_ev:
        delta = abs_tick - prev_tick
        track.append(
            mido.Message(
                "note_on", channel=channel,
                note=pitch, velocity=vel, time=delta,
            )
        )
        prev_tick = abs_tick
    track.append(mido.MetaMessage("end_of_track", time=0))
