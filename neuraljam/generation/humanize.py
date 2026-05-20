"""
neuraljam/generation/humanize.py

Post-procesado de la respuesta de MelodyRNN antes de reproducirla.

Dos efectos:
  - Swing: empuja las corcheas en tiempo débil hacia atrás.
    El feel jazz clásico donde el "and" llega un poco tarde.
    Independiente del tempo — el offset se calcula en función del beat.

  - Humanize: variación aleatoria de velocidad nota a nota.
    Simula la dinámica natural de un músico. Sin esto todas las notas
    salen al mismo volumen y suena a MIDI de los 90.

Aplicar DESPUÉS de generate(), ANTES de player.play().
Solo toca la respuesta generada — nunca el primer del usuario.
"""

import random

from note_seq.protobuf import music_pb2


def humanize(
    seq: music_pb2.NoteSequence,
    swing: float = 0.08,
    velocity_variance: int = 12,
    qpm: float = 120.0,
) -> music_pb2.NoteSequence:
    """
    Aplica swing + variación de velocidad a una NoteSequence.

    Args:
        seq:               respuesta de MelodyRNN. Se modifica in-place.
        swing:             fracción del beat que se desplaza el tiempo débil.
                           0.0 = recto, 0.33 = swing de tresillos perfecto.
                           0.08–0.12 es sutil pero audible.
        velocity_variance: variación máxima de velocidad MIDI (±N).
                           12 da un rango humano sin exagerar.
        qpm:               tempo de la sesión para calcular beat duration.

    Returns:
        La misma secuencia modificada.
    """
    if not seq.notes:
        return seq

    beat_dur = 60.0 / qpm          # negra en segundos
    eighth_dur = beat_dur / 2.0    # corchea en segundos
    swing_offset = swing * beat_dur

    for note in seq.notes:
        # ---- Swing ----
        # Posición dentro del beat actual (0 = en el beat, ~eighth_dur = offbeat)
        beat_pos = note.start_time % beat_dur
        if beat_pos > eighth_dur * 0.6:
            note.start_time += swing_offset
            note.end_time += swing_offset

        # ---- Velocity ----
        delta = random.randint(-velocity_variance, velocity_variance)
        note.velocity = max(30, min(127, note.velocity + delta))

    # Actualizar total_time por si el swing extendió el final
    seq.total_time = max(n.end_time for n in seq.notes)

    return seq
