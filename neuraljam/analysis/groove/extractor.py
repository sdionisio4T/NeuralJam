"""
neuraljam/analysis/groove/extractor.py

Lee una NoteSequence y devuelve un RhythmProfile.

Métricas:
  density          → notas / duración total
  syncopation      → fracción de notas fuera del beat (grid de 8vas)
  pulse_regularity → regularidad de los intervalos entre notas
  avg_note_duration → duración promedio de notas
  tension          → contraste de densidad entre frase actual y referencia
"""

from typing import Optional

from note_seq.protobuf import music_pb2

from .profile import RhythmProfile


def extract_profile(
    seq: music_pb2.NoteSequence,
    qpm: float = 120.0,
    reference: Optional[music_pb2.NoteSequence] = None,
) -> RhythmProfile:
    """
    Extrae el perfil rítmico de una NoteSequence.

    Args:
        seq:       frase a analizar.
        qpm:       tempo en BPM para calcular el grid de beats.
        reference: frase de referencia para calcular tensión (ej. última IA).
                   None → tension=0.0.

    Returns:
        RhythmProfile con todos los campos calculados.
    """
    if not seq.notes:
        return RhythmProfile()

    notes = sorted(seq.notes, key=lambda n: n.start_time)
    duration = seq.total_time or (notes[-1].end_time if notes else 1.0)

    profile = RhythmProfile()
    profile.note_count = len(notes)
    profile.density = len(notes) / max(duration, 0.1)
    profile.avg_note_duration = sum(
        n.end_time - n.start_time for n in notes
    ) / len(notes)
    profile.syncopation = _syncopation(notes, qpm)
    profile.pulse_regularity = _pulse_regularity(notes)
    profile.tension = _tension(seq, reference, qpm)

    return profile


# ===========================================================================
# Métricas internas
# ===========================================================================

def _syncopation(notes, qpm: float) -> float:
    """
    Fracción de notas que caen fuera de la grilla de corcheas (8th notes).
    Una corchea = 0.5 beats. Tolerancia: ±15ms.
    """
    beat_dur = 60.0 / qpm
    eighth_dur = beat_dur / 2.0
    tol = 0.015  # 15ms

    offbeat = 0
    for n in notes:
        phase = n.start_time % eighth_dur
        # Si la fase está cerca de 0 o cerca de eighth_dur → en el grid
        on_grid = phase < tol or (eighth_dur - phase) < tol
        if not on_grid:
            offbeat += 1

    return offbeat / len(notes)


def _pulse_regularity(notes) -> float:
    """
    Qué tan equiespaciadas están las notas entre sí.
    1.0 = todos los intervalos iguales (pulso perfecto).
    0.0 = intervalos muy irregulares.
    """
    if len(notes) < 2:
        return 1.0

    intervals = [
        notes[i + 1].start_time - notes[i].start_time
        for i in range(len(notes) - 1)
    ]
    mean = sum(intervals) / len(intervals)
    if mean == 0:
        return 1.0

    variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
    cv = (variance ** 0.5) / mean  # coeficiente de variación

    # cv=0 → perfectamente regular (1.0), cv≥1 → muy irregular (0.0)
    return max(0.0, 1.0 - min(cv, 1.0))


def _tension(
    seq: music_pb2.NoteSequence,
    reference: Optional[music_pb2.NoteSequence],
    qpm: float,
) -> float:
    """
    Contraste de densidad entre `seq` y `reference`.
    Si no hay referencia, tensión = 0.0.
    """
    if reference is None or not reference.notes:
        return 0.0

    ref_dur = reference.total_time or (
        max(n.end_time for n in reference.notes)
    )
    ref_density = len(reference.notes) / max(ref_dur, 0.1)

    seq_dur = seq.total_time or (
        max(n.end_time for n in seq.notes) if seq.notes else 1.0
    )
    seq_density = len(seq.notes) / max(seq_dur, 0.1)

    # Diferencia relativa normalizada entre 0 y 1
    max_d = max(ref_density, seq_density, 0.1)
    return min(1.0, abs(seq_density - ref_density) / max_d)
