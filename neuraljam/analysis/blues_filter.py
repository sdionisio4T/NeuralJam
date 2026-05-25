"""
neuraljam/analysis/blues_filter.py

Post-procesado armónico para blues: ajusta los pitches generados por MelodyRNN
para que queden dentro de la escala de blues detectada por Krumhansl-Kessler.

No cambia ritmo ni dinámica — solo corrige notas fuera de la escala.
Se aplica después de generate(), antes de humanize().

Escala jazz-blues (8 notas): R, 2, b3, M3, 4, #4, 5, b7
→ El M3 y el 2 dan color jazzístico sin perder el sabor bluesy.
→ El #4 (blue note) es la nota característica del blues.

Escala blues estándar (6 notas): R, b3, 4, #4, 5, b7
→ Más oscura y directa. Para un sonido más "crudo".
"""

from note_seq.protobuf import music_pb2

# Intervalos desde la tónica (en semitonos)
_BLUES_INTERVALS      = [0, 3, 5, 6, 7, 10]            # R  b3  4  #4  5  b7
_JAZZ_BLUES_INTERVALS = [0, 2, 3, 4, 5, 6, 7, 10]      # R  2  b3  M3  4  #4  5  b7

_PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pc_from_name(name: str) -> int:
    """Convierte nombre de nota a pitch class (0=C, 1=C#, ..., 11=B)."""
    return _PC_NAMES.index(name) if name in _PC_NAMES else 0


def apply_blues_filter(
    seq: music_pb2.NoteSequence,
    tonic_pc: int,
    style: str = "jazz",
) -> music_pb2.NoteSequence:
    """
    Ajusta cada nota al grado más cercano de la escala de blues.

    Args:
        seq:       NoteSequence generada (se modifica in-place).
        tonic_pc:  pitch class de la tónica (0=C, 1=C#, ..., 11=B).
        style:     "standard" (6 notas) o "jazz" (8 notas, default).

    Returns:
        La misma secuencia con pitches corregidos.
    """
    if not seq.notes:
        return seq

    intervals = _JAZZ_BLUES_INTERVALS if style == "jazz" else _BLUES_INTERVALS

    for note in seq.notes:
        # Pitch class relativo a la tónica
        pc_rel = (note.pitch - tonic_pc) % 12

        # Grado de escala más cercano (distancia circular)
        best = min(
            intervals,
            key=lambda s: min(abs(pc_rel - s), 12 - abs(pc_rel - s)),
        )

        # Diferencia con signo — mantiene la octava, solo ajusta el semitono
        diff = best - pc_rel
        if diff > 6:
            diff -= 12
        elif diff < -6:
            diff += 12

        note.pitch = max(21, min(108, note.pitch + diff))

    return seq
