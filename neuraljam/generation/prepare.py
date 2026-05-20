"""
neuraljam/generation/prepare.py

Convierte una frase del usuario en un NoteSequence listo para Magenta.

Soporta dos modos:
- Con progresión armónica (ImprovRNN): agrega chord text_annotations.
- Sin progresión (MelodyRNN): solo notas + tempo, sin acordes.

Decisiones:
1. PRESERVA duración real del fraseo. compress_primer=False por default.
2. CUANTIZA al step interno del modelo (limitación inherente).
3. Si progression=None, se calcula bar_dur con QPM_FALLBACK del config.
4. context_seq opcional: NoteSequence de historia que se antepone al primer
   del usuario. Permite al modelo ver frases anteriores como contexto.
"""

import math
from typing import List, Optional

from note_seq.protobuf import music_pb2

from neuraljam import config
from neuraljam.midi.phrase_detector import NoteEvent
from neuraljam.harmony.progression import Progression


# ===========================================================================
# Cuantización
# ===========================================================================

def _seconds_per_step(qpm: float, steps_per_quarter: int) -> float:
    return (60.0 / qpm) / steps_per_quarter


def _quantize_to_step(t: float, step_dur: float) -> float:
    return round(t / step_dur) * step_dur


# ===========================================================================
# Build input
# ===========================================================================

def build_input_sequence(
    phrase: List[NoteEvent],
    progression: Optional[Progression],
    response_bars: int,
    compress_primer: bool = False,
    velocity_default: int = 80,
    steps_per_quarter: int = 4,
    context_seq: Optional[music_pb2.NoteSequence] = None,
    max_primer_notes: int = 32,
) -> music_pb2.NoteSequence:
    """
    Construye el input para Magenta.

    Args:
        phrase: lista de NoteEvent (del detector, sin nota de señal).
        progression: Progression o None. Si None, no se agregan chords
                     y el BPM viene de config.QPM_FALLBACK.
        response_bars: compases extra para la respuesta del modelo.
        compress_primer: True para escalar primer a 1 compás (alternativo).
        velocity_default: si una nota tiene velocity=0, usa este.
        steps_per_quarter: resolución del modelo (de magenta_config).
        context_seq: NoteSequence histórica para anteponer al primer.
                     Se inserta antes de la frase del usuario para que el
                     modelo vea continuidad de sesión. None = sin contexto.
        max_primer_notes: presupuesto total de notas (contexto + frase).
                          Si se excede, se recorta por el principio.
    """
    if not phrase:
        raise ValueError("phrase no puede ser vacía")

    # BPM y bar_dur
    if progression is not None:
        bpm = progression.bpm
        bar_dur = progression.bar_duration_sec
    else:
        bpm = config.QPM_FALLBACK
        bar_dur = (60.0 / bpm) * 4.0  # 4/4 implícito

    step_dur = _seconds_per_step(bpm, steps_per_quarter)

    # ---- Contexto histórico (opcional) -----------------------------------
    # Tomar hasta max_primer_notes//2 notas del final del contexto.
    # Las notas del contexto van primero; la frase del usuario las sigue.

    ctx_notes_raw = []
    context_offset = 0.0

    if context_seq is not None and context_seq.notes:
        budget = max_primer_notes // 2
        sorted_ctx = sorted(context_seq.notes, key=lambda n: n.start_time)
        if len(sorted_ctx) > budget:
            sorted_ctx = sorted_ctx[-budget:]  # las más recientes

        ctx_t0 = sorted_ctx[0].start_time
        ctx_dur = max(n.end_time for n in sorted_ctx) - ctx_t0
        context_offset = _quantize_to_step(ctx_dur, step_dur)
        if context_offset < step_dur:
            context_offset = step_dur
        ctx_notes_raw = sorted_ctx

    # ---- Dimensiones temporales de la frase del usuario ------------------

    primer_end_raw = phrase[-1].start_time + phrase[-1].duration

    if compress_primer:
        scale = bar_dur / primer_end_raw if primer_end_raw > 0 else 1.0
        phrase_bars = 1
    else:
        scale = 1.0
        phrase_bars = max(1, math.ceil(primer_end_raw / bar_dur))

    context_bars = max(1, math.ceil(context_offset / bar_dur)) if context_offset > 0 else 0
    primer_end = (context_bars + phrase_bars) * bar_dur
    total_end = primer_end + response_bars * bar_dur

    # ---- NoteSequence ----------------------------------------------------

    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=bpm)

    # Contexto histórico (viene antes de la frase del usuario)
    if ctx_notes_raw:
        ctx_t0 = ctx_notes_raw[0].start_time
        for n in ctx_notes_raw:
            qs = _quantize_to_step(n.start_time - ctx_t0, step_dur)
            qe = _quantize_to_step(n.end_time - ctx_t0, step_dur)
            if qe <= qs:
                qe = qs + step_dur
            if qs >= primer_end:
                continue
            note = seq.notes.add()
            note.pitch = n.pitch
            note.start_time = qs
            note.end_time = min(qe, primer_end)
            note.velocity = n.velocity if n.velocity > 0 else velocity_default
            note.instrument = 0
            note.program = 0

    # Notas del primer (frase del usuario, desplazadas por context_offset)
    for evt in phrase:
        scaled_start = evt.start_time * scale + context_offset
        scaled_dur = evt.duration * scale

        start_q = _quantize_to_step(scaled_start, step_dur)
        end_q = _quantize_to_step(scaled_start + scaled_dur, step_dur)

        if end_q <= start_q:
            end_q = start_q + step_dur
        if start_q >= primer_end:
            continue
        if end_q > primer_end:
            end_q = primer_end

        note = seq.notes.add()
        note.pitch = evt.pitch
        note.start_time = start_q
        note.end_time = end_q
        note.velocity = evt.velocity if evt.velocity > 0 else velocity_default
        note.instrument = 0
        note.program = 0

    # Acordes (solo si hay progresión)
    if progression is not None:
        chord_events = progression.chords_in_range(0.0, total_end)
        for ce in chord_events:
            ann = seq.text_annotations.add()
            ann.text = ce.symbol
            ann.annotation_type = music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
            ann.time = ce.time_sec

    seq.total_time = total_end
    return seq


# ===========================================================================
# Helpers
# ===========================================================================

def get_primer_end(input_seq: music_pb2.NoteSequence) -> float:
    """End time de la última nota del primer."""
    if not input_seq.notes:
        return 0.0
    return max(n.end_time for n in input_seq.notes)


def extract_generated(
    full_output: music_pb2.NoteSequence,
    primer_end: float,
) -> music_pb2.NoteSequence:
    """Filtra notas generadas (sin eco del primer) y rebasa a t=0."""
    out = music_pb2.NoteSequence()
    out.tempos.add(qpm=full_output.tempos[0].qpm if full_output.tempos else 120)

    eps = 0.01
    generated = [
        n for n in full_output.notes
        if n.start_time >= primer_end - eps
    ]
    if not generated:
        out.total_time = 0.0
        return out

    first_start = min(n.start_time for n in generated)
    for n in generated:
        new_note = out.notes.add()
        new_note.pitch = n.pitch
        new_note.start_time = n.start_time - first_start
        new_note.end_time = n.end_time - first_start
        new_note.velocity = n.velocity if n.velocity > 0 else 80
        new_note.instrument = n.instrument
        new_note.program = n.program

    out.total_time = max(n.end_time for n in out.notes)
    return out
