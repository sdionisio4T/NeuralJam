"""
neuraljam/memory/bank.py

Cola circular de NoteSequences. Guarda frases del usuario y respuestas
de la IA por separado. Operación thread-safe solo en lectura concurrente;
los writes los hace el SubconsciousEngine desde su propio thread.

Fase 6: preload() carga MIDIs externos al banco al inicio de la sesión.
Los MIDIs largos se dividen en chunks de chunk_bars compases para que
cada fragmento sea un contexto musical manejable.
"""

import logging
import random
from collections import deque
from pathlib import Path
from typing import List, Optional

from note_seq.protobuf import music_pb2


log = logging.getLogger(__name__)


class MemoryBank:
    """
    Banco de frases de la sesión.

    Guarda hasta `maxlen` frases de usuario y `maxlen` de la IA.
    `_all` mezcla ambas en orden de llegada (2*maxlen slots).

    preload() carga MIDIs externos antes de empezar — el subconciente
    los usa como contexto desde el turno 1.
    """

    def __init__(self, maxlen: int = 8):
        self.user_phrases: deque = deque(maxlen=maxlen)
        self.ai_phrases: deque = deque(maxlen=maxlen)
        self._all: deque = deque(maxlen=maxlen * 2)

    # ------------------------------------------------------------------
    # Escritura (sesión en vivo)
    # ------------------------------------------------------------------

    def add(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: music_pb2.NoteSequence,
    ) -> None:
        """Agregar un turno completo al banco."""
        self.user_phrases.append(user_seq)
        self.ai_phrases.append(ai_seq)
        self._all.append(user_seq)
        self._all.append(ai_seq)

    # ------------------------------------------------------------------
    # Preload de MIDIs externos (Fase 6)
    # ------------------------------------------------------------------

    def preload(
        self,
        midi_folder: str,
        max_files: int = 20,
        chunk_bars: int = 4,
    ) -> int:
        """
        Carga MIDIs externos al banco al inicio de la sesión.

        Cada archivo se divide en chunks de chunk_bars compases para que
        el contexto sea fragmentos musicales, no solos enteros de 5 minutos.

        Args:
            midi_folder: carpeta con archivos .mid / .midi.
            max_files:   máximo de archivos a leer.
            chunk_bars:  compases por chunk (default 4 = ~8s a 120 BPM).

        Returns:
            Número de archivos cargados exitosamente.
        """
        try:
            from note_seq import midi_file_to_note_sequence
        except ImportError:
            log.error("note_seq no disponible para preload")
            return 0

        folder = Path(midi_folder)
        if not folder.exists():
            log.warning(f"Preload: carpeta no existe: {midi_folder}")
            return 0

        exts = {".mid", ".midi"}
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)
        files = files[:max_files]

        if not files:
            log.warning(f"Preload: no hay archivos .mid en {midi_folder}")
            return 0

        loaded = 0
        total_chunks = 0
        for path in files:
            try:
                seq = midi_file_to_note_sequence(str(path))
                chunks = _split_into_chunks(seq, chunk_bars)
                for chunk in chunks:
                    if chunk.notes:
                        self._all.append(chunk)
                        total_chunks += 1
                loaded += 1
                log.info(
                    f"Preload: {path.name} → {len(chunks)} chunk(s)"
                )
            except Exception:
                log.warning(f"Preload: no se pudo cargar {path.name}", exc_info=True)

        log.info(
            f"Preload completo: {loaded}/{len(files)} archivos, "
            f"{total_chunks} chunks en el banco."
        )
        return loaded

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def get_random(self) -> Optional[music_pb2.NoteSequence]:
        """Devuelve una frase aleatoria del banco. None si está vacío."""
        if not self._all:
            return None
        return random.choice(list(self._all))

    def get_last_ai(self) -> Optional[music_pb2.NoteSequence]:
        if not self.ai_phrases:
            return None
        return self.ai_phrases[-1]

    def get_last_user(self) -> Optional[music_pb2.NoteSequence]:
        if not self.user_phrases:
            return None
        return self.user_phrases[-1]

    def get_most_similar(self, profile) -> Optional[music_pb2.NoteSequence]:
        """
        Devuelve la frase de usuario del banco más parecida rítmicamente
        al perfil dado (RhythmProfile del turno actual).

        Si el banco tiene menos de 2 frases, devuelve la última (igual que
        get_last_user). La similaridad se mide por distancia en tres ejes:
        density, pulse_regularity, syncopation.
        """
        if not self.user_phrases:
            return None
        if len(self.user_phrases) < 2:
            return self.user_phrases[-1]

        from neuraljam.analysis.groove.extractor import extract_profile

        best_seq = None
        best_dist = float("inf")

        for seq in self.user_phrases:
            try:
                qpm = seq.tempos[0].qpm if seq.tempos else 120.0
                p = extract_profile(seq, qpm=qpm)
                dist = (
                    abs(p.density - profile.density) / max(profile.density, 1.0) * 0.4
                    + abs(p.pulse_regularity - profile.pulse_regularity) * 0.3
                    + abs(p.syncopation - profile.syncopation) * 0.3
                )
                if dist < best_dist:
                    best_dist = dist
                    best_seq = seq
            except Exception:
                continue

        return best_seq or self.user_phrases[-1]

    def get_all_user(self) -> List[music_pb2.NoteSequence]:
        """Devuelve todas las frases del usuario acumuladas en la sesión."""
        return list(self.user_phrases)

    def all_phrases(self) -> List[music_pb2.NoteSequence]:
        return list(self._all)

    def is_empty(self) -> bool:
        return len(self._all) == 0

    def __len__(self) -> int:
        return len(self._all)


# ------------------------------------------------------------------
# Helpers privados para preload
# ------------------------------------------------------------------

def _split_into_chunks(
    seq: music_pb2.NoteSequence,
    chunk_bars: int,
) -> List[music_pb2.NoteSequence]:
    """Divide un NoteSequence largo en fragmentos de chunk_bars compases."""
    qpm = seq.tempos[0].qpm if seq.tempos else 120.0
    bar_dur = (60.0 / qpm) * 4.0
    chunk_dur = chunk_bars * bar_dur

    if seq.total_time <= chunk_dur:
        return [seq]

    chunks = []
    t = 0.0
    while t < seq.total_time:
        chunk = _slice_sequence(seq, t, t + chunk_dur, qpm)
        if chunk.notes:
            chunks.append(chunk)
        t += chunk_dur

    return chunks


def _slice_sequence(
    seq: music_pb2.NoteSequence,
    start: float,
    end: float,
    qpm: float,
) -> music_pb2.NoteSequence:
    """Extrae notas de [start, end) y las rebasa a t=0."""
    out = music_pb2.NoteSequence()
    out.tempos.add(qpm=qpm)

    notes = [n for n in seq.notes if start <= n.start_time < end]
    for n in notes:
        new_n = out.notes.add()
        new_n.CopyFrom(n)
        new_n.start_time = n.start_time - start
        new_n.end_time = min(n.end_time, end) - start

    if out.notes:
        out.total_time = max(n.end_time for n in out.notes)
    return out
