"""
neuraljam/memory/bank.py

Cola circular de NoteSequences. Guarda frases del usuario y respuestas
de la IA por separado. Operación thread-safe solo en lectura concurrente;
los writes los hace el SubconsciousEngine desde su propio thread.
"""

import random
from collections import deque
from typing import List, Optional

from note_seq.protobuf import music_pb2


class MemoryBank:
    """
    Banco de frases de la sesión.

    Guarda hasta `maxlen` frases de usuario y `maxlen` de la IA.
    `_all` mezcla ambas en orden de llegada (2*maxlen slots).
    """

    def __init__(self, maxlen: int = 8):
        self.user_phrases: deque = deque(maxlen=maxlen)
        self.ai_phrases: deque = deque(maxlen=maxlen)
        self._all: deque = deque(maxlen=maxlen * 2)

    # ------------------------------------------------------------------
    # Escritura
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

    def all_phrases(self) -> List[music_pb2.NoteSequence]:
        return list(self._all)

    def is_empty(self) -> bool:
        return len(self._all) == 0

    def __len__(self) -> int:
        return len(self._all)
