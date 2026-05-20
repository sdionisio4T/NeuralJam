"""
neuraljam/subconscious/engine.py

Orquesta los procesos de background que enriquecen el primer.

Fase 1: guarda el turno en el banco y elige una frase aleatoria como
        contexto para el próximo turno. Sin análisis tonal todavía.
Fase 3: reemplazar _build_context() con ImprovRNN / JazzRNN.
Fase 4: agregar MusicVAE para interpolación en espacio latente.

Contrato crítico: get_context() NUNCA bloquea. Si el thread de background
no terminó, devuelve el contexto anterior (o None en el primer turno).
"""

import logging
import threading
from typing import List, Optional

from note_seq.protobuf import music_pb2

from neuraljam.memory.bank import MemoryBank
from neuraljam.midi.phrase_detector import NoteEvent


log = logging.getLogger(__name__)


class SubconsciousEngine:
    """
    Motor de contexto de background.

    Ciclo de vida por turno:
        1. MelodyRNN termina de generar → caller llama trigger()
        2. thread de background corre _run() (actualiza banco + prepara contexto)
        3. Mientras tanto, MelodyRNN toca. Tiempo gratis.
        4. El usuario vuelve a tocar → caller llama get_context()
        5. get_context() devuelve el contexto listo (o el anterior si no terminó)
    """

    def __init__(self, bank: MemoryBank):
        self.bank = bank
        self._lock = threading.Lock()
        self._context: Optional[music_pb2.NoteSequence] = None
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def trigger(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: music_pb2.NoteSequence,
    ) -> None:
        """
        Iniciar actualización de background. No bloquea.
        Llamar justo después de que MelodyRNN dispara su respuesta.
        """
        if self._thread and self._thread.is_alive():
            log.debug("Subconscious: thread anterior aún corriendo, iniciando nuevo")

        self._thread = threading.Thread(
            target=self._run,
            args=(user_seq, ai_seq),
            name="Subconscious",
            daemon=True,
        )
        self._thread.start()

    def get_context(self) -> Optional[music_pb2.NoteSequence]:
        """
        Devuelve el mejor contexto disponible. Nunca bloquea.
        Puede ser None si es el primer turno.
        """
        with self._lock:
            return self._context

    def is_ready(self) -> bool:
        """True si ya hay al menos un contexto preparado."""
        return self._context is not None

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def _run(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: music_pb2.NoteSequence,
    ) -> None:
        try:
            self.bank.add(user_seq, ai_seq)
            context = self._build_context()
            with self._lock:
                self._context = context
            log.debug(
                f"Subconscious actualizado. Banco: {len(self.bank)} frases. "
                f"Contexto: {'sí' if context else 'no'}"
            )
        except Exception:
            log.exception("Error en thread de subconciente (no fatal)")

    def _build_context(self) -> Optional[music_pb2.NoteSequence]:
        """
        Fase 1: elige una frase aleatoria del banco.
        Fase 3: reemplazar con ImprovRNN / JazzRNN.
        Fase 4: reemplazar con MusicVAE interpolation.
        """
        return self.bank.get_random()


# ------------------------------------------------------------------
# Helper: convertir List[NoteEvent] → NoteSequence
# ------------------------------------------------------------------

def phrase_to_seq(
    events: List[NoteEvent],
    qpm: float,
) -> music_pb2.NoteSequence:
    """
    Convierte una frase del detector en NoteSequence para el banco.
    No cuantiza — preserva timing real del usuario.
    """
    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=qpm)
    for evt in events:
        n = seq.notes.add()
        n.pitch = evt.pitch
        n.start_time = evt.start_time
        n.end_time = evt.start_time + evt.duration
        n.velocity = evt.velocity if evt.velocity > 0 else 80
        n.instrument = 0
        n.program = 0
    if events:
        seq.total_time = max(e.start_time + e.duration for e in events)
    return seq
