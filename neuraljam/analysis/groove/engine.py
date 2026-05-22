"""
neuraljam/analysis/groove/engine.py

GrooveEngine — calcula y expone el perfil rítmico de la sesión.

Corre en el mismo thread que el loop principal (es barato — solo aritmética).
No genera audio. No emite MIDI.

El Scheduler lo consulta para ajustar temperatura y response_bars.
El SubconsciousEngine lo consulta para decidir el contenido del primer.

Uso:
    groove = GrooveEngine()
    groove.update(user_seq, ai_seq, qpm=live_qpm)   # una vez por turno
    profile = groove.current                          # RhythmProfile actual
    temp_delta = groove.temperature_delta()           # ajuste sugerido de temp
    bars_hint = groove.bars_hint()                    # compases sugeridos
"""

import logging
from typing import Optional

from note_seq.protobuf import music_pb2

from .extractor import extract_profile
from .profile import RhythmProfile

log = logging.getLogger(__name__)


class GrooveEngine:
    """
    Mantiene el perfil rítmico actualizado turno a turno.

    El profile se calcula sobre la frase del usuario.
    La referencia para calcular tensión es la última respuesta de la IA.
    """

    def __init__(self) -> None:
        self._current: Optional[RhythmProfile] = None
        self._last_ai: Optional[music_pb2.NoteSequence] = None

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def update(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: Optional[music_pb2.NoteSequence] = None,
        qpm: float = 120.0,
    ) -> RhythmProfile:
        """
        Actualiza el perfil con la frase del usuario de este turno.
        Guarda la respuesta de la IA como referencia para el próximo turno.
        """
        profile = extract_profile(user_seq, qpm=qpm, reference=self._last_ai)
        self._current = profile

        if ai_seq is not None and ai_seq.notes:
            self._last_ai = ai_seq

        log.debug(f"[GROOVE] {profile}")
        return profile

    @property
    def current(self) -> Optional[RhythmProfile]:
        """Último perfil calculado. None si aún no hubo ningún turno."""
        return self._current

    def temperature_delta(self) -> float:
        """
        Ajuste de temperatura sugerido basado en el groove actual.

        +0.10 si la frase es densa (muchas notas) → más creatividad
        +0.08 si hay alta tensión rítmica → más exploración
        -0.05 si es muy regular → el modelo puede ser más conservador

        El Scheduler aplica este delta DENTRO de los límites del modo.
        Nunca rompe los techos definidos en ModeConfig.
        """
        if self._current is None:
            return 0.0

        delta = 0.0
        if self._current.is_dense:
            delta += 0.10
        if self._current.is_tense:
            delta += 0.08
        if self._current.pulse_regularity > 0.8:
            delta -= 0.05

        return round(delta, 3)

    def bars_hint(self) -> int:
        """
        Sugerencia de compases de respuesta basada en densidad y tensión.

        Frase densa + tensa → respuesta más larga (desarrollar la tensión).
        Frase esparsa o regular → respuesta corta (dejar respirar).

        El Scheduler usa esto como sugerencia, no como valor fijo.
        """
        if self._current is None:
            return 0  # 0 = sin sugerencia, usar lógica normal del scheduler

        if self._current.is_dense and self._current.is_tense:
            return 3
        if self._current.is_dense or self._current.is_syncopated:
            return 2
        return 1
