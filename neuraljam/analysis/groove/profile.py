"""
neuraljam/analysis/groove/profile.py

RhythmProfile — descripción numérica del carácter rítmico de una frase.

No genera audio. No emite MIDI. Es solo datos.
El Scheduler y el SubconsciousEngine lo consultan para ajustar
temperatura, response_bars y el contenido del primer.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RhythmProfile:
    """
    Perfil rítmico de una frase.

    Todos los valores están normalizados 0.0–1.0 salvo donde se indica.
    """

    # Notas por segundo. Frase típica de jazz: 2–6 n/s.
    density: float = 0.0

    # Fracción de notas que caen fuera del beat (offbeat).
    # 0.0 = todo en el beat, 1.0 = todo sincopado.
    syncopation: float = 0.0

    # Regularidad del pulso: 1.0 = notas muy equiespaciadas, 0.0 = irregular.
    pulse_regularity: float = 0.0

    # Contraste rítmico contra la última respuesta de la IA (0.0–1.0).
    # Alto = el usuario cambió drásticamente el ritmo.
    tension: float = 0.0

    # Duración promedio de las notas en segundos.
    avg_note_duration: float = 0.0

    # Número de notas de la frase (sin normalizar).
    note_count: int = 0

    def __str__(self) -> str:
        return (
            f"density={self.density:.2f} n/s  "
            f"sync={self.syncopation:.0%}  "
            f"pulse={self.pulse_regularity:.0%}  "
            f"tension={self.tension:.0%}"
        )

    @property
    def is_dense(self) -> bool:
        """Frase densa: más de 4 notas por segundo."""
        return self.density > 4.0

    @property
    def is_syncopated(self) -> bool:
        """Más de la mitad de las notas caen fuera del beat."""
        return self.syncopation > 0.5

    @property
    def is_tense(self) -> bool:
        """Contraste rítmico alto con la respuesta anterior."""
        return self.tension > 0.6
