"""
neuraljam/modes.py

Configuración de los modos de improvisación.

Cada modo define:
  - Límites de temperatura (piso y techo, None = sin techo)
  - Máximo de compases de respuesta
  - Si se usa el banco de memoria de sesión
  - Qué capas del subconciente se activan
  - Si el scheduler puede callar (always_respond=True = siempre responde)

El modelo activo (1/2/3) es independiente del modo — el modo controla
el comportamiento del primer y la temperatura, no qué modelo genera.

Capas del subconciente por modo:
  normal:       memoria de sesión (banco circular) — sin MusicVAE, sin ImprovRNN
  imitación:    sin contexto, sin memoria
  libre:        MusicVAE + ImprovRNN + banco  ← probado, suena bien
  experimental: MusicVAE + ImprovRNN + banco (sin techo de temperatura)

DECISIÓN TESTEADA (2026-05-21):
  ImprovRNN en modo normal fue removido. Con progresión fija crashea
  (AssertionError en chords.end_step). Con chord dinámico del pitch
  histogram, el fallback al banco sonaba *mejor* que el fragmento generado.
  El banco de sesión solo (última frase del usuario) es el contexto
  más coherente para el modelo principal en uso normal.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ModeConfig:
    name: str
    display: str
    temp_min: float
    temp_max: Optional[float]       # None = sin techo de temperatura
    response_bars_max: int
    use_memory: bool                # guarda frases en el banco de sesión
    use_music_vae: bool             # interpola con MusicVAE en el primer
    use_improv_background: bool     # genera fragmento ImprovRNN con chord dinámico
    always_respond: bool            # True = nunca silencio probabilístico


MODES: dict[str, ModeConfig] = {
    "normal": ModeConfig(
        name="normal",
        display="NORMAL",
        temp_min=0.6,
        temp_max=0.95,
        response_bars_max=3,
        use_memory=True,
        use_music_vae=False,
        use_improv_background=False,    # banco de sesión solo — testeado, funciona mejor
        always_respond=False,
    ),
    "imitation": ModeConfig(
        name="imitation",
        display="IMITACIÓN",
        temp_min=0.45,
        temp_max=0.45,                  # temperatura fija — espejo directo
        response_bars_max=2,
        use_memory=False,
        use_music_vae=False,
        use_improv_background=False,
        always_respond=False,
    ),
    "free": ModeConfig(
        name="free",
        display="LIBRE",
        temp_min=0.6,
        temp_max=None,                  # sin techo — creatividad máxima
        response_bars_max=5,
        use_memory=True,
        use_music_vae=True,
        use_improv_background=True,
        always_respond=True,
    ),
    "experimental": ModeConfig(
        name="experimental",
        display="EXPERIMENTAL",
        temp_min=0.6,
        temp_max=None,
        response_bars_max=4,
        use_memory=True,
        use_music_vae=True,
        use_improv_background=True,
        always_respond=False,
    ),
}

MODE_CYCLE = ["normal", "imitation", "free", "experimental"]


def next_mode(current: str) -> str:
    """Devuelve el nombre del siguiente modo en el ciclo."""
    idx = MODE_CYCLE.index(current) if current in MODE_CYCLE else 0
    return MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
