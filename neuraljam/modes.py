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
  a/b:          dos respuestas por turno — primero sin contexto, luego con contexto

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
    dual_response: bool = False     # True = genera A y B por turno
    b_hears_a: bool = False         # True = B recibe respuesta de A como contexto extra
    a_plays: bool = True            # False = A genera en silencio, solo suena B
    match_user_bars: bool = False   # True = respuesta exactamente igual de larga que tu frase
    use_blues_filter: bool = False  # True = ajusta pitches a escala de blues detectada


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
    # Modo directo: scheduler y groove activos, pero sin memoria de sesión.
    # Equivale a la respuesta A del modo A/B — el modelo responde solo
    # a la frase actual, sin contexto previo.
    "direct": ModeConfig(
        name="direct",
        display="DIRECTO",
        temp_min=0.6,
        temp_max=0.95,
        response_bars_max=3,
        use_memory=False,
        use_music_vae=False,
        use_improv_background=False,
        always_respond=False,
    ),
    # Modo comparación: por cada frase genera dos respuestas —
    # primero sin contexto (modelo limpio), luego con contexto (banco de sesión).
    # Sirve para escuchar en tiempo real si el contexto mejora o empeora.
    "ab": ModeConfig(
        name="ab",
        display="A/B",
        temp_min=0.6,
        temp_max=0.95,
        response_bars_max=6,
        use_memory=True,
        use_music_vae=False,
        use_improv_background=False,
        always_respond=True,
        dual_response=True,
        match_user_bars=True,
    ),
    # Modo B+: A genera en silencio (no suena), B recibe banco + A + tu frase.
    # Una sola respuesta audible pero con el contexto más rico posible.
    "bplus": ModeConfig(
        name="bplus",
        display="B+",
        temp_min=0.6,
        temp_max=0.95,
        response_bars_max=6,
        use_memory=True,
        use_music_vae=False,
        use_improv_background=False,
        always_respond=True,
        dual_response=True,
        b_hears_a=True,
        a_plays=False,
        match_user_bars=True,
    ),
    # Modo blues: filtro armónico activo.
    # Krumhansl-Kessler detecta la tónica desde el banco de sesión.
    # Cada nota generada se ajusta al grado más cercano de la escala jazz-blues:
    # R, 2, b3, M3, 4, #4, 5, b7 — el AI suena "dentro" del blues.
    # Siempre responde. Usa memoria de sesión para mejorar la detección de clave.
    "blues": ModeConfig(
        name="blues",
        display="BLUES",
        temp_min=0.6,
        temp_max=0.95,
        response_bars_max=4,
        use_memory=True,
        use_music_vae=False,
        use_improv_background=False,
        always_respond=True,
        use_blues_filter=True,
        match_user_bars=True,   # responde igual de largo que tu frase → más denso
    ),
    # Modo diálogo: call-and-response estricto.
    # La respuesta tiene exactamente los mismos compases que tu frase.
    # Siempre responde, sin silencio probabilístico.
    "dialogue": ModeConfig(
        name="dialogue",
        display="DIÁLOGO",
        temp_min=0.6,
        temp_max=0.95,
        response_bars_max=6,
        use_memory=True,
        use_music_vae=False,
        use_improv_background=False,
        always_respond=True,
        match_user_bars=True,
    ),
    # Modo comparación 2: igual que A/B pero B también escucha lo que tocó A.
    # B recibe: banco de sesión + respuesta de A + tu frase actual.
    # Permite comparar si escuchar al "otro músico" mejora la respuesta.
    "ab2": ModeConfig(
        name="ab2",
        display="A/B 2",
        temp_min=0.6,
        temp_max=0.95,
        response_bars_max=6,
        use_memory=True,
        use_music_vae=False,
        use_improv_background=False,
        always_respond=True,
        dual_response=True,
        b_hears_a=True,
        match_user_bars=True,
    ),
}

MODE_CYCLE = ["normal", "blues", "dialogue", "imitation", "free", "experimental", "ab", "ab2", "bplus", "direct"]


def next_mode(current: str) -> str:
    """Devuelve el nombre del siguiente modo en el ciclo."""
    idx = MODE_CYCLE.index(current) if current in MODE_CYCLE else 0
    return MODE_CYCLE[(idx + 1) % len(MODE_CYCLE)]
