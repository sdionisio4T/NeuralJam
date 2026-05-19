"""
neuraljam.harmony — Progresiones de acordes.

Etapas (ver ROADMAP):
1. Progresión hardcoded en config.CHORD_PROGRESSION (actual)
2. Progresiones cargables por archivo o argumento CLI (futuro)
3. Detección automática de acordes desde MIDI input del piano (futuro)

Submódulos:
- progression.py: parsea la progresión activa, sabe qué acorde suena
  en cada tiempo, expone la string que Magenta consume.
- detector.py: futuro, detecta acordes en tiempo real desde MIDI.

Formato canónico (validado en el spike): string con acordes separados
por espacios, un acorde por compás. La progresión loopea si la generación
es más larga. Soporta triadas, séptimas, alteraciones, slash chords.

Contrato: expone "¿qué acorde suena en t=X?" y "todos los acordes en
[t1, t2]". Para uso de generation/ al armar el input al modelo.

Lazy imports (PEP 562): mismo patrón que neuraljam.midi.
"""

__all__ = ["Progression", "ChordEvent"]


def __getattr__(name):
    if name in ("Progression", "ChordEvent"):
        from neuraljam.harmony.progression import Progression, ChordEvent
        return {"Progression": Progression, "ChordEvent": ChordEvent}[name]
    raise AttributeError(f"module 'neuraljam.harmony' has no attribute {name!r}")
