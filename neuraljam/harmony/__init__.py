"""
neuraljam.harmony — Progresiones de acordes.

Etapas (ver ROADMAP):
1. Progresión hardcoded en config.CHORD_PROGRESSION (actual)
2. Progresiones cargables por archivo o argumento CLI (futuro)
3. Detección automática de acordes desde MIDI input del piano (futuro)

Submódulos planeados:
- progression.py: parsea y entrega la progresión activa, sabe qué acorde
  suena en cada tiempo
- detector.py: futuro, detecta acordes en tiempo real desde MIDI

Formato canónico (validado en el spike): string con acordes separados
por espacios, un acorde por compás. La progresión loopea si la generación
es más larga. Soporta triadas, séptimas, alteraciones, slash chords.

Contrato: expone "¿qué acorde suena en t=X?" y "todos los acordes en
[t1, t2]". Para uso de generation/ al armar el input al modelo.
"""
