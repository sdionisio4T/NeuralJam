"""
neuraljam.memory — Almacén de frases del usuario y primers importados.

Paso 6-7 del roadmap. Importante post-spike: como el primer determina la
calidad del output, este módulo es palanca directa sobre la musicalidad
del sistema. Permite:

- Guardar frases que tocó el usuario (estilo propio)
- Importar solos externos (Bill Evans, Chucho, etc.) como style primers
- Pasar esos primers al engine como conditioning extra

Submódulos planeados:
- store.py: guarda/recupera frases en disco. Formato: JSON (metadata) +
  MIDI (notas). Schema a definir antes de implementar.
- importer.py: lee archivos MIDI externos y los normaliza como primers
- selector.py: elige qué frase/primer usar en cada turno (similitud
  armónica, similitud rítmica, aleatorio, etc.)

Schema mínimo propuesto para una frase guardada:
- id, timestamp, duration_sec
- bpm, chord_context (lista de acordes activos al tocar)
- notas: lista de (pitch, start, duration, velocity)

Contrato: devuelve NoteSequences ya formateados, listos para usar como
contexto en generation/.
"""
