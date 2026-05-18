"""
neuraljam.generation — Núcleo lógico del sistema.

Orquesta el flujo desde la frase del usuario hasta el output crudo del
modelo:

    phrase → context → input_seq → modelo → raw_output → postprocess

Submódulos planeados:
- engine.py: orquestador principal. Entry point: respond(phrase) → seq
- context.py: decide qué pasar al modelo (frase, memoria, primer importado)
- prepare.py: convierte el contexto a NoteSequence con chord annotations
- postprocess.py: cuantización, humanize, swing (paso 9, no crítico)

Contrato: única entry point pública es engine.respond(phrase). Devuelve
un NoteSequence listo para playback. Los módulos de afuera no llaman
directamente al modelo ni manipulan NoteSequences crudos.
"""
