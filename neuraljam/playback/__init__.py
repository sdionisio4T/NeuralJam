"""
neuraljam.playback — Reproducción hacia el DAW.

Responsabilidades:
- Recibir un NoteSequence de generation/
- Programar eventos note-on/note-off con timing preciso
- Esperar al próximo downbeat para disparar (cuando entre clock sync)
- Garantizar que no se solapen dos respuestas (turn-based estricto)

Submódulos planeados:
- player.py: scheduler de eventos MIDI, materializa NoteSequence
- scheduler.py: lógica de "cuándo disparar" (próximo downbeat, etc.)

Contrato: API simple del tipo play(seq) o schedule(seq, at_downbeat=True).
Internamente usa neuraljam.midi.output para el envío real.
"""
