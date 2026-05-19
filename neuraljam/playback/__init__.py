"""
neuraljam.playback — Reproducción hacia el DAW.

Toma NoteSequences que entrega generation/ y los emite al puerto MIDI
de salida (loopMIDI → Studio One) con timing real.

Submódulos:
- player.py: scheduler de eventos, materializa NoteSequence

Contrato: Player.play(seq) bloquea hasta que termina la reproducción.
Esto garantiza el turn-based: el siguiente turno del usuario no arranca
hasta que la IA terminó de tocar.
"""

__all__ = ["Player"]


def __getattr__(name):
    if name == "Player":
        from neuraljam.playback.player import Player
        return Player
    raise AttributeError(f"module 'neuraljam.playback' has no attribute {name!r}")
