"""
neuraljam.midi — Capa MIDI.

Lazy imports (PEP 562) para evitar cargar mido innecesariamente.
"""

__all__ = ["PhraseDetector", "NoteEvent", "Phrase", "MidiOutput"]


def __getattr__(name):
    if name in ("PhraseDetector", "NoteEvent", "Phrase"):
        from neuraljam.midi.phrase_detector import PhraseDetector, NoteEvent, Phrase
        return {
            "PhraseDetector": PhraseDetector,
            "NoteEvent": NoteEvent,
            "Phrase": Phrase,
        }[name]
    if name == "MidiOutput":
        from neuraljam.midi.output import MidiOutput
        return MidiOutput
    raise AttributeError(f"module 'neuraljam.midi' has no attribute {name!r}")
