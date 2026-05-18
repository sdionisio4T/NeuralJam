"""
neuraljam.midi — Capa MIDI.

Responsabilidades:
- Entrada: leer del teclado físico (puerto MIDI_INPUT_NAME)
- Salida: enviar al DAW vía loopMIDI (puerto MIDI_OUTPUT_NAME)
- Detección de frases del usuario (silencio + captura fiel de timing)
- Clock sync con el DAW (paso 8 del roadmap)
- Resolución de nombres de puerto por prefijo (ports.py)

Submódulos:
- ports.py: helpers para resolver puertos por nombre/prefijo. Existe
  porque los nombres de puertos en Windows tienen sufijos numéricos
  inestables entre sesiones.
- phrase_detector.py: captura frases del usuario con fidelidad rítmica.
- output.py: envía eventos MIDI con timing preciso (pendiente).
- clock_sync.py: lee MIDI Clock del DAW (paso 8, pendiente).

Contrato: ningún módulo fuera de neuraljam.midi debe importar `mido`
directamente. Todo acceso MIDI pasa por acá.

Lazy imports (PEP 562): el paquete no carga los submódulos al
importarse. Solo cuando alguien accede a PhraseDetector o NoteEvent
como atributos del paquete, se hace el import real. Esto evita el
warning de runpy al ejecutar submódulos con `python -m`.
"""

__all__ = ["PhraseDetector", "NoteEvent"]


def __getattr__(name):
    """Carga submódulos on-demand."""
    if name in ("PhraseDetector", "NoteEvent"):
        from neuraljam.midi.phrase_detector import PhraseDetector, NoteEvent
        return {"PhraseDetector": PhraseDetector, "NoteEvent": NoteEvent}[name]
    raise AttributeError(f"module 'neuraljam.midi' has no attribute {name!r}")
