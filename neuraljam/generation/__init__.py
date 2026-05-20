"""
neuraljam.generation — Núcleo lógico del sistema.

Orquesta el flujo desde la frase del usuario hasta el output del modelo:

    phrase → prepare → generator → extract → respuesta

Submódulos:
- prepare.py: convierte NoteEvent + Progression en NoteSequence
- engine.py: GenerationEngine, orquestador principal

Contrato: única API pública es GenerationEngine.respond(phrase). Devuelve
un NoteSequence listo para playback (rebasado a t=0). Los módulos de
afuera no llaman directamente al generator de Magenta.

Lazy imports: el paquete no carga prepare/engine al importarse, así
quien solo necesite leer constantes de config no paga TF.
"""

__all__ = ["GenerationEngine"]


def __getattr__(name):
    if name == "GenerationEngine":
        from neuraljam.generation.engine import GenerationEngine
        return GenerationEngine
    raise AttributeError(f"module 'neuraljam.generation' has no attribute {name!r}")
