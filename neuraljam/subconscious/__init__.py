"""
neuraljam.subconscious — Procesos de background que enriquecen el primer.

Fase 1: MemoryBank + SubconsciousEngine base (contexto aleatorio del banco).
Fase 3: ImprovRNN y JazzRNN generan fragmentos chord-aware.
Fase 4: MusicVAE interpola entre frases en el espacio latente.
"""

from neuraljam.subconscious.engine import SubconsciousEngine

__all__ = ["SubconsciousEngine"]
