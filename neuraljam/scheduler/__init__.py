"""
neuraljam.scheduler — Decide cuándo y cómo responder.

En lugar de responder mecánicamente a cada frase, el Scheduler evalúa
el estado de la sesión y puede:
  - 'enter':  responder ahora
  - 'silent': guardar silencio intencional (el sistema "escucha")

Esto hace al sistema sentirse menos como un bot y más como un músico.
"""

from neuraljam.scheduler.scheduler import Scheduler

__all__ = ["Scheduler"]
