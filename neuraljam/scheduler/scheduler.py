"""
neuraljam/scheduler/scheduler.py

Decide si MelodyRNN responde en este turno y con qué temperatura.

Reglas:
1. Si hubo demasiados silencios seguidos → forzar respuesta (el sistema
   no puede callarse para siempre).
2. Con probabilidad (1 - response_probability) → silencio intencional.
   Simula que el músico "escucha" antes de volver a tocar.
3. Temperatura dinámica: más creativa al inicio, más coherente con la
   sesión avanzada. Sube si el usuario toca denso, baja si toca esparso.
"""

import random


class Scheduler:
    """
    Toma decisiones de turno-a-turno.

    Parámetros:
        response_probability:     chance de responder (0.0–1.0). Default 0.85.
        max_consecutive_silences: si llegamos acá, forzamos respuesta.
        base_temperature:         temperatura de inicio (primer turno).
        min_temperature:          piso de temperatura (sesión muy avanzada).
    """

    def __init__(
        self,
        response_probability: float = 0.85,
        max_consecutive_silences: int = 2,
        base_temperature: float = 1.0,
        min_temperature: float = 0.75,
    ):
        self.response_probability = response_probability
        self.max_consecutive_silences = max_consecutive_silences
        self.base_temperature = base_temperature
        self.min_temperature = min_temperature

        self._turn_count: int = 0
        self._consecutive_silences: int = 0

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def should_respond(self) -> str:
        """
        Decide si el sistema responde en este turno.

        Returns:
            'enter'  — responder ahora.
            'silent' — silencio intencional; el caller salta el turno.

        Llamar UNA vez por frase recibida (incrementa el contador interno).
        """
        self._turn_count += 1

        # Demasiados silencios seguidos → forzar respuesta
        if self._consecutive_silences >= self.max_consecutive_silences:
            self._consecutive_silences = 0
            return "enter"

        # Silencio probabilístico (el sistema "escucha")
        if random.random() > self.response_probability:
            self._consecutive_silences += 1
            return "silent"

        self._consecutive_silences = 0
        return "enter"

    def temperature(self, phrase_note_count: int, phrase_duration: float) -> float:
        """
        Temperatura dinámica para MelodyRNN.

        Lógica:
        - Empieza en base_temperature y decae suavemente con los turnos
          (el sistema se vuelve más conservador conforme avanza la sesión).
        - Sube hasta +0.2 si el usuario toca denso (muchas notas por segundo).

        Args:
            phrase_note_count: número de notas en la frase del usuario.
            phrase_duration:   duración total de la frase en segundos.
        """
        # Factor de sesión: decae de 1.0 a 0.7 en 30 turnos
        session_factor = max(0.7, 1.0 - (self._turn_count / 30) * 0.3)

        # Factor de densidad: notas/segundo del usuario
        density = phrase_note_count / max(phrase_duration, 0.1)
        density_bonus = min(0.2, density * 0.04)

        raw = self.base_temperature * session_factor + density_bonus
        return max(self.min_temperature, min(1.5, raw))

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def consecutive_silences(self) -> int:
        return self._consecutive_silences
