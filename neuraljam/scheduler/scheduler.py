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

Los parámetros de temperatura y response_bars se limitan según el
ModeConfig activo. Si no se pasa modo, usa los límites de modo normal.
"""

import random
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from neuraljam.modes import ModeConfig


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
        base_temperature: float = 0.8,
        min_temperature: float = 0.6,
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

    def should_respond(self, mode: Optional["ModeConfig"] = None) -> str:
        """
        Decide si el sistema responde en este turno.

        Returns:
            'enter'  — responder ahora.
            'silent' — silencio intencional; el caller salta el turno.

        Llamar UNA vez por frase recibida (incrementa el contador interno).
        """
        self._turn_count += 1

        # Modo libre: siempre responde, sin silencio probabilístico
        if mode is not None and mode.always_respond:
            self._consecutive_silences = 0
            return "enter"

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

    def response_bars(
        self,
        phrase_duration: float,
        bpm: float = 120.0,
        mode: Optional["ModeConfig"] = None,
    ) -> int:
        """
        Cuántos compases debe responder el modelo.

        Sigue la longitud de la frase del usuario con variación:
          ~35% : réplica corta (1-2 compases)
          ~40% : respuesta proporcional a tu frase (±1 compás)
          ~25% : desarrollo largo (varios compases extra)

        El techo se toma del ModeConfig activo (normal=3, libre=5, etc.).
        """
        bars_max = mode.response_bars_max if mode is not None else 3

        bar_dur = (60.0 / bpm) * 4.0
        user_bars = max(1, round(phrase_duration / bar_dur))
        base = min(6, user_bars)  # seguimos tu longitud, techo en 6

        roll = random.random()
        if roll < 0.35:
            # Réplica corta — puntual, rítmica
            bars = random.randint(1, max(1, base // 2 + 1))
        elif roll < 0.75:
            # Proporcional con leve variación
            bars = base + random.choice([-1, 0, 0, 1])
        else:
            # Desarrollo — extiende la idea (máx +2, no +4)
            bars = base + random.randint(1, 2)

        return max(1, min(bars_max, bars))

    def temperature(
        self,
        phrase_note_count: int,
        phrase_duration: float,
        mode: Optional["ModeConfig"] = None,
    ) -> float:
        """
        Temperatura dinámica para MelodyRNN.

        Lógica:
        - Empieza en base_temperature y decae suavemente con los turnos
          (el sistema se vuelve más conservador conforme avanza la sesión).
        - Sube hasta +0.2 si el usuario toca denso (muchas notas por segundo).
        - Se clampea entre mode.temp_min y mode.temp_max (None = sin techo).

        Para imitación (temp_min == temp_max), devuelve la temperatura fija.
        """
        temp_min = mode.temp_min if mode is not None else self.min_temperature
        temp_max = mode.temp_max if mode is not None else 0.95

        # Imitación: temperatura fija, sin cálculo dinámico
        if temp_max is not None and temp_min == temp_max:
            return temp_min

        # Factor de sesión: decae de 1.0 a 0.7 en 30 turnos
        session_factor = max(0.7, 1.0 - (self._turn_count / 30) * 0.3)

        # Factor de densidad: notas/segundo del usuario
        density = phrase_note_count / max(phrase_duration, 0.1)
        density_bonus = min(0.2, density * 0.04)

        raw = self.base_temperature * session_factor + density_bonus
        raw = max(temp_min, raw)
        if temp_max is not None:
            raw = min(temp_max, raw)
        return raw

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return self._turn_count

    @property
    def consecutive_silences(self) -> int:
        return self._consecutive_silences
