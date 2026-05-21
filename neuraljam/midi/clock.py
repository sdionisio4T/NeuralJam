"""
neuraljam/midi/clock.py

Receptor de MIDI Clock. Studio One es master; Python es slave.

MIDI clock = 24 pulsos por negra (beat). Este módulo:
  1. Escucha esos pulsos en un puerto loopMIDI separado
  2. Calcula el BPM promediando los ultimos 24 pulsos (1 beat)
  3. Expone wait_for_downbeat() para que la respuesta empiece en el compas

Por que puerto separado: el clock de Studio One va a interferir con
las notas MIDI si lo mezclamos en el mismo puerto de entrada.

Configuracion en Studio One:
    Song > Song Setup > MIDI > MIDI Clock > enviar a "S1-Clock" (loopMIDI)
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import mido


log = logging.getLogger(__name__)

PULSES_PER_BEAT = 24
BEATS_PER_BAR = 4
PULSES_PER_BAR = PULSES_PER_BEAT * BEATS_PER_BAR


class MidiClock:
    """
    Receptor de MIDI Clock en background thread.

    Uso:
        clock = MidiClock("S1-Clock")
        clock.start()

        # En el loop principal:
        qpm = clock.qpm           # BPM detectado en vivo
        clock.wait_for_downbeat() # espera al inicio del proximo compas
    """

    def __init__(self, port_name: str):
        self._port_name = port_name
        # Ventana de ~1 beat: maxlen=32 alcanza para calcular QPM al primer beat
        self._pulse_times: deque = deque(maxlen=PULSES_PER_BEAT + 8)
        self._pulse_count: int = 0
        self._qpm: float = 120.0
        self._is_running: bool = False
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Arranque
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._listen,
            name="MidiClock",
            daemon=True,
        )
        self._thread.start()
        log.info(f"MIDI Clock: escuchando en '{self._port_name}'")

    # ------------------------------------------------------------------
    # Propiedades publicas
    # ------------------------------------------------------------------

    @property
    def qpm(self) -> float:
        """BPM actual calculado de los pulsos del DAW."""
        with self._lock:
            return self._qpm

    @property
    def is_running(self) -> bool:
        """True si el DAW esta reproduciendo (START/CONTINUE o pulsos recibidos)."""
        with self._lock:
            return self._is_running

    @property
    def has_sync(self) -> bool:
        """True si ya tenemos suficientes pulsos para un QPM valido."""
        with self._lock:
            return len(self._pulse_times) >= PULSES_PER_BEAT

    # ------------------------------------------------------------------
    # Sincronizacion
    # ------------------------------------------------------------------

    def wait_for_downbeat(self, max_wait_bars: float = 0.75) -> bool:
        """
        Bloquea hasta el inicio del proximo compas.

        Si el DAW no esta corriendo o el wait excede max_wait_bars,
        devuelve False y el caller reproduce igual (degradacion elegante).

        Args:
            max_wait_bars: maximo de compases a esperar antes de rendirse.
                           0.75 = hasta 3/4 de compas de espera maxima.
        """
        if not self.is_running:
            return False

        with self._lock:
            bar_dur = (60.0 / self._qpm) * BEATS_PER_BAR
        timeout = bar_dur * max_wait_bars
        deadline = time.perf_counter() + timeout

        prev_pos = self._pulse_position
        while time.perf_counter() < deadline:
            pos = self._pulse_position
            if pos < prev_pos:  # el contador dio vuelta -> inicio de nuevo compas
                return True
            prev_pos = pos
            time.sleep(0.001)

        log.debug("wait_for_downbeat: timeout, reproduciendo igualmente")
        return False

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def _pulse_position(self) -> int:
        with self._lock:
            return self._pulse_count % PULSES_PER_BAR

    def _listen(self) -> None:
        try:
            from neuraljam.midi.ports import resolve_input_port
            real_name = resolve_input_port(self._port_name)
            port = mido.open_input(real_name)
            log.info(f"MIDI Clock: puerto abierto '{real_name}'")
        except Exception as e:
            log.error(
                f"MIDI Clock: no se pudo abrir '{self._port_name}': {e}\n"
                f"  Puertos disponibles: {mido.get_input_names()}"
            )
            return

        for msg in port:
            if msg.type == "clock":
                self._on_clock()
            elif msg.type in ("start", "continue"):
                with self._lock:
                    self._pulse_count = 0
                    self._pulse_times.clear()  # descarta timestamps de sesion anterior
                    self._is_running = True
                log.info(f"MIDI Clock: {msg.type.upper()} sincronizado")
            elif msg.type == "stop":
                with self._lock:
                    self._is_running = False
                    self._pulse_times.clear()  # evita timestamps rancios al reiniciar
                log.info("MIDI Clock: STOP")

    def _on_clock(self) -> None:
        now = time.perf_counter()
        with self._lock:
            self._pulse_times.append(now)
            self._pulse_count += 1

            # Activar aunque Studio One no haya mandado START explicito
            if not self._is_running:
                self._is_running = True
                log.info("MIDI Clock: pulsos detectados, sincronizado")

            # QPM con EMA para suavizar jitter de Windows.
            # Ventana de 1 beat (24 pulsos): en cuanto tenemos PULSES_PER_BEAT
            # timestamps en el deque calculamos BPM y suavizamos con EMA alpha=0.15.
            # Esto corrige el bug anterior donde el deque (maxlen=48) nunca llegaba
            # a 96 elementos y _qpm se quedaba en 120 para siempre.
            if len(self._pulse_times) >= PULSES_PER_BEAT:
                n_intervals = len(self._pulse_times) - 1  # intervalos cubiertos
                dt = self._pulse_times[-1] - self._pulse_times[0]
                if dt > 0.01:
                    beats_in_window = n_intervals / PULSES_PER_BEAT
                    raw_qpm = (60.0 * beats_in_window) / dt
                    self._qpm = 0.15 * raw_qpm + 0.85 * self._qpm  # EMA

            # Log cada beat para monitoreo sin spam
            if self._pulse_count % PULSES_PER_BEAT == 0:
                log.debug(f"MIDI Clock: {self._qpm:.1f} BPM")
