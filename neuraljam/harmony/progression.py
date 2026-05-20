"""
neuraljam/harmony/progression.py

Representa una progresión de acordes con un acorde por compás.

Etapa 1 del sistema armónico (ver ROADMAP): la progresión está hardcoded
en config.CHORD_PROGRESSION. Este módulo la convierte en un objeto
consultable y la valida al construirse.

Convención (validada en el spike): los acordes se pasan como string
separada por espacios, ej. "Dm7 G7 Cmaj7 Cmaj7". Un acorde por compás.
Si la generación es más larga que la progresión, ésta se loopea.

API mínima:
    prog = Progression.from_config()
    prog.chord_at_time(2.5)             -> ChordEvent
    prog.chords_in_range(0.0, 8.0)      -> List[ChordEvent]
    prog.to_chord_string(num_bars=8)    -> str ("Dm7 G7 Cmaj7 Cmaj7 Dm7 ...")
"""

from dataclasses import dataclass
from typing import List, Optional

from note_seq.chord_symbols_lib import ChordSymbolError, chord_symbol_pitches

from neuraljam import config


# ===========================================================================
# Tipo de salida
# ===========================================================================

@dataclass(frozen=True)
class ChordEvent:
    """
    Un acorde activo en un compás específico de la progresión.

    Inmutable: representa un instante específico, no debe modificarse.

    Atributos:
        symbol:   notación del acorde, ej. "Dm7", "G7", "Cmaj7"
        bar:      número de compás 0-indexed (puede ser mayor que
                  len(chords) si la progresión loopeó)
        time_sec: tiempo absoluto de inicio del compás, en segundos
    """
    symbol: str
    bar: int
    time_sec: float


# ===========================================================================
# Progression
# ===========================================================================

class Progression:
    """
    Progresión de acordes con un acorde por compás.

    Construcción:
        Progression(["Dm7", "G7", "Cmaj7", "Cmaj7"], bpm=120)
        Progression.from_string("Dm7 G7 Cmaj7 Cmaj7", bpm=120)
        Progression.from_config()  # usa config.CHORD_PROGRESSION y QPM

    El BPM se guarda con la progresión. Si en el futuro entra clock
    sync, se podrá actualizar con set_bpm() sin reconstruir el objeto.
    """

    def __init__(self, chords: List[str], bpm: float):
        if not chords:
            raise ValueError("Progression no puede ser vacía")
        if bpm <= 0:
            raise ValueError(f"BPM debe ser positivo, recibí {bpm}")

        # Validar cada símbolo con el parser de Magenta. Falla acá si
        # alguno no es interpretable. chord_symbol_pitches devuelve las
        # notas del acorde; no nos importa el valor, solo que no tire
        # ChordSymbolError.
        for symbol in chords:
            try:
                chord_symbol_pitches(symbol)
            except ChordSymbolError as e:
                raise ValueError(
                    f"Acorde inválido en progresión: {symbol!r} ({e})"
                ) from e

        self._chords = tuple(chords)
        self._bpm = bpm

    # ---------------------------------------------------------------- #
    # Constructores alternativos
    # ---------------------------------------------------------------- #

    @classmethod
    def from_string(cls, s: str, bpm: float) -> "Progression":
        """Crea desde 'Dm7 G7 Cmaj7 Cmaj7'."""
        chords = s.split()
        return cls(chords, bpm)

    @classmethod
    def from_config(cls) -> "Progression":
        """Constructor por defecto: usa config.CHORD_PROGRESSION y QPM_FALLBACK."""
        return cls.from_string(config.CHORD_PROGRESSION, config.QPM_FALLBACK)

    # ---------------------------------------------------------------- #
    # Propiedades
    # ---------------------------------------------------------------- #

    @property
    def length_bars(self) -> int:
        """Cuántos compases tiene la progresión sin loopear."""
        return len(self._chords)

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def bar_duration_sec(self) -> float:
        """Duración de un compás (4/4) en segundos. A 120 BPM = 2.0s."""
        # 4/4 implícito. Si después soportamos 3/4 o 6/8, este getter
        # toma un time_signature y se ajusta.
        return 60.0 / self._bpm * 4.0

    # ---------------------------------------------------------------- #
    # Consulta
    # ---------------------------------------------------------------- #

    def chord_at_bar(self, bar: int) -> str:
        """
        Devuelve el símbolo del acorde activo en un compás dado.

        Loopea: bar 4 sobre una progresión de 4 acordes devuelve el
        acorde del bar 0.

        bar puede ser >= 0; negativos lanzan ValueError.
        """
        if bar < 0:
            raise ValueError(f"bar debe ser >= 0, recibí {bar}")
        return self._chords[bar % len(self._chords)]

    def chord_at_time(self, t_sec: float) -> ChordEvent:
        """
        Qué acorde suena en el tiempo absoluto t_sec.

        El tiempo t_sec=0.0 corresponde al inicio del compás 0.
        El loopeo se maneja transparentemente.
        """
        if t_sec < 0:
            raise ValueError(f"t_sec debe ser >= 0, recibí {t_sec}")

        bar_abs = int(t_sec // self.bar_duration_sec)
        bar_in_loop = bar_abs % len(self._chords)
        return ChordEvent(
            symbol=self._chords[bar_in_loop],
            bar=bar_abs,
            time_sec=bar_abs * self.bar_duration_sec,
        )

    def chords_in_range(
        self,
        start_sec: float,
        end_sec: float,
    ) -> List[ChordEvent]:
        """
        Lista todos los acordes cuyo INICIO de compás cae en [start, end).

        Útil para armar el input al modelo: queremos saber qué acordes
        van a sonar durante la generación, en orden cronológico.

        Si start_sec cae a mitad de un compás, ese compás SÍ se incluye
        (asumimos que el caller quiere saber el acorde activo en start).
        """
        if start_sec < 0 or end_sec < start_sec:
            raise ValueError(
                f"Rango inválido: start={start_sec}, end={end_sec}"
            )

        # Compás de start (entero, redondeado abajo). Si start está a
        # mitad del compás 1.4, incluimos el compás 1 entero.
        first_bar = int(start_sec // self.bar_duration_sec)
        # Compás del último que ARRANCA antes de end (estricto).
        last_bar = int((end_sec - 1e-9) // self.bar_duration_sec)

        events = []
        for bar in range(first_bar, last_bar + 1):
            events.append(ChordEvent(
                symbol=self._chords[bar % len(self._chords)],
                bar=bar,
                time_sec=bar * self.bar_duration_sec,
            ))
        return events

    def to_chord_string(self, num_bars: Optional[int] = None) -> str:
        """
        Devuelve la progresión en formato string para Magenta.

        Si num_bars > length_bars, loopea explícitamente. Si num_bars es
        None, devuelve la progresión sin loopear (length_bars compases).

        Hacemos el loopeo explícito (en lugar de delegarlo a Magenta)
        porque facilita el debug: el caller ve exactamente qué string
        va a recibir el modelo.
        """
        if num_bars is None:
            num_bars = self.length_bars
        if num_bars <= 0:
            raise ValueError(f"num_bars debe ser > 0, recibí {num_bars}")

        expanded = [
            self._chords[i % len(self._chords)]
            for i in range(num_bars)
        ]
        return " ".join(expanded)

    # ---------------------------------------------------------------- #
    # Repr para debug
    # ---------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"Progression({list(self._chords)}, "
            f"bpm={self._bpm}, "
            f"bar_dur={self.bar_duration_sec:.2f}s)"
        )


# ===========================================================================
# Test manual: python -m neuraljam.harmony.progression
# ===========================================================================

if __name__ == "__main__":
    print("=== Progresión desde config ===")
    prog = Progression.from_config()
    print(repr(prog))
    print(f"Longitud:        {prog.length_bars} compases")
    print(f"Duración compás: {prog.bar_duration_sec:.2f}s")
    print()

    print("=== chord_at_time ===")
    test_times = [0.0, 0.5, 2.0, 3.5, 7.9, 8.0, 10.0]
    for t in test_times:
        evt = prog.chord_at_time(t)
        print(f"  t={t:5.2f}s  ->  bar {evt.bar}  symbol {evt.symbol!r}  "
              f"(compás empieza en {evt.time_sec:.2f}s)")
    print()

    print("=== chords_in_range(0, 8.0) (= 4 compases) ===")
    for evt in prog.chords_in_range(0.0, 8.0):
        print(f"  bar {evt.bar}  t={evt.time_sec:.2f}s  {evt.symbol!r}")
    print()

    print("=== chords_in_range(0, 16.0) (= 8 compases, loopea) ===")
    for evt in prog.chords_in_range(0.0, 16.0):
        print(f"  bar {evt.bar}  t={evt.time_sec:.2f}s  {evt.symbol!r}")
    print()

    print("=== to_chord_string ===")
    print(f"  default (sin args): {prog.to_chord_string()!r}")
    print(f"  num_bars=8:         {prog.to_chord_string(8)!r}")
    print(f"  num_bars=2:         {prog.to_chord_string(2)!r}")
    print()

    print("=== Validación: acorde inválido tiene que fallar ===")
    try:
        Progression(["Dm7", "ZZZ", "Cmaj7"], bpm=120)
        print("  FAIL: aceptó un acorde inválido")
    except ValueError as e:
        print(f"  OK: rechazó - {e}")
