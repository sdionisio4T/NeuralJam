"""
neuraljam/analysis/tonality.py

Detección de tonalidad por pitch class histogram.

Lee las frases acumuladas en el banco de memoria y estima:
  - Tónica más probable (C, D, E... B)
  - Modo: mayor o menor
  - Progresión simple para ImprovRNN (ii-V-I del tono detectado)

No es un analizador armónico completo — es una aproximación práctica
para dar a ImprovRNN un contexto más coherente que la progresión fija.
Funciona bien con música tonal o modal. Con cromatismo intenso o música
atonal el resultado es una aproximación neutral (C mayor).

Uso:
    result = detect_tonality(bank)
    progression = result.progression   # "Am7 Dm7 E7 Am7"
    chord = result.tonic_chord         # "Am"
"""

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from note_seq.protobuf import music_pb2

# Nombres de pitch classes
_PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]

# Perfil Krumhansl-Kessler para mayor y menor (correlación con tonalidad)
# Indica qué tan "típica" es cada nota en esa tonalidad
_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                  2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                  2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Progresiones de blues (I7 IV7 I7 V7) — 4 acordes = 4 compases
# Formato compacto para ImprovRNN: un acorde por compás
_BLUES_PROGRESSIONS = {
    "C":  "C7 F7 C7 G7",
    "C#": "C#7 F#7 C#7 Ab7",
    "D":  "D7 G7 D7 A7",
    "Eb": "Eb7 Ab7 Eb7 Bb7",
    "E":  "E7 A7 E7 B7",
    "F":  "F7 Bb7 F7 C7",
    "F#": "F#7 B7 F#7 C#7",
    "G":  "G7 C7 G7 D7",
    "Ab": "Ab7 Db7 Ab7 Eb7",
    "A":  "A7 D7 A7 E7",
    "Bb": "Bb7 Eb7 Bb7 F7",
    "B":  "B7 E7 B7 F#7",
}

# Progresiones ii-V-I por tónica y modo
_MAJOR_PROGRESSIONS = {
    "C":  "Dm7 G7 Cmaj7 Cmaj7",
    "C#": "Ebm7 Ab7 Dbmaj7 Dbmaj7",
    "D":  "Em7 A7 Dmaj7 Dmaj7",
    "Eb": "Fm7 Bb7 Ebmaj7 Ebmaj7",
    "E":  "F#m7 B7 Emaj7 Emaj7",
    "F":  "Gm7 C7 Fmaj7 Fmaj7",
    "F#": "G#m7 C#7 F#maj7 F#maj7",
    "G":  "Am7 D7 Gmaj7 Gmaj7",
    "Ab": "Bbm7 Eb7 Abmaj7 Abmaj7",
    "A":  "Bm7 E7 Amaj7 Amaj7",
    "Bb": "Cm7 F7 Bbmaj7 Bbmaj7",
    "B":  "C#m7 F#7 Bmaj7 Bmaj7",
}

_MINOR_PROGRESSIONS = {
    "C":  "Cm7 Fm7 G7 Cm7",
    "C#": "C#m7 F#m7 G#7 C#m7",
    "D":  "Dm7 Gm7 A7 Dm7",
    "Eb": "Ebm7 Abm7 Bb7 Ebm7",
    "E":  "Em7 Am7 B7 Em7",
    "F":  "Fm7 Bbm7 C7 Fm7",
    "F#": "F#m7 Bm7 C#7 F#m7",
    "G":  "Gm7 Cm7 D7 Gm7",
    "Ab": "Abm7 Dbm7 Eb7 Abm7",
    "A":  "Am7 Dm7 E7 Am7",
    "Bb": "Bbm7 Ebm7 F7 Bbm7",
    "B":  "Bm7 Em7 F#7 Bm7",
}


@dataclass
class TonalityResult:
    tonic: str          # "A", "C", "G", etc.
    mode: str           # "major", "minor" o "blues"
    confidence: float   # 0.0 - 1.0, qué tan clara es la tonalidad
    progression: str    # progresión para ImprovRNN
    tonic_chord: str    # acorde raíz solo, para contexto simple

    def __str__(self) -> str:
        mode_label = {"major": "mayor", "minor": "menor", "blues": "blues"}.get(self.mode, self.mode)
        return (
            f"{self.tonic} {mode_label} "
            f"(confianza: {self.confidence:.0%}) → {self.progression}"
        )


def detect_tonality(bank) -> TonalityResult:
    """
    Detecta la tonalidad a partir del banco de memoria de sesión.

    Extrae todas las notas de las frases del usuario acumuladas,
    construye un pitch class histogram y correlaciona contra los
    perfiles de Krumhansl-Kessler para todas las tonalidades mayores
    y menores.

    Adicionalmente detecta sabor a blues: si la b3 y la b7 son fuertes
    relativas a la tónica detectada, el modo se marca como "blues" y
    la progresión devuelta es I7-IV7-I7-V7.

    Si el banco está vacío o hay muy pocas notas, devuelve C mayor
    como fallback neutral.
    """
    notes = _collect_notes(bank)

    if len(notes) < 4:
        return _fallback()

    histogram = _pitch_histogram(notes)
    tonic, mode, confidence = _best_key(histogram)

    # Detección de blues: b3 y b7 ambas bien representadas → blues
    if _is_blues(histogram, tonic):
        mode = "blues"

    progression = _progression_for(tonic, mode)
    tonic_chord = f"{tonic}7" if mode == "blues" else (tonic if mode == "major" else f"{tonic}m")

    return TonalityResult(
        tonic=tonic,
        mode=mode,
        confidence=confidence,
        progression=progression,
        tonic_chord=tonic_chord,
    )


# ===========================================================================
# Internos
# ===========================================================================

def _collect_notes(bank) -> List[int]:
    """Extrae todos los pitches de las frases del usuario en el banco."""
    pitches = []
    # bank.get_all_user() si existe; fallback a iteración directa
    if hasattr(bank, "get_all_user"):
        seqs = bank.get_all_user()
    else:
        # Compatibilidad: extraer del buffer interno
        seqs = [
            entry[0] for entry in bank._buffer
            if entry is not None
        ]
    for seq in seqs:
        if seq is not None:
            for n in seq.notes:
                pitches.append(n.pitch)
    return pitches


def _pitch_histogram(pitches: List[int]) -> List[float]:
    """Cuenta la frecuencia de cada pitch class, normalizado a suma=1."""
    counts = [0.0] * 12
    for p in pitches:
        counts[p % 12] += 1
    total = sum(counts) or 1.0
    return [c / total for c in counts]


def _correlate(histogram: List[float], profile: List[float]) -> float:
    """Correlación simple entre histogram y perfil de tonalidad."""
    mean_h = sum(histogram) / 12
    mean_p = sum(profile) / 12
    num = sum((histogram[i] - mean_h) * (profile[i] - mean_p) for i in range(12))
    den_h = sum((histogram[i] - mean_h) ** 2 for i in range(12)) ** 0.5
    den_p = sum((profile[i] - mean_p) ** 2 for i in range(12)) ** 0.5
    if den_h == 0 or den_p == 0:
        return 0.0
    return num / (den_h * den_p)


def _best_key(histogram: List[float]):
    """Encuentra la tonalidad (tónica + modo) con mayor correlación."""
    best_corr = -2.0
    best_tonic = "C"
    best_mode = "major"

    for rotation in range(12):
        rotated = histogram[rotation:] + histogram[:rotation]

        corr_major = _correlate(rotated, _MAJOR_PROFILE)
        if corr_major > best_corr:
            best_corr = corr_major
            best_tonic = _PC_NAMES[rotation]
            best_mode = "major"

        corr_minor = _correlate(rotated, _MINOR_PROFILE)
        if corr_minor > best_corr:
            best_corr = corr_minor
            best_tonic = _PC_NAMES[rotation]
            best_mode = "minor"

    # Normalizar confianza: correlación va de -1 a 1, mapear a 0-1
    confidence = (best_corr + 1.0) / 2.0
    return best_tonic, best_mode, confidence


def _is_blues(histogram: List[float], tonic: str) -> bool:
    """
    Detecta sabor a blues: b3 y b7 ambas presentes y fuertes.
    Umbral: al menos 7% de las notas en cada pitch class.
    """
    tonic_pc = _PC_NAMES.index(tonic) if tonic in _PC_NAMES else 0
    b3 = (tonic_pc + 3) % 12    # tercera menor
    b7 = (tonic_pc + 10) % 12   # séptima menor
    threshold = 0.07
    return histogram[b3] > threshold and histogram[b7] > threshold


def _progression_for(tonic: str, mode: str) -> str:
    if mode == "blues":
        return _BLUES_PROGRESSIONS.get(tonic, "C7 F7 C7 G7")
    if mode == "major":
        return _MAJOR_PROGRESSIONS.get(tonic, "Cmaj7 Cmaj7 Cmaj7 Cmaj7")
    return _MINOR_PROGRESSIONS.get(tonic, "Am7 Dm7 E7 Am7")


def _fallback() -> TonalityResult:
    return TonalityResult(
        tonic="C",
        mode="major",
        confidence=0.0,
        progression="Cmaj7 Cmaj7 Cmaj7 Cmaj7",
        tonic_chord="C",
    )
