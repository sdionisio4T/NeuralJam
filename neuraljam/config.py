"""
neuraljam/config.py — Centro del sistema.

Toda constante, threshold, path o nombre de puerto MIDI vive acá.
Ningún otro módulo debe hardcodear estos valores. Si encuentra el código
una constante mágica que no está en este archivo, eso es un bug.

Notas de diseño:
- Variables planas y sueltas, no clases. Suficiente para 3 modos.
- `MODE` define el modelo *default*. Con dos modelos en RAM (MelodyRNN
  + ImprovRNN), el sistema puede cambiar en runtime por señal del piano.
- `active_profile()` es función para soportar override de MODE post-import.
"""

from pathlib import Path


# ===========================================================================
# Rutas base
# ===========================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models_data"
MEMORY_DIR = PROJECT_ROOT / "memory_data"

# MusicVAE (Fase 4) — contexto por interpolación latente
# cat-mel_2bar_big: 2 compases, 512-dim latente, ~26 MB
MUSIC_VAE_CHECKPOINT_DIR = MODELS_DIR / "cat-mel_2bar_big"
MUSIC_VAE_URL = (
    "https://storage.googleapis.com/magentadata/models/"
    "music_vae/checkpoints/cat-mel_2bar_big.tar"
)


# ===========================================================================
# Modo activo (modelo default)
# ===========================================================================
# Decisión post-test del 18/05: MelodyRNN como motor principal porque
# dialoga mejor con las frases del usuario. ImprovRNN queda en RAM como
# modelo alternativo, disparable con SIGNAL_NOTE_MIN/MAX desde el piano.

MODE = "melody"


# ===========================================================================
# MIDI
# ===========================================================================
# Nombres como prefijos: el resolver acepta sufijos numéricos inestables.

MIDI_INPUT_NAME = "CASIO USB-MIDI"
MIDI_OUTPUT_NAME = "AI-Duet-OUT"
MIDI_CLOCK_PORT = "s1-Clock"   # loopMIDI port para recibir clock de Studio One


# ===========================================================================
# Detección de frases
# ===========================================================================

SILENCE_TIMEOUT = 2.5
MIN_NOTES_TO_RESPOND = 2


# ===========================================================================
# Señal de cambio de modelo
# ===========================================================================
# Si la ÚLTIMA nota de la frase cae dentro de este rango (inclusive),
# el sistema descarta esa nota del primer y usa ImprovRNN en ese turno.
# Si no, usa MelodyRNN (default).
#
# Rango actual: A0 (21) a E1 (28). 8 teclas en la octava más grave del
# piano. Difícil tocarlas por accidente improvisando jazz normal.
#
# Cambiarlo si querés rango más amplio o más restrictivo. Para usar UNA
# sola nota: SIGNAL_NOTE_MIN == SIGNAL_NOTE_MAX.

# Señal de cambio de modelo — rango grave dividido en dos zonas:
#   A0–C1  (21-24) → ImprovRNN    (chord-aware, armónico)
#   C#1–E1 (25-28) → PerformanceRNN (polofónico, expresivo, libre)
SIGNAL_IMPROV_MIN = 21   # A0
SIGNAL_IMPROV_MAX = 24   # C1
SIGNAL_PERF_MIN   = 25   # C#1
SIGNAL_PERF_MAX   = 28   # E1

# Aliases de backward-compat (rango total de señal)
SIGNAL_NOTE_MIN = SIGNAL_IMPROV_MIN
SIGNAL_NOTE_MAX = SIGNAL_PERF_MAX


# ===========================================================================
# Tempo y grilla
# ===========================================================================

QPM_FALLBACK = 120
STEPS_PER_QUARTER = 4


# ===========================================================================
# Armonía (solo se usa con el perfil improv)
# ===========================================================================

CHORD_PROGRESSION = "Dm7 G7 Cmaj7 Cmaj7"


# ===========================================================================
# Perfiles por modo
# ===========================================================================
# Soportamos dos familias de modelos:
#   - improv_rnn (chord-conditioned, requiere progresión)
#   - melody_rnn (sin chords, dúo melódico libre)
# El loader hace dispatch según "model_family".

PROFILES = {
    "melody": {
        "model_family": "melody_rnn",
        "model_path": MODELS_DIR / "attention_rnn.mag",
        "model_url": "http://download.magenta.tensorflow.org/models/attention_rnn.mag",
        "model_config_id": "attention_rnn",
        "temperature": 1.0,
        "response_bars": 4,
        "needs_chords": False,
        "polyphonic": False,
    },
    "improv": {
        "model_family": "improv_rnn",
        "model_path": MODELS_DIR / "chord_pitches_improv.mag",
        "model_url": "http://download.magenta.tensorflow.org/models/chord_pitches_improv.mag",
        "model_config_id": "chord_pitches_improv",
        "temperature": 0.8,
        "response_bars": 4,
        "needs_chords": True,
        "polyphonic": False,
    },
    # Performance todavía no implementado.
    "performance": {
        "model_family": "performance_rnn",
        "model_path": MODELS_DIR / "performance_with_dynamics.mag",
        "model_url": "http://download.magenta.tensorflow.org/models/performance_with_dynamics.mag",
        "model_config_id": "performance_with_dynamics",
        "temperature": 1.0,
        "response_bars": 4,
        "needs_chords": False,
        "polyphonic": True,
    },
}


# ===========================================================================
# Memoria (paso 6 del roadmap)
# ===========================================================================

USER_PHRASES_DIR = MEMORY_DIR / "user_phrases"
IMPORTED_PRIMERS_DIR = MEMORY_DIR / "imported_primers"


# ===========================================================================
# Logging
# ===========================================================================

LOG_LEVEL = "INFO"
LOG_FILE = PROJECT_ROOT / "neuraljam.log"


# ===========================================================================
# Helpers
# ===========================================================================

def active_profile():
    """Devuelve el perfil del MODE actual (default)."""
    if MODE not in PROFILES:
        raise ValueError(
            f"MODE '{MODE}' no reconocido. Válidos: {list(PROFILES.keys())}"
        )
    return PROFILES[MODE]


def ensure_dirs():
    """Crea directorios runtime. Llamar al bootstrap, no al import."""
    for d in (MODELS_DIR, MEMORY_DIR, USER_PHRASES_DIR, IMPORTED_PRIMERS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Self-check
# ===========================================================================

if __name__ == "__main__":
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"MODE default:    {MODE}")
    print(f"MIDI in:         {MIDI_INPUT_NAME!r}")
    print(f"MIDI out:        {MIDI_OUTPUT_NAME!r}")
    print(f"SILENCE_TIMEOUT: {SILENCE_TIMEOUT}s")
    print(f"Signal range:    MIDI {SIGNAL_NOTE_MIN}-{SIGNAL_NOTE_MAX}")
    print(f"QPM_FALLBACK:    {QPM_FALLBACK}")
    print(f"Progresión:      {CHORD_PROGRESSION!r}")
    print()
    print("Perfiles disponibles:")
    for mode_name, prof in PROFILES.items():
        marker = "* " if mode_name == MODE else "  "
        print(f"  {marker}{mode_name:12s}  family={prof['model_family']:14s}  "
              f"chords={prof['needs_chords']}")
