"""
neuraljam/config.py — Centro del sistema.

Toda constante, threshold, path o nombre de puerto MIDI vive acá.
Ningún otro módulo debe hardcodear estos valores. Si encuentra el código
una constante mágica que no está en este archivo, eso es un bug.

Notas de diseño:
- Variables planas y sueltas, no clases. Suficiente para 3 modos.
  Si crece a >5 modos o aparece validación compleja, migrar a dataclass.
- `MODE` se setea acá con default. El CLI puede sobrescribirlo en runtime
  haciendo `import neuraljam.config as cfg; cfg.MODE = "melody"` ANTES
  de que cualquier otro módulo lea `cfg.active_profile()`.
- `active_profile()` es función (no constante precomputada) porque MODE
  puede cambiar después del import. Una constante quedaría stale.
"""

from pathlib import Path


# ===========================================================================
# Rutas base
# ===========================================================================

# Raíz del proyecto. Calculada desde la ubicación de este archivo para que
# funcione independiente del cwd desde el cual se ejecute el sistema.
# `__file__` = .../neuraljam/config.py → parent.parent = raíz del proyecto.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models_data"   # bundles .mag descargados
MEMORY_DIR = PROJECT_ROOT / "memory_data"   # frases guardadas + primers


# ===========================================================================
# Modo activo
# ===========================================================================

# Default. Sobrescribible por CLI en el entry point (ver nota de diseño arriba).
# Valores válidos: "improv" | "melody" | "performance"
MODE = "improv"


# ===========================================================================
# MIDI
# ===========================================================================
# Nombres EXACTOS de los puertos. Para listarlos:
#   python -c "import mido; print(mido.get_input_names(), mido.get_output_names())"
#
# El sufijo " 2" en AI-Duet-OUT no es error: es como mido nombra el puerto
# loopMIDI en este sistema. Si cambiás de máquina o reinstalás loopMIDI,
# verificá el nombre exacto antes de tocar otra cosa.

MIDI_INPUT_NAME = "CASIO USB-MIDI 0"
MIDI_OUTPUT_NAME = "AI-Duet-OUT 2"


# ===========================================================================
# Detección de frases (consumido por midi/phrase_detector.py)
# ===========================================================================

# Segundos de silencio absoluto (todas las teclas levantadas) que disparan
# el envío de la frase al engine de generación. Más alto = más tiempo de
# pensamiento entre notas sin que la IA dispare.
# Subir si la IA dispara antes de que termines de tocar.
SILENCE_TIMEOUT = 2.5

# Mínimo de notas para considerar una secuencia como "frase" y disparar
# respuesta. Evita que un toque accidental o una nota suelta dispare al
# modelo.
MIN_NOTES_TO_RESPOND = 2


# ===========================================================================
# Tempo y grilla
# ===========================================================================

# QPM usado cuando no hay MIDI Clock entrante del DAW (caso default actual,
# antes del paso 8 del roadmap). Cuando se implemente clock sync, este
# valor pasa a ser solo fallback inicial.
QPM_FALLBACK = 120

# Resolución temporal interna del modelo. ImprovRNN trabaja en
# semicorcheas (4 steps por negra). No cambiar a menos que se entienda el
# impacto sobre la cuantización del primer y del output.
STEPS_PER_QUARTER = 4


# ===========================================================================
# Armonía
# ===========================================================================
# Etapa 1 del sistema armónico (ver ROADMAP): progresión hardcoded.
# Formato: string con acordes separados por espacios, un acorde por compás.
# Si la progresión es más corta que la generación, el modelo la loopea.
#
# Notación soportada (validada en el spike):
#   triadas (Dm, G, C), séptimas (Dm7, G7, Cmaj7), alteraciones (Em7b5,
#   A7b9), sextas (Dm6), slash (C/E).

CHORD_PROGRESSION = "Dm7 G7 Cmaj7 Cmaj7"


# ===========================================================================
# Perfiles por modo
# ===========================================================================
# Cada perfil agrupa parámetros específicos del modelo de ese modo.
# Acceso vía active_profile() abajo.
#
# - model_path: path local del bundle .mag. Si no existe, el cargador
#   descarga desde model_url.
# - model_config_id: identificador interno del modelo dentro de Magenta.
#   Tiene que matchear el del bundle; si no, falla al cargar.
# - temperature: 0.7 conservador, 1.0 balanceado, 1.3 creativo/caótico.
# - response_bars: cuántos compases genera la IA por respuesta.
#   Latencia (medida en spike): 4=0.12s, 8=0.29s, 16=0.75s.
# - needs_chords: si False, el modelo ignora chord_progression.
# - polyphonic: si True, el modelo puede generar acordes simultáneos.

PROFILES = {
    "improv": {
        "model_path": MODELS_DIR / "chord_pitches_improv.mag",
        "model_url": "http://download.magenta.tensorflow.org/models/chord_pitches_improv.mag",
        "model_config_id": "chord_pitches_improv",
        "temperature": 0.8,
        "response_bars": 4,
        "needs_chords": True,
        "polyphonic": False,
    },
    "melody": {
        "model_path": MODELS_DIR / "attention_rnn.mag",
        "model_url": "http://download.magenta.tensorflow.org/models/attention_rnn.mag",
        "model_config_id": "attention_rnn",
        "temperature": 1.0,
        "response_bars": 4,
        "needs_chords": False,
        "polyphonic": False,
    },
    # Performance todavía no implementado. Bundle a confirmar cuando se
    # active el modo. Dejo el esqueleto para que la forma del config
    # no cambie cuando llegue el momento.
    "performance": {
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
# Subdirectorios donde el sistema guarda frases del usuario e importa
# solos externos como style primers. Vacíos por ahora; el módulo
# memory/ los va a poblar.

USER_PHRASES_DIR = MEMORY_DIR / "user_phrases"
IMPORTED_PRIMERS_DIR = MEMORY_DIR / "imported_primers"


# ===========================================================================
# Logging
# ===========================================================================

LOG_LEVEL = "INFO"          # DEBUG | INFO | WARNING | ERROR
LOG_FILE = PROJECT_ROOT / "neuraljam.log"


# ===========================================================================
# Helpers
# ===========================================================================

def active_profile():
    """
    Devuelve el dict del perfil correspondiente al MODE actual.

    Función (no constante) a propósito: si el CLI sobreescribe MODE
    después del import, los llamantes siguen viendo el modo correcto.
    """
    if MODE not in PROFILES:
        raise ValueError(
            f"MODE '{MODE}' no reconocido. "
            f"Válidos: {list(PROFILES.keys())}"
        )
    return PROFILES[MODE]


def ensure_dirs():
    """
    Crea los directorios runtime si no existen. Llamar UNA vez al
    bootstrap del sistema, no en cada operación.

    Separado de la carga del módulo porque el side effect de crear
    directorios al importar config.py es mala práctica (rompería
    tests que solo quieren leer constantes).
    """
    for d in (MODELS_DIR, MEMORY_DIR, USER_PHRASES_DIR, IMPORTED_PRIMERS_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Self-check (corrible directamente: `python -m neuraljam.config`)
# ===========================================================================

if __name__ == "__main__":
    # Diagnóstico rápido: imprime el estado del config como lo vería el
    # resto del sistema. Útil para verificar paths después de mover el
    # proyecto de carpeta.
    print(f"PROJECT_ROOT:    {PROJECT_ROOT}")
    print(f"MODE actual:     {MODE}")
    print(f"MIDI in:         {MIDI_INPUT_NAME!r}")
    print(f"MIDI out:        {MIDI_OUTPUT_NAME!r}")
    print(f"SILENCE_TIMEOUT: {SILENCE_TIMEOUT}s")
    print(f"QPM_FALLBACK:    {QPM_FALLBACK}")
    print(f"Progresión:      {CHORD_PROGRESSION!r}")
    print()
    profile = active_profile()
    print(f"Perfil activo ('{MODE}'):")
    for k, v in profile.items():
        print(f"  {k}: {v}")
    print()
    print(f"Modelo existe en disco: {profile['model_path'].exists()}")
