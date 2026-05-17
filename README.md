# AI Duet Local

Versión local y standalone del AI Duet de Google. Recibe MIDI de un teclado físico, lo procesa con MelodyRNN (Magenta) y devuelve una respuesta melódica generada por IA hacia un DAW vía loopMIDI.

```
Teclado Casio (USB) ─► Python (mido) ─► MelodyRNN attention ─► Python ─► loopMIDI ─► Studio One ─► Sonido
```

## Qué hace

- Escucha tu teclado MIDI físico en tiempo real
- Detecta el fin de una frase melódica (después de un silencio configurable)
- Pasa la frase al modelo MelodyRNN preentrenado (attention_rnn)
- Genera una respuesta musical y la envía al DAW
- Loop continuo turn-based (vos tocás → IA responde → vos tocás...)

## Requisitos

- **Windows 10 u 11** (probado en Windows con PowerShell)
- **Python 3.10.x** (no 3.11+, magenta 2.1.4 no es compatible)
- **loopMIDI** (gratuito, de Tobias Erichsen)
- **Studio One** u otro DAW compatible con MIDI input
- **Teclado MIDI USB**
- Espacio en disco: ~2 GB (TensorFlow + dependencias)
- Internet para la instalación inicial

## Estructura del proyecto

```
C:\AI-Duet-Local\
├── venv\                  # Entorno virtual de Python
├── ai_duet.py            # Script principal: bucle dueto
├── test_midi.py          # Test input MIDI (lee Casio)
├── test_output.py        # Test output MIDI (escribe a Studio One)
├── test_modelo.py        # Test carga del modelo
├── attention_rnn.mag     # Modelo preentrenado (~3 MB)
├── requirements.lock     # Versiones exactas de paquetes
└── README.md             # Este archivo
```

## Instalación desde cero

### 1. Python 3.10

```powershell
winget install Python.Python.3.10
```

Cerrar PowerShell, abrir uno nuevo. Verificar:

```powershell
py -3.10 --version
```

Debe mostrar `Python 3.10.x`.

### 2. Crear proyecto y venv

```powershell
mkdir C:\AI-Duet-Local
cd C:\AI-Duet-Local
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
```

Si Activate.ps1 da error de política de scripts:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Confirmar con `S` y reintentar.

### 3. Instalar dependencias

Si tenés `requirements.lock`:

```powershell
pip install -r requirements.lock
```

Si **NO** tenés el lock (instalación desde cero), seguir este orden exacto:

```powershell
pip install --upgrade pip
pip install "setuptools<69"
pip install mido python-rtmidi
pip install "numba>=0.59" "llvmlite>=0.42"
pip install librosa
pip install "tensorflow<2.16"
pip install note-seq pretty_midi mir_eval pygtrie
pip install magenta --no-deps
pip install tensorflow_probability tf-slim
```

Importante: el `--no-deps` en `magenta` es clave. Sin él, magenta intenta instalar versiones antiguas de dependencias que ya no tienen wheels precompilados para Python 3.10 en Windows.

### 4. Descargar modelo MelodyRNN

```powershell
Invoke-WebRequest -Uri "http://download.magenta.tensorflow.org/models/attention_rnn.mag" -OutFile "attention_rnn.mag"
```

### 5. Instalar y configurar loopMIDI

- Descargar de: https://www.tobias-erichsen.de/software/loopmidi.html
- Instalar normal.
- Abrir loopMIDI (queda en la bandeja del sistema).
- Crear un puerto llamado `AI-Duet-OUT` (escribirlo en el campo "New port-name" y click en el "+").
- Dejar la app corriendo (minimizada).

### 6. Configurar Studio One

**Registrar loopMIDI como dispositivo externo:**

- Studio One → Opciones → Dispositivos Externos → Agregar... → Nuevo teclado
- Nombre: `loopMIDI-IA`
- Recibir desde: `AI-Duet-OUT`
- Enviar a: Ninguno
- OK, Aplicar.

**Crear pista para recibir la respuesta de la IA:**

- Track → Add Tracks (Ctrl+T)
- Formato: Instrumento
- Nombre: `IA-Respuesta`
- Entrada: `loopMIDI-IA`
- Salida: Nuevo instrumento → **Presence XT**
- OK.
- En la pista, activar el botón **Monitor** (icono de altavoz/parlante).

## Uso diario

1. Conectar el teclado MIDI por USB y encenderlo.
2. Abrir loopMIDI (verificar que aparece en la bandeja).
3. Abrir Studio One con tu canción que tenga la pista `IA-Respuesta` armada.
4. En PowerShell:

```powershell
cd C:\AI-Duet-Local
.\venv\Scripts\Activate.ps1
python ai_duet.py
```

5. Esperar a que cargue el modelo (`Modelo listo.`).
6. Tocar una frase melódica (4–10 notas, una sola línea, sin acordes).
7. Soltar todas las teclas y esperar 2.5 segundos sin tocar.
8. La IA responde con su frase, que suena en Studio One.
9. Repetir: tocar, esperar, escuchar la respuesta, tocar de nuevo.

Para salir: `Ctrl+C` en la terminal.

## Parámetros configurables

Editar las constantes al principio de `ai_duet.py`:

| Parámetro | Default | Qué hace |
|---|---|---|
| `INPUT_NAME` | `'CASIO USB-MIDI 0'` | Nombre del puerto MIDI del teclado |
| `OUTPUT_NAME` | `'AI-Duet-OUT 2'` | Puerto loopMIDI de salida |
| `BUNDLE_PATH` | `'attention_rnn.mag'` | Archivo del modelo |
| `CONFIG_ID` | `'attention_rnn'` | Tipo de modelo (debe matchear el bundle) |
| `SILENCE_TIMEOUT` | `2.5` | Segundos de silencio que disparan la respuesta. Más alto = más tiempo para pensar la frase |
| `TEMPERATURE` | `1.0` | Creatividad de la respuesta. `0.7` = conservador, sigue tu estilo. `1.3` = creativo, más sorpresas |
| `MIN_NOTES_TO_RESPOND` | `2` | Mínimo de notas para que la IA dispare |
| `QPM` | `120` | Tempo asumido en BPM |

Para verificar nombres de puertos MIDI disponibles:

```powershell
python -c "import mido; print('IN:', mido.get_input_names()); print('OUT:', mido.get_output_names())"
```

## Modelos alternativos

Hay 3 modelos preentrenados de MelodyRNN, con tradeoffs distintos:

| Modelo | URL | Característica |
|---|---|---|
| `attention_rnn` | http://download.magenta.tensorflow.org/models/attention_rnn.mag | Mejor estructura larga (default) |
| `lookback_rnn` | http://download.magenta.tensorflow.org/models/lookback_rnn.mag | Repite patrones, más predecible |
| `basic_rnn` | http://download.magenta.tensorflow.org/models/basic_rnn.mag | Baseline simple |

Para cambiar el modelo:

1. Descargar el `.mag` con `Invoke-WebRequest`.
2. En `ai_duet.py` cambiar las dos líneas:

```python
BUNDLE_PATH = 'lookback_rnn.mag'
CONFIG_ID = 'lookback_rnn'
```

## Limitaciones conocidas

- **Monofónico**: MelodyRNN procesa una nota a la vez. Si tocás acordes, las notas extra se ignoran o confunden al modelo.
- **Tempo fijo** (QPM = 120). No detecta el tempo real al que tocás.
- **Sin GPU**: la primera inferencia tarda más (~5–10 s, TF inicializa). Las siguientes son rápidas (0.2–1 s).
- **Magenta 2.1.4 desactualizado**: declara dependencias rígidas de 2021. Se instala con `--no-deps` y se completan manualmente. Los warnings de pip sobre versiones incompatibles son falsos positivos: el modelo funciona.

## Troubleshooting

### `Failed building wheel for numba / llvmlite / python-rtmidi`

Falta Visual C++ Build Tools, pero no hace falta instalarlos. Forzar wheels precompilados:

```powershell
pip install "setuptools<69"
pip install --force-reinstall --no-deps python-rtmidi
pip install --upgrade "numba>=0.59" "llvmlite>=0.42"
```

### `ModuleNotFoundError: No module named 'tensorflow_probability'` o similar

Instalar la dep que falte. Otras comunes:

```powershell
pip install tensorflow_probability tf-slim
```

### `NoSuchKeyThe specified key does not exist` al descargar el modelo

La URL `storage.googleapis.com/magentadata/...` está rota. Usar:

```
http://download.magenta.tensorflow.org/models/attention_rnn.mag
```

### El script corre pero no suena nada en Studio One

Verificar, en este orden:

1. Pista `IA-Respuesta` tiene el botón **Monitor** activado.
2. Presence XT está cargado en esa pista.
3. Volumen de la pista no está en 0, no está muteada.
4. loopMIDI sigue corriendo en la bandeja (icono visible).
5. El puerto `AI-Duet-OUT` sigue existiendo en loopMIDI.
6. La pista tiene como input `loopMIDI-IA`.

### La IA dispara antes de que termines de tocar

Subir `SILENCE_TIMEOUT` a `3.5` o `4.0`.

### La IA nunca responde

- Verificar consola: ¿aparece `→ Frase: X notas...`?
  - Si no aparece, no detecta MIDI input. Revisar `INPUT_NAME` y que `test_midi.py` funcione.
- Bajar `MIN_NOTES_TO_RESPOND` a `1` (responde con cualquier nota).
- Bajar `SILENCE_TIMEOUT` a `1.5`.

### Respuestas muy cortas (1–2 notas)

Subir el cálculo de duración en `ai_duet.py`:

```python
duration = max(input_seq.total_time * 2.0, 6.0)
```

(Ya está aplicado en la versión actual del script.)

### Respuestas demasiado "raras" o muy "predecibles"

Ajustar `TEMPERATURE`:
- `0.7` = más conservador, respuestas que siguen tu melodía
- `1.0` = balanceado (default)
- `1.3` = más creativo, puede salir cualquier cosa

## Cómo se construyó

Inspirado en [AI Duet de Google Creative Lab](https://github.com/googlecreativelab/aiexperiments-ai-duet), pero reimplementado desde cero porque:

- AI Duet original es web-based (navegador + Flask), captura MIDI con Web MIDI API y reproduce con Tone.js. No conecta a un DAW.
- AI Duet está archivado y sus dependencias no funcionan en versiones modernas de Python.
- Esta versión usa el mismo modelo (MelodyRNN attention) pero con una arquitectura standalone Python, conectada a un teclado físico y un DAW vía loopMIDI.

## Créditos

- Modelo MelodyRNN: equipo Magenta, Google.
- Librerías: TensorFlow, Magenta, note-seq, mido, python-rtmidi.
- Inspiración: AI Duet, Google Creative Lab.
