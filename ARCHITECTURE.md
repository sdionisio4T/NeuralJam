# NeuralJam — Arquitectura del sistema

## Visión general

NeuralJam es un dúo de improvisación en tiempo real. El sistema escucha al músico,
analiza la frase, decide si y cómo responder, genera una melodía con un modelo de
Magenta, y la reproduce por MIDI. Todo ocurre en el mismo proceso Python.

```
Piano (Casio) ──MIDI──► PhraseDetector
                                │
                         Phrase(notes)
                                │
                    ┌───────────▼────────────┐
                    │      Loop principal      │
                    │     neuraljam.py        │
                    └──┬────────┬────────┬───┘
                       │        │        │
                  Scheduler  Subconciente  GrooveEngine
                  (cuándo y    (qué primer   (análisis
                  cuánto)       dar)         rítmico)
                       │        │
                       └────────┘
                            │
                     GenerationEngine
                            │
                   (modelo seleccionado)
                   MelodyRNN / ImprovRNN /
                   PerformanceRNN / etc.
                            │
                     NoteSequence
                            │
                    humanize (swing)
                            │
                    Player ──MIDI──► Studio One
```

---

## Entry point

**`neuraljam/neuraljam.py`** — loop principal.

- Arranca con `python neuraljam.py`
- Carga todos los modelos disponibles en RAM al inicio (~20–40s)
- Escucha teclado en threads separados: `[1-6]` modelo, `[m]` modo, `[b]` baseline, `[s]` guardar
- Al `Ctrl+C`: exporta sesión completa como MIDI y cierra limpio

`ai_duet.py` en la raíz es la versión legada — no usar.

---

## Módulos

### `neuraljam/config.py`
Única fuente de verdad de constantes. Ningún otro módulo hardcodea rutas ni valores.

Clave:
- `PROFILES` — diccionario con los 6 modelos disponibles
- `SIGNAL_IMPROV_MIN/MAX`, `SIGNAL_PERF_MIN/MAX` — rango de notas de señal en el grave
- `QPM_FALLBACK = 120`, `SILENCE_TIMEOUT = 2.5s`, `MIN_NOTES_TO_RESPOND = 2`

---

### `neuraljam/modes.py`
Sistema de modos de improvisación. `[m]` cicla entre ellos.

| Modo | Temp | Compases | Memoria | MusicVAE | ImprovBG | Silencio |
|------|------|----------|---------|----------|----------|---------|
| NORMAL | 0.6–0.95 | máx 3 | ✓ banco | ✗ | ✗ | probabilístico |
| IMITACIÓN | 0.45 fijo | máx 2 | ✗ | ✗ | ✗ | probabilístico |
| LIBRE | 0.6–∞ | máx 5 | ✓ banco | ✓ | ✓ | nunca |
| EXPERIMENTAL | 0.6–∞ | máx 4 | ✓ banco | ✓ | ✓ | probabilístico |

`ModeConfig` es un dataclass frozen — los modos no mutan en runtime.

---

### `neuraljam/models/`

**`loader.py`** — descarga y carga modelos de Magenta.

- `load_all_models()` — carga todos los perfiles con `.mag` disponible
- `download_bundle_if_missing()` — descarga automática en el primer arranque
- `LoadedModel` — wrapper: `generator`, `magenta_config`, `profile`, `family`
- Builders por familia: `melody_rnn`, `improv_rnn`, `performance_rnn`, `polyphony_rnn`

**`music_vae_loader.py`** — loader específico para MusicVAE (checkpoint, no bundle).

**Modelos disponibles (teclas 1–6):**

| Tecla | Clave | Familia | Archivo | Descripción |
|-------|-------|---------|---------|-------------|
| [1] | melody | melody_rnn | attention_rnn.mag | Atención 2 capas — default |
| [2] | improv | improv_rnn | chord_pitches_improv.mag | Chord-conditioned |
| [3] | performance | performance_rnn | performance_with_dynamics.mag | Polifónico expresivo |
| [4] | lookback | melody_rnn | lookback_rnn.mag | Ventana 2 compases — motívico |
| [5] | basic | melody_rnn | basic_rnn.mag | Sin atención — respuesta directa |
| [6] | polyphony | polyphony_rnn | polyphony_rnn.mag | Acordes reales, voicing |

---

### `neuraljam/generation/`

**`engine.py`** — `GenerationEngine`.

- Recibe `phrase`, `model_key`, `context_seq`, `temperature`, `response_bars`
- Dispatch por `model_key` → `LoadedModel` seleccionado
- Usa `model_lock` compartido (threading.Lock) — TF1 no es thread-safe
- Devuelve `NoteSequence` rebasado a `t=0`, o `None` si falla

**`prepare.py`** — `build_input_sequence()`.

- Convierte `List[NoteEvent]` → `NoteSequence` con primer + sección a generar
- Si `context_seq` no es None, lo antepone al primer (subconciente)
- Agrega chord annotations solo si `progression` no es None

**`humanize.py`** — `humanize(seq, swing, velocity_variance, qpm)`.

- Swing: desplaza notas en posiciones de corchea off-beat
- Velocity: variación aleatoria ±velocity_variance

---

### `neuraljam/midi/`

**`phrase_detector.py`** — `PhraseDetector`.

- Escucha MIDI input (Casio)
- Detecta fin de frase por `SILENCE_TIMEOUT` (2.5s sin notas)
- Devuelve `Phrase(notes, has_signal)` — `has_signal` si la última nota cayó en zona grave

**`clock.py`** — `MidiClock`.

- Recibe clock MIDI de Studio One vía loopMIDI
- Estima BPM en vivo por promedio móvil (EMA) sobre pulsos de 24ppq
- `has_sync`, `qpm`, `wait_for_downbeat()`

**`output.py`** / **`ports.py`** — MidiOutput, resolución de puertos.

---

### `neuraljam/memory/`

**`bank.py`** — `MemoryBank`.

- Buffer circular de 8 frases del usuario (más recientes)
- `add(seq)`, `get_random()`, `get_all_user()`
- `preload(folder)` — carga MIDIs externos al inicio (`--preload`)

**`saver.py`** — `save_phrase(seq, tag)`.

- Guarda en `saved_phrases/YYYY-MM-DD/` con timestamp
- `tag="user"` → `user_HHMMSS.mid`, `tag="ai"` → `ai_HHMMSS.mid`

---

### `neuraljam/subconscious/`

**`engine.py`** — `SubconsciousEngine`.

Thread de background que construye `context_seq` entre turnos.

Flujo por modo:
```
use_memory=False (imitación) → context_seq = None
use_memory=True, solo banco  → context_seq = última frase del usuario
use_improv_background=True   → ImprovRNN con chord detectado + banco
use_music_vae=True           → MusicVAE interpola entre últimas 2 frases
```

Detección de chord:
- Si hay ≥3 frases en banco → `detect_tonality()` (Krumhansl-Kessler)
- Si no → pitch más frecuente → tríada simple

**`phrase_to_seq(notes, qpm)`** — convierte `List[NoteEvent]` → `NoteSequence`.

---

### `neuraljam/scheduler/`

**`scheduler.py`** — `Scheduler`.

- `should_respond(mode)` → `"enter"` | `"silent"`
  - 15% de chance de silencio intencional (el sistema "escucha")
  - Máximo 2 silencios seguidos, luego fuerza respuesta
  - `always_respond=True` (modo libre) → siempre entra
- `temperature(phrase_note_count, phrase_duration, mode)` → float
  - Empieza en 0.8, decae a 0.7 en 30 turnos
  - Sube hasta +0.2 si el usuario toca denso
  - Clampeado entre `mode.temp_min` y `mode.temp_max`
- `response_bars(phrase_duration, bpm, mode)` → int
  - 35% réplica corta, 40% proporcional, 25% desarrollo largo
  - Techo = `mode.response_bars_max`

---

### `neuraljam/analysis/`

**`tonality.py`** — `detect_tonality(bank)`.

- Histograma de pitch classes de todas las frases del banco
- Correlación de Pearson contra 24 perfiles (12 mayores + 12 menores) de Krumhansl-Kessler
- Devuelve `TonalityResult(tonic, mode, confidence, progression, tonic_chord)`
- Progresión ii-V-I dinámica por tonalidad detectada

**`groove/`** — `GrooveEngine`.

- `profile.py` — `RhythmProfile`: density, syncopation, pulse_regularity, tension, avg_note_duration
- `extractor.py` — `extract_profile(seq, qpm, reference)`: métricas desde NoteSequence
- `engine.py` — `GrooveEngine.update(user_seq, ai_seq, qpm)`: mantiene perfil turno a turno
  - `temperature_delta()` → sugiere ajuste de temp (+0.10 denso, +0.08 tenso, -0.05 regular)
  - `bars_hint()` → sugiere compases de respuesta (1–3)
  - **Estado actual**: calculado y logueado, pendiente de wirear al Scheduler

---

### `neuraljam/recording/`

**`recorder.py`** — `SessionRecorder`.

- `add_turn(user_seq, ai_seq, qpm)` — acumula turnos en memoria
- `export(path=None)` — exporta MIDI tipo 1, 2 pistas (ch0 usuario, ch1 IA)
- Guarda en `sessions/YYYY-MM-DD_HHMM.mid` al `Ctrl+C`

---

### `neuraljam/harmony/`

**`progression.py`** — `Progression`.

- `from_config()` → toma `CHORD_PROGRESSION` del config
- Solo se pasa al engine si el perfil tiene `needs_chords=True` (solo improv)

---

## Threads en runtime

| Thread | Nombre | Descripción |
|--------|--------|-------------|
| Main | — | Loop principal: detect → generate → play |
| KeyboardListener | `[s]` | Activa save_flag para guardar frase |
| ModelSwitcher | `[1-6]` | Cambia model_state["current"] |
| ModeCycler | `[m]` | Cicla mode_state["current"] |
| BaselineListener | `[b]` | Toggle baseline_state["active"] |
| SubconsciousEngine | interno | Construye context_seq en background |
| MidiClock | reloj | Recibe clock MIDI de Studio One |

El `model_lock` (threading.Lock) es el único recurso compartido crítico —
garantiza que solo un thread llame a `generator.generate()` a la vez.

---

## Flujo de datos por turno

```
1. detector.wait_for_phrase()
      → Phrase(notes=[NoteEvent...], has_signal)

2. current_mode = MODES[mode_state["current"]]
   key = model_state["current"]

3. [baseline activo]
      context=None, temp=1.0, bars=2

   [modo normal]
      scheduler.should_respond(mode) → "enter" | "silent"
      context = subconscious.get_context()   si use_memory
      temp    = scheduler.temperature(...)
      bars    = scheduler.response_bars(...)

4. engine.respond(phrase.notes, model_key=key,
                  context_seq=context, temperature=temp,
                  response_bars=bars, qpm_override=live_qpm)
      → NoteSequence

5. humanize(response, swing=0.08, velocity_variance=12, qpm=live_qpm)

6. groove.update(user_ns, response, qpm=live_qpm)   [análisis, log]
   subconscious.trigger(user_ns, response, mode)     [background]
   recorder.add_turn(user_ns, response, qpm)

7. player.play(response)  →  MIDI out  →  Studio One
```

---

## Fine-tuning pipeline

```
saved_phrases/YYYY-MM-DD/user_*.mid
        │
python tools/export_for_training.py
        │
training_export/neuraljam_training_*.zip
        │
[Google Drive]
        │
colab/finetune_neuraljam.ipynb
  → melody_rnn_create_dataset
  → melody_rnn_train (500–2000 steps, GPU T4)
  → neuraljam_YYYY-MM-DD.mag
        │
models_data/neuraljam_YYYY-MM-DD.mag
  + config.py: model_path → nuevo .mag
```
