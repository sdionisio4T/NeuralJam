# NeuralJam — Estado del sistema (2026-05-21)

## Resumen

El sistema está **operativo y en uso**. Todos los módulos core están
implementados. Los modelos base corren. Las capas de inteligencia
(subconciente, scheduler, groove) están en distintas etapas de integración.

---

## Fases implementadas ✅

### Fase 1 — MemoryBank + SubconsciousEngine
`neuraljam/memory/bank.py`, `neuraljam/subconscious/engine.py`

Buffer circular de 8 frases del usuario. El subconciente construye `context_seq`
en background entre turnos. En modo normal usa solo el banco (testeado: suena mejor
que MusicVAE o ImprovRNN como primer en este contexto).

### Fase 2 — Scheduler dinámico
`neuraljam/scheduler/scheduler.py`

Decide si responder, con qué temperatura y cuántos compases. Silencio
probabilístico (15%), máximo 2 silencios seguidos. Temperatura decae con la
sesión y sube con densidad del usuario. Todo se clampea por `ModeConfig`.

### Fase 3 — ImprovRNN en RAM
`neuraljam/models/loader.py`

ImprovRNN cargado como modelo alternativo (tecla [2]). En modo normal fue
removido del subconciente porque el banco de sesión solo sonaba mejor.
Disponible en modos libre/experimental via `use_improv_background=True`.

### Fase 3b — Detección de tonalidad
`neuraljam/analysis/tonality.py`

Krumhansl-Kessler sobre histograma de pitch classes. Activa cuando hay ≥3
frases en el banco. Genera progresión ii-V-I dinámica para ImprovRNN en
modos libre/experimental.

### Fase 4 — MusicVAE
`neuraljam/models/music_vae_loader.py`

`cat-mel_2bar_big` — interpola entre últimas 2 frases del usuario en espacio
latente. Activo solo en modos libre/experimental (`use_music_vae=True`).
En modo normal fue descartado por las mismas razones que ImprovRNN.

### Fase 6 — Sistema de modos
`neuraljam/modes.py`

Tecla [m] cicla: NORMAL → IMITACIÓN → LIBRE → EXPERIMENTAL.
Cada modo controla temperatura, compases, capas activas del subconciente.

### Fase 7b — Baseline toggle
`neuraljam/neuraljam.py` (`_start_baseline_listener`)

Tecla [b] desactiva todas las capas (scheduler, subconciente, MusicVAE).
Modelo corre limpio con temp=1.0, bars=2. Sirve para comparar si las capas
realmente mejoran el resultado.

### Fase 8 — MIDI Clock sync
`neuraljam/midi/clock.py`

Sincronización con Studio One via loopMIDI. EMA sobre pulsos de 24ppq.
`clock.qpm` alimenta el cálculo de compases y el recorder.
`--sync-beat` espera al próximo downbeat antes de reproducir.

### Fase 9 — Humanize
`neuraljam/generation/humanize.py`

Swing (0.08) + velocity variance (±12). Se aplica antes de reproducir,
después de la generación.

### Fase 10 — Groove Engine
`neuraljam/analysis/groove/`

Calcula density, syncopation, pulse_regularity, tension por turno.
`temperature_delta()` y `bars_hint()` están calculados y logueados.
**Pendiente:** wirear al Scheduler (espera que el usuario pueda escuchar
la diferencia antes de activarlo).

### Fase 13 — Pipeline de fine-tuning
`tools/export_for_training.py`, `colab/finetune_neuraljam.ipynb`

Export escanea `saved_phrases/`, filtra por fecha/notas mínimas, crea ZIP.
Notebook de Colab completo: install → Drive → NoteSequences → TFRecord →
melody_rnn_train → .mag. Probado: encontró 16 frases del usuario.

### Fase 15 — Grabación de sesión
`neuraljam/recording/recorder.py`

MIDI tipo 1, 2 pistas (ch0 usuario, ch1 IA). Se exporta automáticamente
al `Ctrl+C` si hubo al menos 1 turno. Guardado en `sessions/`.

---

## Modelos disponibles

| Tecla | Modelo | Estado | Notas |
|-------|--------|--------|-------|
| [1] | attention_rnn | ✅ operativo | Default, el más probado |
| [2] | chord_pitches_improv | ✅ operativo | Requiere chord annotation |
| [3] | performance_with_dynamics | ✅ bundle OK | Sin testear end-to-end |
| [4] | lookback_rnn | ⬇ descarga al arrancar | Motívico, repetición |
| [5] | basic_rnn | ⬇ descarga al arrancar | Más simple, sin atención |
| [6] | polyphony_rnn | ⬇ descarga al arrancar | Acordes reales |

---

## Pendiente / próximos pasos

### Groove → Scheduler (alta prioridad, baja complejidad)
`GrooveEngine.temperature_delta()` y `bars_hint()` ya están calculados en
cada turno. Solo falta pasarlos al Scheduler como offset adicional.
Espera audición del usuario para validar que mejora.

### PerformanceRNN end-to-end
Bundle descargado, builder implementado. No se testeó con frases reales
en el loop completo. Probarlo con tecla [3].

### Modelos nuevos [4][5][6]
Se descargan automáticamente en el primer arranque. Probar uno por uno
y comparar el carácter de cada uno.

### Fine-tuning
16 frases guardadas actualmente — necesita 30 para un cambio sutil.
El pipeline de Colab está listo para cuando haya suficiente material.

### Groove → SubconsciousEngine (baja prioridad)
Usar el perfil rítmico para pesar el primer del subconciente.
Ideas: si la frase es sincopada, preferir frases del banco también sincopadas.

---

## Configuración activa

```
SILENCE_TIMEOUT      = 2.5s
MIN_NOTES_TO_RESPOND = 2
QPM_FALLBACK         = 120
response_probability = 0.85
max_consecutive_silences = 2
base_temperature     = 0.8
MemoryBank maxlen    = 8
MusicVAE             = cat-mel_2bar_big (2 compases)
Humanize swing       = 0.08
Humanize velocity_variance = 12
```

---

## Estructura de archivos de sesión

```
AI-Duet-Local/
├── neuraljam/          ← paquete principal
│   ├── config.py
│   ├── modes.py
│   ├── neuraljam.py    ← entry point
│   ├── generation/
│   ├── memory/
│   ├── midi/
│   ├── models/
│   ├── subconscious/
│   ├── scheduler/
│   ├── analysis/
│   │   ├── tonality.py
│   │   └── groove/
│   ├── recording/
│   ├── harmony/
│   └── playback/
├── models_data/        ← .mag files (gitignored)
├── saved_phrases/      ← frases guardadas con [s] (gitignored)
├── sessions/           ← grabaciones MIDI completas
├── training_export/    ← ZIPs para Colab
├── tools/
│   └── export_for_training.py
├── colab/
│   └── finetune_neuraljam.ipynb
├── ARCHITECTURE.md     ← diseño completo del sistema
└── STATUS.md           ← este archivo
```
