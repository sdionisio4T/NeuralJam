# NeuralJam — Estado del sistema (2026-05-24)

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

### Fase 10 — Groove Engine ✅ wired
`neuraljam/analysis/groove/`

Calcula density, syncopation, pulse_regularity, tension por turno.
`temperature_delta()` y `bars_hint()` ahora se aplican sobre los valores
del Scheduler en cada turno. Wired el 2026-05-24. Pendiente validación auditiva.

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

## Hallazgos sesión 2026-05-24

### Modo A/B — confirmado y validado auditivamente
- Ambas respuestas (A y B) suenan bien por separado
- Con metrónomo B mejora drásticamente — desarrolla las ideas del usuario
- **Por qué B es mejor**: attention_rnn tiene más material para el mecanismo
  de atención (banco + frase actual vs solo frase actual). Doble historia
  rítmica → detecta el groove y lo continúa
- **B = NORMAL**: confirmado que son idénticos en parámetros y contexto
- **A = DIRECTO**: scheduler + groove activos, sin banco. Nuevo modo creado

### Contexto y ritmo
- El contexto mejora el ritmo más que cualquier otro parámetro
- Con metrónomo la diferencia A vs B es muy audible — B "desarrolla ideas"
- Sin metrónomo ambos suenan bien pero menos comparables

### B+ — revisado, validado ✅
- Primer test negativo (sesión corta, banco vacío, A sin guía)
- Segundo test con banco acumulado: funciona bien, B suena mejor que en A/B
- Conclusión: B+ necesita algunas frases en el banco para que A tenga
  material de referencia implícita. Con banco vacío A se desorienta.

### Modos nuevos agregados
- **DIRECTO**: scheduler + groove, sin banco — equivale a la A del A/B
- **A/B**: dos respuestas por turno, A sin contexto, B con banco
- **A/B 2**: igual pero B también hereda la respuesta de A
- **B+**: A en silencio, B con banco + respuesta de A (resultado negativo)
- **DIÁLOGO**: call-and-response estricto — respuesta igual de larga que tu frase ✅ validado

### DIÁLOGO — validado 2026-05-24 ✅
- `match_user_bars=True`: bars = round(phrase_dur / bar_dur), clampado a response_bars_max
- Siempre responde (always_respond=True), usa banco, temp 0.6–0.95
- Resultado: mejora significativa ("mejoró muchísimo") — el modelo no desperdicia
  pasos intentando terminar antes/después. El espacio fijo le da estructura al diálogo.

### Launcher raíz corregido
`neuraljam.py` raíz era versión legada con `phrase.has_signal` obsoleto.
Reemplazado por redirector al paquete real.

---

## Modelos disponibles

| Tecla | Modelo | Estado | Notas |
|-------|--------|--------|-------|
| [1] | attention_rnn | ✅ validado en sesión | Default, el más probado |
| [2] | chord_pitches_improv | ✅ operativo | Requiere chord annotation |
| [3] | performance_with_dynamics | ✅ bundle OK | Sin testear end-to-end |
| [4] | lookback_rnn | ⬇ descarga al arrancar | Sin testear |
| [5] | basic_rnn | ⬇ descarga al arrancar | Sin testear |
| [6] | polyphony_rnn | ⬇ descarga al arrancar | Sin testear |

## Modos disponibles ([m] para ciclar)

| Modo | Tecla | Temp | Memoria | Carácter |
|------|-------|------|---------|----------|
| NORMAL | — | 0.6–0.95 | banco | validado ✅ |
| DIÁLOGO | m×1 | 0.6–0.95 | banco | validado ✅ — mejor estructura |
| IMITACIÓN | m×2 | 0.45 fijo | no | sin validar |
| LIBRE | m×3 | 0.6–∞ | banco+VAE | sin validar |
| EXPERIMENTAL | m×4 | 0.6–∞ | banco+VAE | sin validar |
| A/B | m×5 | 0.6–0.95 | A:no / B:banco | validado ✅ |
| A/B 2 | m×6 | 0.6–0.95 | A:no / B:banco+A | validado ✅ — suena mejor |
| B+ | m×7 | 0.6–0.95 | A:silencio→B | validado ✅ — necesita banco previo |
| DIRECTO | m×8 | 0.6–0.95 | no | sin validar aislado |

---

## Pendiente / próximos pasos

### Modelos nuevos [4][5][6]
Sin testear. Primera prioridad para próxima sesión con audio.

### PerformanceRNN [3] end-to-end
Bundle descargado, builder implementado. Nunca tocado en vivo.

### IMITACIÓN, LIBRE, EXPERIMENTAL
Implementados pero sin comparación auditiva formal.

### A/B 2 ✅ validado
B hereda banco + respuesta de A. Suena bien.

### B+ ✅ validado
Funciona bien con banco acumulado. Con banco vacío A se desorienta.

### Fine-tuning
~17 frases guardadas. Necesita 30 mínimo. Seguir guardando con [s].

### Groove → SubconsciousEngine (baja prioridad)
Usar perfil rítmico para elegir del banco la frase más parecida rítmicamente.

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
