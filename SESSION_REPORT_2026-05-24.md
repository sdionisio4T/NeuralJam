# NeuralJam — Reporte de sesión 2026-05-24

## Resumen ejecutivo

Sesión de desarrollo y testeo en vivo. Se implementaron tres funcionalidades
nuevas: modo DIÁLOGO (call-and-response estricto), match_user_bars para los
modos duales A/B, A/B2 y B+, y el modo BLUES con filtro armónico basado en
Krumhansl-Kessler. Todo testeado en vivo con audio real tocando en F blues.

---

## 1. Modo DIÁLOGO

**Motivación:** el modelo a veces responde con demasiados o pocos compases
respecto a lo que tocó el usuario. Call-and-response natural requiere que el
espacio sea simétrico.

**Implementación:**
- `ModeConfig.match_user_bars = True` (nuevo campo en el dataclass)
- Cálculo: `bars = round(phrase_dur / bar_dur)`, clampado a `response_bars_max`
- `always_respond = True`, usa banco de sesión, temp 0.6–0.95
- Posición en el ciclo `[m]`: segundo lugar (un toque desde NORMAL)

**Archivos tocados:**
- `neuraljam/modes.py` — campo nuevo + modo "dialogue"
- `neuraljam/neuraljam.py` — rama `match_user_bars` antes del scheduler

**Resultado testeado:** ✅ "mejoró muchísimo" — el modelo no desperdicia
pasos intentando terminar antes o después. El espacio fijo le da estructura.

---

## 2. match_user_bars en modos duales (A/B, A/B2, B+)

**Motivación:** los modos de comparación también se benefician de la simetría
de compases. Además facilita comparar A vs B en igualdad de condiciones.

**Implementación:**
- `match_user_bars = True` agregado a los tres modos
- `response_bars_max` subido a 6 para no recortar frases largas
- Sin cambios en `neuraljam.py` — el cálculo de `bars` ya corría antes
  del bloque `dual_response`, así que llega correcto a A y B automáticamente

**Archivos tocados:**
- `neuraljam/modes.py` — ab, ab2, bplus actualizados

---

## 3. Modo BLUES con filtro armónico

### 3a. Detección de blues (Krumhansl-Kessler extendido)

**Archivo:** `neuraljam/analysis/tonality.py`

La función `detect_tonality(bank)` ya existía y detectaba mayor/menor.
Se extendió con:

- Diccionario `_BLUES_PROGRESSIONS` — I7 IV7 I7 V7 para las 12 tónicas
- Función `_is_blues(histogram, tonic)` — detecta sabor a blues verificando
  que b3 y b7 sean ambas > 7% del histograma relativo a la tónica detectada
- Si `_is_blues` es True: `mode = "blues"`, `tonic_chord = "F7"`,
  progresión devuelta es la blues (no ii-V-I)
- `TonalityResult.__str__` actualizado para mostrar "blues" como modo

**Resultado en vivo (F blues):**
```
[TONALIDAD] F blues (confianza: 83%) → F7 Bb7 F7 C7
```

### 3b. Filtro de escala de blues

**Archivo:** `neuraljam/analysis/blues_filter.py` (nuevo)

Post-procesado de pitch post-generación, pre-humanize.

```
Escala jazz-blues: R  2  b3  M3  4  #4  5  b7
                   0  2   3   4  5   6  7  10  (semitonos desde tónica)
```

- `apply_blues_filter(seq, tonic_pc, style="jazz")` — modifica in-place
- Para cada nota: calcula pitch class relativo a tónica, encuentra el grado
  más cercano por distancia circular, ajusta máximo ±6 semitonos
- Rango seguro: `max(21, min(108, new_pitch))`
- El #4 (tritono) es la blue note característica
- El M3 y el 2 dan color jazzístico sin perder el sabor bluesy

### 3c. Modo BLUES en el ciclo

**Archivo:** `neuraljam/modes.py`

```python
"blues": ModeConfig(
    name="blues", display="BLUES",
    temp_min=0.6, temp_max=0.95, response_bars_max=4,
    use_memory=True, always_respond=True,
    use_blues_filter=True,
    match_user_bars=True,    # añadido tras primer test — evita respuestas esparsas
)
```

`use_blues_filter: bool = False` agregado al dataclass como campo opcional.

MODE_CYCLE actualizado:
```python
["normal", "blues", "dialogue", "imitation", "free",
 "experimental", "ab", "ab2", "bplus", "direct"]
```

### 3d. Wiring en el loop principal

**Archivo:** `neuraljam/neuraljam.py`

- `_tonality = {"result": None, "bank_len": 0}` — caché inicializado antes del loop
- Por turno: si `current_mode.use_blues_filter` y banco no vacío, recalcula
  solo cuando `len(bank)` cambió desde el último cálculo
- Umbral de confianza: `> 0.3` para aplicar el filtro (primer turno puede ser bajo)
- `blues_tonic_pc = pc_from_name(tonal.tonic)` se calcula una vez por turno
- Se aplica en los tres puntos de reproducción:
  - Respuesta normal: antes de `humanize()`
  - Dual response A: antes de `humanize(response_a)`
  - Dual response B: antes de `humanize(response_b)`

### 3e. Fix de ritmo post-primer-test

**Problema observado:** primer test generó 5 notas en 7 segundos (muy esparso).
El scheduler asignó 3 compases para una frase de 1.5 compases del usuario.
El filtro no fue el causante (solo toca pitch, no timing), pero el espacio
excesivo diluyó el ritmo.

**Fix:** `match_user_bars = True` en el modo BLUES. La IA responde exactamente
la misma cantidad de compases que el usuario. Mismo espacio → mayor densidad.

---

## Estado del ciclo de modos (tecla [m])

| Posición | Modo | Temp | Memoria | Filtro | match_bars | Dual |
|----------|------|------|---------|--------|------------|------|
| 0 | NORMAL | 0.6–0.95 | banco | — | No | — |
| 1 | BLUES | 0.6–0.95 | banco | jazz-blues | Sí | — |
| 2 | DIÁLOGO | 0.6–0.95 | banco | — | Sí | — |
| 3 | IMITACIÓN | 0.45 fijo | no | — | No | — |
| 4 | LIBRE | 0.6–∞ | banco+VAE | — | No | — |
| 5 | EXPERIMENTAL | 0.6–∞ | banco+VAE | — | No | — |
| 6 | A/B | 0.6–0.95 | A:no/B:banco | — | Sí | A+B |
| 7 | A/B 2 | 0.6–0.95 | A:no/B:banco+A | — | Sí | A+B |
| 8 | B+ | 0.6–0.95 | A:silencio→B | — | Sí | B |
| 9 | DIRECTO | 0.6–0.95 | no | — | No | — |

---

## Archivos modificados en esta sesión

```
neuraljam/
├── analysis/
│   ├── blues_filter.py       ← NUEVO
│   └── tonality.py           ← blues detection + _BLUES_PROGRESSIONS
├── modes.py                  ← ModeConfig.use_blues_filter, BLUES mode,
│                                match_user_bars en ab/ab2/bplus
└── neuraljam.py              ← tonality cache, blues filter wiring x3
```

---

## Pendientes para próxima sesión

### Testeo pendiente
- [ ] **PerformanceRNN [3]** — bundle OK, nunca tocado en vivo
- [ ] **lookback_rnn [4] / basic_rnn [5]** — cargados, sin probar
- [ ] **IMITACIÓN** — implementada, sin comparación auditiva formal
- [ ] **LIBRE / EXPERIMENTAL** — sin validación reciente
- [ ] **DIRECTO** — sin test aislado formal
- [ ] **Groove → bank selection** — implementado, efecto sutil, necesita banco grande
- [ ] **BLUES + ImprovRNN** — actualmente el filtro actúa sobre MelodyRNN;
      ImprovRNN con progresión I7-IV7-V7 daría armonía real por acorde

### Fine-tuning
- ~17 frases guardadas → necesitan 30 mínimo
- Seguir presionando [s] después de buenos turnos
- Exportar con `python tools/export_for_training.py` cuando lleguen a 30

### Ideas futuras
- **Blues con ImprovRNN**: en modo BLUES activar ImprovRNN como generador
  principal con la progresión I7-IV7-V7 detectada por KK. Daría conciencia
  real de qué acorde está sonando, no solo escala global.
- **Detección de posición en el blues**: detectar en qué compás del chorus
  está el usuario (I, IV o V) mediante análisis de pitch del chunk actual.
  Requeriría ventana deslizante de análisis más corta que el banco completo.
- **BLUES + match_user_bars en duales**: extender blues filter a A/B y B+.

---

## Contexto para continuar en nuevo chat

```
Proyecto: NeuralJam — sistema de improvisación jazz con IA (Magenta/TensorFlow)
Repo local: C:\AI-Duet-Local\
Entry point: python neuraljam.py (desde C:\AI-Duet-Local\)

Stack:
  - Python 3.10 + Magenta 2.1.4 + TensorFlow <2.16 (Windows 11)
  - MIDI: loopMIDI + rtmidi — Casio → NeuralJam → Studio One
  - Modelos en RAM: attention_rnn [1], chord_pitches_improv [2],
    performance_rnn [3], lookback_rnn [4], basic_rnn [5], polyphony_rnn [6]

Arquitectura core:
  PhraseDetector → SubconsciousEngine (contexto) + Scheduler (temp/bars)
                → GenerationEngine (MelodyRNN) → blues_filter → humanize → Player

Archivos clave:
  neuraljam/neuraljam.py         — loop principal, wiring de todo
  neuraljam/modes.py             — ModeConfig, MODES dict, MODE_CYCLE
  neuraljam/analysis/tonality.py — Krumhansl-Kessler, detección de blues
  neuraljam/analysis/blues_filter.py — filtro de escala jazz-blues
  neuraljam/generation/humanize.py   — swing + velocity
  neuraljam/memory/bank.py           — banco circular de frases
  neuraljam/subconscious/engine.py   — contexto en background
  neuraljam/scheduler/scheduler.py   — temperatura, compases, silencio
  neuraljam/analysis/groove/         — density, syncopation, tension

Sesión anterior implementó:
  1. DIÁLOGO mode (match_user_bars=True) → testeado ✅ "mejoró muchísimo"
  2. match_user_bars en A/B, A/B2, B+ → testeado ✅
  3. Modo BLUES:
     - Krumhansl-Kessler detecta tónica + modo blues (b3+b7 > 7%)
     - blues_filter.py: ajusta pitches a escala jazz-blues post-generación
     - Testeado en F blues: confianza 83%, F7 Bb7 F7 C7 detectado ✅
     - match_user_bars=True para evitar respuestas esparsas ✅

Modos disponibles con [m] (10 en total):
  normal → blues → dialogue → imitation → free → experimental
  → ab → ab2 → bplus → direct

Próximas prioridades:
  1. Testear PerformanceRNN [3] en vivo (nunca probado)
  2. Testear lookback_rnn [4] y basic_rnn [5]
  3. Blues + ImprovRNN: usar ImprovRNN con progresión I7-IV7-V7 como generador
     principal en modo BLUES (armonía real por acorde, no solo escala global)
  4. Acumular frases para fine-tuning (van ~17, faltan ~13 para los 30 mínimos)

Convenciones del proyecto:
  - El usuario hace el commit, Claude implementa por fases
  - No tocar valores de tuning sin testear primero (ver tuning_musical.md)
  - Ctrl+C funciona pero tarda ~1s (normal, es un sleep loop para Windows)
  - os._exit(0) al final del finally para forzar salida limpia
```
