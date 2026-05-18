# NeuralJam — Roadmap y decisiones

Documento vivo. Trazabilidad de qué se decidió y por qué.
Cuando algo cambia respecto del plan original, se anota acá.

---

## Estado actual

- `ai_duet.py` (raíz): script monolítico funcional, MelodyRNN, monofónico,
  turn-based, conectado a Studio One via loopMIDI. Se mantiene como
  referencia hasta que `neuraljam/` lo reemplace funcionalmente.
- `spike_improv/`: validación técnica de ImprovRNN. Cerrado, resultados
  abajo. Se mantiene como archivo histórico, no se importa desde
  `neuraljam/`.
- `neuraljam/`: refactor en curso. Solo `config.py` hasta ahora.

---

## Spike ImprovRNN — resultados clave

Fecha de cierre: mayo 2026.

### Lo que el spike confirmó

- **URL oficial del bundle funciona**:
  `http://download.magenta.tensorflow.org/models/chord_pitches_improv.mag`
  (5.4 MB).
- **Notación armónica completa**: el parser de Magenta acepta triadas,
  séptimas, alteraciones (`Em7b5`, `A7b9`), `Dm6`, slash chords (`C/E`).
  Solo falla con notación inválida (`XYZ`).
- **Latencia trivial en CPU**: 0.12s para 4 compases, 0.29s para 8,
  0.75s para 16. Dos órdenes de magnitud abajo del objetivo (<3s) del
  plan original.
- **RAM aceptable**: 410 MB pico. Cabe holgado en 8 GB.
- **Cold start**: ~17s, dominado por imports de TensorFlow, no por
  Magenta. Requiere UX de bootstrap claro (mensajes de progreso).
- **Em7b5 entendido**: en test con primer neutral (sin B ni Bb), el
  modelo generó 7 Bb (b5 característica) vs 1 B natural sobre 34 notas.
  No degrada silenciosamente Em7b5 a triada Em.

### El hallazgo decisivo

**El conditioning del primer pesa más que el modelo.**

El spike inicial usaba un primer cuadrado de 4 negras diatónicas a C
mayor. Con ese primer, las generaciones sonaban rítmicamente cuadradas
y armónicamente sesgadas a C mayor. Cambiando el primer a uno bebop, uno
sincopado, y uno con cromatismo (Eb fuera de C mayor), la calidad
musical cambió completamente. El modelo respondió a cada uno con
fraseo coherente al input.

Conclusión: ImprovRNN tiene musicalidad latente. La calidad de la
generación es función de la calidad del primer.

### Limitaciones reales (no resolubles con mejor primer)

- Cromatismo solo como passing tone local, no como vocabulario extendido.
- Bebop superficial; no alcanza densidad lineal tipo Charlie Parker.
- Frases relativamente cortas; sin desarrollo motívico largo.
- Sin microtiming (swing real, anticipación al 4+) por diseño del modelo.

Estas son limitaciones arquitectónicas de RNNs sin atención global.
No se resuelven con postprocess. Se mitigan con conditioning más rico:
primers de mejor calidad, memoria de frases del usuario, solos
importados como style primers.

---

## Cambios al plan original

El roadmap original (ver "ORDEN DE IMPLEMENTACIÓN" del briefing del
proyecto) cambió en tres puntos basado en lo que aprendió el spike.

### 1. Captura MIDI fiel se vuelve crítica

**Antes**: el `phrase_detector` se pensaba simple, lista de pitches +
timeout de silencio.

**Ahora**: como el conditioning del primer determina la calidad, la
captura debe preservar timing exacto, velocity, duraciones reales
(corchea vs negra) y silencios. Si cuantizamos groseramente la entrada,
perdemos la palanca principal que tenemos sobre la calidad del output.

### 2. Phrase Memory sube en prioridad

**Antes**: paso 5-6, después de MIDI Clock Sync.

**Ahora**: paso 6, antes de Clock Sync. Justificación: importar solos
externos (Bill Evans, Chucho, etc.) como style primers es la palanca
más grande sobre la calidad musical *sin tocar el modelo*. No tiene
sentido posponerlo cuando ya validamos que el primer es decisivo.

### 3. Postprocess baja de prioridad

**Antes**: módulo crítico para compensar phrasing cuadrado del modelo.

**Ahora**: polish, no cuello de botella. El modelo con buen primer ya
entrega groove razonable, síncopa funcional y silencios musicales. Swing
y humanize siguen siendo bueno tener, pero después de tener el sistema
funcionando.

---

## Roadmap actualizado

| # | Paso | Estado |
|---|---|---|
| 1 | `neuraljam/config.py` centralizado | en curso |
| 2 | Estructura de carpetas modular (`midi/`, `models/`, etc.) | pendiente |
| 3 | `midi/phrase_detector.py` con captura fiel | pendiente |
| 4 | Sistema armónico hardcoded (`harmony/`) | pendiente |
| 5 | ImprovRNN integrado al engine (`generation/`, `models/`) | pendiente |
| 6 | Phrase Memory: guardar + importar solos | pendiente |
| 7 | Usar memoria como conditioning | pendiente |
| 8 | MIDI Clock Sync con Studio One | pendiente |
| 9 | Postprocess (swing/humanize/anticipate) | pendiente |
| 10 | MusicVAE / PerformanceRNN / GUI | futuro |

---

## Decisiones cerradas (no retomar)

- ImprovRNN modelo principal del modo `improv`, vía bundle
  `chord_pitches_improv.mag`. MelodyRNN se mantiene como `melody` (fallback
  rápido monofónico sin acordes).
- Disparo de respuesta: por silencio (`SILENCE_TIMEOUT`), salida cuantizada
  al próximo downbeat cuando entre el MIDI Clock Sync (paso 8). No hay
  generación asincrónica/predictiva: la latencia del modelo (0.12-0.75s)
  permite generación síncrona sin problema.
- Un modelo cargado en RAM a la vez. Cambio de modo implica reiniciar el
  script. No hay hot-swap.
- Fine-tuning descartado por falta de GPU.
- Postprocess no es prerequisito del sistema funcional. Entra después.
