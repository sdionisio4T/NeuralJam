# NeuralJam — Cosas que necesitan audición para validar

Todo lo de acá está implementado y no crashea. El problema es que
no tiene sentido activarlo sin escuchar si mejora o empeora.
Cada ítem dice qué probar y qué decisión tomar según el resultado.

---

## 1. PerformanceRNN en el loop completo [tecla 3]

**Estado:** bundle descargado, builder implementado, nunca tocado en vivo.

**Cómo probar:**
```
python neuraljam.py
# arranca, modelos cargan
# presioná [3] en la terminal
# tocá una frase
```

**Qué escuchar:**
- ¿Responde con notas? (puede generar silencio o notas muy raras)
- ¿Hay timing extraño? (PerformanceRNN trabaja en steps_per_second, no en compases)
- ¿El volumen es coherente o explota en velocidades extremas?

**Decisión pendiente:**
- Si suena usable → documentar carácter y dejar como opción
- Si suena caótico → ajustar temperatura (bajar de 1.0 a 0.6)
- Si crashea → revisar `_build_performance_generator()` en loader.py

---

## 2. Modelos nuevos: lookback [4], basic [5], polyphony [6]

**Estado:** agregados en esta sesión, se descargan al primer arranque.

**Cómo probar:** igual que PerformanceRNN, presionar la tecla correspondiente.

**Qué escuchar por modelo:**

| Tecla | Qué esperar | Si suena mal |
|-------|-------------|--------------|
| [4] lookback_rnn | Frases con repetición, desarrollo motívico | Bajar temperatura |
| [5] basic_rnn | Respuestas más cortas y directas, menos sofisticado | Subir temperatura |
| [6] polyphony_rnn | Acordes reales, varias notas simultáneas | Puede ser demasiado denso |

**Decisión pendiente:**
- Para cada uno: ¿tiene carácter musical útil para la improvisación?
- ¿Hay alguno que reemplace a attention_rnn como default?
- PolyphonyRNN: ¿el Player lo reproduce bien con notas simultáneas?

---

## 3. Groove Engine → Scheduler

**Estado:** `temperature_delta()` y `bars_hint()` se calculan y loguean en cada turno
pero **no se aplican**. El Scheduler no los recibe todavía.

**Lo que haría cuando se active:**
- Frase densa (muchas notas) → +0.10 de temperatura → respuesta más creativa
- Frase tensa (contraste vs IA anterior) → +0.08 temperatura → más exploración
- Frase muy regular (pulso perfecto) → -0.05 temperatura → más conservador
- Frase densa + tensa → 3 compases mínimo de respuesta

**Cómo probar cuando estés listo:**
1. Verificar en los logs que `temp_delta` y `bars_hint` tienen valores razonables
2. Aplicar en `neuraljam.py` después de calcular `temp` y `bars`:
   ```python
   temp = min(temp + groove.temperature_delta(), current_mode.temp_max or 2.0)
   bars = max(bars, groove.bars_hint()) if groove.bars_hint() > 0 else bars
   ```
3. Comparar con [b] baseline: ¿las respuestas son más coherentes con el groove?

**Decisión pendiente:**
- ¿El sistema se siente más responsivo al ritmo del usuario?
- ¿O agrega variación aleatoria sin sentido musical?

---

## 4. MusicVAE en modos libre/experimental

**Estado:** cargado, activo en modos libre y experimental, nunca validado seriamente.

**Cómo probar:**
```
python neuraljam.py
# presioná [m] dos veces → LIBRE
# tocá 4-5 frases para que el banco acumule material
# escuchá si el primer (contexto) cambia el carácter de las respuestas
```

**Qué escuchar:**
- ¿Las respuestas en LIBRE suenan más "exploradas" que en NORMAL?
- ¿O suenan desconectadas de lo que tocaste?

**Decisión pendiente:**
- Si mejora → mantener, posiblemente activar en más modos
- Si empeora → igual que pasó en normal: banco solo es suficiente

---

## 5. ImprovRNN en background (modos libre/experimental)

**Estado:** `use_improv_background=True` en LIBRE y EXPERIMENTAL. El subconciente
genera un fragmento ImprovRNN con chord detectado por Krumhansl-Kessler y lo
mezcla con el banco. Testeado que no crashea. No testeado musicalmente.

**Cómo probar:**
- Mismo setup que MusicVAE (modo LIBRE, varias frases)
- Comparar con [b] baseline después de cada frase larga

**Qué escuchar:**
- ¿El modelo "responde" de forma más armónica en LIBRE que en NORMAL?
- ¿O el contexto generado por ImprovRNN confunde al modelo principal?

**Decisión pendiente:**
- La misma que se tomó para modo normal: si el banco solo suena mejor, sacar.
- Si suena mejor en LIBRE, documentar por qué funciona ahí y no en normal.

---

## 6. Detección de tonalidad (Krumhansl-Kessler)

**Estado:** `detect_tonality()` corre y devuelve `TonalityResult(tonic, mode, confidence)`.
Nunca se validó si detecta la tonalidad correcta en improvisación jazz real.

**Cómo probar:**
```python
# Agregar temporalmente al loop en neuraljam.py después de groove.update():
from neuraljam.analysis.tonality import detect_tonality
if len(bank.get_all_user()) >= 3:
    t = detect_tonality(bank)
    log.info(f"[TONALIDAD] {t.tonic} {t.mode} conf={t.confidence:.2f} → {t.progression}")
```

**Qué verificar:**
- Si tocás en Do mayor ¿detecta C major?
- Si improvisás en La menor ¿detecta A minor?
- ¿La confianza es alta (>0.7) o baja y cambia mucho?
- ¿La progresión ii-V-I generada es musically correcta?

**Decisión pendiente:**
- Si detecta bien → activar feedback visual en la terminal durante la sesión
- Si falla mucho en jazz (cromatismo, modos) → revisar pesos del perfil o agregar perfiles modales

---

## 7. Modo IMITACIÓN

**Estado:** implementado (temp=0.45 fija, sin memoria, máx 2 compases).
Nunca comparado seriamente contra NORMAL.

**Cómo probar:**
```
python neuraljam.py
# presioná [m] una vez → IMITACIÓN
# tocá una frase simple y repetitiva
```

**Qué escuchar:**
- ¿El modelo imita el ritmo y melodía de lo que tocaste?
- ¿0.45 de temperatura es suficientemente bajo para sentir imitación?
- ¿O es igual que NORMAL a simple vista?

**Decisión pendiente:**
- Si se siente diferente y útil → mantener
- Si no se nota diferencia → bajar temperatura más (0.3) o cambiar `response_bars_max` a 1

---

## 8. Baseline toggle [b] como herramienta de comparación

**Estado:** implementado. [b] desactiva todas las capas. Nunca usado en una sesión real para comparar.

**Protocolo de prueba sugerido:**
1. Tocá 3-4 frases en NORMAL → escuchá las respuestas
2. Presioná [b] → tocá las mismas frases → escuchá las respuestas
3. Presioná [b] de nuevo para volver a NORMAL

**Qué determinar:**
- ¿Las respuestas en NORMAL son mejor que en BASELINE?
- Si no → algo de las capas está perjudicando
- Si sí → qué capa específica aporta más (probar desactivarlas individualmente)

---

## 9. Grabación de sesión — verificar reproducción

**Estado:** el recorder exporta MIDI tipo 1 con 2 pistas. Nunca se abrió en Studio One
para verificar que las pistas estén bien alineadas.

**Cómo verificar:**
1. Correr una sesión corta (3-4 turnos)
2. Ctrl+C → buscar el archivo en `sessions/`
3. Abrir en Studio One → verificar:
   - ¿Las pistas ch0 (usuario) y ch1 (IA) están en sync?
   - ¿El tempo está correcto?
   - ¿Las notas del usuario y la IA no se superponen en tiempo?

---

## Orden sugerido de prueba

1. **[3] PerformanceRNN** — fácil de probar, no requiere sesión larga
2. **[4][5][6] Modelos nuevos** — mismo esfuerzo, comparar carácter
3. **[b] Baseline vs Normal** — referencia para evaluar todo lo demás
4. **Modo IMITACIÓN [m]** — rápido de probar
5. **Modo LIBRE (MusicVAE + ImprovRNN)** — requiere 4-5 frases de calentamiento
6. **Tonalidad** — agregar log temporal, verificar durante improvisación normal
7. **Grabación** — verificar al final de cualquier sesión
8. **Groove → Scheduler** — implementar solo cuando todo lo anterior esté validado
