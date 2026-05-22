# NeuralJam — Qué falta y qué mejorar

---

## Lo que falta implementar

### Groove → Scheduler (pendiente de audición, código listo)
El Groove Engine calcula `temperature_delta` y `bars_hint` en cada turno pero
el Scheduler no los recibe. Es el cambio más pequeño con el mayor potencial:
el sistema respondería al groove real del usuario en vez de solo a la densidad
de notas. Dos líneas de código en `neuraljam.py`, decisión bloqueada por audición.

### Feedback visual en terminal
Ahora la terminal muestra logs crudos. Podría mostrar una línea de estado limpia:
```
[NORMAL] [attention_rnn] QPM=127  Tonalidad: Am  Groove: denso/sincopado
Turno 7 | temp=0.82 | bars=3 | gen=0.4s
```
Útil para entender qué está pensando el sistema sin abrir el log.

### Stats al final de sesión
Al Ctrl+C se exporta el MIDI pero no hay resumen. Podría imprimir:
- Cuántos turnos, cuántos silencios intencionales
- Temperatura promedio y distribución
- Tonalidad más frecuente detectada
- Cuántas frases guardadas con [s]

### Compatibilidad con `--preload` y fine-tuned models
El `--preload` ya existe para cargar MIDIs externos al banco. Falta documentar
el flujo completo: fine-tune en Colab → bajar .mag → agregar a PROFILES →
`--preload` con material del mismo estilo → sesión con todo integrado.

---

## Lo que sugiero mejorar

### 1. MemoryBank ponderado por similitud rítmica (alto impacto)

Ahora el banco devuelve frases al azar entre las últimas 8.
El Groove Engine ya calcula `pulse_regularity` y `density` de cada frase.
Si usamos eso para elegir del banco la frase más parecida rítmicamente
a la actual, el contexto que recibe el modelo es mucho más coherente.

```
usuario toca frase sincopada
→ banco busca la frase más sincopada de las 8
→ ese es el primer que recibe MelodyRNN
→ respuesta más conectada al groove real
```

Requiere guardar el `RhythmProfile` junto a cada frase en el banco.

---

### 2. Dinámica: velocidad MIDI del usuario → AI

Ahora la humanización aplica velocity_variance fija (±12). El AI no sabe
si tocaste piano o forte.

Propuesta: calcular la velocidad promedio de la frase del usuario y usarla
como base para la humanización de la respuesta.

```python
user_avg_vel = mean(n.velocity for n in phrase.notes)
humanize(response, velocity_base=user_avg_vel, variance=12)
```

Muy simple, alto impacto musical: si el usuario toca suave, la IA responde suave.

---

### 3. Modo DIALOGO — alternancia estricta (nuevo modo)

Los modos actuales son sobre temperatura y capas. Falta un modo sobre
la estructura de la conversación.

DIÁLOGO: el sistema garantiza que su respuesta tiene exactamente la misma
longitud que tu frase (en compases). Sin silencio probabilístico.
Simula un contrapunto de call-and-response real.

```python
"dialogue": ModeConfig(
    name="dialogue",
    display="DIÁLOGO",
    temp_min=0.5, temp_max=0.8,
    response_bars_max=4,
    use_memory=True,
    use_music_vae=False,
    use_improv_background=False,
    always_respond=True,
    match_user_bars=True,   # campo nuevo
)
```

Requiere agregar `match_user_bars` a `ModeConfig` y una rama en `response_bars()`.

---

### 4. Feedback de tonalidad visible + acorde sugerido

Cuando `detect_tonality()` tiene confianza >0.7, mostrar en terminal:
```
[TONALIDAD] Am  conf=0.84  →  Am7 Dm7 E7 Am7
```

Útil para el músico: sabe qué detectó el sistema, puede contradecirlo
intencionalmente o acompañarlo. También le da contexto sobre cuándo
el ImprovRNN background va a tener un chord bueno vs uno al azar.

---

### 5. Cooldown después de respuesta larga

Si el AI respondió 5 compases (modo LIBRE), debería haber más probabilidad
de silencio en el turno siguiente. Ahora el scheduler no sabe cuánto acabó
de tocar la IA.

```python
# En el loop, después de player.play():
scheduler.notify_ai_bars(bars)   # nuevo método

# En Scheduler:
def notify_ai_bars(self, bars: int):
    if bars >= 4:
        self._cooldown = 1   # forzar al menos un silencio
```

Evita que el sistema se ponga a hablar solo durante varios turnos seguidos.

---

### 6. Detección de señal musical (no solo notas graves)

Ahora el cambio de modelo se activa por notas en la zona A0–E1 (muy grave).
Eso funciona pero es tosco — tenés que bajar una octava intencionalmente.

Alternativa: detectar una pausa larga (>4s) como señal de "modo siguiente".
O dos frases seguidas muy cortas (1-2 notas) como "señal de cambio".
Sería más musical que pisar teclas en el bajo.

---

### 7. Temperatura adaptativa por modelo

Ahora todos los modelos usan la misma lógica de temperatura.
`basic_rnn` sin atención probablemente necesita temperatura más baja que
`attention_rnn` para no descontrolarse. `polyphony_rnn` genera acordes,
la semántica de temperatura es distinta.

Propuesta: temperatura base por modelo en PROFILES, además de los límites del modo.

```python
"basic": {
    ...
    "base_temperature": 0.7,   # más conservador por diseño
},
"polyphony": {
    ...
    "base_temperature": 0.9,   # más alta, el modelo es inherentemente denso
},
```

---

### 8. Fine-tuning: necesitás más frases

Con 16 frases guardadas el fine-tuning no va a cambiar el modelo de forma audible.
El mínimo útil es 30, el cambio claro está en 150.

El workflow más eficiente es:
- Tocar sesiones normales con `[s]` frecuente (cada frase que suene bien)
- No esperar a "la frase perfecta" — cantidad importa más que calidad para el entrenamiento
- Exportar y acumular en Drive, no entrenar hasta tener ≥50 frases

---

## Resumen de prioridades

| # | Qué | Esfuerzo | Impacto | Bloqueo |
|---|-----|----------|---------|---------|
| 1 | Groove → Scheduler | muy bajo | alto | audición |
| 2 | Velocidad usuario → AI | bajo | alto | — |
| 3 | Feedback visual terminal | bajo | medio | — |
| 4 | MemoryBank ponderado por groove | medio | alto | audición |
| 5 | Stats al cierre | bajo | bajo | — |
| 6 | Cooldown post-respuesta larga | bajo | medio | audición |
| 7 | Tonalidad visible en terminal | bajo | medio | — |
| 8 | Modo DIÁLOGO | medio | alto | — |
| 9 | Temperatura base por modelo | bajo | medio | audición |
| 10 | Fine-tuning con más frases | — | muy alto | tiempo tocando |
