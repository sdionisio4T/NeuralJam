"""
neuraljam/subconscious/engine.py

Orquesta los procesos de background que enriquecen el primer.

Fase 1: guarda el turno en el banco y elige una frase del usuario como
        contexto para el próximo turno.
Fase 3: ImprovRNN se reserva SOLO como modelo de RESPUESTA directa
        (señal grave). No genera contexto — su chord conditioning fijo
        choca con el fraseo libre del usuario.
Fase 4: MusicVAE interpola en espacio latente entre la frase actual del
        usuario y una frase del banco → context_seq enriquecido para
        MelodyRNN. Si MusicVAE falla o no está disponible, vuelve al
        banco directamente (fallback seguro).

PRINCIPIO FUNDAMENTAL:
    MelodyRNN SIEMPRE es quien responde.
    MusicVAE solo enriquece el context_seq que MelodyRNN recibe como primer.

Flujo por turno:
    frase_usuario  ─encode─┐
                           ├─ interpolate(alpha=0.5) ─decode─► context_seq ─► MelodyRNN
    frase_banco    ─encode─┘

Seguridad de threads:
  - self._lock protege self._context y self._music_vae
  - MusicVAE tiene su propia sesión TF independiente de MelodyRNN/ImprovRNN,
    por lo que no necesita el model_lock compartido
  - El background thread nunca bloquea el loop principal
"""

import logging
import threading
from typing import List, Optional

from note_seq.protobuf import generator_pb2, music_pb2

from neuraljam.memory.bank import MemoryBank
from neuraljam.midi.phrase_detector import NoteEvent


log = logging.getLogger(__name__)

# Configuración interna de ImprovRNN como generador de contexto
# (mantenido pero DESACTIVADO — ver docstring de _build_context)
_FRAGMENT_STEPS = 16
_FRAGMENT_TEMPERATURE = 0.9

# Parámetros de MusicVAE
_VAE_STEPS = 32          # 2 compases × 16 steps/bar (steps_per_quarter=4, 4/4)
_VAE_TEMPERATURE = 0.5   # baja temperatura al decodificar: contexto coherente, no aleatorio
_VAE_ALPHA = 0.5         # punto de interpolación: 0=frase_usuario pura, 1=frase_banco pura


class SubconsciousEngine:
    """
    Motor de contexto de background.

    Ciclo de vida por turno:
        1. MelodyRNN termina de generar → caller llama trigger()
        2. Thread de background corre _run() mientras suena la respuesta
        3. El usuario vuelve a tocar → caller llama get_context()
        4. Devuelve el context_seq listo (o el anterior, o None)

    Con MusicVAE activo, el context_seq es una melodía interpolada entre
    la frase actual del usuario y la última frase guardada en el banco.
    Sin MusicVAE, es la última frase del usuario directamente.
    """

    def __init__(
        self,
        bank: MemoryBank,
        model_lock: Optional[threading.Lock] = None,
    ):
        self.bank = bank
        self._lock = threading.Lock()       # protege _context y _music_vae
        self._model_lock = model_lock       # para ImprovRNN si se reactivara
        self._context: Optional[music_pb2.NoteSequence] = None
        self._thread: Optional[threading.Thread] = None
        self._improv_model = None           # reservado para señal grave, no para contexto
        self._music_vae = None              # set via set_music_vae()
        self._last_user_seq: Optional[music_pb2.NoteSequence] = None

    # ------------------------------------------------------------------
    # Configuración post-init
    # ------------------------------------------------------------------

    def set_improv_model(self, loaded_model) -> None:
        """
        Almacena referencia al ImprovRNN ya cargado.
        NOTA: ImprovRNN se reserva para respuesta directa (señal grave),
        no para generación de contexto.
        """
        self._improv_model = loaded_model
        log.info("SubconsciousEngine: ImprovRNN registrado (solo para señal grave).")

    def set_music_vae(self, model) -> None:
        """
        Conecta el MusicVAE ya cargado para interpolación de contexto.
        MelodyRNN SIEMPRE responde — MusicVAE solo enriquece el context_seq.
        """
        with self._lock:
            self._music_vae = model
        log.info(
            "SubconsciousEngine: MusicVAE conectado. "
            "Contexto = interpolacion latente entre frase actual y banco."
        )

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def trigger(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: music_pb2.NoteSequence,
        mode=None,
    ) -> None:
        """
        Iniciar actualización de background. No bloquea.

        Si el modo no usa memoria (ej. imitación), limpia el contexto
        y no lanza el thread — así el próximo turno arranca sin contexto.
        """
        self._last_user_seq = user_seq

        if mode is not None and not mode.use_memory:
            with self._lock:
                self._context = None
            log.debug("SubconsciousEngine: modo sin memoria, contexto limpiado.")
            return

        if self._thread and self._thread.is_alive():
            log.debug("SubconsciousEngine: thread anterior aun corriendo, saltando turno.")
            return  # no acumular threads: el contexto anterior sigue valido

        self._thread = threading.Thread(
            target=self._run,
            args=(user_seq, ai_seq, mode),
            name="Subconscious",
            daemon=True,
        )
        self._thread.start()

    def get_context(self) -> Optional[music_pb2.NoteSequence]:
        """Devuelve el mejor contexto disponible. Nunca bloquea."""
        with self._lock:
            return self._context

    def is_ready(self) -> bool:
        return self._context is not None

    # ------------------------------------------------------------------
    # Background
    # ------------------------------------------------------------------

    def _run(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: music_pb2.NoteSequence,
        mode=None,
    ) -> None:
        try:
            self.bank.add(user_seq, ai_seq)
            context = self._build_context(user_seq, mode)
            with self._lock:
                self._context = context

            n_notas = len(context.notes) if context else 0
            log.debug(
                f"Subconscious listo. Banco: {len(self.bank)} frases. "
                f"Contexto: {n_notas} notas."
            )
        except Exception:
            log.exception("Error en thread de subconciente (no fatal)")

    def _build_context(
        self,
        user_seq: music_pb2.NoteSequence,
        mode=None,
    ) -> Optional[music_pb2.NoteSequence]:
        """
        Construye el context_seq para MelodyRNN según el modo activo.

        normal:       ImprovRNN (chord del pitch histogram) + fallback banco
        libre/exp:    MusicVAE + ImprovRNN concatenados + fallback banco
        imitación:    None (sin contexto — se maneja en trigger() antes de llegar acá)

        Si ImprovRNN y MusicVAE están disponibles y el modo los pide,
        se concatenan (improv primero) y prepare.py recorta al presupuesto.
        """
        use_vae = (mode is None or mode.use_music_vae)
        use_improv = (mode is not None and mode.use_improv_background)

        with self._lock:
            music_vae = self._music_vae if use_vae else None

        ref_seq = self.bank.get_last_user() or self.bank.get_random()

        # --- Capa ImprovRNN (chord dinámico) ---
        improv_ctx = None
        if use_improv:
            improv_ctx = self._generate_improv_fragment(user_seq)

        # --- Capa MusicVAE (interpolación latente) ---
        vae_ctx = None
        if music_vae is not None and ref_seq is not None and user_seq.notes:
            vae_ctx = self._interpolate_with_music_vae(music_vae, user_seq, ref_seq)

        # --- Combinar y devolver ---
        if improv_ctx is not None and vae_ctx is not None:
            combined = _concat_seqs(improv_ctx, vae_ctx)
            log.debug(
                f"Subconscious: ImprovRNN ({len(improv_ctx.notes)} notas) "
                f"+ MusicVAE ({len(vae_ctx.notes)} notas) combinados"
            )
            return combined

        if improv_ctx is not None:
            return improv_ctx

        if vae_ctx is not None:
            return vae_ctx

        # Fallback: última frase del banco directamente
        return ref_seq

    def _interpolate_with_music_vae(
        self,
        model,
        seq_a: music_pb2.NoteSequence,
        seq_b: music_pb2.NoteSequence,
        alpha: float = _VAE_ALPHA,
    ) -> Optional[music_pb2.NoteSequence]:
        """
        Interpola entre seq_a (frase del usuario) y seq_b (frase del banco)
        en el espacio latente de MusicVAE.

        alpha=0.0 → contexto = frase del usuario pura
        alpha=0.5 → contexto = mezcla exacta (default)
        alpha=1.0 → contexto = frase del banco pura

        MusicVAE tiene su propia sesión TF independiente — no necesita
        el model_lock compartido con MelodyRNN/ImprovRNN.
        """
        import numpy as np

        try:
            qpm = seq_a.tempos[0].qpm if seq_a.tempos else 120.0
            # MusicVAE requiere melodia monofonica estricta (sin solapamientos).
            # Si hay notas superpuestas el encoder lanza NoExtractedExamplesError
            # pero igual gasta CPU en la conversion de datos.
            a_padded = _pad_to_two_bars(_monophonize(seq_a), qpm)
            b_padded = _pad_to_two_bars(_monophonize(seq_b), qpm)

            # Minimo de notas para que el encode tenga sentido
            if len(a_padded.notes) < 2 or len(b_padded.notes) < 2:
                log.debug("MusicVAE: secuencias muy cortas para interpolar, fallback.")
                return None

            # Encode: NoteSequence → espacio latente
            # mu es la media del espacio latente (más estable que z muestreado)
            _, mu, _ = model.encode([a_padded, b_padded])

            if mu is None or mu.shape[0] < 2:
                log.debug("MusicVAE: encode devolvio menos de 2 vectores, fallback.")
                return None

            # Interpolate: punto entre los dos vectores
            z_interp = mu[0:1] + alpha * (mu[1:2] - mu[0:1])

            # Decode: vector latente → NoteSequence
            results = model.decode(
                length=_VAE_STEPS,
                z=z_interp,
                temperature=_VAE_TEMPERATURE,
            )

            if not results or not results[0].notes:
                log.debug("MusicVAE: decode devolvio secuencia vacia, fallback.")
                return None

            log.debug(
                f"MusicVAE: interpolacion exitosa "
                f"({len(results[0].notes)} notas, alpha={alpha:.2f})"
            )
            return results[0]

        except Exception:
            log.warning("MusicVAE: interpolacion fallo (no fatal, usando fallback).", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # ImprovRNN — generador de contexto con chord dinámico
    # ------------------------------------------------------------------

    def _generate_improv_fragment(
        self,
        user_seq: music_pb2.NoteSequence,
    ) -> Optional[music_pb2.NoteSequence]:
        """
        Genera un fragmento ImprovRNN usando el acorde derivado del pitch
        histogram de la frase del usuario — sin progresión fija.

        No bloquea: si el model_lock está ocupado por la generación principal,
        se descarta y el contexto anterior sigue válido.
        """
        if self._improv_model is None:
            return None

        model_lock = self._model_lock
        if model_lock is not None and not model_lock.acquire(blocking=False):
            log.debug("Subconscious: model_lock ocupado, saltando ImprovRNN")
            return None

        try:
            # Con ≥3 frases del usuario usar detección de tonalidad real;
            # si hay pocas frases, derivar del pitch histogram de la frase actual.
            if len(self.bank.user_phrases) >= 3:
                from neuraljam.analysis.tonality import detect_tonality
                tonality = detect_tonality(self.bank)
                chord = tonality.tonic_chord
                log.debug(
                    f"Subconscious ImprovRNN: tonalidad detectada = {tonality}"
                )
            else:
                chord = _chord_from_phrase(user_seq)
                log.debug(f"Subconscious ImprovRNN: chord de pitch histogram = {chord!r}")
            return self._call_improv(user_seq, chord)
        except Exception:
            log.warning("Subconscious: ImprovRNN falló (no fatal)", exc_info=True)
            return None
        finally:
            if model_lock is not None:
                model_lock.release()

    def _call_improv(
        self,
        user_seq: music_pb2.NoteSequence,
        chord: str,
    ) -> Optional[music_pb2.NoteSequence]:
        """
        Llamada al generador ImprovRNN con un acorde específico.

        El acorde se repite en cada compás — es un punto de gravedad
        armónica, no una progresión. Evita el caos que causaba la
        progresión fija Dm7-G7-Cmaj7 sobre fraseo libre.
        """
        from neuraljam import config

        model = self._improv_model
        qpm = config.QPM_FALLBACK
        spp = model.magenta_config.steps_per_quarter
        step_dur = (60.0 / qpm) / spp
        fragment_dur = _FRAGMENT_STEPS * step_dur

        primer_end = user_seq.total_time
        total_end = primer_end + fragment_dur

        input_seq = music_pb2.NoteSequence()
        input_seq.tempos.add(qpm=qpm)
        for n in user_seq.notes:
            new_n = input_seq.notes.add()
            new_n.CopyFrom(n)

        bar_dur = (60.0 / qpm) * 4.0
        t = 0.0
        while t < total_end:
            ann = input_seq.text_annotations.add()
            ann.text = chord
            ann.annotation_type = (
                music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
            )
            ann.time = t
            t += bar_dur
        input_seq.total_time = total_end

        options = generator_pb2.GeneratorOptions()
        options.args["temperature"].float_value = _FRAGMENT_TEMPERATURE
        options.generate_sections.add(
            start_time=primer_end + 0.001,
            end_time=total_end,
        )

        full_output = model.generator.generate(input_seq, options)

        frag_notes = [
            n for n in full_output.notes
            if n.start_time >= primer_end - 0.01
        ]
        if not frag_notes:
            return None

        frag = music_pb2.NoteSequence()
        frag.tempos.add(qpm=qpm)
        t0 = min(n.start_time for n in frag_notes)
        for n in frag_notes:
            new_n = frag.notes.add()
            new_n.CopyFrom(n)
            new_n.start_time = n.start_time - t0
            new_n.end_time = n.end_time - t0
        frag.total_time = max(n.end_time for n in frag.notes)
        return frag


# ===========================================================================
# Helpers
# ===========================================================================

def _chord_from_phrase(seq: music_pb2.NoteSequence) -> str:
    """
    Deriva el acorde más probable de una frase por pitch class histogram.

    Devuelve el nombre de la nota más frecuente como raíz sin extensiones
    (ej: "C", "G", "A"). Más neutral que una progresión fija — ImprovRNN
    recibe un punto de gravedad armónica consistente con lo que el usuario
    acaba de tocar.
    """
    if not seq.notes:
        return "C"
    from collections import Counter
    _PC_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
    pcs = [n.pitch % 12 for n in seq.notes]
    root_pc = Counter(pcs).most_common(1)[0][0]
    return _PC_NAMES[root_pc]


def _concat_seqs(
    seq_a: music_pb2.NoteSequence,
    seq_b: music_pb2.NoteSequence,
) -> music_pb2.NoteSequence:
    """
    Concatena seq_b inmediatamente después de seq_a.

    Se usa para combinar el fragmento ImprovRNN con el contexto MusicVAE
    en los modos libre y experimental. prepare.py recortará al presupuesto
    de notas disponible (toma las más recientes).
    """
    out = music_pb2.NoteSequence()
    qpm = seq_a.tempos[0].qpm if seq_a.tempos else 120.0
    out.tempos.add(qpm=qpm)

    offset = seq_a.total_time
    if offset == 0.0 and seq_a.notes:
        offset = max(n.end_time for n in seq_a.notes)

    for n in seq_a.notes:
        new_n = out.notes.add()
        new_n.CopyFrom(n)

    for n in seq_b.notes:
        new_n = out.notes.add()
        new_n.CopyFrom(n)
        new_n.start_time = n.start_time + offset
        new_n.end_time = n.end_time + offset

    if out.notes:
        out.total_time = max(n.end_time for n in out.notes)
    return out


def _monophonize(seq: music_pb2.NoteSequence) -> music_pb2.NoteSequence:
    """
    Convierte a melodia monofonica: si una nota solapa con la siguiente,
    la recorta para que termine justo cuando la siguiente empieza.

    MusicVAE (cat-mel_2bar_big) lanza NoExtractedExamplesError con notas
    superpuestas. El piano real siempre produce leves solapamientos entre
    notas consecutivas — hay que limpiarlos antes de encodear.
    """
    if not seq.notes:
        return seq

    out = music_pb2.NoteSequence()
    for t in seq.tempos:
        out.tempos.add(qpm=t.qpm)
    out.total_time = seq.total_time

    notes = sorted(seq.notes, key=lambda n: n.start_time)
    for i, n in enumerate(notes):
        end = n.end_time
        if i + 1 < len(notes):
            end = min(end, notes[i + 1].start_time)
        if end <= n.start_time:
            continue  # nota de duracion cero tras recorte, descartar
        new_n = out.notes.add()
        new_n.pitch = n.pitch
        new_n.start_time = n.start_time
        new_n.end_time = end
        new_n.velocity = n.velocity if n.velocity > 0 else 80
        new_n.instrument = n.instrument
        new_n.program = n.program

    return out


def _pad_to_two_bars(
    seq: music_pb2.NoteSequence,
    qpm: float,
) -> music_pb2.NoteSequence:
    """
    Asegura que la NoteSequence tenga al menos 2 compases de duración.

    MusicVAE (cat-mel_2bar_big) necesita NoteSequences de exactamente
    2 compases para encode. Si la frase es más corta, el data converter
    no encontraría slices válidos y encode devolvería vectores vacíos.
    """
    two_bar_dur = (60.0 / qpm) * 8.0   # 2 bars × 4 beats
    out = music_pb2.NoteSequence()
    out.CopyFrom(seq)
    if out.total_time < two_bar_dur:
        out.total_time = two_bar_dur
    return out


def phrase_to_seq(
    events: List[NoteEvent],
    qpm: float,
) -> music_pb2.NoteSequence:
    """Convierte una frase del detector en NoteSequence para el banco."""
    seq = music_pb2.NoteSequence()
    seq.tempos.add(qpm=qpm)
    for evt in events:
        n = seq.notes.add()
        n.pitch = evt.pitch
        n.start_time = evt.start_time
        n.end_time = evt.start_time + evt.duration
        n.velocity = evt.velocity if evt.velocity > 0 else 80
        n.instrument = 0
        n.program = 0
    if events:
        seq.total_time = max(e.start_time + e.duration for e in events)
    return seq
