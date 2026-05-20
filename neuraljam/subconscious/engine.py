"""
neuraljam/subconscious/engine.py

Orquesta los procesos de background que enriquecen el primer.

Fase 1: guarda el turno en el banco y elige una frase aleatoria como
        contexto para el próximo turno.
Fase 3: usa ImprovRNN para generar un fragmento chord-aware en background.
        Si ImprovRNN no está disponible o falla, cae al banco (Fase 1).
Fase 4: MusicVAE interpolará entre frases del banco (pendiente).

Seguridad de threads:
  - self._lock protege self._context (leído en el loop principal,
    escrito en el thread de background).
  - model_lock (compartido con GenerationEngine) asegura que solo
    un thread llama a generator.generate() a la vez. Si el lock no
    está disponible (loop principal generando), el subconciente usa
    el banco como fallback. Nunca bloquea el loop principal.
"""

import logging
import threading
from typing import List, Optional

from note_seq.protobuf import generator_pb2, music_pb2

from neuraljam.memory.bank import MemoryBank
from neuraljam.midi.phrase_detector import NoteEvent


log = logging.getLogger(__name__)

# Número de pasos que genera ImprovRNN como contexto (1 bar @ 120 BPM, 4 spp)
_FRAGMENT_STEPS = 16
_FRAGMENT_TEMPERATURE = 0.9


class SubconsciousEngine:
    """
    Motor de contexto de background.

    Ciclo de vida por turno:
        1. MelodyRNN termina de generar → caller llama trigger()
        2. Thread de background corre _run() mientras suena la respuesta
        3. El usuario vuelve a tocar → caller llama get_context()
        4. Devuelve el fragmento listo (o el anterior, o None)
    """

    def __init__(
        self,
        bank: MemoryBank,
        model_lock: Optional[threading.Lock] = None,
    ):
        self.bank = bank
        self._lock = threading.Lock()       # protege self._context
        self._model_lock = model_lock       # compartido con GenerationEngine
        self._context: Optional[music_pb2.NoteSequence] = None
        self._thread: Optional[threading.Thread] = None
        self._improv_model = None           # se setea en neuraljam.py
        self._last_user_seq: Optional[music_pb2.NoteSequence] = None

    # ------------------------------------------------------------------
    # Configuración post-init
    # ------------------------------------------------------------------

    def set_improv_model(self, loaded_model) -> None:
        """Llamar después de load_all_models() si 'improv' cargó."""
        self._improv_model = loaded_model
        log.info("Subconscious: ImprovRNN disponible para generación de contexto")

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def trigger(
        self,
        user_seq: music_pb2.NoteSequence,
        ai_seq: music_pb2.NoteSequence,
    ) -> None:
        """Iniciar actualización de background. No bloquea."""
        self._last_user_seq = user_seq  # guardar antes de arrancar el thread

        if self._thread and self._thread.is_alive():
            log.debug("Subconscious: thread anterior aún corriendo, iniciando nuevo")

        self._thread = threading.Thread(
            target=self._run,
            args=(user_seq, ai_seq),
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
    ) -> None:
        try:
            self.bank.add(user_seq, ai_seq)
            context = self._build_context(user_seq)
            with self._lock:
                self._context = context
            log.debug(
                f"Subconscious listo. Banco: {len(self.bank)} frases. "
                f"Contexto: {'improv' if context and context.notes else 'banco/vacío'}"
            )
        except Exception:
            log.exception("Error en thread de subconciente (no fatal)")

    def _build_context(
        self,
        user_seq: music_pb2.NoteSequence,
    ) -> Optional[music_pb2.NoteSequence]:
        """
        Fase 3: intenta generar con ImprovRNN.
        Fallback: frase aleatoria del banco (Fase 1).
        """
        if self._improv_model is not None and user_seq.notes:
            frag = self._generate_improv_fragment(user_seq)
            if frag is not None:
                return frag

        return self.bank.get_random()

    def _generate_improv_fragment(
        self,
        user_seq: music_pb2.NoteSequence,
    ) -> Optional[music_pb2.NoteSequence]:
        """
        Genera un fragmento corto con ImprovRNN sobre el primer acorde
        de la progresión. No bloquea si el model_lock está tomado.
        """
        model_lock = self._model_lock
        if model_lock is not None and not model_lock.acquire(blocking=False):
            log.debug("Subconscious: model_lock ocupado, usando banco")
            return None

        try:
            return self._call_improv(user_seq)
        except Exception:
            log.exception("Subconscious: fallo en ImprovRNN (no fatal)")
            return None
        finally:
            if model_lock is not None:
                model_lock.release()

    def _call_improv(
        self,
        user_seq: music_pb2.NoteSequence,
    ) -> Optional[music_pb2.NoteSequence]:
        """Llamada real al generador ImprovRNN."""
        from neuraljam import config

        model = self._improv_model
        qpm = config.QPM_FALLBACK
        spp = model.magenta_config.steps_per_quarter
        step_dur = (60.0 / qpm) / spp
        fragment_dur = _FRAGMENT_STEPS * step_dur

        primer_end = user_seq.total_time
        total_end = primer_end + fragment_dur

        # Construir el input: notas del usuario + chord annotation
        input_seq = music_pb2.NoteSequence()
        input_seq.tempos.add(qpm=qpm)
        for n in user_seq.notes:
            new_n = input_seq.notes.add()
            new_n.CopyFrom(n)

        # Usar el primer acorde de la progresión configurada
        first_chord = config.CHORD_PROGRESSION.split()[0]
        ann = input_seq.text_annotations.add()
        ann.text = first_chord
        ann.annotation_type = (
            music_pb2.NoteSequence.TextAnnotation.CHORD_SYMBOL
        )
        ann.time = 0.0
        input_seq.total_time = total_end

        options = generator_pb2.GeneratorOptions()
        options.args["temperature"].float_value = _FRAGMENT_TEMPERATURE
        options.generate_sections.add(
            start_time=primer_end + 0.001,
            end_time=total_end,
        )

        full_output = model.generator.generate(input_seq, options)

        # Extraer solo el fragmento generado (sin el primer)
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


# ------------------------------------------------------------------
# Helper: convertir List[NoteEvent] → NoteSequence
# ------------------------------------------------------------------

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
