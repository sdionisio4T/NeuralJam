"""
neuraljam/generation/engine.py

Engine de generación. Mantiene los modelos cargados y orquesta el flujo:

    phrase + model_key
      ↓
    prepare.build_input_sequence(phrase, progression_si_aplica, ...)
      ↓
    NoteSequence input
      ↓
    generator.generate(input, options)
      ↓
    prepare.extract_generated(output, primer_end)
      ↓
    NoteSequence respuesta (rebasada a t=0)

Cambio respecto del refactor original:
- Antes: un solo `model: LoadedModel`. Ahora: `models: Dict[str, LoadedModel]`.
- `respond(phrase, model_key)` permite elegir modelo por turno.
- La progresión se pasa al modelo solo si el perfil dice needs_chords=True.
"""

import logging
import threading
import time
from typing import Dict, List, Optional

from note_seq.protobuf import generator_pb2, music_pb2

from neuraljam.generation.prepare import (
    build_input_sequence,
    get_primer_end,
    extract_generated,
)
from neuraljam.harmony.progression import Progression
from neuraljam.midi.phrase_detector import NoteEvent
from neuraljam.models.loader import LoadedModel


log = logging.getLogger(__name__)


# ===========================================================================
# Engine
# ===========================================================================

class GenerationEngine:
    """
    Engine con múltiples modelos en RAM.

    El caller (entry point) decide qué modelo usar por turno pasando
    `model_key` a respond(). El engine no conoce nada de la "señal" del
    detector — eso es responsabilidad del entry point.

    Si un model_key requerido no está cargado (ej. quisiste 'improv' pero
    solo cargó 'melody'), respond() devuelve None y loguea warning.
    """

    def __init__(
        self,
        models: Dict[str, LoadedModel],
        progression: Optional[Progression] = None,
        model_lock: Optional[threading.Lock] = None,
    ):
        if not models:
            raise ValueError("models no puede estar vacío")
        self.models = models
        self.progression = progression
        self._model_lock = model_lock or threading.Lock()

    # ----------------------------------------------------------------- #
    # API pública
    # ----------------------------------------------------------------- #

    def respond(
        self,
        phrase: List[NoteEvent],
        model_key: str,
        context_seq: Optional[music_pb2.NoteSequence] = None,
        temperature: Optional[float] = None,
        response_bars: Optional[int] = None,
        qpm_override: Optional[float] = None,
    ) -> Optional[music_pb2.NoteSequence]:
        """
        Genera respuesta usando el modelo indicado por model_key.

        Args:
            phrase: lista de NoteEvent (ya sin nota de señal).
            model_key: "melody" | "improv" | "performance" (si están cargados).
            context_seq: NoteSequence histórica para enriquecer el primer.
                         Generada por SubconsciousEngine. None = sin contexto.
            temperature: temperatura dinámica (del Scheduler). Si None, usa
                         el valor fijo del perfil del modelo.
            response_bars: compases de respuesta (del Scheduler). Si None,
                           usa el valor fijo del perfil del modelo.

        Returns:
            NoteSequence rebasado a t=0 con la respuesta, o None si falla.
        """
        if not phrase:
            log.warning("respond() llamado con phrase vacía, devuelvo None")
            return None

        if model_key not in self.models:
            log.warning(
                f"Model key '{model_key}' no está cargado. "
                f"Disponibles: {list(self.models.keys())}"
            )
            return None

        loaded = self.models[model_key]

        try:
            input_seq = self._prepare(phrase, loaded, context_seq, response_bars, qpm_override)
            full_output = self._generate(input_seq, loaded, temperature)
            response = self._extract(full_output, input_seq)
        except Exception:
            log.exception(f"Falló generación con {model_key}, devuelvo None")
            return None

        if not response.notes:
            log.warning(f"{model_key}: generación devolvió 0 notas")
            return None

        log.info(
            f"{model_key}: respuesta generada {len(response.notes)} notas, "
            f"{response.total_time:.2f}s"
        )
        return response

    # ----------------------------------------------------------------- #
    # Pasos internos
    # ----------------------------------------------------------------- #

    def _prepare(
        self,
        phrase: List[NoteEvent],
        loaded: LoadedModel,
        context_seq: Optional[music_pb2.NoteSequence] = None,
        response_bars: Optional[int] = None,
        qpm_override: Optional[float] = None,
    ) -> music_pb2.NoteSequence:
        """Convierte la frase al input que entiende el modelo."""
        prog = self.progression if loaded.profile["needs_chords"] else None
        bars = response_bars if response_bars is not None else int(loaded.profile["response_bars"])

        return build_input_sequence(
            phrase=phrase,
            progression=prog,
            response_bars=bars,
            compress_primer=False,
            steps_per_quarter=loaded.magenta_config.steps_per_quarter,
            context_seq=context_seq,
            qpm_override=qpm_override,
        )

    def _generate(
        self,
        input_seq: music_pb2.NoteSequence,
        loaded: LoadedModel,
        temperature: Optional[float] = None,
    ) -> music_pb2.NoteSequence:
        """Llama al generator y mide latencia."""
        primer_end = get_primer_end(input_seq)
        total_end = input_seq.total_time

        # temperature dinámica del Scheduler prevalece sobre el perfil fijo
        temp = temperature if temperature is not None else float(loaded.profile["temperature"])

        options = generator_pb2.GeneratorOptions()
        options.args["temperature"].float_value = temp
        options.generate_sections.add(
            start_time=primer_end + 0.001,  # epsilon, bug del modelo
            end_time=total_end,
        )

        t0 = time.time()
        with self._model_lock:
            out = loaded.generator.generate(input_seq, options)
        log.debug(
            f"{loaded.family}: primer_end={primer_end:.2f}, "
            f"total_end={total_end:.2f}, temp={temp:.2f}, lat={time.time() - t0:.2f}s"
        )
        return out

    def _extract(
        self,
        full_output: music_pb2.NoteSequence,
        input_seq: music_pb2.NoteSequence,
    ) -> music_pb2.NoteSequence:
        """Filtra notas generadas y rebasa."""
        primer_end = get_primer_end(input_seq)
        return extract_generated(full_output, primer_end)


# ===========================================================================
# Test manual: python -m neuraljam.generation.engine
# ===========================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("tensorflow").setLevel(logging.ERROR)

    from neuraljam import config
    from neuraljam.models import load_all_models

    config.ensure_dirs()

    fake_phrase = [
        NoteEvent(pitch=60, start_time=0.00, duration=0.45, velocity=80),
        NoteEvent(pitch=62, start_time=0.50, duration=0.45, velocity=75),
        NoteEvent(pitch=64, start_time=1.00, duration=0.45, velocity=82),
        NoteEvent(pitch=65, start_time=1.50, duration=0.45, velocity=78),
    ]

    print("=== Cargando TODOS los modelos ===")
    models = load_all_models(do_warmup=True)
    print()

    print("=== Construyendo engine ===")
    progression = Progression.from_config()
    engine = GenerationEngine(models, progression)
    print(f"  Modelos: {list(models.keys())}")
    print(f"  Progresión: {progression!r}")
    print()

    # Probar cada modelo cargado
    for key in models.keys():
        print(f"=== Generando con '{key}' ===")
        t0 = time.time()
        response = engine.respond(fake_phrase, model_key=key)
        elapsed = time.time() - t0
        print(f"  Latencia total: {elapsed:.2f}s")
        if response is None:
            print(f"  FALLO: None")
        else:
            print(f"  Notas: {len(response.notes)}, dur: {response.total_time:.2f}s")
        print()
