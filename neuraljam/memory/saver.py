"""
neuraljam/memory/saver.py

Guarda frases marcadas como interesantes durante la sesión.

Flujo completo:
    Sesión en vivo → presionás 's' → se guardan user + ai como MIDI
    → saved_phrases/YYYY-MM-DD/user_HHMMSS.mid
                               ai_HHMMSS.mid
    → subís esa carpeta a Google Drive
    → Colab hace fine-tuning desde attention_rnn.mag
    → bajás el nuevo .mag
    → en config.py cambiás model_path al nuevo checkpoint
    → próxima sesión MelodyRNN ya aprendió tu estilo
"""

import datetime
import logging
from pathlib import Path
from typing import Optional

from note_seq.protobuf import music_pb2


log = logging.getLogger(__name__)

_DEFAULT_FOLDER = "saved_phrases"


def save_phrase(
    seq: music_pb2.NoteSequence,
    tag: str = "phrase",
    folder: str = _DEFAULT_FOLDER,
) -> Optional[str]:
    """
    Guarda una NoteSequence como MIDI en saved_phrases/YYYY-MM-DD/.

    Args:
        seq:    NoteSequence a guardar.
        tag:    prefijo del archivo ('user' o 'ai').
        folder: carpeta raíz (default 'saved_phrases').

    Returns:
        Path del archivo guardado, o None si falló.
    """
    try:
        from note_seq import sequence_proto_to_midi_file
    except ImportError:
        log.error("note_seq no disponible para guardar frases")
        return None

    if not seq or not seq.notes:
        log.debug("save_phrase: secuencia vacía, ignorando")
        return None

    try:
        date_folder = Path(folder) / datetime.date.today().isoformat()
        date_folder.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%H%M%S%f")
        path = date_folder / f"{tag}_{ts}.mid"
        sequence_proto_to_midi_file(seq, str(path))
        log.info(f"Guardado: {path}")
        return str(path)
    except Exception:
        log.exception("Error guardando frase MIDI")
        return None
