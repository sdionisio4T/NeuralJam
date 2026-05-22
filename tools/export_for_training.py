"""
tools/export_for_training.py

Prepara las frases guardadas para fine-tuning en Google Colab.

Qué hace:
  1. Escanea saved_phrases/ y colecta todos los MIDIs del usuario
  2. Los copia a training_export/YYYY-MM-DD_HHMMSS/
  3. Crea un zip listo para subir a Google Drive
  4. Imprime estadísticas y las próximas instrucciones

Uso:
  python tools/export_for_training.py
  python tools/export_for_training.py --include-ai     # incluir respuestas IA también
  python tools/export_for_training.py --since 2026-05-21  # solo desde esa fecha
  python tools/export_for_training.py --min-notes 4    # filtrar frases muy cortas

Las frases se exportan como MIDIs individuales — el formato que
melody_rnn_create_dataset espera para construir el dataset de entrenamiento.

Referencia de cantidad:
  30  frases → cambio sutil de estilo
  150 frases → tonalidad más estable
  300 frases → estilo claramente tuyo
"""

import argparse
import shutil
import sys
import zipfile
from datetime import datetime, date
from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAVED_DIR = PROJECT_ROOT / "saved_phrases"
EXPORT_DIR = PROJECT_ROOT / "training_export"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exporta frases guardadas para fine-tuning en Colab."
    )
    parser.add_argument(
        "--include-ai",
        action="store_true",
        help="Incluir también las respuestas de la IA (default: solo usuario).",
    )
    parser.add_argument(
        "--since",
        metavar="YYYY-MM-DD",
        default=None,
        help="Solo incluir frases desde esta fecha.",
    )
    parser.add_argument(
        "--min-notes",
        type=int,
        default=2,
        help="Filtrar frases con menos de N notas (default: 2).",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="No crear el archivo zip (solo copiar los MIDIs).",
    )
    return parser.parse_args()


def collect_midis(include_ai: bool, since: date | None) -> list[Path]:
    """Escanea saved_phrases/ y devuelve los MIDIs que aplican."""
    if not SAVED_DIR.exists():
        print(f"ERROR: no existe {SAVED_DIR}")
        print("Guardá frases primero con la tecla [s] durante una sesión.")
        sys.exit(1)

    midis = []
    for date_folder in sorted(SAVED_DIR.iterdir()):
        if not date_folder.is_dir():
            continue

        # Filtrar por fecha si se pidió
        if since is not None:
            try:
                folder_date = date.fromisoformat(date_folder.name)
                if folder_date < since:
                    continue
            except ValueError:
                pass

        for midi in sorted(date_folder.glob("*.mid")):
            is_user = midi.name.startswith("user_")
            is_ai = midi.name.startswith("ai_")

            if is_user or (include_ai and is_ai):
                midis.append(midi)

    return midis


def count_notes(midi_path: Path) -> int:
    """Cuenta notas en un MIDI sin depender de note_seq."""
    try:
        import mido
        mid = mido.MidiFile(str(midi_path))
        return sum(
            1 for track in mid.tracks
            for msg in track
            if msg.type == "note_on" and msg.velocity > 0
        )
    except Exception:
        return 999  # si no se puede leer, incluir igual


def main():
    args = parse_args()

    since = None
    if args.since:
        try:
            since = date.fromisoformat(args.since)
        except ValueError:
            print(f"ERROR: formato de fecha inválido: {args.since!r} (usar YYYY-MM-DD)")
            sys.exit(1)

    print("Escaneando saved_phrases/ ...")
    all_midis = collect_midis(args.include_ai, since)

    if not all_midis:
        print("No hay frases que exportar.")
        print(f"Buscado en: {SAVED_DIR}")
        sys.exit(0)

    # Filtrar por notas mínimas
    filtered = []
    skipped = 0
    for midi in all_midis:
        n = count_notes(midi)
        if n >= args.min_notes:
            filtered.append(midi)
        else:
            skipped += 1

    if not filtered:
        print(f"Todas las frases tienen menos de {args.min_notes} notas. Bajá --min-notes.")
        sys.exit(0)

    # Crear carpeta de exportación
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_dir = EXPORT_DIR / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # Copiar MIDIs
    for midi in filtered:
        shutil.copy2(midi, out_dir / midi.name)

    # Crear zip
    zip_path = None
    if not args.no_zip:
        zip_path = EXPORT_DIR / f"neuraljam_training_{ts}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for midi in filtered:
                zf.write(out_dir / midi.name, midi.name)

    # Resumen
    user_count = sum(1 for m in filtered if m.name.startswith("user_"))
    ai_count = len(filtered) - user_count

    print()
    print("=" * 55)
    print(f"  Frases exportadas:  {len(filtered)}")
    print(f"    - usuario:        {user_count}")
    if ai_count:
        print(f"    - IA:             {ai_count}")
    if skipped:
        print(f"  Filtradas (cortas): {skipped}")
    print(f"  Carpeta:            {out_dir}")
    if zip_path:
        size_mb = zip_path.stat().st_size / (1024 * 1024)
        print(f"  ZIP:                {zip_path.name}  ({size_mb:.1f} MB)")
    print("=" * 55)

    _print_next_steps(user_count, zip_path)


def _print_next_steps(count: int, zip_path: Path | None):
    if count < 30:
        needed = 30 - count
        print(f"\n  Tenés {count} frases — necesitás {needed} más para un cambio sutil.")
        print("  Seguí tocando y presionando [s] en los momentos buenos.")
        return

    level = "sutil" if count < 150 else ("estable" if count < 300 else "tu estilo")
    print(f"\n  Con {count} frases → resultado esperado: {level}")
    print()
    print("  Próximos pasos:")
    if zip_path:
        print(f"  1. Subí {zip_path.name} a Google Drive")
    else:
        print(f"  1. Subí la carpeta training_export/ a Google Drive")
    print("  2. Abrí colab/finetune_neuraljam.ipynb en Colab")
    print("  3. Conectá tu Drive y corré todas las celdas (~20-40 min, GPU T4 gratis)")
    print("  4. Bajá el nuevo .mag generado")
    print("  5. En config.py cambiá model_path al nuevo archivo")
    print("  6. La próxima sesión MelodyRNN ya habló con vos.")


if __name__ == "__main__":
    main()
