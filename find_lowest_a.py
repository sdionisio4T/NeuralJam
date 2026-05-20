"""
find_lowest_a.py — Identifica la nota A más grave de tu Casio.

Tocá notas (en especial los As del rango grave). El script imprime el
número MIDI y el nombre de cada nota, resaltando los A. Cuando
encuentres la A más grave de tu teclado, anotá el número MIDI: ése será
el valor de SIGNAL_NOTE_MIDI en config.

Uso:
    python find_lowest_a.py

Ctrl+C para salir.
"""

import logging
import sys
import time

import mido

from neuraljam.midi.ports import resolve_input_port
from neuraljam import config


# Nombres de las notas (MIDI 0 = C-1 en convención estándar usada por mido)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def midi_to_name(midi: int) -> str:
    """Convierte un número MIDI a notación tipo 'A1', 'C#3', etc."""
    octave = (midi // 12) - 1
    note = NOTE_NAMES[midi % 12]
    return f"{note}{octave}"


def is_a(midi: int) -> bool:
    """True si la nota es un A (cualquier octava)."""
    return midi % 12 == 9


def main():
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    try:
        port_name = resolve_input_port(config.MIDI_INPUT_NAME)
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    port = mido.open_input(port_name)
    print(f"Escuchando en '{port_name}'")
    print()
    print("Tocá notas en el rango grave (las teclas más a la izquierda).")
    print("Los A van a aparecer destacados con [A].")
    print("Cuando toques el A MÁS GRAVE físicamente posible, anotá su MIDI.")
    print("Ctrl+C para salir.")
    print()

    lowest_a_seen = None

    try:
        while True:
            for msg in port.iter_pending():
                if msg.type != "note_on" or msg.velocity == 0:
                    continue

                midi = msg.note
                name = midi_to_name(midi)
                marker = "[A]" if is_a(midi) else "   "

                # Resaltar si es el A más grave visto hasta ahora
                tag = ""
                if is_a(midi):
                    if lowest_a_seen is None or midi < lowest_a_seen:
                        lowest_a_seen = midi
                        tag = "  <-- A más grave visto hasta ahora"

                print(f"  {marker}  MIDI {midi:3d}  =  {name:4s}  "
                      f"vel={msg.velocity:3d}{tag}")

            time.sleep(0.005)

    except KeyboardInterrupt:
        print()
        print("─" * 50)
        if lowest_a_seen is not None:
            print(f"A más grave detectado: MIDI {lowest_a_seen} ({midi_to_name(lowest_a_seen)})")
            print()
            print("Anotá ese número. En el config va a ser:")
            print(f"    SIGNAL_NOTE_MIDI = {lowest_a_seen}")
        else:
            print("No se detectó ningún A en esta sesión.")
        port.close()


if __name__ == "__main__":
    main()
