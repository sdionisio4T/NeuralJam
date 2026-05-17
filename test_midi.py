import mido

INPUT_NAME = "CASIO USB-MIDI 0"

print(f"Escuchando {INPUT_NAME}. Toca teclas en Casio. Ctrl+C para salir.\n")

with mido.open_input(INPUT_NAME) as port:
    for msg in port:
        print(msg)
