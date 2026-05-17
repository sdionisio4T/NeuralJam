import time

import mido

OUTPUT_NAME = "AI-Duet-OUT 2"

print(f"Enviando notas a {OUTPUT_NAME}...")

with mido.open_output(OUTPUT_NAME) as port:
    # Acorde Do mayor arpegio: Do, Mi, Sol
    for note in [60, 64, 67]:
        port.send(mido.Message("note_on", note=note, velocity=80))
        print(f"  Nota {note} ON")
        time.sleep(0.5)
        port.send(mido.Message("note_off", note=note, velocity=0))
        time.sleep(0.1)

print("Listo.")
