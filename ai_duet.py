import threading
import time

import mido
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2, music_pb2

# === CONFIGURACIÓN ===
INPUT_NAME = "CASIO USB-MIDI 0"
OUTPUT_NAME = "AI-Duet-OUT 2"
BUNDLE_PATH = "attention_rnn.mag"
CONFIG_ID = "attention_rnn"
SILENCE_TIMEOUT = 2.5  # seg de silencio = fin de frase
TEMPERATURE = 1.0  # creatividad: 0.5=conservador, 1.5=loco
MIN_NOTES_TO_RESPOND = 2  # mínimo notas para gatillar respuesta
QPM = 120  # tempo asumido

# === CARGAR MODELO ===
print("Cargando MelodyRNN attention_rnn...")
bundle = sequence_generator_bundle.read_bundle_file(BUNDLE_PATH)
generator_map = melody_rnn_sequence_generator.get_generator_map()
generator = generator_map[CONFIG_ID](checkpoint=None, bundle=bundle)
generator.initialize()
print("Modelo listo.\n")

# === ESTADO ===
captured_notes = []
last_event_time = None
state_lock = threading.Lock()


# === HELPERS ===
def make_note_sequence(notes_list):
    seq = music_pb2.NoteSequence()
    seq.tempos.add().qpm = QPM
    seq.ticks_per_quarter = 220
    if not notes_list:
        return seq
    t0 = notes_list[0]["start"]
    for n in notes_list:
        note = seq.notes.add()
        note.pitch = n["pitch"]
        note.velocity = max(1, min(127, n["velocity"]))
        note.start_time = n["start"] - t0
        note.end_time = (n["end"] or n["start"] + 0.2) - t0
        note.instrument = 0
        note.program = 0
    seq.total_time = max(n.end_time for n in seq.notes)
    return seq


def generate_response(input_seq, duration_seconds):
    last_end = input_seq.total_time
    gen_options = generator_pb2.GeneratorOptions()
    gen_options.args["temperature"].float_value = TEMPERATURE
    gen_options.generate_sections.add(
        start_time=last_end, end_time=last_end + duration_seconds
    )
    full = generator.generate(input_seq, gen_options)
    response = music_pb2.NoteSequence()
    response.tempos.add().qpm = QPM
    response.ticks_per_quarter = 220
    for n in full.notes:
        if n.start_time >= last_end - 0.01:
            new = response.notes.add()
            new.pitch = n.pitch
            new.velocity = 80
            new.start_time = n.start_time - last_end
            new.end_time = n.end_time - last_end
            new.instrument = 0
    response.total_time = max((n.end_time for n in response.notes), default=0)
    return response


def play_sequence(seq, out_port):
    if not seq.notes:
        return
    events = []
    for n in seq.notes:
        events.append((n.start_time, "on", n.pitch, n.velocity))
        events.append((n.end_time, "off", n.pitch, 0))
    events.sort(key=lambda e: e[0])
    t0 = time.time()
    for ev in events:
        wait = ev[0] - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
        msg_type = "note_on" if ev[1] == "on" else "note_off"
        out_port.send(mido.Message(msg_type, note=ev[2], velocity=ev[3]))


# === MAIN ===
def main():
    input_port = mido.open_input(INPUT_NAME)
    output_port = mido.open_output(OUTPUT_NAME)
    print(f"Input:  {INPUT_NAME}")
    print(f"Output: {OUTPUT_NAME}")
    print(f"Silencio para gatillar: {SILENCE_TIMEOUT}s")
    print("Tocá una frase. Ctrl+C para salir.\n")

    global captured_notes, last_event_time

    try:
        while True:
            for msg in input_port.iter_pending():
                now = time.time()
                if msg.type == "note_on" and msg.velocity > 0:
                    with state_lock:
                        captured_notes.append(
                            {
                                "pitch": msg.note,
                                "start": now,
                                "end": None,
                                "velocity": msg.velocity,
                            }
                        )
                        last_event_time = now
                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    with state_lock:
                        for n in reversed(captured_notes):
                            if n["pitch"] == msg.note and n["end"] is None:
                                n["end"] = now
                                break
                        last_event_time = now

            snapshot = None
            with state_lock:
                any_active = any(n["end"] is None for n in captured_notes)
                if not any_active and last_event_time is not None:
                    if (
                        time.time() - last_event_time > SILENCE_TIMEOUT
                        and len(captured_notes) >= MIN_NOTES_TO_RESPOND
                    ):
                        snapshot = list(captured_notes)
                        captured_notes.clear()
                        last_event_time = None

            if snapshot:
                input_seq = make_note_sequence(snapshot)
                duration = max(input_seq.total_time * 2.0, 6.0)
                print(f"→ Frase: {len(snapshot)} notas, {duration:.1f}s. Generando...")
                t0 = time.time()
                response = generate_response(input_seq, duration)
                print(
                    f"← Respuesta: {len(response.notes)} notas. Inferencia: {time.time() - t0:.1f}s. Tocando..."
                )
                play_sequence(response, output_port)
                print("  Listo. Tocá otra frase.\n")

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nSaliendo.")
    finally:
        input_port.close()
        output_port.close()


if __name__ == "__main__":
    main()
