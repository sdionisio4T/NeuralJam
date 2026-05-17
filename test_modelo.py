from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle

print("Cargando bundle attention_rnn.mag...")
bundle = sequence_generator_bundle.read_bundle_file("attention_rnn.mag")

print("Construyendo generador...")
generator_map = melody_rnn_sequence_generator.get_generator_map()
generator = generator_map["attention_rnn"](checkpoint=None, bundle=bundle)
generator.initialize()

print("Modelo cargado y listo ✓")
