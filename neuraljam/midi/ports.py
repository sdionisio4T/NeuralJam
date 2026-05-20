"""
neuraljam/midi/ports.py

Helpers para resolver puertos MIDI por nombre.

Por qué existe este módulo: mido en Windows con python-rtmidi agrega un
sufijo numérico a los nombres de puerto ("CASIO USB-MIDI 0", "CASIO
USB-MIDI 1", etc.). Ese número depende del orden de enumeración de
dispositivos y NO es estable entre sesiones. Hardcodear el nombre
completo en config es frágil.

Solución: el config define el prefijo del nombre ("CASIO USB-MIDI"), y
acá resolvemos al puerto real disponible, sin importar el sufijo.
"""

import logging
from typing import List

import mido

log = logging.getLogger(__name__)


def find_port_by_name(wanted: str, available: List[str]) -> str:
    """
    Resuelve el nombre real de un puerto MIDI a partir de un nombre/prefijo.

    Estrategia:
    1. Match exacto: si `wanted` está literalmente en `available`, se usa.
    2. Match por prefijo: si exactamente un puerto en `available` empieza
       con `wanted`, se usa ese.
    3. Si hay múltiples matches por prefijo, error (ambigüedad real).
    4. Si no hay match, error con la lista de puertos disponibles.

    Args:
        wanted: nombre o prefijo del puerto que se quiere abrir.
        available: lista de puertos visibles (de mido.get_input_names() o
                   mido.get_output_names()).

    Returns:
        El nombre real del puerto, listo para pasar a mido.open_input/output.

    Raises:
        RuntimeError si no hay match único.
    """
    # 1. Match exacto
    if wanted in available:
        return wanted

    # 2. Match por prefijo
    matches = [p for p in available if p.startswith(wanted)]
    if len(matches) == 1:
        log.info(f"Puerto '{wanted}' resolvió a '{matches[0]}' (match por prefijo)")
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Múltiples puertos MIDI matchean '{wanted}': {matches}. "
            f"Ajustá el nombre en config para que sea unívoco."
        )

    raise RuntimeError(
        f"No hay puerto MIDI que matchee '{wanted}'. Disponibles: {available}"
    )


def resolve_input_port(wanted: str) -> str:
    """Conveniencia: find_port_by_name contra los inputs actuales."""
    return find_port_by_name(wanted, mido.get_input_names())


def resolve_output_port(wanted: str) -> str:
    """Conveniencia: find_port_by_name contra los outputs actuales."""
    return find_port_by_name(wanted, mido.get_output_names())
