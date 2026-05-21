"""
neuraljam/models/music_vae_loader.py

Descarga, extrae y carga MusicVAE (cat-mel_2bar_big).

ROL EN EL SISTEMA:
    MusicVAE NO responde directamente. Su único rol es enriquecer el
    context_seq que recibe MelodyRNN como primer extendido:

        frase_usuario  ─encode─┐
                               ├─ interpolate(alpha=0.5) ─decode─► context_seq ─► MelodyRNN ─► respuesta
        frase_banco    ─encode─┘

    MelodyRNN SIEMPRE es quien responde. MusicVAE es un generador de contexto.

MODELO:
    cat-mel_2bar_big (~26 MB)
    - Ventana: 2 compases, steps_per_quarter=4 (32 pasos totales)
    - Espacio latente: 512 dimensiones
    - Input/output: NoteSequences monofónicas cuantizadas

DESCARGA:
    Se hace automáticamente la primera vez que carga el sistema.
    El .tar se guarda en models_data/ y se extrae en models_data/cat-mel_2bar_big/.
"""

import logging
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_CONFIG_ID = "cat-mel_2bar_big"


# ===========================================================================
# Descarga y extracción
# ===========================================================================

def _download_and_extract(url: str, dest_dir: Path) -> bool:
    """Descarga el .tar del checkpoint y lo extrae en dest_dir."""
    tar_path = dest_dir.parent / (dest_dir.name + ".tar")

    if not tar_path.exists():
        log.info("MusicVAE: descargando checkpoint (~26 MB)...")
        log.info(f"  URL: {url}")
        try:
            t0 = time.time()
            urllib.request.urlretrieve(url, tar_path)
            mb = tar_path.stat().st_size / (1024 * 1024)
            log.info(f"  Descargado: {mb:.1f} MB en {time.time() - t0:.1f}s")
        except Exception as e:
            log.error(f"MusicVAE: falló la descarga: {e}")
            return False

    # Verificar que no sea un HTML de error
    mb = tar_path.stat().st_size / (1024 * 1024)
    if mb < 1.0:
        log.error(
            f"MusicVAE: archivo .tar sospechosamente pequeño ({mb:.2f} MB). "
            "Posible URL rota. Borrando para reintentar."
        )
        tar_path.unlink()
        return False

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Si ya hay archivos extraídos, no repetir
    if any(dest_dir.iterdir()):
        log.info(f"MusicVAE: checkpoint ya extraído en {dest_dir.name}/")
        return True

    log.info("MusicVAE: extrayendo checkpoint...")
    try:
        with tarfile.open(tar_path) as tf:
            tf.extractall(dest_dir)
        log.info(f"MusicVAE: extraído en {dest_dir.name}/")
        return True
    except Exception as e:
        log.error(f"MusicVAE: falló la extracción: {e}")
        return False


# ===========================================================================
# Búsqueda del checkpoint
# ===========================================================================

def _find_checkpoint_path(base_dir: Path) -> Optional[Path]:
    """
    Busca el archivo .index del checkpoint (en base_dir o en subdirectorios).
    Devuelve el path sin la extensión .index (el prefix que usa TF).

    El .tar de Google Cloud no incluye el archivo 'checkpoint' que
    tf.train.latest_checkpoint() necesita para autodiscovery, así que
    buscamos el .index directamente y derivamos el prefix.
    """
    # Buscar en el directorio base primero, luego en subdirectorios
    for search_dir in [base_dir, *sorted(base_dir.rglob("*"))]:
        if isinstance(search_dir, Path) and not search_dir.is_dir():
            continue
        index_files = list(search_dir.glob("*.index")) if search_dir.is_dir() else []
        if index_files:
            # Tomar el más reciente si hay varios
            index_file = sorted(index_files)[-1]
            # El prefix es el path sin ".index"
            prefix = index_file.with_suffix("")
            log.debug(f"MusicVAE: checkpoint encontrado en {prefix}")
            return prefix

    return None


# ===========================================================================
# Carga
# ===========================================================================

def load_music_vae(
    checkpoint_dir: Path,
    url: str,
) -> Optional[object]:
    """
    Carga MusicVAE (cat-mel_2bar_big). Descarga si no existe.

    Args:
        checkpoint_dir: carpeta destino de extracción (models_data/cat-mel_2bar_big/).
        url:            URL del .tar en Google Cloud Storage.

    Returns:
        TrainedModel listo para encode() / decode(), o None si falla.
    """
    try:
        # numpy >= 1.24 eliminó np.bool, np.int, np.float, etc.
        # Magenta 2.1.4 todavía los usa en music_vae/data.py.
        # Este patch los restaura ANTES de que se importe el módulo.
        import numpy as _np
        for _alias, _builtin in [
            ("bool", bool), ("int", int), ("float", float), ("complex", complex),
            ("object", object), ("str", str),
        ]:
            if not hasattr(_np, _alias):
                setattr(_np, _alias, _builtin)

        from magenta.models.music_vae import configs as vae_configs
        from magenta.models.music_vae.trained_model import TrainedModel
    except Exception as e:
        log.error(
            f"MusicVAE: no se pudo importar magenta.models.music_vae: {e}\n"
            "  Verificá que Magenta esté instalado correctamente."
        )
        return None

    # Descargar y extraer si no existe
    if not checkpoint_dir.exists() or not any(checkpoint_dir.iterdir()):
        ok = _download_and_extract(url, checkpoint_dir)
        if not ok:
            return None

    # Encontrar el path exacto del checkpoint.
    # El .tar no incluye el archivo "checkpoint" que TF necesita para
    # autodiscovery, así que buscamos el .index y usamos su stem.
    ckpt_path = _find_checkpoint_path(checkpoint_dir)
    if ckpt_path is None:
        log.error(
            f"MusicVAE: no se encontró ningún checkpoint en {checkpoint_dir}. "
            "Borrá la carpeta y volvé a correr para redescargar."
        )
        return None

    try:
        log.info(f"MusicVAE: cargando {ckpt_path.name}...")
        t0 = time.time()
        cfg = vae_configs.CONFIG_MAP[_CONFIG_ID]
        model = TrainedModel(
            cfg,
            batch_size=4,
            checkpoint_dir_or_path=str(ckpt_path),
        )
        log.info(f"MusicVAE: listo en {time.time() - t0:.1f}s")
        return model
    except Exception as e:
        log.error(f"MusicVAE: falló al cargar el modelo: {e}")
        return None
