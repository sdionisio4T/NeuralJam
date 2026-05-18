"""
neuraljam.models — Gestión de modelos de Magenta.

Responsabilidades:
- Descargar bundles .mag desde la URL del perfil activo (si no existen)
- Cargar el modelo correspondiente al MODE actual
- Hacer warmup pass para evitar latencia alta en la primera frase real
- Exponer un objeto generator listo para que generation/ lo use

Submódulos planeados:
- loader.py: carga el bundle del perfil activo, hace warmup
- downloader.py: descarga el .mag si no está local

Restricciones:
- Un solo modelo cargado en RAM a la vez (no hay RAM para más).
- Cambio de modo implica reiniciar el sistema. No hay hot-swap.

Contrato: el resto del sistema recibe un objeto generator ya warmupeado.
No conoce el tipo interno del modelo.
"""
