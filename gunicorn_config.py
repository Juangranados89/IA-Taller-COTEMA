# gunicorn_config.py
import os

# Aumentar el timeout para permitir el procesamiento de archivos grandes.
# El valor por defecto es 30 segundos. Lo aumentamos a 120 segundos.
timeout = 120

# Número de workers (recomendado por Render.com)
workers = int(os.environ.get('WEB_CONCURRENCY', 2))

# Enlazar al puerto y host correctos
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"

# Nivel de logging
loglevel = 'info'

print(f"Gunicorn config loaded: timeout={timeout}s, workers={workers}, bind={bind}")
