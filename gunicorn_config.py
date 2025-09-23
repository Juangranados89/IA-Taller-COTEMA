#!/usr/bin/env python3

"""
Configuración de Gunicorn para COTEMA Analytics
Optimizada para manejo de archivos Excel grandes y procesamiento de datos
"""

import multiprocessing
import os

# Binding
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker processes (reducir para archivos grandes)
workers = min(2, multiprocessing.cpu_count())  # Máximo 2 workers para evitar OOM
worker_class = "sync"
worker_connections = 500  # Reducido para mejor memoria por conexión

# Timeouts CRÍTICOS para archivos Excel grandes
timeout = 600  # 10 minutos para archivos grandes (era 300)
keepalive = 5
graceful_timeout = 120  # 2 minutos para shutdown graceful

# Request limits AUMENTADOS para archivos Excel
max_requests = 100  # Reducido para forzar restart de workers
max_requests_jitter = 25
worker_memory_limit = "1024MB"  # Límite explícito de memoria por worker

# File upload limits
client_max_body_size = "50M"
limit_request_field_size = 16384  # Aumentado para headers grandes

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Security
limit_request_line = 4096
limit_request_fields = 100
limit_request_field_size = 8192

# Preload para mejor rendimiento
preload_app = True

# Worker temp directory
worker_tmp_dir = "/dev/shm"

# Memory optimizations for Excel processing
def when_ready(server):
    server.log.info("Server is ready. Optimized for Excel file processing.")

def worker_int(worker):
    worker.log.info("worker received INT or QUIT signal")

def pre_fork(server, worker):
    server.log.info("Worker spawned (pid: %s)", worker.pid)
