#!/usr/bin/env python3

"""
Configuración de Gunicorn para COTEMA Analytics
Optimizada para manejo de archivos Excel grandes y procesamiento de datos
"""

import multiprocessing
import os

# Binding
bind = f"0.0.0.0:{os.environ.get('PORT', '8000')}"

# Worker processes
workers = min(4, (multiprocessing.cpu_count() * 2) + 1)
worker_class = "sync"
worker_connections = 1000

# Timeouts (importantes para archivos Excel grandes)
timeout = 300  # 5 minutos para procesamiento de archivos grandes
keepalive = 2
graceful_timeout = 60

# Request limits
max_requests = 500
max_requests_jitter = 50

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
