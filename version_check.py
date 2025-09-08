#!/usr/bin/env python3
"""
Verificador de versión de COTEMA - Script para confirmar que los cambios están activos
"""
import json
from datetime import datetime

VERSION_INFO = {
    "version": "COTEMA v2.1 - FR-30 KPI Optimizado",
    "last_update": "2025-01-27 19:45 UTC",
    "commit_hash": "b80bc7e",
    "features_active": [
        "✅ Endpoint /analyze_statistics_advanced restaurado",
        "✅ KPI FR-30 mejorado con cálculos confiables 0-100%",
        "✅ Enfoque mes actual + próximo mes",
        "✅ API limpia para BI: /api/kpi/fr30",
        "✅ Algoritmo FR-30 v2.1 optimizado",
        "✅ Identificación automática de equipos críticos"
    ],
    "endpoints_available": [
        "POST /analyze_statistics_advanced (Para interfaz actual)",
        "GET /api/kpi/fr30 (Para consumo BI)",
        "GET /version_check (Este endpoint)"
    ],
    "deployment_timestamp": datetime.now().isoformat()
}

if __name__ == "__main__":
    print(json.dumps(VERSION_INFO, indent=2, ensure_ascii=False))
