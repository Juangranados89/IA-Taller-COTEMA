#!/usr/bin/env python3
"""
Script para probar todos los endpoints y verificar que los botones funcionen correctamente
"""
import requests
import json
import sys

# URLs a probar
ENDPOINTS = [
    # Endpoints principales
    ("POST", "/analyze_statistics_advanced", "KPI FR-30 Avanzado"),
    ("GET", "/api/kpi/fr30", "API limpia para BI"),
    ("POST", "/analyze-fr30", "Análisis FR-30 específico"),
    ("POST", "/deep-analysis", "Análisis profundo"),
    ("GET", "/api/train-progress", "Progreso de entrenamiento"),
    ("GET", "/version_check", "Verificación de versión"),
    
    # Endpoints existentes
    ("POST", "/quick-analysis", "Análisis rápido"),
    ("POST", "/train-models", "Entrenar modelos"),
    ("GET", "/progress", "Progreso general"),
    ("POST", "/ml_analysis", "Análisis ML"),
    ("GET", "/api/status", "Estado de la API"),
]

BASE_URL = "http://localhost:5000"

def test_endpoint(method, endpoint, description):
    """Probar un endpoint específico"""
    url = BASE_URL + endpoint
    try:
        if method == "GET":
            response = requests.get(url, timeout=5)
        else:
            response = requests.post(url, timeout=5, json={})
        
        status = "✅ OK" if response.status_code < 400 else f"❌ {response.status_code}"
        print(f"{status} {method:4} {endpoint:30} - {description}")
        
        if response.status_code < 400:
            return True
        else:
            print(f"     Error: {response.text[:100]}...")
            return False
            
    except requests.exceptions.ConnectionRefused:
        print(f"🔌 CONN {method:4} {endpoint:30} - {description} (Servidor no iniciado)")
        return False
    except Exception as e:
        print(f"❌ ERR  {method:4} {endpoint:30} - {description} ({str(e)[:50]})")
        return False

def main():
    print("🔍 VERIFICACIÓN DE ENDPOINTS - COTEMA v2.1")
    print("=" * 70)
    
    success_count = 0
    total_count = len(ENDPOINTS)
    
    for method, endpoint, description in ENDPOINTS:
        if test_endpoint(method, endpoint, description):
            success_count += 1
    
    print("=" * 70)
    print(f"📊 RESULTADOS: {success_count}/{total_count} endpoints funcionando")
    
    if success_count == total_count:
        print("🎉 ¡TODOS LOS BOTONES DEBERÍAN FUNCIONAR CORRECTAMENTE!")
    else:
        print(f"⚠️  {total_count - success_count} endpoints necesitan atención")
    
    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
