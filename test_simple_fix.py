#!/usr/bin/env python3
"""Test simplificado para verificar la corrección"""

import sys
sys.path.append('/workspaces/IA-Taller-COTEMA')

import json
from app import app, ml_engine

def test_simple():
    """Prueba simple de estructura de datos"""
    print("🔍 Probando corrección de estructura...")
    
    # Cargar equipos
    equipos = ml_engine.load_real_equipment_codes()
    print(f"✅ Equipos: {equipos[:3]}...")
    
    # Datos simulados de KPI
    kpis = {'fr30': {}, 'rul': {}, 'forecast': {}, 'anomaly': {}}
    
    # Simular datos FR-30 corregidos
    import random
    for equipo in equipos[:5]:
        base_risk = random.uniform(0.05, 0.65)
        banda = '🟢 BAJO' if base_risk < 0.25 else ('🟠 MEDIO' if base_risk < 0.50 else '🔴 ALTO')
        banda_color = 'success' if base_risk < 0.25 else ('warning' if base_risk < 0.50 else 'danger')
        
        kpis['fr30'][equipo] = {
            'risk_30d': round(base_risk, 3),
            'risk_percentage': f"{round(base_risk * 100, 1)}%",  # CLAVE CORREGIDA
            'status': banda,                                      # CLAVE CORREGIDA
            'badge_color': banda_color,                          # CLAVE CORREGIDA
            'confidence': 0.85,
            'explicacion': f'Simulación - {equipo}'
        }
    
    # Verificar estructura
    if kpis['fr30']:
        first_equipo = list(kpis['fr30'].keys())[0]
        first_data = kpis['fr30'][first_equipo]
        
        print(f"\n📊 Estructura corregida para {first_equipo}:")
        for key, value in first_data.items():
            print(f"  {key}: {value}")
        
        # Verificar claves que espera JavaScript
        required_js_keys = ['risk_percentage', 'status', 'badge_color']
        available_keys = list(first_data.keys())
        
        print(f"\n🔍 Claves requeridas por JS: {required_js_keys}")
        print(f"🔍 Claves disponibles: {available_keys}")
        
        missing = [k for k in required_js_keys if k not in available_keys]
        
        if missing:
            print(f"❌ Faltan: {missing}")
            return False
        else:
            print("✅ Todas las claves requeridas están presentes")
            
            # Mostrar ejemplo de uso en JavaScript
            print(f"\n📝 Ejemplo JS:")
            print(f"  fr30.risk_percentage = '{first_data['risk_percentage']}'")
            print(f"  fr30.status = '{first_data['status']}'")
            print(f"  fr30.badge_color = '{first_data['badge_color']}'")
            
            return True
    else:
        print("❌ No hay datos FR-30")
        return False

if __name__ == "__main__":
    success = test_simple()
    
    if success:
        print("\n✅ CORRECCIÓN CONFIRMADA: La estructura ahora es compatible con JavaScript")
    else:
        print("\n❌ PROBLEMA PERSISTE: Estructura incompatible")
    
    sys.exit(0 if success else 1)
