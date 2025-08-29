#!/usr/bin/env python3
"""Test script para verificar la corrección del error de estructura de datos"""

import sys
sys.path.append('/workspaces/IA-Taller-COTEMA')

from app import app, ml_engine
import json

def test_kpi_structure():
    """Prueba la estructura de datos de KPIs"""
    print("🔍 Probando estructura de datos KPIs...")
    
    with app.app_context():
        try:
            # Cargar códigos de equipos reales
            equipos = ml_engine.load_real_equipment_codes()
            print(f"✅ Equipos cargados: {equipos[:5]}...")
            
            # Simular llamada a KPIs
            from app import calculate_kpis
            
            # Crear una solicitud simulada
            with app.test_request_context('/kpis/enero'):
                result = calculate_kpis('enero')
                
                if hasattr(result, 'data'):
                    data = json.loads(result.data)
                    
                    print(f"✅ Respuesta obtenida para {data.get('total_equipos', 0)} equipos")
                    
                    # Verificar estructura FR-30
                    if 'kpis' in data and 'fr30' in data['kpis']:
                        fr30_data = data['kpis']['fr30']
                        
                        if fr30_data:
                            first_equipo = list(fr30_data.keys())[0]
                            first_data = fr30_data[first_equipo]
                            
                            print(f"\n📊 Estructura FR-30 para {first_equipo}:")
                            for key, value in first_data.items():
                                print(f"  {key}: {value}")
                            
                            # Verificar claves requeridas por JavaScript
                            required_keys = ['risk_percentage', 'status', 'badge_color']
                            missing_keys = [key for key in required_keys if key not in first_data]
                            
                            if missing_keys:
                                print(f"❌ Faltan claves: {missing_keys}")
                                return False
                            else:
                                print("✅ Todas las claves requeridas están presentes")
                                return True
                        else:
                            print("❌ No hay datos FR-30")
                            return False
                    else:
                        print("❌ No hay estructura kpis.fr30")
                        return False
                else:
                    print("❌ No se pudo obtener datos de respuesta")
                    return False
                    
        except Exception as e:
            print(f"❌ Error en prueba: {e}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    print("🚀 Iniciando test de corrección de estructura de datos...")
    success = test_kpi_structure()
    
    if success:
        print("\n✅ PRUEBA EXITOSA: La estructura de datos está corregida")
    else:
        print("\n❌ PRUEBA FALLIDA: Aún hay problemas en la estructura")
    
    sys.exit(0 if success else 1)
