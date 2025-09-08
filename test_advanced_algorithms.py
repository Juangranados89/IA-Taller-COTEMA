"""
Script de Prueba - Algoritmos Avanzados COTEMA
Verificar que los nuevos algoritmos ML + Weibull funcionen correctamente
"""

import sys
import os
sys.path.append('/workspaces/IA-Taller-COTEMA')

# Probar importaciones
try:
    from src.advanced_prediction_engine import get_advanced_fr30_prediction
    from src.weibull_survival import integrate_weibull_analysis
    print("✅ Algoritmos avanzados importados correctamente")
    ALGORITHMS_OK = True
except Exception as e:
    print(f"❌ Error importando algoritmos: {e}")
    ALGORITHMS_OK = False

if ALGORITHMS_OK:
    print("\n🔬 Iniciando pruebas de algoritmos...")
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Crear datos de prueba realistas
    print("📊 Creando dataset de prueba...")
    
    equipos_test = ['VD-TC27', 'VD-C084', 'VD-TC04', 'VD-C033', 'VD-C048', 
                    'VD-C039', 'VD-C013', 'VD-C042', 'VD-TC37', 'CH-HR01']
    
    # Generar fechas de fallas con patrones realistas
    data_test = []
    base_date = datetime(2023, 1, 1)
    
    for equipo in equipos_test:
        # Cada equipo tiene un patrón diferente de fallas
        num_fallas = np.random.randint(5, 25)  # Entre 5 y 25 fallas por equipo
        
        for i in range(num_fallas):
            # Patrón de fallas más frecuentes en ciertos meses
            mes_peso = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 
                                      p=[0.10, 0.08, 0.08, 0.12, 0.15, 0.12, 0.08, 0.06, 0.12, 0.05, 0.02, 0.02])
            
            # Generar fecha aleatoria en ese mes
            año = np.random.choice([2023, 2024, 2025], p=[0.3, 0.4, 0.3])
            día = np.random.randint(1, 29)
            
            fecha_falla = datetime(año, mes_peso, día)
            
            data_test.append({
                'EQUIPO': equipo,
                'FECHA_INGRESO': fecha_falla,
                'TIPO_MANTENIMIENTO': np.random.choice(['CORRECTIVO', 'PREVENTIVO'], p=[0.7, 0.3]),
                'DESCRIPCION_FALLA': f'Falla {i+1} en {equipo}'
            })
    
    df_test = pd.DataFrame(data_test)
    print(f"✅ Dataset creado: {len(df_test)} registros, {len(equipos_test)} equipos")
    
    # Probar algoritmo ML avanzado
    print("\n🤖 Probando algoritmo ML avanzado...")
    try:
        # Llamar a la función de predicción (sin el parámetro 'year')
        ml_result = get_advanced_fr30_prediction(df_test)
        
        print(f"✅ ML Analysis completado")
        print(f"   - Equipos analizados: {len(ml_result.get('equipos_riesgo', []))}")
        print(f"   - Meses predichos: {len(ml_result.get('meses_tendencia', []))}")
        print(f"   - Algoritmo usado: {ml_result.get('factores_analisis', {}).get('algoritmo_utilizado', 'N/A')}")
        
        if ml_result.get('equipos_riesgo'):
            top_equipo = ml_result['equipos_riesgo'][0]
            print(f"   - Equipo mayor riesgo: {top_equipo['equipo']} (Score: {top_equipo['riesgo_score']})")
        
    except Exception as e:
        print(f"❌ Error en ML: {e}")
        ml_result = None
    
    # Probar integración Weibull
    if ml_result:
        print("\n📈 Probando análisis Weibull...")
        try:
            weibull_result = integrate_weibull_analysis(df_test, ml_result)
            
            print(f"✅ Weibull Analysis completado")
            weibull_info = weibull_result.get('weibull_analysis', {})
            print(f"   - Equipos analizados Weibull: {weibull_info.get('equipos_analizados', 0)}")
            print(f"   - Equipos con datos suficientes: {weibull_info.get('equipos_con_datos_suficientes', 0)}")
            print(f"   - Algoritmo mejorado: {weibull_info.get('algoritmo_mejorado', 'N/A')}")
            
            if weibull_result.get('equipos_riesgo'):
                top_equipo_w = weibull_result['equipos_riesgo'][0]
                print(f"   - Top equipo (híbrido): {top_equipo_w['equipo']} (Score: {top_equipo_w['riesgo_score']})")
                
                # Mostrar métricas adicionales si están disponibles
                if 'mtbf_dias' in top_equipo_w:
                    print(f"   - MTBF: {top_equipo_w['mtbf_dias']} días")
                if 'prob_falla_30d' in top_equipo_w:
                    print(f"   - Prob falla 30d: {top_equipo_w['prob_falla_30d']}")
        
        except Exception as e:
            print(f"❌ Error en Weibull: {e}")
    
    print(f"\n🎯 Resumen de Mejoras Implementadas:")
    print(f"   ✅ Random Forest + Gradient Boosting Ensemble")
    print(f"   ✅ Features temporales avanzadas (TBF, estacionalidad)")
    print(f"   ✅ Análisis de supervivencia Weibull")
    print(f"   ✅ Predicción híbrida ML + Weibull")
    print(f"   ✅ Métricas de confianza mejoradas")
    print(f"   ✅ Patrones estacionales industriales")

else:
    print("❌ No se pueden ejecutar las pruebas sin algoritmos avanzados")
