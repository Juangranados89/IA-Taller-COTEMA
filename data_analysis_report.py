#!/usr/bin/env python3
"""
ANÁLISIS DE DATOS PARA PREDICCIONES ML - TALLER COTEMA
======================================================
Este script analiza la estructura de datos disponible y diseña
la estrategia de predicciones basada en datos reales del taller.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def analyze_cotema_data():
    """Análisis completo de los datos del taller COTEMA"""
    
    print("🔍 ANÁLISIS DE DATOS COTEMA PARA PREDICCIONES ML")
    print("=" * 60)
    
    try:
        # Leer datos reales
        df = pd.read_excel('sample_data/Registro_Entrada_Taller_COTEMA.xlsx', 
                          sheet_name='REG', skiprows=4, usecols='B:Y')
        
        print(f"📊 RESUMEN GENERAL:")
        print(f"   Total registros: {len(df):,}")
        print(f"   Equipos únicos: {df['CODIGO'].nunique()}")
        print(f"   Período: {df['FECHA IN'].min().strftime('%Y-%m-%d')} a {df['FECHA IN'].max().strftime('%Y-%m-%d')}")
        
        # Análisis de equipos por código
        print(f"\n🏭 ANÁLISIS DE EQUIPOS:")
        equipment_analysis = {}
        
        for codigo in df['CODIGO'].dropna().unique()[:20]:  # Top 20 equipos
            equipo_data = df[df['CODIGO'] == codigo]
            if len(equipo_data) > 0:
                equipment_analysis[codigo] = {
                    'total_fallas': len(equipo_data),
                    'descripcion': equipo_data['DESCRIPCION'].iloc[0] if pd.notna(equipo_data['DESCRIPCION'].iloc[0]) else 'N/A',
                    'flota': equipo_data['FLOTA'].iloc[0] if pd.notna(equipo_data['FLOTA'].iloc[0]) else 'N/A',
                    'correctivas': len(equipo_data[equipo_data['TIPO ATENCION'] == 'CORRECTIVA']),
                    'preventivas': len(equipo_data[equipo_data['TIPO ATENCION'] == 'PREVENTIVA']),
                    'mttr_promedio': equipo_data['MTTR'].mean() if pd.notna(equipo_data['MTTR']).any() else 0,
                    'sistemas_afectados': equipo_data['SISTEMA AFECTADO'].nunique()
                }
        
        # Mostrar análisis de equipos
        for codigo, data in sorted(equipment_analysis.items(), key=lambda x: x[1]['total_fallas'], reverse=True)[:10]:
            tipo_equipo = codigo.split('-')[0] if '-' in codigo else 'UNKNOWN'
            print(f"   {codigo} ({tipo_equipo}): {data['total_fallas']} fallas, MTTR: {data['mttr_promedio']:.1f}h")
        
        # Análisis por tipo de equipo
        print(f"\n🔧 ANÁLISIS POR TIPO DE EQUIPO:")
        tipos_equipo = {}
        for codigo in df['CODIGO'].dropna():
            if '-' in str(codigo):
                tipo = str(codigo).split('-')[0]
                if tipo not in tipos_equipo:
                    tipos_equipo[tipo] = []
                tipos_equipo[tipo].append(codigo)
        
        for tipo, equipos in sorted(tipos_equipo.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            equipos_unicos = len(set(equipos))
            total_registros = len([e for e in equipos])
            print(f"   {tipo}: {equipos_unicos} equipos, {total_registros} registros")
        
        # Análisis de sistemas afectados
        print(f"\n⚙️ SISTEMAS MÁS AFECTADOS:")
        sistemas = df['SISTEMA AFECTADO'].value_counts().head(10)
        for sistema, count in sistemas.items():
            if pd.notna(sistema):
                print(f"   {sistema}: {count} fallas")
        
        # Análisis temporal
        print(f"\n📅 ANÁLISIS TEMPORAL:")
        df['FECHA IN'] = pd.to_datetime(df['FECHA IN'])
        df['mes_año'] = df['FECHA IN'].dt.to_period('M')
        fallas_mensuales = df['mes_año'].value_counts().sort_index()
        print(f"   Promedio fallas/mes: {fallas_mensuales.mean():.1f}")
        print(f"   Meses con más fallas: {fallas_mensuales.tail(3).to_dict()}")
        
        return create_prediction_strategy(df, equipment_analysis, tipos_equipo)
        
    except Exception as e:
        print(f"❌ Error en análisis: {e}")
        return None

def create_prediction_strategy(df, equipment_analysis, tipos_equipo):
    """Crea estrategia de predicción basada en datos disponibles"""
    
    print(f"\n🎯 ESTRATEGIA DE PREDICCIÓN RECOMENDADA:")
    print("=" * 50)
    
    # Nivel 1: Predicciones con datos actuales
    print(f"📊 NIVEL 1 - PREDICCIONES CON DATOS ACTUALES:")
    print(f"✅ FR-30 (Probabilidad falla 30 días):")
    print(f"   - Basado en frecuencia histórica por equipo")
    print(f"   - Factores: tipo equipo, sistema afectado, MTTR histórico")
    print(f"   - Ajuste por estacionalidad")
    
    print(f"✅ RUL (Vida útil restante):")
    print(f"   - Basado en horometro/km y patrones de falla")
    print(f"   - MTTR como indicador de complejidad")
    print(f"   - Perfil por tipo de equipo")
    
    print(f"✅ ANOMALÍAS:")
    print(f"   - Detección de patrones inusuales en frecuencia")
    print(f"   - MTTR excesivos vs. promedio histórico")
    print(f"   - Sistemas afectados fuera de patrón normal")
    
    # Nivel 2: Campos adicionales
    print(f"\n🚀 NIVEL 2 - MEJORAS RECOMENDADAS:")
    print(f"📈 Campos críticos a agregar:")
    print(f"   🌡️  TEMPERATURA: Motores, hidráulicos, transmisiones")
    print(f"   📳 VIBRACIÓN: Componentes rotativos, motores")
    print(f"   💧 PRESIÓN: Sistemas hidráulicos y neumáticos") 
    print(f"   ⏰ HORAS_OPERACION_DIARIAS: Carga de trabajo")
    print(f"   🌍 CONDICIONES_TRABAJO: Interno/externo, tipo de carga")
    print(f"   🔧 MANTENIMIENTO_PREVENTIVO: Historial y scheduling")
    
    # Implementación específica por tipo
    print(f"\n🏭 IMPLEMENTACIÓN POR TIPO DE EQUIPO:")
    
    equipment_profiles = {
        'VD': {
            'name': 'Volquetas/Camiones',
            'critical_systems': ['MOTOR', 'TRANSMISION', 'FRENOS'],
            'sensors_priority': ['temperatura_motor', 'presion_aceite', 'temperatura_transmision'],
            'prediction_focus': 'Desgaste por kilometraje y carga'
        },
        'CG': {
            'name': 'Camiones Grúa',
            'critical_systems': ['HIDRAULICO', 'MOTOR', 'GRUA'],
            'sensors_priority': ['presion_hidraulica', 'temperatura_motor', 'carga_grua'],
            'prediction_focus': 'Stress hidráulico y mecánico'
        },
        'EX': {
            'name': 'Excavadoras',
            'critical_systems': ['HIDRAULICO', 'MOTOR', 'ORUGAS'],
            'sensors_priority': ['presion_hidraulica', 'temperatura_motor', 'vibracion_brazo'],
            'prediction_focus': 'Desgaste hidráulico intensivo'
        },
        'CV': {
            'name': 'Cintas/Compactadores',
            'critical_systems': ['VIBRATORIO', 'MOTOR', 'TRANSMISION'],
            'sensors_priority': ['vibracion_tambor', 'temperatura_motor', 'horas_vibracion'],
            'prediction_focus': 'Fatiga por vibración continua'
        }
    }
    
    for tipo_code, equipos in sorted(tipos_equipo.items(), key=lambda x: len(x[1]), reverse=True)[:4]:
        if tipo_code in equipment_profiles:
            profile = equipment_profiles[tipo_code]
            count = len(set(equipos))
            print(f"   {tipo_code} - {profile['name']} ({count} equipos):")
            print(f"      🎯 Foco: {profile['prediction_focus']}")
            print(f"      🔧 Sistemas críticos: {', '.join(profile['critical_systems'])}")
            print(f"      📊 Sensores prioridad: {', '.join(profile['sensors_priority'])}")
    
    return {
        'total_equipos': df['CODIGO'].nunique(),
        'tipos_principales': list(tipos_equipo.keys())[:5],
        'sistemas_criticos': df['SISTEMA AFECTADO'].value_counts().head(5).index.tolist(),
        'estrategia': 'historical_pattern_ml',
        'nivel_implementacion': 1,
        'mejoras_recomendadas': equipment_profiles
    }

def generate_prediction_features():
    """Genera las features que podemos extraer de los datos actuales"""
    
    print(f"\n🔬 FEATURES DISPONIBLES PARA ML:")
    print("=" * 40)
    
    features_available = {
        'primary': [
            'equipo_codigo',           # Identificación única
            'tipo_equipo',             # VD, CG, EX, etc.
            'horometro_acumulado',     # Desgaste total
            'km_acumulados',           # Desgaste por distancia
            'dias_desde_ultima_falla', # Tiempo desde último evento
            'frecuencia_fallas_30d',   # Fallas en últimos 30 días
            'mttr_historico_promedio', # Tiempo promedio reparación
            'sistema_afectado_freq',   # Frecuencia por sistema
        ],
        'derived': [
            'intensidad_uso',          # Horometro/tiempo
            'patron_estacional',       # Época del año
            'criticidad_historica',    # Basada en MTTR y frecuencia
            'tendencia_deterioro',     # Cambio en frecuencia fallas
            'eficiencia_mantenimiento', # Ratio preventivo/correctivo
        ],
        'target_variables': [
            'prob_falla_30d',          # Probabilidad falla próximos 30 días
            'rul_estimado_dias',       # Días hasta próximo mantenimiento
            'sistema_falla_probable',   # Sistema más probable a fallar
            'mttr_esperado',           # Tiempo esperado de reparación
        ]
    }
    
    for category, features in features_available.items():
        print(f"\n{category.upper()}:")
        for feature in features:
            print(f"   ✅ {feature}")
    
    return features_available

if __name__ == "__main__":
    strategy = analyze_cotema_data()
    features = generate_prediction_features()
    
    print(f"\n🎉 ANÁLISIS COMPLETADO")
    print(f"Estrategia guardada para implementación en app.py")
