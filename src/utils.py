"""
Utilidades y funciones de apoyo
Incluye generación de explicaciones con LLM y otras utilidades
"""

import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_llm_explanations(model_type, equipo, details):
    """
    Genera explicaciones usando LLM para los diferentes modelos
    """
    try:
        if model_type == 'fr30':
            return generate_fr30_explanation(equipo, details)
        elif model_type == 'rul':
            return generate_rul_explanation(equipo, details)
        elif model_type == 'forecast':
            return generate_forecast_explanation(equipo, details)
        elif model_type == 'anomaly':
            return generate_anomaly_explanation(equipo, details)
        else:
            return {"explanation": "Tipo de modelo no reconocido"}
    
    except Exception as e:
        print(f"Error generando explicación LLM: {str(e)}")
        return {"explanation": f"Error generando explicación: {str(e)}"}

def generate_fr30_explanation(equipo, details):
    """
    Genera explicación para FR-30
    """
    risk_30d = details.get('risk_30d', 0)
    features = details.get('features', {})
    
    # Análisis de drivers principales
    drivers = []
    acciones = []
    
    # Analizar TBF
    tbf_dias = features.get('tbf_dias', 0)
    tbf_med_60d = features.get('tbf_med_60d', 0)
    if tbf_dias > 0 and tbf_med_60d > 0 and tbf_dias < tbf_med_60d * 0.7:
        drivers.append(f"TBF actual ({tbf_dias:.0f} días) significativamente menor al histórico ({tbf_med_60d:.0f} días)")
        acciones.append("Revisar últimas intervenciones y calidad de reparaciones")
    
    # Analizar reincidencia
    ingresos_30d = features.get('ingresos_30d', 0)
    if ingresos_30d >= 2:
        drivers.append(f"Alta frecuencia de ingresos recientes: {ingresos_30d} en 30 días")
        acciones.append("Analizar causa raíz de fallas recurrentes")
    
    # Analizar MTTR
    mttr_eq = features.get('mttr_eq', 0)
    if mttr_eq > 48:
        drivers.append(f"MTTR elevado ({mttr_eq:.0f} horas) indica complejidad de reparaciones")
        acciones.append("Evaluar disponibilidad de repuestos y capacitación técnica")
    
    # Analizar sistema
    sistema_count_60d = features.get('sistema_count_60d', 0)
    if sistema_count_60d >= 2:
        drivers.append(f"Reincidencia en mismo sistema: {sistema_count_60d} veces en 60 días")
        acciones.append("Inspección profunda del sistema afectado")
    
    # Si no hay drivers claros
    if not drivers:
        if risk_30d < 0.3:
            drivers.append("Operación estable con patrones normales de falla")
            acciones.append("Mantener programa de mantenimiento preventivo")
        else:
            drivers.append("Múltiples factores menores contribuyen al riesgo")
            acciones.append("Monitoreo cercano y análisis de tendencias")
    
    # Acciones para próximas semanas
    acciones_1_2_sem = []
    if risk_30d >= 0.5:
        acciones_1_2_sem.extend([
            "Inspección inmediata de sistemas críticos",
            "Verificar disponibilidad de repuestos prioritarios"
        ])
    elif risk_30d >= 0.3:
        acciones_1_2_sem.extend([
            "Programar inspección preventiva",
            "Revisar historial de mantenimiento"
        ])
    else:
        acciones_1_2_sem.append("Continuar con programa normal de mantenimiento")
    
    explanation = {
        "resumen": f"Equipo {equipo} presenta riesgo de falla 30d = {risk_30d:.2f}. " +
                  ("Requiere atención inmediata." if risk_30d >= 0.5 else 
                   "Monitoreo recomendado." if risk_30d >= 0.3 else "Operación normal."),
        "drivers": drivers[:3],  # Top 3 drivers
        "acciones_1_2_sem": acciones_1_2_sem,
        "factores_influyentes": {
            "tbf_actual": tbf_dias,
            "ingresos_recientes": ingresos_30d,
            "mttr_promedio": mttr_eq,
            "reincidencia_sistema": sistema_count_60d
        }
    }
    
    return explanation

def generate_rul_explanation(equipo, details):
    """
    Genera explicación para RUL
    """
    rul50_d = details.get('rul50_d', 0)
    rul90_d = details.get('rul90_d', 0)
    parameters = details.get('parameters', {})
    
    # Determinar urgencia
    if rul90_d < 7:
        urgencia = "CRÍTICA"
        plazo = "inmediata"
        riesgo = "muy alto de falla inminente"
    elif rul90_d < 21:
        urgencia = "ALTA"
        plazo = "esta semana"
        riesgo = "alto si no se interviene pronto"
    elif rul50_d < 45:
        urgencia = "MEDIA"
        plazo = "próximas 2 semanas"
        riesgo = "moderado con ventana de intervención"
    else:
        urgencia = "BAJA"
        plazo = "próximo mes"
        riesgo = "bajo con tiempo suficiente para planificar"
    
    # Plan de 5 pasos
    plan_pasos = []
    
    if urgencia == "CRÍTICA":
        plan_pasos = [
            f"DÍA 1-2: Inspección inmediata y evaluación de estado",
            f"DÍA 3-4: Adquisición urgente de repuestos críticos",
            f"DÍA 5-6: Programación de parada y preparación de taller",
            f"DÍA 7: Ejecución de mantenimiento mayor",
            f"DÍA 8-10: Pruebas y puesta en servicio"
        ]
    elif urgencia == "ALTA":
        plan_pasos = [
            f"SEMANA 1: Diagnóstico detallado y evaluación de criticidad",
            f"SEMANA 1-2: Gestión de repuestos y recursos",
            f"SEMANA 2: Coordinación con operaciones para parada programada",
            f"SEMANA 3: Ejecución de mantenimiento preventivo/correctivo",
            f"SEMANA 3-4: Verificación y retorno a operación normal"
        ]
    else:
        plan_pasos = [
            f"SEMANA 1-2: Planificación detallada de intervención",
            f"SEMANA 3-4: Adquisición de repuestos y materiales",
            f"SEMANA 5-6: Coordinación con cronograma operativo",
            f"SEMANA 7-8: Ejecución de mantenimiento programado",
            f"SEMANA 9: Análisis post-mantenimiento y lecciones aprendidas"
        ]
    
    # Repuestos críticos basados en parámetros del modelo
    repuestos_criticos = [
        "Componentes de desgaste principal del sistema",
        "Elementos de filtración y lubricación",
        "Piezas con histórico de falla frecuente"
    ]
    
    explanation = {
        "resumen": f"Equipo {equipo}: RUL-50 = {rul50_d:.0f} días, RUL-90 = {rul90_d:.0f} días. Urgencia: {urgencia}",
        "ventana_intervencion": f"Intervención recomendada {plazo}",
        "plan_5_pasos": plan_pasos,
        "urgencia_dias": int(rul90_d),
        "riesgos_si_no": f"Riesgo {riesgo}",
        "repuestos_criticos": repuestos_criticos,
        "parametros_modelo": parameters
    }
    
    return explanation

def generate_forecast_explanation(equipo, details):
    """
    Genera explicación para pronóstico de uso
    """
    forecast_7d = details.get('forecast_7d_h', 0)
    forecast_30d = details.get('forecast_30d_h', 0)
    horas_por_dia = details.get('horas_por_dia', 0)
    confianza = details.get('confianza', 'MEDIA')
    
    # Evaluar intensidad de uso
    if horas_por_dia > 16:
        intensidad = "MUY ALTA"
        impacto = "acelerado desgaste y mayor probabilidad de falla"
    elif horas_por_dia > 12:
        intensidad = "ALTA"
        impacto = "desgaste acelerado requiere monitoreo cercano"
    elif horas_por_dia > 8:
        intensidad = "MEDIA"
        impacto = "uso normal dentro de parámetros esperados"
    else:
        intensidad = "BAJA"
        impacto = "uso reducido permite mantenimiento programado"
    
    # Bullets accionables
    bullets = [
        f"Proyección uso: {forecast_7d:.0f}h/7d y {forecast_30d:.0f}h/30d (intensidad {intensidad})",
        f"Impacto: {impacto}",
        f"Confianza del modelo: {confianza} - {'Usar para planificación detallada' if confianza == 'ALTA' else 'Validar con datos operativos'}",
    ]
    
    # Recomendaciones específicas
    if intensidad in ["MUY ALTA", "ALTA"]:
        bullets.append("SCHEDULER: Considerar rotación de equipos para reducir carga")
        bullets.append("COMPRAS: Acelerar adquisición de repuestos de desgaste")
    else:
        bullets.append("SCHEDULER: Ventana disponible para mantenimiento programado")
        bullets.append("COMPRAS: Proceder con cronograma normal de repuestos")
    
    explanation = {
        "resumen": f"Proyección uso para {equipo}: {forecast_7d:.0f}h/7d y {forecast_30d:.0f}h/30d",
        "intensidad_uso": intensidad,
        "horas_diarias": f"{horas_por_dia:.1f}",
        "bullets_accionables": bullets,
        "recomendaciones": {
            "scheduler": "Rotación de equipos" if intensidad in ["MUY ALTA", "ALTA"] else "Mantenimiento programado",
            "compras": "Acelerar repuestos" if intensidad in ["MUY ALTA", "ALTA"] else "Cronograma normal"
        }
    }
    
    return explanation

def generate_anomaly_explanation(equipo, details):
    """
    Genera explicación para anomalías
    """
    anomaly_score = details.get('anomaly_score', 0)
    features = details.get('features', {})
    
    # Analizar patrones anómalos
    anomalias_detectadas = []
    
    # TBF
    tbf_med_30d = features.get('tbf_med_30d', 0)
    tbf_trend = features.get('tbf_trend', 0)
    if tbf_trend < -1:
        anomalias_detectadas.append(f"Tendencia decreciente en TBF: {tbf_trend:.1f} días/evento")
    if tbf_med_30d > 0 and tbf_med_30d < 10:
        anomalias_detectadas.append(f"TBF anormalmente bajo: {tbf_med_30d:.0f} días promedio")
    
    # Reincidencia
    ingresos_30d = features.get('ingresos_30d', 0)
    ratio_actividad = features.get('ratio_actividad_30_60', 1)
    if ingresos_30d >= 3:
        anomalias_detectadas.append(f"Alta frecuencia: {ingresos_30d} ingresos en 30 días")
    if ratio_actividad > 2:
        anomalias_detectadas.append(f"Actividad reciente {ratio_actividad:.1f}x mayor que histórica")
    
    # MTTR
    cambio_mttr = features.get('cambio_mttr', 1)
    if cambio_mttr > 1.5:
        anomalias_detectadas.append(f"MTTR aumentó {cambio_mttr:.1f}x respecto a histórico")
    
    # Sistemas
    sistemas_unicos_30d = features.get('sistemas_unicos_30d', 0)
    if sistemas_unicos_30d > 2:
        anomalias_detectadas.append(f"Múltiples sistemas afectados: {sistemas_unicos_30d}")
    
    # Si no hay anomalías específicas detectadas
    if not anomalias_detectadas:
        if anomaly_score > 0.6:
            anomalias_detectadas.append("Combinación de factores menores genera patrón atípico")
        else:
            anomalias_detectadas.append("Operación dentro de patrones normales")
    
    # Chequeos recomendados
    chequeos = []
    if anomaly_score >= 0.8:
        chequeos.extend([
            "Inspección visual completa del equipo HOY",
            "Verificación de parámetros operativos críticos"
        ])
    elif anomaly_score >= 0.6:
        chequeos.extend([
            "Revisión de registros de mantenimiento reciente",
            "Análisis de condiciones operativas actuales"
        ])
    else:
        chequeos.append("Continuar con programa normal de monitoreo")
    
    explanation = {
        "resumen": f"Equipo {equipo} presenta score de anomalía = {anomaly_score:.2f}",
        "anomalias_detectadas": anomalias_detectadas[:3],  # Top 3
        "chequeos_recomendados": chequeos,
        "relacion_tbf_reincidencia_mttr": {
            "tbf_30d": tbf_med_30d,
            "ingresos_30d": ingresos_30d,
            "cambio_mttr": cambio_mttr,
            "sistemas_afectados": sistemas_unicos_30d
        }
    }
    
    return explanation

def format_number(value, format_type='decimal'):
    """
    Formatea números para presentación
    """
    try:
        if pd.isna(value) or value is None:
            return "N/A"
        
        if format_type == 'percentage':
            return f"{float(value):.1%}"
        elif format_type == 'decimal':
            return f"{float(value):.2f}"
        elif format_type == 'integer':
            return f"{int(value):,}"
        elif format_type == 'days':
            return f"{int(value)} días"
        elif format_type == 'hours':
            return f"{float(value):.1f} hrs"
        else:
            return str(value)
    
    except:
        return "Error"

def calculate_kpi_summary(kpis_data):
    """
    Calcula resumen de KPIs
    """
    summary = {
        'total_equipos': 0,
        'fr30_alto': 0,
        'fr30_medio': 0,
        'fr30_bajo': 0,
        'rul_critico': 0,
        'rul_alerta': 0,
        'rul_normal': 0,
        'anomalia_alta': 0,
        'anomalia_media': 0,
        'anomalia_baja': 0
    }
    
    fr30_data = kpis_data.get('fr30', {})
    rul_data = kpis_data.get('rul', {})
    anomaly_data = kpis_data.get('anomaly', {})
    
    equipos = set(fr30_data.keys()) | set(rul_data.keys()) | set(anomaly_data.keys())
    summary['total_equipos'] = len(equipos)
    
    # Contar FR-30
    for equipo in fr30_data:
        risk = fr30_data[equipo].get('risk_30d', 0)
        if risk >= 0.5:
            summary['fr30_alto'] += 1
        elif risk >= 0.3:
            summary['fr30_medio'] += 1
        else:
            summary['fr30_bajo'] += 1
    
    # Contar RUL
    for equipo in rul_data:
        rul90 = rul_data[equipo].get('rul90_d', 0)
        if rul90 < 7:
            summary['rul_critico'] += 1
        elif rul90 < 21:
            summary['rul_alerta'] += 1
        else:
            summary['rul_normal'] += 1
    
    # Contar anomalías
    for equipo in anomaly_data:
        score = anomaly_data[equipo].get('anomaly_score', 0)
        if score >= 0.8:
            summary['anomalia_alta'] += 1
        elif score >= 0.6:
            summary['anomalia_media'] += 1
        else:
            summary['anomalia_baja'] += 1
    
    return summary

def export_to_csv(kpis_data, filename):
    """
    Exporta los KPIs a CSV
    """
    try:
        all_data = []
        
        for mes, kpis in kpis_data.items():
            fr30_data = kpis.get('fr30', {})
            rul_data = kpis.get('rul', {})
            forecast_data = kpis.get('forecast', {})
            anomaly_data = kpis.get('anomaly', {})
            
            equipos = set(fr30_data.keys()) | set(rul_data.keys()) | set(forecast_data.keys()) | set(anomaly_data.keys())
            
            for equipo in equipos:
                row = {
                    'mes': mes,
                    'equipo': equipo,
                    'fr30_risk': fr30_data.get(equipo, {}).get('risk_30d', 0),
                    'fr30_banda': fr30_data.get(equipo, {}).get('banda', ''),
                    'rul50_d': rul_data.get(equipo, {}).get('rul50_d', 0),
                    'rul90_d': rul_data.get(equipo, {}).get('rul90_d', 0),
                    'forecast_7d_h': forecast_data.get(equipo, {}).get('forecast_7d_h', 0),
                    'forecast_30d_h': forecast_data.get(equipo, {}).get('forecast_30d_h', 0),
                    'horas_por_dia': forecast_data.get(equipo, {}).get('horas_por_dia', 0),
                    'anomaly_score': anomaly_data.get(equipo, {}).get('anomaly_score', 0),
                    'anomaly_banda': anomaly_data.get(equipo, {}).get('banda', '')
                }
                all_data.append(row)
        
        df = pd.DataFrame(all_data)
        df.to_csv(filename, index=False)
        print(f"Datos exportados a {filename}")
        return True
    
    except Exception as e:
        print(f"Error exportando a CSV: {str(e)}")
        return False

def validate_configuration():
    """
    Valida la configuración del sistema
    """
    checks = {
        'directories': True,
        'dependencies': True,
        'models_path': True
    }
    
    # Verificar directorios
    required_dirs = ['uploads', 'models', 'static/plots', 'static/exports']
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except:
                checks['directories'] = False
    
    # Verificar dependencias críticas
    try:
        import pandas
        import sklearn
        import plotly
        import flask
    except ImportError as e:
        print(f"Dependencia faltante: {e}")
        checks['dependencies'] = False
    
    # Verificar path de modelos
    if not os.path.exists('models'):
        try:
            os.makedirs('models', exist_ok=True)
        except:
            checks['models_path'] = False
    
    return checks

def get_system_status():
    """
    Obtiene el estado del sistema
    """
    status = {
        'timestamp': datetime.now().isoformat(),
        'configuration': validate_configuration(),
        'memory_usage': get_memory_usage(),
        'disk_space': get_disk_space()
    }
    
    return status

def get_memory_usage():
    """
    Obtiene el uso de memoria del sistema
    """
    try:
        import psutil
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'percent': memory.percent,
            'used': memory.used
        }
    except:
        return {'error': 'No se pudo obtener información de memoria'}

def get_disk_space():
    """
    Obtiene el espacio en disco disponible
    """
    try:
        import psutil
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'free': disk.free,
            'used': disk.used,
            'percent': (disk.used / disk.total) * 100
        }
    except:
        return {'error': 'No se pudo obtener información de disco'}
