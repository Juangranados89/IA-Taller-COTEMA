"""
COTEMA Processor - Motor de Procesamiento Especializado para Datos de Mantenimiento
==============================================================================

Este módulo contiene la lógica especializada para procesar y normalizar
datos de mantenimiento del taller COTEMA.

Funciones principales:
- process_cotema_data: Procesamiento principal con normalización inteligente
- generate_quality_report: Reporte de calidad y integridad de datos
- create_catalogs: Generación de catálogos para análisis ML
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re
from typing import Dict, List, Tuple, Any

def process_cotema_data(df_raw: pd.DataFrame) -> Tuple[List[Dict], Dict, Dict]:
    """
    Procesa y normaliza datos COTEMA con análisis de calidad integrado.
    
    Args:
        df_raw: DataFrame crudo desde Excel
        
    Returns:
        Tuple con:
        - dataset: Lista de registros normalizados
        - quality_report: Reporte de calidad y estadísticas
        - catalogos: Catálogos para análisis ML
    """
    logging.info("🔄 Iniciando procesamiento especializado COTEMA...")
    
    try:
        # Paso 1: Normalización de columnas
        df = _normalize_columns(df_raw.copy())
        
        # Paso 2: Limpieza básica
        df = _basic_cleaning(df)
        
        # Paso 3: Análisis de calidad
        quality_report = _generate_quality_report(df)
        
        # Paso 4: Normalización de datos
        df_normalized = _normalize_data_values(df)
        
        # Paso 5: Creación de catálogos
        catalogos = _create_catalogs(df_normalized)
        
        # Paso 6: Conversión a dataset final
        dataset = df_normalized.to_dict('records')
        
        logging.info(f"✅ COTEMA processing completado. {len(dataset)} registros procesados.")
        
        return dataset, quality_report, catalogos
        
    except Exception as e:
        logging.error(f"❌ Error en process_cotema_data: {e}")
        # Retorno de emergencia con datos mínimos
        emergency_dataset = df_raw.head(100).to_dict('records') if not df_raw.empty else []
        emergency_report = {
            'total_registros': len(df_raw),
            'registros_abiertos': 0,
            'registros_cerrados': 0,
            'errores': {'processing_error': str(e)}
        }
        emergency_catalogs = {}
        
        return emergency_dataset, emergency_report, emergency_catalogs


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas."""
    
    # Mapeo de columnas comunes
    column_mapping = {
        # Variaciones de equipo
        'equipo': 'equipo',
        'codigo_equipo': 'equipo', 
        'cod_equipo': 'equipo',
        'equipment': 'equipo',
        
        # Variaciones de fecha
        'fecha': 'fecha',
        'fecha_inicio': 'fecha',
        'date': 'fecha',
        'fecha_evento': 'fecha',
        
        # Variaciones de estado
        'estado': 'estado',
        'status': 'estado',
        'estado_actual': 'estado',
        
        # Otros campos comunes
        'descripcion': 'descripcion',
        'description': 'descripcion',
        'detalle': 'descripcion',
        'observaciones': 'observaciones',
        'comments': 'observaciones',
        'tipo': 'tipo',
        'type': 'tipo',
        'prioridad': 'prioridad',
        'priority': 'prioridad'
    }
    
    # Normalizar nombres de columnas
    df.columns = df.columns.astype(str)
    df.columns = [col.lower().strip() for col in df.columns]
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', col) for col in df.columns]
    
    # Aplicar mapeo si encontramos coincidencias
    new_columns = {}
    for old_col in df.columns:
        for pattern, new_name in column_mapping.items():
            if pattern in old_col or old_col in pattern:
                new_columns[old_col] = new_name
                break
    
    if new_columns:
        df = df.rename(columns=new_columns)
        logging.info(f"🔄 Columnas normalizadas: {new_columns}")
    
    return df


def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica de datos."""
    
    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    
    # Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how='all')
    
    # Limpiar espacios en strings
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        try:
            # Convertir a string y limpiar espacios de forma segura
            df[col] = df[col].astype(str)
            df[col] = [str(val).strip() if pd.notna(val) and str(val) != 'nan' else np.nan for val in df[col]]
            df[col] = df[col].replace('nan', np.nan)
            df[col] = df[col].replace('', np.nan)
        except Exception as e:
            logging.warning(f"⚠️ No se pudo limpiar la columna {col}: {e}")
            continue
    
    logging.info(f"🧹 Limpieza básica completada. Shape final: {df.shape}")
    return df


def _generate_quality_report(df: pd.DataFrame) -> Dict:
    """Genera reporte de calidad de datos."""
    
    total_registros = len(df)
    
    # Análisis básico de estados si existe la columna
    registros_abiertos = 0
    registros_cerrados = 0
    
    if 'estado' in df.columns:
        estados_abiertos = ['abierto', 'pendiente', 'en proceso', 'activo', 'open', 'pending']
        estados_cerrados = ['cerrado', 'completado', 'finalizado', 'closed', 'completed']
        
        estado_values = df['estado'].fillna('').astype(str).str.lower()
        
        for estado in estados_abiertos:
            registros_abiertos += estado_values.str.contains(estado, na=False).sum()
            
        for estado in estados_cerrados:
            registros_cerrados += estado_values.str.contains(estado, na=False).sum()
    
    # Detección de errores comunes
    errores = {}
    
    # Fechas inválidas
    if 'fecha' in df.columns:
        try:
            pd.to_datetime(df['fecha'], errors='coerce')
            fechas_nulas = df['fecha'].isna().sum()
            if fechas_nulas > 0:
                errores['fechas_invalidas'] = int(fechas_nulas)
        except:
            errores['fechas_no_procesables'] = total_registros
    
    # Valores faltantes por columna
    missing_values = df.isnull().sum()
    critical_missing = missing_values[missing_values > total_registros * 0.5]
    if not critical_missing.empty:
        errores['columnas_criticas_faltantes'] = len(critical_missing)  # Solo el número, no el diccionario
    
    quality_report = {
        'total_registros': total_registros,
        'registros_abiertos': registros_abiertos,
        'registros_cerrados': registros_cerrados,
        'errores': errores,
        'columnas_disponibles': list(df.columns),
        'completitud_general': round((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    }
    
    logging.info(f"📊 Reporte de calidad generado. Completitud: {quality_report['completitud_general']}%")
    return quality_report


def _normalize_data_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza valores de datos."""
    
    df_norm = df.copy()
    
    # Normalizar fechas si existen - Versión robusta
    date_columns = ['fecha', 'fecha_inicio', 'fecha_fin', 'date']
    for col in date_columns:
        if col in df_norm.columns:
            try:
                # Hacer una copia para evitar problemas de índice duplicado
                original_values = df_norm[col].copy().reset_index(drop=True)
                
                # Intentar conversión a datetime
                converted_dates = pd.to_datetime(original_values, errors='coerce')
                
                # Verificar que la conversión fue exitosa
                converted_count = converted_dates.notna().sum()
                total_count = original_values.notna().sum()
                
                if converted_count > 0:
                    # Asignar los valores directamente para evitar problemas de índice
                    df_norm[col] = converted_dates.values
                    logging.info(f"📅 Columna {col} normalizada como fecha ({converted_count}/{total_count} valores convertidos)")
                else:
                    logging.warning(f"⚠️ No se pudieron convertir fechas en {col}, manteniendo como texto")
                    
            except Exception as e:
                logging.warning(f"⚠️ Error normalizando fechas en {col}: {e}")
                # Mantener valores originales en caso de error
    
    # Normalizar códigos de equipo
    if 'equipo' in df_norm.columns:
        try:
            # Normalización segura de equipos
            df_norm['equipo'] = [str(val).upper().strip() if pd.notna(val) else 'UNKNOWN' for val in df_norm['equipo']]
            # Generar FR-30 KPI básico
            df_norm['es_fr30'] = [bool(re.search(r'FR.*30|30.*FR', str(val), re.IGNORECASE)) for val in df_norm['equipo']]
            logging.info("🏷️ Códigos de equipo normalizados y FR-30 identificados")
        except Exception as e:
            logging.warning(f"⚠️ Error normalizando equipos: {e}")
    
    # Normalizar estados
    if 'estado' in df_norm.columns:
        try:
            estado_mapping = {
                'abierto': 'ABIERTO',
                'cerrado': 'CERRADO', 
                'pendiente': 'PENDIENTE',
                'en proceso': 'EN_PROCESO',
                'completado': 'CERRADO',
                'finalizado': 'CERRADO'
            }
            
            # Normalización segura de estados
            estados_normalizados = []
            for val in df_norm['estado']:
                if pd.isna(val):
                    estados_normalizados.append('DESCONOCIDO')
                else:
                    val_lower = str(val).lower().strip()
                    mapped_state = estado_mapping.get(val_lower, val_lower.upper())
                    estados_normalizados.append(mapped_state)
            
            df_norm['estado_normalizado'] = estados_normalizados
            logging.info("🔄 Estados normalizados")
        except Exception as e:
            logging.warning(f"⚠️ Error normalizando estados: {e}")
    
    return df_norm


def _create_catalogs(df: pd.DataFrame) -> Dict:
    """Crea catálogos para análisis ML."""
    
    catalogos = {}
    
    # Catálogo de equipos
    if 'equipo' in df.columns:
        equipos_unicos = df['equipo'].dropna().unique()
        catalogos['equipos'] = {
            'total': len(equipos_unicos),
            'lista': sorted(equipos_unicos.astype(str)),
            'fr30_count': df['es_fr30'].sum() if 'es_fr30' in df.columns else 0
        }
    
    # Catálogo de estados
    if 'estado_normalizado' in df.columns:
        estados_unicos = df['estado_normalizado'].dropna().unique()
        catalogos['estados'] = {
            'total': len(estados_unicos),
            'lista': sorted(estados_unicos.astype(str)),
            'distribucion': df['estado_normalizado'].value_counts().to_dict()
        }
    
    # Estadísticas temporales - Versión robusta
    if 'fecha' in df.columns:
        fechas_validas = df['fecha'].dropna()
        if not fechas_validas.empty:
            try:
                # Verificar si las fechas son datetime o strings
                if pd.api.types.is_datetime64_any_dtype(fechas_validas):
                    # Las fechas son datetime, podemos hacer operaciones temporales
                    catalogos['temporal'] = {
                        'fecha_minima': fechas_validas.min(),
                        'fecha_maxima': fechas_validas.max(),
                        'rango_dias': (fechas_validas.max() - fechas_validas.min()).days,
                        'registros_con_fecha': len(fechas_validas),
                        'tipo_fecha': 'datetime'
                    }
                else:
                    # Las fechas son strings, solo estadísticas básicas
                    muestra_valores = [str(val) for val in fechas_validas.head(3)]
                    catalogos['temporal'] = {
                        'fecha_minima': str(fechas_validas.iloc[0]),
                        'fecha_maxima': str(fechas_validas.iloc[-1]),
                        'rango_dias': 'No calculable (formato texto)',
                        'registros_con_fecha': len(fechas_validas),
                        'tipo_fecha': 'string',
                        'muestra_valores': muestra_valores
                    }
                logging.info("📅 Estadísticas temporales creadas")
            except Exception as e:
                logging.warning(f"⚠️ Error creando estadísticas temporales: {e}")
                # Catálogo temporal básico como fallback
                catalogos['temporal'] = {
                    'registros_con_fecha': len(fechas_validas),
                    'error': str(e)
                }
    
    logging.info(f"📚 Catálogos creados: {list(catalogos.keys())}")
    return catalogos


def get_fr30_analysis(df: pd.DataFrame) -> Dict:
    """Análisis específico para equipos FR-30."""
    
    if 'es_fr30' not in df.columns:
        return {'error': 'Análisis FR-30 no disponible - datos no procesados'}
    
    fr30_data = df[df['es_fr30'] == True]
    
    analysis = {
        'total_registros_fr30': len(fr30_data),
        'porcentaje_fr30': round((len(fr30_data) / len(df)) * 100, 2) if len(df) > 0 else 0,
        'equipos_fr30_unicos': fr30_data['equipo'].nunique() if 'equipo' in fr30_data.columns else 0
    }
    
    # Análisis de estados FR-30
    if 'estado_normalizado' in fr30_data.columns:
        analysis['estados_fr30'] = fr30_data['estado_normalizado'].value_counts().to_dict()
    
    # Análisis temporal FR-30
    if 'fecha' in fr30_data.columns:
        fechas_fr30 = fr30_data['fecha'].dropna()
        if not fechas_fr30.empty:
            try:
                # Verificar si las fechas son datetime
                if pd.api.types.is_datetime64_any_dtype(fechas_fr30):
                    analysis['periodo_fr30'] = {
                        'inicio': fechas_fr30.min(),
                        'fin': fechas_fr30.max(),
                        'registros_con_fecha': len(fechas_fr30),
                        'tipo_fecha': 'datetime'
                    }
                else:
                    # Fechas como string
                    analysis['periodo_fr30'] = {
                        'inicio': str(fechas_fr30.iloc[0]),
                        'fin': str(fechas_fr30.iloc[-1]),
                        'registros_con_fecha': len(fechas_fr30),
                        'tipo_fecha': 'string'
                    }
            except Exception as e:
                analysis['periodo_fr30'] = {
                    'registros_con_fecha': len(fechas_fr30),
                    'error': f'Error procesando fechas: {str(e)}'
                }
    
    return analysis
