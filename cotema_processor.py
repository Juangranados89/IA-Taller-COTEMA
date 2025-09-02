"""
COTEMA Processor - Motor de Procesamiento Especializado para Datos de Mantenimiento
===================================================================================

Este módulo contiene la lógica especializada para procesar y normalizar
datos de mantenimiento del taller COTEMA.

Funciones principales:
- process_cotema_data: Procesamiento principal con normalización inteligente
- _generate_quality_report: Reporte de calidad e integridad de datos
- _create_catalogs: Generación de catálogos para análisis ML
- get_fr30_analysis: Extracto analítico específico FR-30
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re
from typing import Dict, List, Tuple, Any


# --------------------------------------------------------------------------------------
# API PRINCIPAL
# --------------------------------------------------------------------------------------

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
        emergency_dataset = df_raw.head(100).to_dict('records') if df_raw is not None and not df_raw.empty else []
        emergency_report = {
            'total_registros': int(len(df_raw)) if df_raw is not None else 0,
            'registros_abiertos': 0,
            'registros_cerrados': 0,
            'errores': {'processing_error': str(e)}
        }
        emergency_catalogs = {}
        
        return emergency_dataset, emergency_report, emergency_catalogs


# --------------------------------------------------------------------------------------
# NORMALIZACIÓN DE COLUMNAS (ROBUSTA Y SIN COLISIONES)
# --------------------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas (robusto, sin colisiones peligrosas)."""
    df = df.copy()
    df.columns = df.columns.astype(str)
    # Normaliza: minúsculas, subrayado, sin caracteres especiales
    df.columns = [re.sub(r'[^a-zA-Z0-9_]+', '_', c.lower().strip()) for c in df.columns]

    # Mapeo EXÁCTO (no substring). Mantener campos de fecha distintos.
    exact_map = {
        # Equipo
        'equipo': 'equipo',
        'codigo_equipo': 'equipo',
        'cod_equipo': 'equipo',
        'equipment': 'equipo',
        'codigo': 'equipo',  # opcional si a veces viene como "codigo"

        # Fechas (mantener campos separados)
        'fecha': 'fecha',
        'fecha_evento': 'fecha',
        'date': 'fecha',
        'fecha_inicio': 'fecha_inicio',
        'start_date': 'fecha_inicio',
        'inicio': 'fecha_inicio',
        'fecha_fin': 'fecha_fin',
        'end_date': 'fecha_fin',
        'fin': 'fecha_fin',
        'fecha_in': 'fecha_in',
        'fecha_ingreso': 'fecha_in',
        'fecha_entrada': 'fecha_in',
        'in': 'fecha_in',
        'fecha_out': 'fecha_out',
        'fecha_salida': 'fecha_out',
        'out': 'fecha_out',

        # Estado
        'estado': 'estado',
        'status': 'estado',
        'estado_actual': 'estado',

        # Otros
        'descripcion': 'descripcion',
        'description': 'descripcion',
        'detalle': 'descripcion',
        'observaciones': 'observaciones',
        'comments': 'observaciones',
        'tipo': 'tipo',
        'type': 'tipo',
        'prioridad': 'prioridad',
        'priority': 'prioridad',
    }

    # Renombrar solo por coincidencia EXACTA (sin contains)
    df.rename(columns=lambda c: exact_map.get(c, c), inplace=True)

    # Resolver columnas duplicadas combinando por primera no nula por fila
    if df.columns.duplicated().any():
        dups = df.columns[df.columns.duplicated()].unique()
        resolved = {}
        for name in dups:
            block = df.loc[:, df.columns == name]
            combined = block.bfill(axis=1).iloc[:, 0]
            resolved[name] = combined
            # Eliminar todas las columnas duplicadas de ese nombre
            df = df.loc[:, df.columns != name]
        # Reinsertar columna única combinada
        for name, series in resolved.items():
            df[name] = series.values
        logging.warning(f"🔧 Resueltas colisiones de nombres: {list(resolved.keys())}")

    logging.info(f"🔄 Columnas normalizadas. Final: {list(df.columns)}")
    return df


# --------------------------------------------------------------------------------------
# LIMPIEZA BÁSICA
# --------------------------------------------------------------------------------------

def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica de datos."""
    df = df.copy()

    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    # Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how='all')

    # Limpiar espacios en strings
    string_columns = df.select_dtypes(include=['object']).columns
    for col in string_columns:
        try:
            df[col] = df[col].astype(str)
            df[col] = [s.strip() if (pd.notna(s) and s != 'nan') else np.nan for s in df[col]]
            df[col] = df[col].replace({'nan': np.nan, '': np.nan})
        except Exception as e:
            logging.warning(f"⚠️ No se pudo limpiar la columna {col}: {e}")
            continue

    logging.info(f"🧹 Limpieza básica completada. Shape final: {df.shape}")
    return df


# --------------------------------------------------------------------------------------
# QUALITY REPORT (BLINDADO SI 'fecha' ESTÁ DUPLICADA)
# --------------------------------------------------------------------------------------

def _generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Genera reporte de calidad de datos."""
    total_registros = int(len(df))

    # Análisis de estados si existe la columna
    registros_abiertos = 0
    registros_cerrados = 0
    if 'estado' in df.columns:
        estados_abiertos = ['abierto', 'pendiente', 'en proceso', 'activo', 'open', 'pending']
        estados_cerrados = ['cerrado', 'completado', 'finalizado', 'closed', 'completed']
        estado_values = df['estado'].fillna('').astype(str).str.lower()
        for estado in estados_abiertos:
            registros_abiertos += int(estado_values.str.contains(estado, na=False).sum())
        for estado in estados_cerrados:
            registros_cerrados += int(estado_values.str.contains(estado, na=False).sum())

    errores: Dict[str, Any] = {}

    # Fechas inválidas en 'fecha' si existe
    if 'fecha' in df.columns:
        try:
            fecha_col = df['fecha']
            if isinstance(fecha_col, pd.DataFrame):  # columnas duplicadas con el mismo nombre
                fecha_col = fecha_col.bfill(axis=1).iloc[:, 0]
            converted_dates = pd.to_datetime(fecha_col, errors='coerce', dayfirst=True)
            fechas_nulas = int(converted_dates.isna().sum())
            if fechas_nulas > 0:
                errores['fechas_invalidas'] = fechas_nulas
        except Exception as e:
            logging.warning(f"⚠️ Error analizando fechas en quality report: {e}")
            errores['fechas_no_procesables'] = total_registros

    # Valores faltantes por columna
    missing_values = df.isnull().sum()
    critical_missing = missing_values[missing_values > total_registros * 0.5]
    if not critical_missing.empty:
        errores['columnas_criticas_faltantes'] = int(len(critical_missing))

    completitud_general = 100.0
    denom = float(df.shape[0] * max(df.shape[1], 1))
    if denom > 0:
        completitud_general = round((1 - df.isnull().sum().sum() / denom) * 100, 2)

    quality_report: Dict[str, Any] = {
        'total_registros': total_registros,
        'registros_abiertos': registros_abiertos,
        'registros_cerrados': registros_cerrados,
        'errores': errores,
        'columnas_disponibles': list(df.columns),
        'completitud_general': completitud_general
    }

    logging.info(f"📊 Reporte de calidad generado. Completitud: {quality_report['completitud_general']}%")
    return quality_report


# --------------------------------------------------------------------------------------
# NORMALIZACIÓN DE VALORES (FECHAS, EQUIPO, ESTADO)
# --------------------------------------------------------------------------------------

def _normalize_data_values(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza valores de datos (fechas, equipos, estados)."""
    df_norm = df.copy()

    # Extra safety: resolver duplicados de nombres si los hay
    if df_norm.columns.duplicated().any():
        for name in df_norm.columns[df_norm.columns.duplicated()].unique():
            block = df_norm.loc[:, df_norm.columns == name]
            combined = block.bfill(axis=1).iloc[:, 0]
            df_norm = df_norm.loc[:, df_norm.columns != name]
            df_norm[name] = combined.values
        logging.warning("🔧 Colisiones resueltas en _normalize_data_values")

    # 1) Fechas
    date_columns = ['fecha', 'fecha_inicio', 'fecha_fin', 'date', 'fecha_in', 'fecha_out']
    for col in date_columns:
        if col in df_norm.columns:
            try:
                col_data = df_norm[col]
                # Si es DataFrame (duplicados), aplanar
                if isinstance(col_data, pd.DataFrame):
                    col_data = col_data.bfill(axis=1).iloc[:, 0]

                if 'out' in col.lower():
                    # Respetar registros abiertos: solo convertir no-vacíos
                    mask = col_data.notna() & (col_data.astype(str).str.strip() != '')
                    if mask.any():
                        converted = pd.to_datetime(col_data[mask], errors='coerce', dayfirst=True)
                        col_data.loc[mask] = converted
                        df_norm[col] = col_data.values
                        logging.info(f"📅 {col} normalizada ({int(converted.notna().sum())}/{int(mask.sum())})")
                    else:
                        logging.info(f"📅 {col} sin valores; registros abiertos")
                else:
                    converted = pd.to_datetime(col_data, errors='coerce', dayfirst=True)
                    df_norm[col] = converted.values
                    logging.info(f"📅 {col} normalizada ({int(converted.notna().sum())}/{int(col_data.notna().sum())})")
            except Exception as e:
                logging.warning(f"⚠️ Error normalizando fechas en {col}: {e}")
                # mantener original

    # Si no existe 'fecha' pero hay 'fecha_out' o 'fecha_in', crearla (preferir OUT)
    if 'fecha' not in df_norm.columns:
        if 'fecha_out' in df_norm.columns or 'fecha_in' in df_norm.columns:
            fecha_out = df_norm['fecha_out'] if 'fecha_out' in df_norm.columns else pd.Series([pd.NaT] * len(df_norm))
            fecha_in = df_norm['fecha_in'] if 'fecha_in' in df_norm.columns else pd.Series([pd.NaT] * len(df_norm))
            df_norm['fecha'] = fecha_out.fillna(fecha_in)
            df_norm['fecha'] = pd.to_datetime(df_norm['fecha'], errors='coerce', dayfirst=True)
            logging.info("📅 Columna 'fecha' creada como preferencia(fecha_out, fecha_in)")

    # 2) Normalizar códigos de equipo
    if 'equipo' in df_norm.columns:
        try:
            df_norm['equipo'] = [str(val).upper().strip() if pd.notna(val) else 'UNKNOWN' for val in df_norm['equipo']]
            # Tag simple para FR-30 según el código del equipo (opcional)
            df_norm['es_fr30'] = [bool(re.search(r'FR.*30|30.*FR', str(val), re.IGNORECASE)) for val in df_norm['equipo']]
            logging.info("🏷️ Códigos de equipo normalizados y FR-30 identificados")
        except Exception as e:
            logging.warning(f"⚠️ Error normalizando equipos: {e}")

    # 3) Normalizar estados
    if 'estado' in df_norm.columns:
        try:
            estado_mapping = {
                'abierto': 'ABIERTO',
                'cerrado': 'CERRADO',
                'pendiente': 'PENDIENTE',
                'en proceso': 'EN_PROCESO',
                'completado': 'CERRADO',
                'finalizado': 'CERRADO',
                'open': 'ABIERTO',
                'closed': 'CERRADO',
                'completed': 'CERRADO',
                'active': 'ABIERTO',
                'pending': 'PENDIENTE',
            }
            estados_normalizados: List[str] = []
            for val in df_norm['estado']:
                if pd.isna(val):
                    estados_normalizados.append('DESCONOCIDO')
                else:
                    raw = str(val).lower().strip()
                    estados_normalizados.append(estado_mapping.get(raw, raw.upper()))
            df_norm['estado_normalizado'] = estados_normalizados
            logging.info("🔄 Estados normalizados")
        except Exception as e:
            logging.warning(f"⚠️ Error normalizando estados: {e}")

    return df_norm


# --------------------------------------------------------------------------------------
# CATÁLOGOS PARA ML
# --------------------------------------------------------------------------------------

def _create_catalogs(df: pd.DataFrame) -> Dict[str, Any]:
    """Crea catálogos para análisis ML."""
    catalogos: Dict[str, Any] = {}

    # Catálogo de equipos
    if 'equipo' in df.columns:
        equipos_unicos = df['equipo'].dropna().astype(str).unique()
        catalogos['equipos'] = {
            'total': int(len(equipos_unicos)),
            'lista': sorted(map(str, equipos_unicos)),
            'fr30_count': int(df['es_fr30'].sum()) if 'es_fr30' in df.columns else 0
        }

    # Catálogo de estados
    if 'estado_normalizado' in df.columns:
        estados_unicos = df['estado_normalizado'].dropna().astype(str).unique()
        catalogos['estados'] = {
            'total': int(len(estados_unicos)),
            'lista': sorted(map(str, estados_unicos)),
            'distribucion': {str(k): int(v) for k, v in df['estado_normalizado'].value_counts(dropna=False).to_dict().items()}
        }

    # Estadísticas temporales
    if 'fecha' in df.columns:
        fechas_validas = df['fecha'].dropna()
        if not fechas_validas.empty:
            try:
                if pd.api.types.is_datetime64_any_dtype(fechas_validas):
                    catalogos['temporal'] = {
                        'fecha_minima': pd.to_datetime(fechas_validas.min()).isoformat(),
                        'fecha_maxima': pd.to_datetime(fechas_validas.max()).isoformat(),
                        'rango_dias': int((fechas_validas.max() - fechas_validas.min()).days),
                        'registros_con_fecha': int(len(fechas_validas)),
                        'tipo_fecha': 'datetime'
                    }
                else:
                    muestra_valores = [str(val) for val in fechas_validas.head(3)]
                    catalogos['temporal'] = {
                        'fecha_minima': str(fechas_validas.iloc[0]),
                        'fecha_maxima': str(fechas_validas.iloc[-1]),
                        'rango_dias': 'No calculable (formato texto)',
                        'registros_con_fecha': int(len(fechas_validas)),
                        'tipo_fecha': 'string',
                        'muestra_valores': muestra_valores
                    }
                logging.info("📅 Estadísticas temporales creadas")
            except Exception as e:
                logging.warning(f"⚠️ Error creando estadísticas temporales: {e}")
                catalogos['temporal'] = {
                    'registros_con_fecha': int(len(fechas_validas)),
                    'error': str(e)
                }

    logging.info(f"📚 Catálogos creados: {list(catalogos.keys())}")
    return catalogos


# --------------------------------------------------------------------------------------
# ANÁLISIS ESPECÍFICO FR-30
# --------------------------------------------------------------------------------------

def get_fr30_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """Análisis específico para equipos FR-30."""
    if 'es_fr30' not in df.columns:
        return {'error': 'Análisis FR-30 no disponible - datos no procesados'}

    fr30_data = df[df['es_fr30'] == True]

    analysis: Dict[str, Any] = {
        'total_registros_fr30': int(len(fr30_data)),
        'porcentaje_fr30': round((len(fr30_data) / len(df)) * 100, 2) if len(df) > 0 else 0.0,
        'equipos_fr30_unicos': int(fr30_data['equipo'].nunique()) if 'equipo' in fr30_data.columns else 0
    }

    # Estados FR-30
    if 'estado_normalizado' in fr30_data.columns:
        analysis['estados_fr30'] = {str(k): int(v) for k, v in fr30_data['estado_normalizado'].value_counts().to_dict().items()}

    # Periodo FR-30
    if 'fecha' in fr30_data.columns:
        fechas_fr30 = fr30_data['fecha'].dropna()
        if not fechas_fr30.empty:
            try:
                if pd.api.types.is_datetime64_any_dtype(fechas_fr30):
                    analysis['periodo_fr30'] = {
                        'inicio': pd.to_datetime(fechas_fr30.min()).isoformat(),
                        'fin': pd.to_datetime(fechas_fr30.max()).isoformat(),
                        'registros_con_fecha': int(len(fechas_fr30)),
                        'tipo_fecha': 'datetime'
                    }
                else:
                    analysis['periodo_fr30'] = {
                        'inicio': str(fechas_fr30.iloc[0]),
                        'fin': str(fechas_fr30.iloc[-1]),
                        'registros_con_fecha': int(len(fechas_fr30)),
                        'tipo_fecha': 'string'
                    }
            except Exception as e:
                analysis['periodo_fr30'] = {
                    'registros_con_fecha': int(len(fechas_fr30)),
                    'error': f'Error procesando fechas: {str(e)}'
                }

    return analysis
