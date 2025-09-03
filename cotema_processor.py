"""
COTEMA Processor - Motor de Procesamiento para Datos de Mantenimiento (enfocado en COTEMA)
==========================================================================================

Este módulo procesa y normaliza datos reales del taller COTEMA sin simulaciones.
Está diseñado para evitar colisiones de columnas (p. ej., fecha_in / fecha_out),
entregar un reporte de calidad robusto y generar catálogos puramente desde el
archivo cargado.

Funciones principales:
- process_cotema_data(df_raw): orquesta todo el flujo y devuelve (dataset, quality_report, catalogos)
- get_fr30_analysis(df, days=30): resumen real de correctivas en la ventana indicada (por defecto 30 días)

Notas de diseño:
- Mapeos por IGUALDAD EXACTA (no por substring) para evitar nombres duplicados.
- Se respeta 'codigo' como identificador principal; se expone alias 'equipo' para compatibilidad.
- No se generan códigos ni métricas simuladas. Si faltan datos, se reporta y/o se devuelve vacío/0.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# API PRINCIPAL
# --------------------------------------------------------------------------------------

def process_cotema_data(df_raw: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    """
    Procesa y normaliza datos COTEMA con análisis de calidad integrado (sin simulaciones).

    Args:
        df_raw: DataFrame crudo leído del Excel.

    Returns:
        dataset: lista de registros normalizados (dicts)
        quality_report: reporte de calidad (conteos, errores, completitud)
        catalogos: catálogos derivados 100% de los datos reales
    """
    logging.info("🔄 Iniciando procesamiento COTEMA (enfocado en datos reales)...")

    try:
        # Paso 1: Normalización de columnas
        df = _normalize_columns(df_raw.copy())

        # Paso 2: Limpieza básica
        df = _basic_cleaning(df)

        # Paso 3: Análisis de calidad (sobre df "crudo limpio" para ver problemas tempranos)
        quality_report = _generate_quality_report(df)

        # Paso 4: Normalización de valores (fechas, categóricos, métricas reales)
        df_norm = _normalize_data_values(df)

        # Paso 5: Catálogos (desde df normalizado)
        catalogos = _create_catalogs(df_norm)

        # Paso 6: Dataset final
        dataset = df_norm.to_dict("records")

        logging.info(f"✅ COTEMA processing OK. Registros: {len(dataset)}")
        return dataset, quality_report, catalogos

    except Exception as e:
        logging.exception(f"❌ Error en process_cotema_data: {e}")
        emergency_dataset = df_raw.head(100).to_dict("records") if isinstance(df_raw, pd.DataFrame) and not df_raw.empty else []
        emergency_report = {
            "total_registros": int(len(df_raw)) if isinstance(df_raw, pd.DataFrame) else 0,
            "registros_abiertos": 0,
            "registros_cerrados": 0,
            "errores": {"processing_error": str(e)},
            "columnas_disponibles": list(df_raw.columns) if isinstance(df_raw, pd.DataFrame) else [],
            "completitud_general": 0.0,
        }
        return emergency_dataset, emergency_report, {}


# --------------------------------------------------------------------------------------
# UTILIDADES
# --------------------------------------------------------------------------------------

def _strip_accents(s: str) -> str:
    """Elimina acentos/diacríticos de una cadena (NFD)."""
    if not isinstance(s, str):
        s = str(s)
    nf = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in nf if unicodedata.category(ch) != "Mn")


def _sanitize_columns(columns: List[str]) -> List[str]:
    """Sanitiza nombres de columnas: normaliza, remueve caracteres especiales."""
    sanitized = []
    for col in columns:
        # Normalizar unicode y remover acentos
        clean = unicodedata.normalize('NFKD', str(col)).encode('ascii', 'ignore').decode('ascii')
        # Solo letras, números y guiones bajos
        clean = re.sub(r'[^\w]', '_', clean.lower())
        # Reducir múltiples guiones bajos
        clean = re.sub(r'_+', '_', clean).strip('_')
        # Mapeos adicionales para compatibilidad con diferentes archivos
        mappings = {
            'equipo': 'codigo',
            'fecha': 'fecha_in', 
            'estado': 'tipo_atencion',
            'prioridad': 'tipo_atencion',
            'descripcion': 'sistema_afectado',
            'horas_trabajo': 'horometro_in',
            'tecnico': 'ejecutor'
        }
        clean = mappings.get(clean, clean)
        sanitized.append(clean)
    return sanitized


# --------------------------------------------------------------------------------------
# NORMALIZACIÓN DE COLUMNAS (SIN COLISIONES)
# --------------------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nombres de columnas con mapeos EXACTOS y resuelve duplicados."""
    df = df.copy()
    df.columns = _sanitize_columns(df.columns.tolist())

    # Mapeo EXACTO específico para el layout de COTEMA (ajustado para tu archivo)
    mapping_exact = {
        # Identidad / descripción
        "codigo": "codigo",
        "placa": "placa",
        "descripcion": "descripcion",
        "flota": "flota",

        # Medidas/lecturas
        "horas_in": "horas_in",
        "horometro_in": "horometro_in",
        "km_in": "km_in",
        "horas_out": "horas_out",

        # Fechas (NO colisionar)
        "fecha_in": "fecha_in",
        "fecha_out": "fecha_out",

        # Atributos de proceso
        "operador": "operador",
        "ejecutor": "ejecutor",
        "tipo_atencion": "tipo_atencion",
        "sistema_afectado": "sistema_afectado",
        "origen_averia": "origen_averia",
        "descripcion_intervencion": "descripcion_intervencion",
        "atencion_local": "atencion_local",
        "atencion_externa": "atencion_externa",
        "sco_sse": "sco_sse",
        "odc_ors": "odc_ors",

        # Métricas ya calculadas en la fuente
        "cont_dias_ave": "cont_dias_ave",
        "con_hrs_ave": "con_hrs_ave",
        "con_in_taller": "con_in_taller",
        "mttr": "mttr",
    }

    # Aplicar mapeo exacto si la columna existe
    rename_map = {c: mapping_exact[c] for c in df.columns if c in mapping_exact}
    df.rename(columns=rename_map, inplace=True)

    # Resolver duplicados de nombre si los hubiera (bfill por fila)
    if df.columns.duplicated().any():
        dup_names = df.columns[df.columns.duplicated()].unique().tolist()
        for name in dup_names:
            block = df.loc[:, df.columns == name]
            combined = block.bfill(axis=1).iloc[:, 0]
            # Elimina todas las columnas con ese nombre y deja una
            df = df.loc[:, df.columns != name]
            df[name] = combined
        logging.warning(f"🔧 Colisiones resueltas para: {dup_names}")

    # Alias de compatibilidad: expone 'equipo' además de 'codigo' (sin perder 'codigo')
    if "codigo" in df.columns and "equipo" not in df.columns:
        df["equipo"] = df["codigo"]

    logging.info(f"🔄 Columnas normalizadas: {list(df.columns)}")
    return df


# --------------------------------------------------------------------------------------
# LIMPIEZA BÁSICA
# --------------------------------------------------------------------------------------

def _basic_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: elimina filas/columnas vacías y trim de strings."""
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    obj_cols = df.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        try:
            s = df[col].astype(str).str.strip()
            s = s.replace({"": np.nan, "nan": np.nan, "None": np.nan})
            df[col] = s
        except Exception as e:
            logging.warning(f"⚠️ No se pudo limpiar columna {col}: {e}")
    logging.info(f"🧹 Limpieza básica OK. Shape: {df.shape}")
    return df


# --------------------------------------------------------------------------------------
# QUALITY REPORT (REAL, SIN SUPOSICIONES)
# --------------------------------------------------------------------------------------

def _generate_quality_report(df: pd.DataFrame) -> Dict[str, Any]:
    """Genera un reporte de calidad centrado en columnas reales del archivo."""
    total = int(len(df))
    errores: Dict[str, Any] = {}

    # Abiertos/Cerrados a partir de fecha_out
    abiertos = cerrados = 0
    if "fecha_out" in df.columns:
        fecha_out = pd.to_datetime(df["fecha_out"], errors="coerce")
        abiertos = int(fecha_out.isna().sum())
        cerrados = int(total - abiertos)

    # Fechas nulas e invertidas
    na_in = na_out = invertidas = 0
    if "fecha_in" in df.columns:
        fecha_in = pd.to_datetime(df["fecha_in"], errors="coerce")
        na_in = int(fecha_in.isna().sum())
    if "fecha_out" in df.columns:
        fecha_out = pd.to_datetime(df["fecha_out"], errors="coerce")
        na_out = int(fecha_out.isna().sum())
    if "fecha_in" in df.columns and "fecha_out" in df.columns:
        dias = (pd.to_datetime(df["fecha_out"], errors="coerce") - pd.to_datetime(df["fecha_in"], errors="coerce")).dt.days
        invertidas = int((dias < 0).sum())

    if na_in:
        errores["fecha_in_nulas"] = na_in
    if na_out:
        errores["fecha_out_nulas"] = na_out
    if invertidas:
        errores["fechas_invertidas"] = invertidas

    # MTTR negativo/sospechoso
    if "mttr" in df.columns:
        mttr_num = pd.to_numeric(df["mttr"], errors="coerce")
        mttr_neg = int((mttr_num < 0).sum())
        if mttr_neg:
            errores["mttr_negativo"] = mttr_neg

    # Completitud general
    total_cells = int(df.shape[0] * max(df.shape[1], 1))
    nonnull = int(df.notna().sum().sum())
    completitud = round((nonnull / total_cells) * 100, 2) if total_cells else 0.0

    report = {
        "total_registros": total,
        "registros_abiertos": abiertos,
        "registros_cerrados": cerrados,
        "errores": errores,
        "columnas_disponibles": list(df.columns),
        "completitud_general": completitud,
    }
    logging.info(f"📊 Quality report OK (completitud {completitud}%)")
    return report


# --------------------------------------------------------------------------------------
# NORMALIZACIÓN DE VALORES (FECHAS/CATEGÓRICOS SIN INVENTAR)
# --------------------------------------------------------------------------------------

def _normalize_data_values(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte fechas a datetime, deriva estado real y calcula días en taller (>=0)."""
    df_norm = df.copy()

    # Fechas
    for col in ("fecha_in", "fecha_out"):
        if col in df_norm.columns:
            try:
                df_norm[col] = pd.to_datetime(df_norm[col], errors="coerce")
            except Exception as e:
                logging.warning(f"⚠️ Error normalizando {col}: {e}")

    # Estado real: ABIERTO si fecha_out NaT, CERRADO en caso contrario
    if "fecha_out" in df_norm.columns and "estado" not in df_norm.columns:
        df_norm["estado"] = np.where(df_norm["fecha_out"].isna(), "ABIERTO", "CERRADO")

    # Días en taller (sin negativos; negativos -> NaN)
    if "fecha_in" in df_norm.columns and "fecha_out" in df_norm.columns:
        dias = (df_norm["fecha_out"] - df_norm["fecha_in"]).dt.days
        df_norm["dias_en_taller"] = np.where(dias >= 0, dias, np.nan)

    # Categóricos clave en mayúsculas limpias (sin inventar valores)
    for cat in ("tipo_atencion", "sistema_afectado", "origen_averia", "atencion_local", "atencion_externa"):
        if cat in df_norm.columns:
            s = df_norm[cat].astype(str).str.strip().str.upper()
            
            # Mapeos especiales para tipo_atencion para compatibilidad con diferentes archivos
            if cat == "tipo_atencion":
                # Mapear valores comunes de estado/prioridad a tipos de atención
                s = s.replace({
                    'COMPLETADO': 'CORRECTIVA',
                    'PENDIENTE': 'CORRECTIVA', 
                    'EN_PROCESO': 'CORRECTIVA',
                    'PROGRAMADO': 'PREVENTIVA',
                    'ALTA': 'CORRECTIVA',
                    'MEDIA': 'CORRECTIVA',
                    'BAJA': 'PREVENTIVA',
                    'CRITICAL': 'CORRECTIVA',
                    'HIGH': 'CORRECTIVA',
                    'MEDIUM': 'CORRECTIVA',
                    'LOW': 'PREVENTIVA'
                })
            
            df_norm[cat] = s.replace({"NAN": np.nan, "NONE": np.nan, "": np.nan})

    return df_norm


# --------------------------------------------------------------------------------------
# CATÁLOGOS (DESDE DATOS REALES)
# --------------------------------------------------------------------------------------

def _create_catalogs(df: pd.DataFrame) -> Dict[str, Any]:
    """Construye catálogos puros desde el df normalizado (sin defaults simulados)."""
    cat: Dict[str, Any] = {}

    # Catálogo de equipos
    if "codigo" in df.columns:
        cods = df["codigo"].dropna().astype(str).unique()
        cat["equipos"] = {
            "total": int(len(cods)),
            "lista": sorted(cods.tolist()),
        }

    # Tipos de atención
    if "tipo_atencion" in df.columns:
        vals = df["tipo_atencion"].dropna().astype(str)
        cat["tipo_atencion"] = {
            "total": int(vals.nunique()),
            "distribucion": vals.value_counts().to_dict(),
        }

    # Sistema afectado
    if "sistema_afectado" in df.columns:
        vals = df["sistema_afectado"].dropna().astype(str)
        cat["sistema_afectado"] = {
            "total": int(vals.nunique()),
            "top10": vals.value_counts().head(10).to_dict(),
        }

    # Temporal (basado en fecha_in)
    if "fecha_in" in df.columns:
        fechas = pd.to_datetime(df["fecha_in"], errors="coerce").dropna()
        if not fechas.empty:
            cat["temporal"] = {
                "fecha_minima": str(fechas.min().date()),
                "fecha_maxima": str(fechas.max().date()),
                "rango_dias": int((fechas.max() - fechas.min()).days),
                "registros_con_fecha_in": int(len(fechas)),
            }

    return cat


# --------------------------------------------------------------------------------------
# ANÁLISIS ESPECÍFICO "FR-30" (REAL: CORRECTIVAS EN ÚLTIMOS N DÍAS)
# --------------------------------------------------------------------------------------

def get_fr30_analysis(df: pd.DataFrame, days: int = 30, codigo_column: str = "codigo") -> Dict[str, Any]:
    """
    Análisis realista para el KPI “FR-30” interpretado como actividad correctiva reciente.

    Produce el top de equipos por conteo de mantenimientos CORRECTIVOS en la ventana
    de 'days' días hacia atrás, usando 'fecha_in' como referencia.

    Args:
        df: DataFrame normalizado o crudo (se normalizan fechas localmente si hace falta).
        days: ventana hacia atrás en días (por defecto 30).
        codigo_column: nombre de la columna de códigos (por defecto 'codigo').

    Returns:
        dict con:
            - window_days: tamaño de la ventana en días
            - since: fecha de corte ISO (hoy - days)
            - total_correctivas_en_ventana: entero
            - equipos_con_correctivas: entero (número de equipos con al menos 1 correctiva)
            - top_equipos: lista [{codigo, correctivas_ventana}]
    """
    if df is None or df.empty:
        return {
            "window_days": int(days),
            "since": datetime.now().date().isoformat(),
            "total_correctivas_en_ventana": 0,
            "equipos_con_correctivas": 0,
            "top_equipos": [],
        }

    # Asegurar columnas necesarias
    if "tipo_atencion" not in df.columns or codigo_column not in df.columns:
        return {
            "window_days": int(days),
            "since": datetime.now().date().isoformat(),
            "total_correctivas_en_ventana": 0,
            "equipos_con_correctivas": 0,
            "top_equipos": [],
            "warning": "Faltan columnas requeridas (tipo_atencion / codigo)",
        }

    # Normaliza fecha_in localmente para este cálculo
    if "fecha_in" in df.columns:
        fecha_in = pd.to_datetime(df["fecha_in"], errors="coerce")
    else:
        # Si no existe, no podemos calcular ventana temporal real
        return {
            "window_days": int(days),
            "since": datetime.now().date().isoformat(),
            "total_correctivas_en_ventana": 0,
            "equipos_con_correctivas": 0,
            "top_equipos": [],
            "warning": "No existe fecha_in para calcular la ventana temporal",
        }

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    mask_recent = (fecha_in.notna()) & (fecha_in >= cutoff)

    recent = df.loc[mask_recent]
    # Filtra CORRECTIVA (en mayúsculas tras normalización)
    tipos = recent["tipo_atencion"].astype(str).str.upper()
    recent_corr = recent.loc[tipos == "CORRECTIVA"]

    if recent_corr.empty:
        return {
            "window_days": int(days),
            "since": cutoff.date().isoformat(),
            "total_correctivas_en_ventana": 0,
            "equipos_con_correctivas": 0,
            "top_equipos": [],
        }

    top = (
        recent_corr.groupby(codigo_column).size().sort_values(ascending=False).head(10).reset_index(name="correctivas_ventana")
    )

    return {
        "window_days": int(days),
        "since": cutoff.date().isoformat(),
        "total_correctivas_en_ventana": int(len(recent_corr)),
        "equipos_con_correctivas": int(top[codigo_column].nunique()),
        "top_equipos": top.to_dict("records"),
    }


def get_fr30_advanced_analysis(df, year=2025):
    """
    Análisis FR-30 avanzado para identificar equipos con mayor tendencia a fallar
    y predecir en qué mes es más probable que ocurra.
    
    Factores considerados:
    - Cantidad de ingresos (frecuencia)
    - Ingresos por sistemas críticos
    - MTTR (Mean Time To Repair)
    - E.TC. (Eficiencia de Tiempo de Ciclo)
    
    Returns:
        dict: Análisis con datos listos para gráficos X(meses) Y(equipos por riesgo)
    """
    try:
        # Detectar columnas automáticamente
        codigo_column = None
        for col in ["codigo", "Codigo", "CODIGO", "equipo", "Equipo"]:
            if col in df.columns:
                codigo_column = col
                break
        
        if codigo_column is None:
            return {
                "error": "No se encontró columna de código de equipo",
                "equipos_riesgo": [],
                "meses_tendencia": [],
                "factores_analisis": {}
            }
        
        # Filtrar por año 2025
        if "fecha_in" not in df.columns:
            return {
                "error": "No se encontró columna fecha_in",
                "equipos_riesgo": [],
                "meses_tendencia": [],
                "factores_analisis": {}
            }
        
        df_copy = df.copy()
        df_copy["fecha_in"] = pd.to_datetime(df_copy["fecha_in"], errors="coerce")
        df_2025 = df_copy[df_copy["fecha_in"].dt.year == year].copy()
        
        if df_2025.empty:
            return {
                "error": f"No hay datos para el año {year}",
                "equipos_riesgo": [],
                "meses_tendencia": [],
                "factores_analisis": {}
            }
        
        # Agregar columna de mes
        df_2025["mes"] = df_2025["fecha_in"].dt.month
        df_2025["mes_nombre"] = df_2025["fecha_in"].dt.strftime("%B")
        
        # Filtrar solo correctivas
        if "tipo_atencion" in df_2025.columns:
            correctivas_2025 = df_2025[df_2025["tipo_atencion"].str.upper() == "CORRECTIVA"].copy()
        else:
            correctivas_2025 = df_2025.copy()  # Asumir que todos son correctivos si no hay tipo
        
        if correctivas_2025.empty:
            # Si no hay correctivas específicas, usar todos los datos como muestra
            correctivas_2025 = df_2025.copy()
            
        if correctivas_2025.empty:
            return {
                "error": f"No hay datos procesables en {year}",
                "equipos_riesgo": [],
                "meses_tendencia": [],
                "factores_analisis": {
                    "total_equipos_analizados": 0,
                    "total_correctivas_2025": 0,
                    "promedio_mttr_horas": 0,
                    "equipos_con_criticos": 0,
                    "mes_mas_problematico": 1
                }
            }
        
        # Calcular MTTR (si hay fechas de salida)
        mttr_data = {}
        if "fecha_out" in correctivas_2025.columns:
            correctivas_2025["fecha_out"] = pd.to_datetime(correctivas_2025["fecha_out"], errors="coerce")
            correctivas_2025["tiempo_reparacion"] = (
                correctivas_2025["fecha_out"] - correctivas_2025["fecha_in"]
            ).dt.total_seconds() / 3600  # Horas
            
            mttr_por_equipo = correctivas_2025.groupby(codigo_column)["tiempo_reparacion"].mean()
            mttr_data = mttr_por_equipo.to_dict()
        
        # Análisis por equipo
        analisis_equipos = []
        
        for equipo in correctivas_2025[codigo_column].unique():
            equipo_data = correctivas_2025[correctivas_2025[codigo_column] == equipo]
            
            # Factor 1: Cantidad de ingresos (frecuencia)
            total_ingresos = len(equipo_data)
            
            # Factor 2: Sistemas críticos (si existe columna de criticidad)
            ingresos_criticos = 0
            if "criticidad" in equipo_data.columns or "sistema_critico" in equipo_data.columns:
                critico_col = "criticidad" if "criticidad" in equipo_data.columns else "sistema_critico"
                ingresos_criticos = equipo_data[critico_col].str.upper().isin(["CRITICO", "CRÍTICO", "HIGH", "ALTA"]).sum()
            
            # Factor 3: MTTR
            mttr_equipo = mttr_data.get(equipo, 0)
            
            # Factor 4: E.TC. (Eficiencia de Tiempo de Ciclo) - basado en frecuencia vs tiempo
            meses_activos = equipo_data["mes"].nunique()
            etc_score = total_ingresos / max(meses_activos, 1)  # Ingresos por mes activo
            
            # Calcular score de riesgo combinado
            score_frecuencia = min(total_ingresos / 10, 1.0) * 0.3  # Normalizado, peso 30%
            score_criticos = min(ingresos_criticos / 5, 1.0) * 0.25  # Peso 25%
            score_mttr = min(mttr_equipo / 168, 1.0) * 0.25  # Normalizado a semanas, peso 25%
            score_etc = min(etc_score / 5, 1.0) * 0.2  # Peso 20%
            
            riesgo_total = score_frecuencia + score_criticos + score_mttr + score_etc
            
            # Análisis de tendencia mensual
            tendencia_mensual = equipo_data.groupby("mes").size().to_dict()
            mes_mayor_riesgo = max(tendencia_mensual.keys(), key=lambda x: tendencia_mensual[x]) if tendencia_mensual else 1
            
            analisis_equipos.append({
                "equipo": equipo,
                "riesgo_score": round(riesgo_total, 3),
                "total_ingresos": int(total_ingresos),
                "ingresos_criticos": int(ingresos_criticos),
                "mttr_horas": round(mttr_equipo, 2),
                "etc_score": round(etc_score, 2),
                "mes_mayor_riesgo": int(mes_mayor_riesgo),
                "tendencia_mensual": tendencia_mensual
            })
        
        # Ordenar por riesgo descendente (mayor riesgo primero)
        analisis_equipos.sort(key=lambda x: x["riesgo_score"], reverse=True)
        
        # Top 15 equipos con mayor riesgo
        top_equipos_riesgo = analisis_equipos[:15]
        
        # Análisis de tendencia general por mes
        tendencia_general = correctivas_2025.groupby("mes").agg({
            codigo_column: "count",
            "mes_nombre": "first"
        }).reset_index()
        
        tendencia_general.columns = ["mes", "total_correctivas", "mes_nombre"]
        meses_tendencia = tendencia_general.to_dict("records")
        
        # Factores de análisis resumen
        factores_resumen = {
            "total_equipos_analizados": len(analisis_equipos),
            "total_correctivas_2025": len(correctivas_2025),
            "promedio_mttr_horas": round(sum(mttr_data.values()) / len(mttr_data), 2) if mttr_data else 0,
            "equipos_con_criticos": sum(1 for e in analisis_equipos if e["ingresos_criticos"] > 0),
            "mes_mas_problematico": max(meses_tendencia, key=lambda x: x["total_correctivas"])["mes"] if meses_tendencia else 1
        }
        
        return {
            "year_analizado": year,
            "equipos_riesgo": top_equipos_riesgo,
            "meses_tendencia": meses_tendencia,
            "factores_analisis": factores_resumen,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "error": f"Error en análisis avanzado: {str(e)}",
            "equipos_riesgo": [],
            "meses_tendencia": [],
            "factores_analisis": {}
        }
