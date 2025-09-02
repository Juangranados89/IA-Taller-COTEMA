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


def _sanitize_columns(cols: List[str]) -> List[str]:
    """Normaliza nombres: lower, sin acentos, separadores a _, solo [a-z0-9_]."""
    out = []
    for c in cols:
        c = str(c).strip()
        c = _strip_accents(c).lower()
        c = re.sub(r"[^a-z0-9]+", "_", c)  # no letras/numeros -> _
        c = c.strip("_")
        out.append(c or "col")
    return out


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
