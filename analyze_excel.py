#!/usr/bin/env python3
"""
Análisis específico del archivo Excel COTEMA - Pestaña REG
Enfoque en el rango B5:Y5 hacia abajo
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

def analyze_cotema_reg_sheet(file_path):
    """
    Analiza específicamente la pestaña REG en el rango B5:Y5 hacia abajo
    """
    print("🔍 ANÁLISIS ESPECÍFICO - PESTAÑA REG (B5:Y)")
    print("=" * 60)
    
    try:
        # Cargar específicamente la pestaña REG
        print("📂 Cargando pestaña 'REG'...")
        
        # Leer solo la pestaña REG, empezando desde la fila 5 (índice 4)
        df = pd.read_excel(file_path, sheet_name='REG', skiprows=4)
        
        # Tomar solo las columnas desde B hasta Y (asumiendo que A es índice 0, B es 1, Y es 24)
        # Pandas cuenta columnas desde 0, así que B=1, Y=24
        # Pero al usar skiprows=4, la primera columna será la que estaba en B5
        
        print(f"📊 DATOS CARGADOS:")
        print(f"  • Filas totales: {len(df):,}")
        print(f"  • Columnas: {len(df.columns)}")
        
        # Mostrar las primeras columnas para entender la estructura
        print(f"\n🏗️ ESTRUCTURA DE COLUMNAS DETECTADA:")
        for i, col in enumerate(df.columns, 1):
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            null_count = df[col].isnull().sum()
            if non_null > 0:  # Solo mostrar columnas con datos
                sample_val = df[col].dropna().iloc[0] if non_null > 0 else "N/A"
                print(f"  {i:2d}. {col:<25} | {str(dtype):<12} | Datos: {non_null:4d} | Ejemplo: {sample_val}")
        
        # Filtrar solo filas que tienen datos (no completamente vacías)
        df_clean = df.dropna(how='all')
        print(f"\n🧹 DESPUÉS DE LIMPIAR FILAS VACÍAS:")
        print(f"  • Filas con datos: {len(df_clean):,}")
        
        # Mostrar muestra de datos reales
        print(f"\n📋 MUESTRA DE DATOS REALES (Primeras 5 filas):")
        if len(df_clean) > 0:
            # Mostrar solo las primeras 8 columnas para que sea legible
            sample_df = df_clean.head(5).iloc[:, :8]
            print(sample_df.to_string(max_colwidth=15))
        
        # Análisis específico de campos esperados para taller
        print(f"\n🔍 ANÁLISIS DE CAMPOS PARA TALLER COTEMA:")
        
        columns_found = {}
        
        # Buscar patrones específicos en nombres de columnas
        patterns = {
            'codigo_equipo': ['CODIGO', 'COD', 'EQUIPO', 'EQUIPMENT', 'ID'],
            'fecha_ingreso': ['FECHA', 'INGRESO', 'ENTRADA', 'IN', 'INICIO'],
            'fecha_salida': ['SALIDA', 'OUT', 'FIN', 'TERMINO'],
            'sistema': ['SISTEMA', 'SYSTEM', 'TIPO', 'CATEGORIA'],
            'descripcion': ['DESCRIPCION', 'DETALLE', 'OBSERV', 'NOTA'],
            'tiempo': ['TIEMPO', 'HORA', 'DURACION', 'TIME'],
            'estado': ['ESTADO', 'STATUS', 'CONDICION']
        }
        
        for category, keywords in patterns.items():
            found_cols = []
            for col in df_clean.columns:
                col_upper = str(col).upper()
                if any(keyword in col_upper for keyword in keywords):
                    found_cols.append(col)
            
            if found_cols:
                columns_found[category] = found_cols
                print(f"\n  🎯 {category.upper().replace('_', ' ')}:")
                for col in found_cols:
                    unique_vals = df_clean[col].nunique()
                    sample_data = df_clean[col].dropna().head(3).tolist()
                    print(f"     • {col}: {unique_vals} valores únicos")
                    print(f"       Ejemplos: {sample_data}")
        
        # Análisis temporal si encontramos fechas
        if 'fecha_ingreso' in columns_found or 'fecha_salida' in columns_found:
            print(f"\n📅 ANÁLISIS TEMPORAL:")
            
            date_columns = []
            if 'fecha_ingreso' in columns_found:
                date_columns.extend(columns_found['fecha_ingreso'])
            if 'fecha_salida' in columns_found:
                date_columns.extend(columns_found['fecha_salida'])
            
            for date_col in date_columns[:2]:  # Analizar máximo 2 columnas de fecha
                try:
                    # Intentar convertir a fecha
                    dates = pd.to_datetime(df_clean[date_col], errors='coerce')
                    valid_dates = dates.dropna()
                    
                    if len(valid_dates) > 0:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        total_days = (max_date - min_date).days
                        
                        print(f"\n  📊 {date_col}:")
                        print(f"     Período: {min_date.strftime('%Y-%m-%d')} a {max_date.strftime('%Y-%m-%d')}")
                        print(f"     Duración: {total_days} días ({total_days/365.25:.1f} años)")
                        print(f"     Registros válidos: {len(valid_dates):,}")
                        
                        # Análisis por año
                        years = valid_dates.dt.year.value_counts().sort_index()
                        print(f"     Distribución por año:")
                        for year, count in years.items():
                            print(f"       {year}: {count:,} registros")
                        
                except Exception as e:
                    print(f"     ❌ Error procesando {date_col}: {str(e)}")
        
        # Análisis de equipos si encontramos código de equipo
        if 'codigo_equipo' in columns_found:
            print(f"\n🏭 ANÁLISIS DE EQUIPOS:")
            equipo_col = columns_found['codigo_equipo'][0]
            
            equipos_stats = df_clean[equipo_col].value_counts()
            total_equipos = equipos_stats.nunique() if not equipos_stats.empty else 0
            
            print(f"  Total equipos únicos: {total_equipos}")
            
            if total_equipos > 0:
                print(f"  Registros por equipo:")
                print(f"    Promedio: {equipos_stats.mean():.1f}")
                print(f"    Mediana: {equipos_stats.median():.1f}")
                print(f"    Mínimo: {equipos_stats.min()}")
                print(f"    Máximo: {equipos_stats.max()}")
                
                print(f"\n  Top 10 equipos con más ingresos:")
                for i, (equipo, count) in enumerate(equipos_stats.head(10).items(), 1):
                    print(f"    {i:2d}. {equipo}: {count} ingresos")
        
        # Análisis de sistemas/tipos de falla
        if 'sistema' in columns_found:
            print(f"\n🔧 ANÁLISIS DE SISTEMAS/FALLAS:")
            sistema_col = columns_found['sistema'][0]
            
            sistemas_stats = df_clean[sistema_col].value_counts()
            
            print(f"  Tipos de sistema/falla únicos: {len(sistemas_stats)}")
            print(f"\n  Top 10 sistemas más frecuentes:")
            for i, (sistema, count) in enumerate(sistemas_stats.head(10).items(), 1):
                pct = (count / len(df_clean)) * 100
                print(f"    {i:2d}. {sistema}: {count} ({pct:.1f}%)")
        
        # Calidad de datos
        print(f"\n📊 CALIDAD DE DATOS:")
        total_cells = len(df_clean) * len(df_clean.columns)
        null_cells = df_clean.isnull().sum().sum()
        completeness = ((total_cells - null_cells) / total_cells) * 100
        
        print(f"  Completitud general: {completeness:.1f}%")
        print(f"  Filas completas: {len(df_clean.dropna()):,} ({(len(df_clean.dropna())/len(df_clean)*100):.1f}%)")
        
        # Estadísticas por columna importante
        key_columns = []
        if 'codigo_equipo' in columns_found:
            key_columns.extend(columns_found['codigo_equipo'])
        if 'fecha_ingreso' in columns_found:
            key_columns.extend(columns_found['fecha_ingreso'])
        if 'sistema' in columns_found:
            key_columns.extend(columns_found['sistema'])
        
        if key_columns:
            print(f"\n  Completitud de columnas clave:")
            for col in key_columns[:5]:  # Máximo 5 columnas
                null_count = df_clean[col].isnull().sum()
                completeness_col = ((len(df_clean) - null_count) / len(df_clean)) * 100
                print(f"    {col}: {completeness_col:.1f}% completo")
        
        # Recomendaciones específicas para COTEMA
        print(f"\n💡 RECOMENDACIONES ESPECÍFICAS PARA COTEMA:")
        print(f"  1. CONFIGURACIÓN DEL DATA PROCESSOR:")
        
        if columns_found:
            print(f"     Mapeo sugerido de columnas:")
            for category, cols in columns_found.items():
                if cols:
                    print(f"       • {category}: '{cols[0]}'")
        
        print(f"\n  2. CONFIGURACIÓN DE MODELOS IA:")
        if 'codigo_equipo' in columns_found:
            equipo_col = columns_found['codigo_equipo'][0]
            total_equipos = df_clean[equipo_col].nunique()
            total_registros = len(df_clean)
            registros_por_equipo = total_registros / total_equipos if total_equipos > 0 else 0
            
            print(f"     • Equipos para analizar: {total_equipos}")
            print(f"     • Promedio registros/equipo: {registros_por_equipo:.1f}")
            
            if registros_por_equipo >= 10:
                print(f"     • ✅ Suficientes datos para modelos predictivos")
            else:
                print(f"     • ⚠️  Pocos datos históricos, modelos con menor precisión")
        
        print(f"\n  3. PRÓXIMOS PASOS:")
        print(f"     • Adaptar data_processor.py con mapeo de columnas")
        print(f"     • Configurar validaciones específicas")
        print(f"     • Ajustar parámetros de modelos IA")
        print(f"     • Probar carga y procesamiento")
        
        return {
            'df': df_clean,
            'columns_found': columns_found,
            'total_records': len(df_clean),
            'completeness': completeness,
            'recommendations': columns_found
        }
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    file_path = "/workspaces/IA-Taller-COTEMA/sample_data/Registro_Entrada_Taller_COTEMA.xlsx"
    analysis = analyze_cotema_reg_sheet(file_path)
    
    if analysis:
        print(f"\n✅ ANÁLISIS ESPECÍFICO COMPLETADO")
        print(f"� Total registros procesados: {analysis['total_records']:,}")
        print(f"🎯 Columnas identificadas: {len(analysis['columns_found'])}")
    else:
        print(f"\n❌ ERROR EN EL ANÁLISIS ESPECÍFICO")
