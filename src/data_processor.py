"""
Procesador de datos para el análisis de taller
Limpia yclass DataProcessor:
    def __init__(self):
        # Mapeo específico para COTEMA basado en análisis real del archivo
        self.column_mapping = {
            'codigo_equipo': ['CODIGO'],
            'fecha_ingreso': ['FECHA IN'],
            'fecha_salida': ['FECHA OUT'],
            'descripcion': ['DESCRIPCION INTERVENCION'],
            'sistema': ['SISTEMA AFECTADO'],
            'tipo_atencion': ['TIPO ATENCION'],
            'mttr': ['MTTR'],
            'dias_averia': ['Cont.Dias.Ave'],
            'horas_averia': ['Con.Hrs.Ave'],
            'contador_ingresos': ['Con.In.Taller'],
            'descripcion_equipo': ['DESCRIPCION'],
            'flota': ['FLOTA'],
            'placa': ['PLACA']
        } los datos del Excel para el análisis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self):
        self.required_columns = ['CODIGO', 'FECHA_IN', 'FECHA_OUT']
        self.optional_columns = ['SISTEMA_AFECTADO', 'FLOTA', 'CLASE', 'DESCRIPCION']
    
    def load_excel(self, file_path):
        """
        Carga archivo Excel específico de COTEMA
        """
        try:
            print("📂 Cargando archivo Excel de COTEMA...")
            
            # Cargar específicamente la pestaña REG desde la fila 5
            df = pd.read_excel(file_path, sheet_name='REG', skiprows=4)
            
            # Limpiar filas completamente vacías
            df = df.dropna(how='all')
            
            print(f"✅ Archivo cargado: {len(df):,} registros")
            return df
            
        except Exception as e:
            print(f"❌ Error cargando archivo: {str(e)}")
            raise
    
    def _map_similar_columns(self, df):
        """
        Intenta mapear nombres de columnas similares a los requeridos
        """
        column_mapping = {
            'CODIGO': ['CODIGO', 'COD', 'EQUIPO', 'ID_EQUIPO', 'EQUIPMENT'],
            'FECHA_IN': ['FECHA_IN', 'FECHA_INGRESO', 'INGRESO', 'DATE_IN', 'ENTRADA'],
            'FECHA_OUT': ['FECHA_OUT', 'FECHA_SALIDA', 'SALIDA', 'DATE_OUT', 'EGRESO'],
            'SISTEMA_AFECTADO': ['SISTEMA_AFECTADO', 'SISTEMA', 'SUBSISTEMA', 'COMPONENT'],
            'FLOTA': ['FLOTA', 'FLEET', 'TIPO_FLOTA'],
            'CLASE': ['CLASE', 'CLASS', 'CATEGORIA', 'TIPO']
        }
        
        for target_col, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns and target_col not in df.columns:
                    df = df.rename(columns={possible_name: target_col})
                    print(f"Mapeado {possible_name} -> {target_col}")
                    break
        
        return df
    
    def _clean_data(self, df):
        """
        Limpia los datos del DataFrame
        """
        # Convertir fechas
        date_columns = ['FECHA_IN', 'FECHA_OUT']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Eliminar filas con fechas inválidas
        df = df.dropna(subset=['FECHA_IN'])
        
        # Limpiar códigos de equipo
        df['CODIGO'] = df['CODIGO'].astype(str).str.strip().str.upper()
        df = df[df['CODIGO'] != 'NAN']
        df = df[df['CODIGO'] != '']
        
        # Limpiar sistema afectado si existe
        if 'SISTEMA_AFECTADO' in df.columns:
            df['SISTEMA_AFECTADO'] = df['SISTEMA_AFECTADO'].fillna('NO_ESPECIFICADO')
            df['SISTEMA_AFECTADO'] = df['SISTEMA_AFECTADO'].astype(str).str.strip().str.upper()
        else:
            df['SISTEMA_AFECTADO'] = 'NO_ESPECIFICADO'
        
        # Agregar columnas faltantes con valores por defecto
        if 'FLOTA' not in df.columns:
            df['FLOTA'] = 'GENERAL'
        if 'CLASE' not in df.columns:
            df['CLASE'] = 'EQUIPO_GENERAL'
        
        # Filtrar fechas válidas
        df = df[df['FECHA_IN'] >= '2020-01-01']
        df = df[df['FECHA_IN'] <= datetime.now()]
        
        # Si hay FECHA_OUT, validar que sea posterior a FECHA_IN
        if 'FECHA_OUT' in df.columns:
            df = df[(df['FECHA_OUT'].isna()) | (df['FECHA_OUT'] >= df['FECHA_IN'])]
        
        return df.sort_values(['CODIGO', 'FECHA_IN']).reset_index(drop=True)
    
    def _calculate_metrics(self, df):
        """
        Calcula métricas adicionales necesarias para los modelos
        """
        # Ordenar por equipo y fecha
        df = df.sort_values(['CODIGO', 'FECHA_IN'])
        
        # Calcular TBF (Time Between Failures) en días
        df['tbf_dias'] = df.groupby('CODIGO')['FECHA_IN'].diff().dt.days
        
        # Calcular MTTR (Mean Time To Repair) si tenemos FECHA_OUT
        if 'FECHA_OUT' in df.columns:
            df['ciclo_horas'] = (df['FECHA_OUT'] - df['FECHA_IN']).dt.total_seconds() / 3600
            df['ciclo_horas'] = df['ciclo_horas'].fillna(24)  # Default 24 horas si no hay salida
        else:
            df['ciclo_horas'] = 24  # Valor por defecto
        
        # Calcular días desde última salida
        df['dias_desde_ultima_salida'] = (datetime.now() - df['FECHA_OUT'].fillna(df['FECHA_IN'])).dt.days
        
        # Métricas rolling por equipo
        df = self._calculate_rolling_metrics(df)
        
        # Métricas por sistema
        df = self._calculate_system_metrics(df)
        
        return df
    
    def _calculate_rolling_metrics(self, df):
        """
        Calcula métricas rolling por equipo
        """
        # Ingresos en ventanas de tiempo
        for window in [30, 60, 90]:
            df[f'ingresos_{window}d'] = df.groupby('CODIGO').apply(
                lambda x: x['FECHA_IN'].rolling(f'{window}D', on='FECHA_IN').count()
            ).values
        
        # TBF rolling
        df['tbf_med_60d'] = df.groupby('CODIGO')['tbf_dias'].transform(
            lambda x: x.rolling(window=min(5, len(x)), min_periods=1).median()
        )
        
        df['tbf_std_60d'] = df.groupby('CODIGO')['tbf_dias'].transform(
            lambda x: x.rolling(window=min(5, len(x)), min_periods=1).std().fillna(0)
        )
        
        # MTTR rolling por equipo
        df['mttr_eq'] = df.groupby('CODIGO')['ciclo_horas'].transform('mean')
        df['mttr_med_90d'] = df.groupby('CODIGO')['ciclo_horas'].transform(
            lambda x: x.rolling(window=min(10, len(x)), min_periods=1).median()
        )
        
        return df
    
    def _calculate_system_metrics(self, df):
        """
        Calcula métricas por sistema afectado
        """
        # Contador de sistema en ventana de 60 días
        df['sistema_count_60d'] = df.groupby(['CODIGO', 'SISTEMA_AFECTADO']).apply(
            lambda x: x['FECHA_IN'].rolling('60D', on='FECHA_IN').count()
        ).values
        
        # Sistemas únicos en ventana
        df['sistemas_unicos_60d'] = df.groupby('CODIGO').apply(
            lambda x: x.groupby(x['FECHA_IN'].dt.to_period('D'))['SISTEMA_AFECTADO'].nunique().rolling(60).sum()
        ).values
        
        return df
    
    def get_data_summary(self, df):
        """
        Genera un resumen de los datos procesados
        """
        summary = {
            'total_registros': len(df),
            'equipos_unicos': df['CODIGO'].nunique(),
            'periodo': {
                'inicio': df['FECHA_IN'].min().strftime('%Y-%m-%d'),
                'fin': df['FECHA_IN'].max().strftime('%Y-%m-%d')
            },
            'sistemas_afectados': df['SISTEMA_AFECTADO'].nunique(),
            'flotas': df['FLOTA'].nunique() if 'FLOTA' in df.columns else 1,
            'clases': df['CLASE'].nunique() if 'CLASE' in df.columns else 1,
            'tbf_promedio': df['tbf_dias'].mean(),
            'mttr_promedio': df['ciclo_horas'].mean()
        }
        
        return summary
    
    def validate_data_quality(self, df):
        """
        Valida la calidad de los datos
        """
        issues = []
        
        # Verificar duplicados
        duplicates = df.duplicated(['CODIGO', 'FECHA_IN']).sum()
        if duplicates > 0:
            issues.append(f"Se encontraron {duplicates} registros duplicados")
        
        # Verificar fechas futuras
        future_dates = (df['FECHA_IN'] > datetime.now()).sum()
        if future_dates > 0:
            issues.append(f"Se encontraron {future_dates} fechas futuras")
        
        # Verificar TBF anómalos
        extreme_tbf = df[(df['tbf_dias'] > 365) | (df['tbf_dias'] < 0)].shape[0]
        if extreme_tbf > 0:
            issues.append(f"Se encontraron {extreme_tbf} valores de TBF anómalos")
        
        # Verificar completitud
        completeness = {}
        for col in df.columns:
            missing_pct = (df[col].isna().sum() / len(df)) * 100
            if missing_pct > 0:
                completeness[col] = f"{missing_pct:.1f}%"
        
        return {
            'issues': issues,
            'completeness': completeness,
            'data_quality_score': max(0, 100 - len(issues) * 10)
        }
