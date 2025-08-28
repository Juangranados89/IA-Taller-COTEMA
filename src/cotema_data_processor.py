"""
Data Processor específico para COTEMA
Adaptado para la estructura real del archivo Excel
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class CotemaDataProcessor:
    def __init__(self):
        # Mapeo específico para COTEMA basado en análisis real
        self.column_mapping = {
            'codigo_equipo': 'CODIGO',
            'fecha_ingreso': 'FECHA IN',
            'fecha_salida': 'FECHA OUT',
            'sistema_afectado': 'SISTEMA AFECTADO',
            'tipo_atencion': 'TIPO ATENCION',
            'descripcion_intervencion': 'DESCRIPCION INTERVENCION',
            'mttr_horas': 'MTTR',
            'dias_desde_averia': 'Cont.Dias.Ave',
            'horas_desde_averia': 'Con.Hrs.Ave',
            'contador_ingresos': 'Con.In.Taller',
            'descripcion_equipo': 'DESCRIPCION',
            'placa': 'PLACA',
            'flota': 'FLOTA',
            'operador': 'OPERADOR',
            'ejecutor': 'EJECUTOR',
            'atencion_local': 'ATENCION LOCAL',
            'atencion_externa': 'ATENCION EXTERNA',
            'origen_averia': 'ORIGEN AVERIA'
        }
        
        # Valores esperados para validación
        self.expected_systems = [
            'HIDRÁULICO', 'MOTOR', 'VIBRATORIO', 'AIRE ACONDICIONADO Y CLIMATIZACIÓN',
            'SISTEMA ELÉCTRICO', 'SISTEMA DE FRENOS', 'TRANSMISIÓN', 'DIRECCIÓN',
            'SISTEMA DE ENFRIAMIENTO', 'CARROCERÍA', 'COMBUSTIBLE', 'NEUMÁTICOS'
        ]
        
        self.expected_tipos = ['CORRECTIVA', 'PREVENTIVA', 'ALISTAMIENTO-TC']
        
    def load_and_process(self, file_path, sheet_name='REG'):
        """
        Carga y procesa el archivo Excel específico de COTEMA
        """
        try:
            print("🔄 Cargando datos de COTEMA...")
            
            # Cargar desde fila 5 (skiprows=4) como identificado en el análisis
            df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows=4)
            
            # Limpiar filas completamente vacías
            df = df.dropna(how='all')
            
            print(f"📊 Datos cargados: {len(df):,} registros")
            
            # Procesar datos
            df_processed = self._process_data(df)
            
            print(f"✅ Datos procesados: {len(df_processed):,} registros válidos")
            
            return df_processed
            
        except Exception as e:
            print(f"❌ Error cargando datos: {str(e)}")
            raise
    
    def _process_data(self, df):
        """
        Procesa y limpia los datos específicos de COTEMA
        """
        # Crear DataFrame procesado con columnas estandarizadas
        processed_data = {}
        
        # Mapear columnas principales
        for std_name, original_name in self.column_mapping.items():
            if original_name in df.columns:
                processed_data[std_name] = df[original_name].copy()
            else:
                print(f"⚠️  Columna no encontrada: {original_name}")
                processed_data[std_name] = None
        
        df_proc = pd.DataFrame(processed_data)
        
        # Limpiar y validar fechas
        df_proc = self._process_dates(df_proc)
        
        # Calcular TBF (Time Between Failures)
        df_proc = self._calculate_tbf(df_proc)
        
        # Agregar métricas de rolling
        df_proc = self._add_rolling_metrics(df_proc)
        
        # Validar y limpiar datos
        df_proc = self._validate_data(df_proc)
        
        return df_proc
    
    def _process_dates(self, df):
        """
        Procesa las columnas de fecha específicas de COTEMA
        """
        print("📅 Procesando fechas...")
        
        # Convertir fechas
        date_columns = ['fecha_ingreso', 'fecha_salida']
        
        for col in date_columns:
            if col in df.columns and df[col] is not None:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calcular duración en taller (si tenemos ambas fechas)
        if ('fecha_ingreso' in df.columns and 'fecha_salida' in df.columns and 
            df['fecha_ingreso'] is not None and df['fecha_salida'] is not None):
            
            df['duracion_taller_dias'] = (df['fecha_salida'] - df['fecha_ingreso']).dt.days
            df['duracion_taller_horas'] = (df['fecha_salida'] - df['fecha_ingreso']).dt.total_seconds() / 3600
            
            # Usar MTTR si está disponible, sino calcular
            if 'mttr_horas' in df.columns and df['mttr_horas'] is not None:
                # Completar valores faltantes de MTTR con la duración calculada
                df['mttr_horas'] = df['mttr_horas'].fillna(df['duracion_taller_horas'])
            else:
                df['mttr_horas'] = df['duracion_taller_horas']
        
        return df
    
    def _calculate_tbf(self, df):
        """
        Calcula Time Between Failures específico para COTEMA
        """
        print("⏱️  Calculando TBF...")
        
        # Usar los datos existentes de COTEMA
        if 'dias_desde_averia' in df.columns and df['dias_desde_averia'] is not None:
            df['tbf_dias'] = df['dias_desde_averia'].copy()
        else:
            # Calcular TBF manualmente si no está disponible
            df_sorted = df.sort_values(['codigo_equipo', 'fecha_ingreso'])
            df_sorted['fecha_ingreso_anterior'] = df_sorted.groupby('codigo_equipo')['fecha_ingreso'].shift(1)
            df_sorted['tbf_dias'] = (df_sorted['fecha_ingreso'] - df_sorted['fecha_ingreso_anterior']).dt.days
            df['tbf_dias'] = df_sorted['tbf_dias']
        
        # Usar horas desde avería si está disponible
        if 'horas_desde_averia' in df.columns and df['horas_desde_averia'] is not None:
            df['tbf_horas'] = df['horas_desde_averia'].copy()
        else:
            # Estimar horas basado en días (asumiendo 8 horas operativas por día)
            df['tbf_horas'] = df['tbf_dias'] * 8
        
        return df
    
    def _add_rolling_metrics(self, df):
        """
        Agrega métricas de ventana móvil específicas para COTEMA
        """
        print("📊 Calculando métricas de ventana móvil...")
        
        # Ordenar por equipo y fecha
        df = df.sort_values(['codigo_equipo', 'fecha_ingreso'])
        
        # Métricas por equipo con ventanas móviles
        for equipo in df['codigo_equipo'].dropna().unique():
            mask = df['codigo_equipo'] == equipo
            equipo_data = df[mask].copy()
            
            if len(equipo_data) >= 3:  # Mínimo 3 registros para rolling
                # TBF promedio móvil (30 días)
                df.loc[mask, 'tbf_mean_30d'] = equipo_data['tbf_dias'].rolling(
                    window=min(3, len(equipo_data)), min_periods=1).mean()
                
                # MTTR promedio móvil
                df.loc[mask, 'mttr_mean_30d'] = equipo_data['mttr_horas'].rolling(
                    window=min(3, len(equipo_data)), min_periods=1).mean()
                
                # Frecuencia de fallas (ingresos en últimos 30 días)
                for idx in equipo_data.index:
                    fecha_actual = equipo_data.loc[idx, 'fecha_ingreso']
                    if pd.notna(fecha_actual):
                        fecha_limite = fecha_actual - timedelta(days=30)
                        ingresos_30d = len(equipo_data[
                            (equipo_data['fecha_ingreso'] >= fecha_limite) & 
                            (equipo_data['fecha_ingreso'] <= fecha_actual)
                        ])
                        df.loc[idx, 'ingresos_30d'] = ingresos_30d
        
        # Rellenar valores faltantes con estadísticas globales
        df['tbf_mean_30d'] = df['tbf_mean_30d'].fillna(df['tbf_dias'].median())
        df['mttr_mean_30d'] = df['mttr_mean_30d'].fillna(df['mttr_horas'].median())
        df['ingresos_30d'] = df['ingresos_30d'].fillna(1)
        
        return df
    
    def _validate_data(self, df):
        """
        Valida y limpia datos específicos de COTEMA
        """
        print("🔍 Validando datos...")
        
        # Filtrar registros con código de equipo válido
        df = df[df['codigo_equipo'].notna() & (df['codigo_equipo'] != '')]
        
        # Validar fechas
        df = df[df['fecha_ingreso'].notna()]
        
        # Limpiar valores extremos de TBF
        if 'tbf_dias' in df.columns:
            # Eliminar TBF negativos o mayores a 2 años
            df = df[(df['tbf_dias'] >= 0) & (df['tbf_dias'] <= 730)]
        
        # Limpiar valores extremos de MTTR
        if 'mttr_horas' in df.columns:
            # Eliminar MTTR negativos o mayores a 30 días (720 horas)
            df = df[(df['mttr_horas'] >= 0) & (df['mttr_horas'] <= 720)]
        
        # Normalizar tipos de atención
        if 'tipo_atencion' in df.columns and df['tipo_atencion'] is not None:
            df['tipo_atencion'] = df['tipo_atencion'].str.upper().str.strip()
        
        # Normalizar sistemas afectados
        if 'sistema_afectado' in df.columns and df['sistema_afectado'] is not None:
            df['sistema_afectado'] = df['sistema_afectado'].str.upper().str.strip()
        
        print(f"✅ Validación completada: {len(df):,} registros válidos")
        
        return df
    
    def get_monthly_data(self, df, year_month):
        """
        Filtra datos para un mes específico (formato: 'YYYY-MM')
        """
        try:
            year, month = map(int, year_month.split('-'))
            
            # Filtrar por mes de ingreso
            mask = (df['fecha_ingreso'].dt.year == year) & (df['fecha_ingreso'].dt.month == month)
            monthly_data = df[mask].copy()
            
            print(f"📅 Datos para {year_month}: {len(monthly_data)} registros")
            
            return monthly_data
            
        except Exception as e:
            print(f"❌ Error filtrando datos mensuales: {str(e)}")
            return pd.DataFrame()
    
    def get_equipment_summary(self, df):
        """
        Genera resumen por equipo específico para COTEMA
        """
        summary = []
        
        for equipo in df['codigo_equipo'].unique():
            equipo_data = df[df['codigo_equipo'] == equipo]
            
            if len(equipo_data) > 0:
                summary.append({
                    'codigo': equipo,
                    'total_ingresos': len(equipo_data),
                    'tbf_promedio': equipo_data['tbf_dias'].median(),
                    'mttr_promedio': equipo_data['mttr_horas'].median(),
                    'ultimo_ingreso': equipo_data['fecha_ingreso'].max(),
                    'sistema_principal': equipo_data['sistema_afectado'].mode().iloc[0] if len(equipo_data['sistema_afectado'].mode()) > 0 else 'N/A',
                    'tipo_atencion_principal': equipo_data['tipo_atencion'].mode().iloc[0] if len(equipo_data['tipo_atencion'].mode()) > 0 else 'N/A'
                })
        
        return pd.DataFrame(summary)
    
    def export_processed_data(self, df, output_path):
        """
        Exporta datos procesados a Excel
        """
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Datos_Procesados', index=False)
                
                # Agregar resumen por equipo
                summary = self.get_equipment_summary(df)
                summary.to_excel(writer, sheet_name='Resumen_Equipos', index=False)
                
            print(f"💾 Datos exportados a: {output_path}")
            
        except Exception as e:
            print(f"❌ Error exportando datos: {str(e)}")

if __name__ == "__main__":
    # Prueba del procesador con datos reales
    processor = CotemaDataProcessor()
    
    file_path = "/workspaces/IA-Taller-COTEMA/sample_data/Registro_Entrada_Taller_COTEMA.xlsx"
    
    try:
        df_processed = processor.load_and_process(file_path)
        print(f"\n✅ PROCESAMIENTO COMPLETADO")
        print(f"📊 Registros procesados: {len(df_processed):,}")
        print(f"🏭 Equipos únicos: {df_processed['codigo_equipo'].nunique()}")
        
        # Mostrar muestra de datos procesados
        print(f"\n📋 MUESTRA DE DATOS PROCESADOS:")
        print(df_processed[['codigo_equipo', 'fecha_ingreso', 'tbf_dias', 'mttr_horas']].head())
        
    except Exception as e:
        print(f"❌ Error en procesamiento: {str(e)}")
