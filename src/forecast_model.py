"""
Modelo de Pronóstico de Uso
Predice horas de operación futuras usando Prophet o ARIMA
"""

import pandas as pd
import numpy as np
from prophet import Prophet
import warnings
import joblib
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class ForecastModel:
    def __init__(self):
        self.models = {}  # Modelos por equipo o clase
        self.is_trained = False
        
    def prepare_usage_series(self, df):
        """
        Prepara series temporales de uso por equipo
        """
        usage_series = {}
        
        for equipo in df['CODIGO'].unique():
            equipo_data = df[df['CODIGO'] == equipo].sort_values('FECHA_IN')
            
            if len(equipo_data) < 3:  # Muy pocos datos
                continue
            
            # Crear serie temporal de uso diario
            # Aproximar horas de uso basado en ciclo_horas
            daily_usage = equipo_data.set_index('FECHA_IN')['ciclo_horas'].resample('D').sum().fillna(0)
            
            # Suavizar la serie
            daily_usage = daily_usage.rolling(window=3, min_periods=1).mean()
            
            # Convertir a formato Prophet
            ts_data = pd.DataFrame({
                'ds': daily_usage.index,
                'y': daily_usage.values
            })
            
            # Filtrar valores extremos
            q99 = ts_data['y'].quantile(0.99)
            ts_data['y'] = np.where(ts_data['y'] > q99, q99, ts_data['y'])
            
            # Solo mantener si hay suficiente variabilidad
            if ts_data['y'].std() > 0.1 and len(ts_data) >= 10:
                usage_series[equipo] = {
                    'data': ts_data,
                    'clase': equipo_data.iloc[-1].get('CLASE', 'GENERAL'),
                    'stats': {
                        'mean_daily_hours': ts_data['y'].mean(),
                        'std_daily_hours': ts_data['y'].std(),
                        'max_daily_hours': ts_data['y'].max(),
                        'days_with_usage': (ts_data['y'] > 0).sum()
                    }
                }
        
        return usage_series
    
    def fit_prophet_models(self, usage_series):
        """
        Ajusta modelos Prophet por equipo o clase
        """
        fitted_models = {}
        
        # Intentar ajustar modelo individual por equipo
        for equipo, series_info in usage_series.items():
            ts_data = series_info['data']
            
            if len(ts_data) < 15:  # Muy pocos datos para Prophet
                continue
                
            try:
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.05,
                    seasonality_prior_scale=10.0,
                    uncertainty_samples=100
                )
                
                model.fit(ts_data)
                
                fitted_models[equipo] = {
                    'model': model,
                    'tipo': 'individual',
                    'stats': series_info['stats'],
                    'clase': series_info['clase']
                }
                
                print(f"Modelo Prophet ajustado para equipo {equipo}")
                
            except Exception as e:
                print(f"Error ajustando Prophet para equipo {equipo}: {str(e)}")
                continue
        
        # Si hay pocos modelos individuales, crear modelos por clase
        if len(fitted_models) < len(usage_series) * 0.3:
            fitted_models.update(self._fit_class_models(usage_series))
        
        return fitted_models
    
    def _fit_class_models(self, usage_series):
        """
        Ajusta modelos por clase cuando hay pocos datos individuales
        """
        class_models = {}
        
        # Agrupar por clase
        class_data = {}
        for equipo, series_info in usage_series.items():
            clase = series_info['clase']
            if clase not in class_data:
                class_data[clase] = []
            class_data[clase].append(series_info['data'])
        
        # Ajustar modelo por clase
        for clase, data_list in class_data.items():
            if len(data_list) < 2:
                continue
                
            try:
                # Combinar datos de la clase
                combined_data = pd.concat(data_list, ignore_index=True)
                combined_data = combined_data.groupby('ds')['y'].mean().reset_index()
                
                if len(combined_data) < 10:
                    continue
                
                model = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=False,
                    changepoint_prior_scale=0.1
                )
                
                model.fit(combined_data)
                
                class_models[f"CLASE_{clase}"] = {
                    'model': model,
                    'tipo': 'clase',
                    'stats': {
                        'mean_daily_hours': combined_data['y'].mean(),
                        'equipos_en_clase': len(data_list)
                    }
                }
                
                print(f"Modelo Prophet ajustado para clase {clase}")
                
            except Exception as e:
                print(f"Error ajustando Prophet para clase {clase}: {str(e)}")
                continue
        
        return class_models
    
    def predict_usage(self, df):
        """
        Predice uso futuro para todos los equipos
        """
        try:
            # Preparar series de uso
            usage_series = self.prepare_usage_series(df)
            
            if not usage_series:
                return self._simple_forecast_fallback(df)
            
            # Ajustar modelos
            fitted_models = self.fit_prophet_models(usage_series)
            
            if not fitted_models:
                return self._simple_forecast_fallback(df)
            
            self.models = fitted_models
            self.is_trained = True
            
            # Generar pronósticos
            forecast_results = {}
            
            for equipo in df['CODIGO'].unique():
                try:
                    # Buscar modelo apropiado
                    model_key = None
                    if equipo in fitted_models:
                        model_key = equipo
                    else:
                        # Buscar por clase
                        equipo_data = df[df['CODIGO'] == equipo]
                        if len(equipo_data) > 0:
                            clase = equipo_data.iloc[-1].get('CLASE', 'GENERAL')
                            class_key = f"CLASE_{clase}"
                            if class_key in fitted_models:
                                model_key = class_key
                    
                    if not model_key:
                        # Usar el primer modelo disponible
                        model_key = list(fitted_models.keys())[0]
                    
                    model_info = fitted_models[model_key]
                    model = model_info['model']
                    
                    # Crear fechas futuras
                    future = model.make_future_dataframe(periods=30)
                    forecast = model.predict(future)
                    
                    # Extraer pronósticos
                    current_date = datetime.now().date()
                    
                    # Pronóstico 7 días
                    forecast_7d = forecast[
                        (forecast['ds'].dt.date > current_date) &
                        (forecast['ds'].dt.date <= current_date + timedelta(days=7))
                    ]['yhat'].sum()
                    
                    # Pronóstico 30 días
                    forecast_30d = forecast[
                        (forecast['ds'].dt.date > current_date) &
                        (forecast['ds'].dt.date <= current_date + timedelta(days=30))
                    ]['yhat'].sum()
                    
                    # Asegurar valores positivos
                    forecast_7d = max(0, forecast_7d)
                    forecast_30d = max(0, forecast_30d)
                    
                    # Calcular horas por día
                    horas_por_dia = forecast_30d / 30 if forecast_30d > 0 else model_info['stats'].get('mean_daily_hours', 8)
                    
                    forecast_results[equipo] = {
                        'forecast_7d_h': float(forecast_7d),
                        'forecast_30d_h': float(forecast_30d),
                        'horas_por_dia': float(horas_por_dia),
                        'modelo_usado': model_key,
                        'tipo_modelo': model_info['tipo'],
                        'confianza': 'ALTA' if model_info['tipo'] == 'individual' else 'MEDIA'
                    }
                    
                except Exception as e:
                    print(f"Error pronosticando para equipo {equipo}: {str(e)}")
                    # Usar valores fallback
                    forecast_results[equipo] = self._get_fallback_forecast(df, equipo)
            
            # Guardar modelo
            self._save_model()
            
            return forecast_results
            
        except Exception as e:
            print(f"Error en ForecastModel: {str(e)}")
            return self._simple_forecast_fallback(df)
    
    def _simple_forecast_fallback(self, df):
        """
        Pronóstico simple cuando no se puede usar Prophet
        """
        forecast_results = {}
        
        # Calcular promedio global de horas por día
        horas_promedio_global = df['ciclo_horas'].mean()
        if pd.isna(horas_promedio_global) or horas_promedio_global <= 0:
            horas_promedio_global = 8  # Default 8 horas/día
        
        for equipo in df['CODIGO'].unique():
            equipo_data = df[df['CODIGO'] == equipo]
            
            # Promedio específico del equipo
            horas_promedio_equipo = equipo_data['ciclo_horas'].mean()
            if pd.isna(horas_promedio_equipo) or horas_promedio_equipo <= 0:
                horas_promedio_equipo = horas_promedio_global
            
            # Ajustar por tendencia reciente (últimos 30 días)
            fechas_recientes = equipo_data[
                equipo_data['FECHA_IN'] >= (equipo_data['FECHA_IN'].max() - pd.Timedelta(days=30))
            ]
            
            if len(fechas_recientes) > 0:
                factor_actividad = len(fechas_recientes) / 30 * 7  # Normalizar a semana
                factor_actividad = min(2.0, max(0.1, factor_actividad))  # Limitar factor
            else:
                factor_actividad = 1.0
            
            horas_por_dia = horas_promedio_equipo * factor_actividad
            
            forecast_results[equipo] = {
                'forecast_7d_h': float(horas_por_dia * 7),
                'forecast_30d_h': float(horas_por_dia * 30),
                'horas_por_dia': float(horas_por_dia),
                'modelo_usado': 'SIMPLE',
                'tipo_modelo': 'fallback',
                'confianza': 'BAJA',
                'factor_actividad': float(factor_actividad)
            }
        
        return forecast_results
    
    def _get_fallback_forecast(self, df, equipo):
        """
        Obtiene pronóstico fallback para un equipo específico
        """
        equipo_data = df[df['CODIGO'] == equipo]
        horas_promedio = equipo_data['ciclo_horas'].mean()
        
        if pd.isna(horas_promedio) or horas_promedio <= 0:
            horas_promedio = 8
        
        return {
            'forecast_7d_h': float(horas_promedio * 7),
            'forecast_30d_h': float(horas_promedio * 30),
            'horas_por_dia': float(horas_promedio),
            'modelo_usado': 'FALLBACK',
            'tipo_modelo': 'simple',
            'confianza': 'BAJA'
        }
    
    def get_maintenance_recommendations(self, equipo_forecast, rul_data):
        """
        Genera recomendaciones de mantenimiento basadas en pronóstico y RUL
        """
        forecast_7d = equipo_forecast.get('forecast_7d_h', 0)
        forecast_30d = equipo_forecast.get('forecast_30d_h', 0)
        horas_por_dia = equipo_forecast.get('horas_por_dia', 8)
        
        rul50_d = rul_data.get('rul50_d', 30)
        rul90_d = rul_data.get('rul90_d', 45)
        
        # Calcular fecha sugerida de servicio
        if horas_por_dia > 0 and rul50_d > 0:
            dias_hasta_servicio = max(1, int(rul50_d * 0.8))  # 80% del RUL-50
        else:
            dias_hasta_servicio = 15  # Default
        
        fecha_servicio_sugerida = datetime.now() + timedelta(days=dias_hasta_servicio)
        
        # Evaluar riesgo
        riesgo = 'BAJO'
        if rul90_d < 7:
            riesgo = 'ALTO'
        elif rul50_d < 14 and horas_por_dia > 12:
            riesgo = 'MEDIO'
        
        return {
            'fecha_servicio_sugerida': fecha_servicio_sugerida.strftime('%Y-%m-%d'),
            'dias_hasta_servicio': dias_hasta_servicio,
            'riesgo_disponibilidad': riesgo,
            'horas_proyectadas_hasta_servicio': float(horas_por_dia * dias_hasta_servicio),
            'recomendacion': self._generate_recommendation(riesgo, dias_hasta_servicio, horas_por_dia)
        }
    
    def _generate_recommendation(self, riesgo, dias, horas_dia):
        """
        Genera recomendación textual
        """
        if riesgo == 'ALTO':
            return f"🔴 URGENTE: Programar mantenimiento en {dias} días. Alta probabilidad de falla."
        elif riesgo == 'MEDIO':
            return f"🟠 ATENCIÓN: Ventana de {dias} días para mantenimiento. Monitorear uso intensivo."
        else:
            return f"🟢 NORMAL: Mantenimiento programado en {dias} días. Operación estable."
    
    def _save_model(self):
        """
        Guarda los modelos entrenados
        """
        if self.is_trained:
            try:
                # Guardar solo los parámetros esenciales, no los objetos Prophet completos
                model_data = {}
                for key, model_info in self.models.items():
                    model_data[key] = {
                        'tipo': model_info['tipo'],
                        'stats': model_info['stats'],
                        'modelo_usado': True
                    }
                
                joblib.dump(model_data, 'models/forecast_model.pkl')
                print("Modelo Forecast guardado exitosamente")
            except Exception as e:
                print(f"Error guardando modelo Forecast: {str(e)}")
    
    def load_model(self, model_path='models/forecast_model.pkl'):
        """
        Carga un modelo previamente entrenado
        """
        try:
            self.models = joblib.load(model_path)
            self.is_trained = True
            print("Modelo Forecast cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo Forecast: {str(e)}")
            return False
