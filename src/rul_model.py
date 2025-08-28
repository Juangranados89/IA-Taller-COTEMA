"""
Modelo RUL: Remaining Useful Life (Vida Útil Restante)
Implementa modelos de supervivencia Weibull para RUL-50 y RUL-90
"""

import pandas as pd
import numpy as np
from lifelines import WeibullFitter, LogNormalFitter
from lifelines.utils import concordance_index
import warnings
import joblib
from scipy.optimize import fsolve
warnings.filterwarnings('ignore')

class RULModel:
    def __init__(self):
        self.fitted_models = {}  # Modelos por CLASE o global
        self.model_params = {}
        self.is_trained = False
        
    def prepare_survival_data(self, df):
        """
        Prepara los datos para análisis de supervivencia
        """
        # Crear copia
        data = df.copy().sort_values(['CODIGO', 'FECHA_IN'])
        
        # Calcular tiempos entre fallas (TBF)
        survival_data = []
        
        for equipo in data['CODIGO'].unique():
            equipo_data = data[data['CODIGO'] == equipo].sort_values('FECHA_IN')
            
            if len(equipo_data) < 2:
                continue
                
            # Para cada par de ingresos consecutivos
            for i in range(1, len(equipo_data)):
                duration = (equipo_data.iloc[i]['FECHA_IN'] - 
                          equipo_data.iloc[i-1]['FECHA_IN']).days
                
                if duration > 0 and duration < 1000:  # Filtrar valores extremos
                    record = {
                        'CODIGO': equipo,
                        'duration': duration,
                        'event': 1,  # Siempre es un evento (falla) observado
                        'CLASE': equipo_data.iloc[i]['CLASE'] if 'CLASE' in equipo_data.columns else 'GENERAL',
                        'FLOTA': equipo_data.iloc[i]['FLOTA'] if 'FLOTA' in equipo_data.columns else 'GENERAL',
                        'fecha_inicio': equipo_data.iloc[i-1]['FECHA_IN'],
                        'fecha_evento': equipo_data.iloc[i]['FECHA_IN']
                    }
                    survival_data.append(record)
        
        if not survival_data:
            raise ValueError("No se pudieron extraer datos de supervivencia válidos")
            
        survival_df = pd.DataFrame(survival_data)
        print(f"Datos de supervivencia extraídos: {len(survival_df)} observaciones")
        
        return survival_df
    
    def fit_weibull_models(self, survival_df):
        """
        Ajusta modelos Weibull por grupo (CLASE)
        """
        fitted_models = {}
        
        # Agrupar por CLASE para tener suficientes datos
        clases = survival_df['CLASE'].unique()
        
        for clase in clases:
            clase_data = survival_df[survival_df['CLASE'] == clase]
            
            if len(clase_data) < 5:  # Muy pocos datos para esta clase
                continue
                
            try:
                # Ajustar modelo Weibull
                wf = WeibullFitter()
                wf.fit(clase_data['duration'], clase_data['event'])
                
                # Verificar ajuste
                if hasattr(wf, 'lambda_') and hasattr(wf, 'rho_'):
                    fitted_models[clase] = {
                        'model': wf,
                        'lambda': float(wf.lambda_),
                        'k': float(wf.rho_),  # rho es el parámetro de forma (k)
                        'n_observations': len(clase_data),
                        'mean_duration': clase_data['duration'].mean()
                    }
                    print(f"Modelo Weibull ajustado para clase {clase}: λ={wf.lambda_:.2f}, k={wf.rho_:.2f}")
                
            except Exception as e:
                print(f"Error ajustando Weibull para clase {clase}: {str(e)}")
                continue
        
        # Si no se pudo ajustar ningún modelo por clase, ajustar modelo global
        if not fitted_models:
            try:
                wf = WeibullFitter()
                wf.fit(survival_df['duration'], survival_df['event'])
                
                fitted_models['GLOBAL'] = {
                    'model': wf,
                    'lambda': float(wf.lambda_),
                    'k': float(wf.rho_),
                    'n_observations': len(survival_df),
                    'mean_duration': survival_df['duration'].mean()
                }
                print(f"Modelo Weibull global ajustado: λ={wf.lambda_:.2f}, k={wf.rho_:.2f}")
                
            except Exception as e:
                print(f"Error ajustando modelo Weibull global: {str(e)}")
                return None
        
        return fitted_models
    
    def calculate_rul_for_equipment(self, modelo_params, edad_actual):
        """
        Calcula RUL-50 y RUL-90 para un equipo dado su edad actual
        
        Fórmula Weibull:
        S(t) = exp[-(t/λ)^k]
        RUL_q = (t_a^k - λ^k * ln(q))^(1/k) - t_a
        """
        try:
            lambda_param = modelo_params['lambda']
            k_param = modelo_params['k']
            
            if edad_actual <= 0:
                edad_actual = 1  # Evitar edad 0
            
            # Calcular RUL-50 (mediana)
            try:
                rul50 = ((edad_actual**k_param - lambda_param**k_param * np.log(0.5))**(1/k_param)) - edad_actual
            except:
                rul50 = lambda_param * 0.693**(1/k_param)  # Aproximación
            
            # Calcular RUL-90 (conservador)
            try:
                rul90 = ((edad_actual**k_param - lambda_param**k_param * np.log(0.1))**(1/k_param)) - edad_actual
            except:
                rul90 = lambda_param * 2.303**(1/k_param)  # Aproximación
            
            # Asegurar valores positivos y razonables
            rul50 = max(0, min(rul50, lambda_param * 2))
            rul90 = max(0, min(rul90, lambda_param * 3))
            
            return {
                'rul50_d': float(rul50),
                'rul90_d': float(rul90)
            }
            
        except Exception as e:
            print(f"Error calculando RUL: {str(e)}")
            # Valores fallback basados en parámetros del modelo
            return {
                'rul50_d': float(modelo_params.get('mean_duration', 30) * 0.5),
                'rul90_d': float(modelo_params.get('mean_duration', 30) * 0.8)
            }
    
    def calculate_rul(self, df):
        """
        Calcula RUL para todos los equipos
        """
        try:
            # Preparar datos de supervivencia
            survival_df = self.prepare_survival_data(df)
            
            # Ajustar modelos Weibull
            fitted_models = self.fit_weibull_models(survival_df)
            
            if not fitted_models:
                return self._simple_rul_fallback(df)
            
            self.fitted_models = fitted_models
            self.is_trained = True
            
            # Calcular RUL para cada equipo
            rul_results = {}
            
            # Obtener estado actual de cada equipo
            for equipo in df['CODIGO'].unique():
                equipo_data = df[df['CODIGO'] == equipo].sort_values('FECHA_IN')
                
                if len(equipo_data) == 0:
                    continue
                
                # Información del equipo
                ultimo_registro = equipo_data.iloc[-1]
                clase = ultimo_registro.get('CLASE', 'GENERAL')
                
                # Buscar modelo apropiado
                modelo_key = clase if clase in fitted_models else 'GLOBAL'
                if modelo_key not in fitted_models:
                    modelo_key = list(fitted_models.keys())[0]  # Usar el primer modelo disponible
                
                modelo_params = fitted_models[modelo_key]
                
                # Calcular edad actual (días desde último ingreso)
                edad_actual = (pd.Timestamp.now() - ultimo_registro['FECHA_IN']).days
                if edad_actual < 0:
                    edad_actual = 0
                
                # Calcular RUL
                rul_values = self.calculate_rul_for_equipment(modelo_params, edad_actual)
                
                # Agregar información adicional
                rul_results[equipo] = {
                    **rul_values,
                    'edad_actual_d': edad_actual,
                    'clase': clase,
                    'modelo_usado': modelo_key,
                    'parametros': {
                        'lambda': modelo_params['lambda'],
                        'k': modelo_params['k']
                    },
                    'n_observaciones': modelo_params['n_observations']
                }
            
            # Guardar modelo
            self._save_model()
            
            return rul_results
            
        except Exception as e:
            print(f"Error en RULModel: {str(e)}")
            return self._simple_rul_fallback(df)
    
    def _simple_rul_fallback(self, df):
        """
        Modelo simple cuando no se puede ajustar Weibull
        """
        rul_results = {}
        
        # Calcular TBF promedio global
        tbf_global = df['tbf_dias'].median()
        if pd.isna(tbf_global) or tbf_global <= 0:
            tbf_global = 30  # Default
        
        for equipo in df['CODIGO'].unique():
            equipo_data = df[df['CODIGO'] == equipo]
            
            # TBF específico del equipo
            tbf_equipo = equipo_data['tbf_dias'].median()
            if pd.isna(tbf_equipo) or tbf_equipo <= 0:
                tbf_equipo = tbf_global
            
            # Edad actual
            ultimo_registro = equipo_data.iloc[-1]
            edad_actual = (pd.Timestamp.now() - ultimo_registro['FECHA_IN']).days
            
            # RUL simple basado en TBF histórico
            rul50 = max(1, tbf_equipo * 0.7 - edad_actual * 0.5)
            rul90 = max(1, tbf_equipo * 0.9 - edad_actual * 0.3)
            
            rul_results[equipo] = {
                'rul50_d': float(rul50),
                'rul90_d': float(rul90),
                'edad_actual_d': edad_actual,
                'clase': ultimo_registro.get('CLASE', 'GENERAL'),
                'modelo_usado': 'SIMPLE',
                'tbf_base': float(tbf_equipo)
            }
        
        return rul_results
    
    def get_equipment_details(self, equipo):
        """
        Obtiene detalles del modelo RUL para un equipo específico
        """
        # Implementar obtención de detalles específicos
        return {
            'rul50_d': 0,
            'rul90_d': 0,
            'parameters': {}
        }
    
    def _save_model(self):
        """
        Guarda los modelos entrenados
        """
        if self.is_trained:
            try:
                # Guardar solo los parámetros, no los objetos lifelines
                model_data = {}
                for key, model_info in self.fitted_models.items():
                    model_data[key] = {
                        'lambda': model_info['lambda'],
                        'k': model_info['k'],
                        'n_observations': model_info['n_observations'],
                        'mean_duration': model_info['mean_duration']
                    }
                
                joblib.dump(model_data, 'models/rul_model.pkl')
                print("Modelo RUL guardado exitosamente")
            except Exception as e:
                print(f"Error guardando modelo RUL: {str(e)}")
    
    def load_model(self, model_path='models/rul_model.pkl'):
        """
        Carga un modelo previamente entrenado
        """
        try:
            self.fitted_models = joblib.load(model_path)
            self.is_trained = True
            print("Modelo RUL cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo RUL: {str(e)}")
            return False
    
    def predict_maintenance_window(self, equipo_rul, forecast_hours_per_day):
        """
        Predice ventana óptima de mantenimiento
        """
        rul50_d = equipo_rul.get('rul50_d', 0)
        rul90_d = equipo_rul.get('rul90_d', 0)
        
        # Convertir a horas si se proporciona el forecast
        if forecast_hours_per_day > 0:
            rul50_h = rul50_d * forecast_hours_per_day
            rul90_h = rul90_d * forecast_hours_per_day
            
            return {
                'ventana_optima_inicio': max(1, int(rul90_d * 0.7)),
                'ventana_optima_fin': int(rul50_d * 0.9),
                'urgencia': 'ALTA' if rul90_d < 7 else 'MEDIA' if rul90_d < 21 else 'BAJA',
                'rul50_h': rul50_h,
                'rul90_h': rul90_h
            }
        
        return {
            'ventana_optima_inicio': max(1, int(rul90_d * 0.7)),
            'ventana_optima_fin': int(rul50_d * 0.9),
            'urgencia': 'ALTA' if rul90_d < 7 else 'MEDIA' if rul90_d < 21 else 'BAJA'
        }
