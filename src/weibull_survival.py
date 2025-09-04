"""
Algoritmo de Análisis de Supervivencia - Distribución de Weibull
Para predicción precisa de tiempo hasta la falla (TTF) y probabilidad de supervivencia
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class WeibullSurvivalAnalysis:
    """Análisis de supervivencia usando distribución de Weibull para equipos COTEMA"""
    
    def __init__(self):
        self.weibull_params = {}
        self.survival_functions = {}
        self.reliability_curves = {}
    
    def calculate_time_between_failures(self, df):
        """Calcular tiempo entre fallas para cada equipo"""
        try:
            df_sorted = df.copy()
            df_sorted['FECHA_INGRESO'] = pd.to_datetime(df_sorted['FECHA_INGRESO'])
            df_sorted = df_sorted.sort_values(['EQUIPO', 'FECHA_INGRESO'])
            
            # Solo considerar mantenimientos correctivos
            df_correctivos = df_sorted[df_sorted['TIPO_MANTENIMIENTO'] == 'CORRECTIVO']
            
            tbf_data = {}
            
            for equipo in df_correctivos['EQUIPO'].unique():
                equipo_data = df_correctivos[df_correctivos['EQUIPO'] == equipo]
                
                if len(equipo_data) < 2:  # Necesitamos al menos 2 eventos
                    continue
                
                fechas = equipo_data['FECHA_INGRESO'].tolist()
                tiempos_entre_fallas = []
                
                for i in range(1, len(fechas)):
                    dias = (fechas[i] - fechas[i-1]).days
                    if dias > 0:  # Solo tiempos positivos
                        tiempos_entre_fallas.append(dias)
                
                if tiempos_entre_fallas:
                    tbf_data[equipo] = tiempos_entre_fallas
            
            return tbf_data
            
        except Exception as e:
            print(f"Error calculando TBF: {e}")
            return {}
    
    def fit_weibull_distribution(self, tbf_data):
        """Ajustar distribución de Weibull para cada equipo"""
        weibull_results = {}
        
        for equipo, tiempos in tbf_data.items():
            if len(tiempos) < 3:  # Necesitamos datos suficientes
                continue
                
            try:
                # Ajustar distribución de Weibull usando maximum likelihood
                # Weibull tiene parámetros: shape (beta) y scale (eta)
                tiempos_array = np.array(tiempos)
                
                # Método 1: Usar scipy.stats
                shape, loc, scale = stats.weibull_min.fit(tiempos_array, floc=0)
                
                # Método 2: Estimación más robusta con log-likelihood
                def weibull_log_likelihood(params):
                    beta, eta = params
                    if beta <= 0 or eta <= 0:
                        return np.inf
                    
                    # Log-likelihood de Weibull
                    n = len(tiempos_array)
                    ll = n * np.log(beta/eta) + (beta-1) * np.sum(np.log(tiempos_array/eta)) - np.sum((tiempos_array/eta)**beta)
                    return -ll  # Minimizar el negativo
                
                # Optimización
                initial_guess = [1.5, np.mean(tiempos_array)]
                bounds = [(0.1, 10), (0.1, np.max(tiempos_array) * 2)]
                
                result = minimize(weibull_log_likelihood, initial_guess, bounds=bounds, method='L-BFGS-B')
                
                if result.success:
                    beta_opt, eta_opt = result.x
                else:
                    # Fallback a método scipy
                    beta_opt, eta_opt = shape, scale
                
                # Calcular métricas de bondad de ajuste
                ks_stat, p_value = stats.kstest(tiempos_array, 
                                              lambda x: stats.weibull_min.cdf(x, beta_opt, loc=0, scale=eta_opt))
                
                # Calcular MTBF (Mean Time Between Failures)
                import math
                mtbf = eta_opt * math.gamma(1 + 1/beta_opt)
                
                # Calcular confiabilidad a diferentes tiempos
                reliability_30d = np.exp(-(30/eta_opt)**beta_opt)
                reliability_90d = np.exp(-(90/eta_opt)**beta_opt)
                reliability_180d = np.exp(-(180/eta_opt)**beta_opt)
                
                weibull_results[equipo] = {
                    'beta': beta_opt,      # Parámetro de forma (shape)
                    'eta': eta_opt,        # Parámetro de escala (scale)
                    'mtbf': mtbf,          # Tiempo medio entre fallas
                    'reliability_30d': reliability_30d,
                    'reliability_90d': reliability_90d,
                    'reliability_180d': reliability_180d,
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'sample_size': len(tiempos),
                    'data_quality': 'Buena' if p_value > 0.05 else 'Regular'
                }
                
            except Exception as e:
                print(f"Error ajustando Weibull para {equipo}: {e}")
                continue
        
        self.weibull_params = weibull_results
        return weibull_results
    
    def predict_failure_probability(self, equipo, days_ahead=30):
        """Predecir probabilidad de falla en los próximos X días"""
        if equipo not in self.weibull_params:
            return {'error': 'Equipo no tiene datos Weibull'}
        
        params = self.weibull_params[equipo]
        beta, eta = params['beta'], params['eta']
        
        # Probabilidad de falla en los próximos días
        prob_falla = 1 - np.exp(-(days_ahead/eta)**beta)
        
        # Calcular percentiles de tiempo hasta falla
        percentiles = [10, 25, 50, 75, 90]
        tiempo_percentiles = {}
        
        for p in percentiles:
            # Inversa de la CDF de Weibull
            t_p = eta * (-np.log(1 - p/100))**(1/beta)
            tiempo_percentiles[f'P{p}'] = round(t_p, 1)
        
        return {
            'probabilidad_falla_30d': round(prob_falla, 4),
            'mtbf_dias': round(params['mtbf'], 1),
            'percentiles_ttf': tiempo_percentiles,
            'confiabilidad_actual': round(params['reliability_30d'], 4),
            'estado_weibull': params['data_quality']
        }
    
    def rank_equipment_by_failure_risk(self, target_days=30):
        """Ranking de equipos por riesgo de falla en los próximos días"""
        risk_ranking = []
        
        for equipo in self.weibull_params:
            prediction = self.predict_failure_probability(equipo, target_days)
            
            if 'error' not in prediction:
                params = self.weibull_params[equipo]
                
                # Score compuesto de riesgo
                prob_weight = prediction['probabilidad_falla_30d'] * 0.4  # 40% probabilidad
                mtbf_weight = (1 / max(prediction['mtbf_dias'], 1)) * 0.3  # 30% inversa de MTBF
                reliability_weight = (1 - prediction['confiabilidad_actual']) * 0.3  # 30% baja confiabilidad
                
                risk_score = prob_weight + mtbf_weight + reliability_weight
                
                risk_ranking.append({
                    'equipo': equipo,
                    'riesgo_weibull': round(min(risk_score * 2, 1.0) * 100, 1),  # Convertir a escala 0-100
                    'prob_falla_30d': prediction['probabilidad_falla_30d'],
                    'mtbf_dias': prediction['mtbf_dias'],
                    'confiabilidad': prediction['confiabilidad_actual'],
                    'beta': round(params['beta'], 2),
                    'eta': round(params['eta'], 1),
                    'calidad_datos': params['data_quality'],
                    'muestra_tamaño': params['sample_size']
                })
        
        # Ordenar por riesgo descendente
        risk_ranking.sort(key=lambda x: x['riesgo_weibull'], reverse=True)
        
        return risk_ranking
    
    def get_maintenance_recommendations(self, equipo, confidence_level=0.9):
        """Recomendaciones de mantenimiento basadas en Weibull"""
        if equipo not in self.weibull_params:
            return {'error': 'No hay datos suficientes para este equipo'}
        
        params = self.weibull_params[equipo]
        beta, eta = params['beta']
        
        # Tiempo óptimo de mantenimiento preventivo
        # Para maximizar disponibilidad y minimizar costos
        optimal_time = eta * (-np.log(1 - confidence_level))**(1/beta)
        
        # Clasificación del patrón de falla según beta
        if beta < 1:
            failure_pattern = "Fallas tempranas (mortalidad infantil)"
            recommendation = "Mejorar calidad de instalación/repuestos"
        elif 1 <= beta <= 2:
            failure_pattern = "Fallas aleatorias"
            recommendation = "Mantenimiento basado en condición"
        else:
            failure_pattern = "Fallas por desgaste"
            recommendation = "Mantenimiento preventivo programado"
        
        return {
            'tiempo_optimo_mantenimiento': round(optimal_time, 1),
            'patron_falla': failure_pattern,
            'recomendacion': recommendation,
            'intervalo_preventivo_sugerido': round(optimal_time * 0.8, 1),  # 80% del tiempo óptimo
            'confianza_recomendacion': confidence_level
        }


# Integración con el sistema principal
def integrate_weibull_analysis(df, existing_analysis):
    """Integrar análisis Weibull con el análisis FR-30 existente"""
    try:
        weibull = WeibullSurvivalAnalysis()
        
        # Calcular tiempos entre fallas
        tbf_data = weibull.calculate_time_between_failures(df)
        
        if not tbf_data:
            return existing_analysis  # Sin datos suficientes
        
        # Ajustar distribuciones Weibull
        weibull_params = weibull.fit_weibull_distribution(tbf_data)
        
        if not weibull_params:
            return existing_analysis
        
        # Obtener ranking por riesgo Weibull
        weibull_ranking = weibull.rank_equipment_by_failure_risk(30)
        
        # Combinar con análisis existente
        equipos_combinados = []
        for equipo_existente in existing_analysis.get('equipos_riesgo', []):
            equipo_nombre = equipo_existente['equipo']
            
            # Buscar datos Weibull para este equipo
            weibull_data = next((w for w in weibull_ranking if w['equipo'] == equipo_nombre), None)
            
            if weibull_data:
                # Combinar scores: 60% ML + 40% Weibull (ambos ya en escala 0-100)
                score_ml = equipo_existente.get('riesgo_score', 0)
                score_weibull = weibull_data['riesgo_weibull']
                score_combinado = 0.6 * score_ml + 0.4 * score_weibull
                
                equipo_existente['riesgo_score'] = round(score_combinado, 1)
                equipo_existente['mtbf_dias'] = weibull_data['mtbf_dias']
                equipo_existente['prob_falla_30d'] = weibull_data['prob_falla_30d']
                equipo_existente['algoritmo'] = 'ML + Weibull Hybrid'
                equipo_existente['confianza_prediccion'] = min(equipo_existente.get('confianza_prediccion', 0.5) + 0.2, 1.0)
            
            equipos_combinados.append(equipo_existente)
        
        # Re-ordenar por nuevo score combinado
        equipos_combinados.sort(key=lambda x: x['riesgo_score'], reverse=True)
        
        # Actualizar análisis
        existing_analysis['equipos_riesgo'] = equipos_combinados
        existing_analysis['weibull_analysis'] = {
            'equipos_analizados': len(weibull_params),
            'equipos_con_datos_suficientes': len([w for w in weibull_ranking if w['calidad_datos'] == 'Buena']),
            'algoritmo_mejorado': 'ML + Weibull Survival Analysis'
        }
        
        return existing_analysis
        
    except Exception as e:
        print(f"Error integrando Weibull: {e}")
        return existing_analysis
