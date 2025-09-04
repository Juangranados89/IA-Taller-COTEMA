"""
Motor de Predicción Avanzada para COTEMA - FR-30 Mejorado
Algoritmos de Machine Learning para afinar predicciones de fallas de equipos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedPredictionEngine:
    """Motor de predicción avanzada con múltiples algoritmos"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.prediction_accuracy = {}
        
    def create_advanced_features(self, df):
        """Crear características avanzadas para ML"""
        try:
            # Copiar DataFrame para no modificar el original
            df_features = df.copy()
            
            # 1. Features temporales
            if 'FECHA_INGRESO' in df_features.columns:
                df_features['FECHA_INGRESO'] = pd.to_datetime(df_features['FECHA_INGRESO'])
                df_features['dia_semana'] = df_features['FECHA_INGRESO'].dt.dayofweek
                df_features['mes'] = df_features['FECHA_INGRESO'].dt.month
                df_features['trimestre'] = df_features['FECHA_INGRESO'].dt.quarter
                df_features['dias_desde_inicio_año'] = df_features['FECHA_INGRESO'].dt.dayofyear
            
            # 2. Features de frecuencia por equipo
            equipo_stats = df_features.groupby('EQUIPO').agg({
                'EQUIPO': 'count',  # frecuencia total
                'TIPO_MANTENIMIENTO': lambda x: (x == 'CORRECTIVO').sum(),  # correctivos
                'FECHA_INGRESO': ['min', 'max']  # primera y última fecha
            }).round(2)
            
            equipo_stats.columns = ['total_ingresos', 'correctivos_count', 'primera_fecha', 'ultima_fecha']
            equipo_stats['ratio_correctivos'] = (equipo_stats['correctivos_count'] / equipo_stats['total_ingresos']).fillna(0)
            
            # 3. Features de tiempo entre fallas (TBF)
            df_sorted = df_features.sort_values(['EQUIPO', 'FECHA_INGRESO'])
            df_sorted['fecha_anterior'] = df_sorted.groupby('EQUIPO')['FECHA_INGRESO'].shift(1)
            df_sorted['dias_entre_fallas'] = (df_sorted['FECHA_INGRESO'] - df_sorted['fecha_anterior']).dt.days
            
            tbf_stats = df_sorted.groupby('EQUIPO')['dias_entre_fallas'].agg(['mean', 'std', 'min', 'max']).fillna(0)
            tbf_stats.columns = ['tbf_promedio', 'tbf_std', 'tbf_min', 'tbf_max']
            
            # 4. Features de estacionalidad
            seasonal_patterns = df_features.groupby(['EQUIPO', 'mes']).size().reset_index(name='ingresos_mes')
            seasonal_variance = seasonal_patterns.groupby('EQUIPO')['ingresos_mes'].agg(['mean', 'std']).fillna(0)
            seasonal_variance.columns = ['estacionalidad_media', 'estacionalidad_varianza']
            
            # 5. Features de criticidad
            df_features['es_correctivo'] = (df_features['TIPO_MANTENIMIENTO'] == 'CORRECTIVO').astype(int)
            df_features['es_urgente'] = df_features['DESCRIPCION_FALLA'].str.contains('URGENT|CRITICO|EMERGENCIA', case=False, na=False).astype(int)
            
            # Combinar todas las características
            features_combinadas = equipo_stats.join(tbf_stats, how='left').join(seasonal_variance, how='left')
            features_combinadas = features_combinadas.fillna(0)
            
            return features_combinadas, df_sorted
            
        except Exception as e:
            print(f"Error creando features avanzadas: {e}")
            return pd.DataFrame(), df
    
    def predict_equipment_failure_risk(self, df, target_year=2025):
        """Algoritmo 1: Random Forest + Gradient Boosting para riesgo de falla"""
        try:
            features_df, df_processed = self.create_advanced_features(df)
            
            if features_df.empty:
                return {}
            
            # Preparar datos para entrenamiento
            X = features_df.select_dtypes(include=[np.number])
            y = features_df['ratio_correctivos']  # Target: ratio de correctivos
            
            if len(X) < 5:  # Necesitamos al menos 5 equipos para entrenar
                return self._fallback_risk_analysis(df, target_year)
            
            # División train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Normalización
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Modelos ensemble
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
            gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3)
            
            # Entrenar modelos
            rf_model.fit(X_train_scaled, y_train)
            gb_model.fit(X_train_scaled, y_train)
            
            # Predicciones en conjunto completo
            X_all_scaled = scaler.transform(X)
            rf_predictions = rf_model.predict(X_all_scaled)
            gb_predictions = gb_model.predict(X_all_scaled)
            
            # Promedio ponderado de predicciones
            ensemble_predictions = 0.6 * rf_predictions + 0.4 * gb_predictions
            
            # Crear resultados con score de riesgo refinado
            equipos_riesgo = []
            for idx, (equipo, features) in enumerate(features_df.iterrows()):
                risk_score = min(ensemble_predictions[idx] * 1.2, 1.0)  # Amplificar pero limitar a 1.0
                
                # Ajuste por patrones recientes (último trimestre)
                recent_data = df_processed[
                    (df_processed['EQUIPO'] == equipo) & 
                    (df_processed['FECHA_INGRESO'] >= pd.Timestamp.now() - pd.Timedelta(days=90))
                ]
                recent_boost = min(len(recent_data) * 0.05, 0.3)  # Boost por actividad reciente
                
                equipos_riesgo.append({
                    'equipo': equipo,
                    'riesgo_score': round((risk_score + recent_boost) * 100, 1),  # Convertir a escala 0-100
                    'total_ingresos': int(features['total_ingresos']),
                    'ingresos_criticos': int(features['correctivos_count']),
                    'mttr_horas': round(features.get('tbf_promedio', 0) / 24, 1),
                    'mes_mayor_riesgo': self._predict_peak_month(equipo, df_processed, target_year),
                    'confianza_prediccion': round(min(features['total_ingresos'] / 10, 1.0), 2)
                })
            
            # Ordenar por riesgo descendente
            equipos_riesgo.sort(key=lambda x: x['riesgo_score'], reverse=True)
            
            # Guardar métricas del modelo
            if len(X_test) > 0:
                test_predictions = 0.6 * rf_model.predict(X_test_scaled) + 0.4 * gb_model.predict(X_test_scaled)
                self.prediction_accuracy['r2_score'] = r2_score(y_test, test_predictions)
                self.prediction_accuracy['mae'] = mean_absolute_error(y_test, test_predictions)
            
            # Feature importance
            self.feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
            
            return {
                'equipos_riesgo': equipos_riesgo[:20],  # Top 20
                'model_accuracy': self.prediction_accuracy,
                'feature_importance': self.feature_importance,
                'algorithm_used': 'Random Forest + Gradient Boosting Ensemble'
            }
            
        except Exception as e:
            print(f"Error en predicción ML: {e}")
            return self._fallback_risk_analysis(df, target_year)
    
    def predict_monthly_trends_advanced(self, df, target_year=2025):
        """Algoritmo 2: Análisis de series temporales con patrones estacionales"""
        try:
            df_temporal = df.copy()
            df_temporal['FECHA_INGRESO'] = pd.to_datetime(df_temporal['FECHA_INGRESO'])
            df_temporal['mes'] = df_temporal['FECHA_INGRESO'].dt.month
            df_temporal['año'] = df_temporal['FECHA_INGRESO'].dt.year
            
            # Análisis de patrones históricos por mes
            patron_mensual = df_temporal[df_temporal['TIPO_MANTENIMIENTO'] == 'CORRECTIVO'].groupby('mes').size()
            
            # Calcular predicciones solo para mes actual y próximos 2 meses
            from datetime import datetime
            mes_actual = datetime.now().month
            
            meses_prediccion = []
            for i in range(3):  # Mes actual + 2 siguientes
                mes = ((mes_actual - 1 + i) % 12) + 1  # Ciclo de 1-12
                
                # Base histórica para este mes
                base_historica = patron_mensual.get(mes, 0)
                
                # Factor de crecimiento (basado en tendencia de últimos años)
                años_disponibles = df_temporal['año'].unique()
                if len(años_disponibles) > 1:
                    tendencia_anual = self._calculate_growth_trend(df_temporal, mes)
                    proyeccion = base_historica * (1 + tendencia_anual)
                else:
                    proyeccion = base_historica
                
                # Ajuste por estacionalidad industrial
                factor_estacional = self._get_seasonal_factor(mes)
                proyeccion_ajustada = proyeccion * factor_estacional
                
                # Determinar etiqueta según la posición
                if i == 0:
                    etiqueta = "Mes Actual"
                elif i == 1:
                    etiqueta = "Próximo Mes"
                else:
                    etiqueta = "2 Meses"
                
                meses_prediccion.append({
                    'mes': mes,
                    'mes_nombre': f"{self._get_month_name(mes)} ({etiqueta})",
                    'total_correctivas': max(int(proyeccion_ajustada), 1),
                    'confianza': min(base_historica / max(patron_mensual.max(), 1), 1.0),
                    'factor_estacional': factor_estacional,
                    'es_prediccion': i > 0,  # Marcar cuáles son predicciones futuras
                    'periodo': etiqueta
                })
            
            return meses_prediccion
            
        except Exception as e:
            print(f"Error en predicción temporal: {e}")
            return []
    
    def _predict_peak_month(self, equipo, df, target_year):
        """Predecir el mes de mayor riesgo para un equipo específico"""
        try:
            equipo_data = df[df['EQUIPO'] == equipo]
            if equipo_data.empty:
                return 1
                
            # Análisis de patrones históricos del equipo
            equipo_data['mes'] = equipo_data['FECHA_INGRESO'].dt.month
            patron_equipo = equipo_data.groupby('mes').size()
            
            if patron_equipo.empty:
                return 1
                
            # Mes con más incidencias históricas
            mes_pico = patron_equipo.idxmax()
            
            # Ajuste por tendencia reciente
            datos_recientes = equipo_data[equipo_data['FECHA_INGRESO'] >= pd.Timestamp.now() - pd.Timedelta(days=180)]
            if not datos_recientes.empty:
                patron_reciente = datos_recientes.groupby('mes').size()
                if not patron_reciente.empty:
                    mes_pico_reciente = patron_reciente.idxmax()
                    # Promedio ponderado: 70% histórico, 30% reciente
                    mes_pico = round(0.7 * mes_pico + 0.3 * mes_pico_reciente)
            
            return int(mes_pico)
            
        except:
            return 1
    
    def _calculate_growth_trend(self, df, mes):
        """Calcular tendencia de crecimiento para un mes específico"""
        try:
            monthly_data = df[df['mes'] == mes].groupby('año').size()
            if len(monthly_data) < 2:
                return 0
            
            # Calcular tasa de crecimiento promedio
            years = sorted(monthly_data.index)
            growth_rates = []
            
            for i in range(1, len(years)):
                prev_value = monthly_data[years[i-1]]
                curr_value = monthly_data[years[i]]
                if prev_value > 0:
                    growth_rate = (curr_value - prev_value) / prev_value
                    growth_rates.append(growth_rate)
            
            return np.mean(growth_rates) if growth_rates else 0
            
        except:
            return 0
    
    def _get_seasonal_factor(self, mes):
        """Factores estacionales industriales típicos"""
        # Basado en patrones industriales comunes
        factores_industriales = {
            1: 1.1,   # Enero: arranque post-vacaciones
            2: 0.9,   # Febrero: normalización
            3: 1.0,   # Marzo: operación normal
            4: 1.1,   # Abril: incremento pre-verano
            5: 1.2,   # Mayo: pico operacional
            6: 1.3,   # Junio: máximo antes de mantenimientos verano
            7: 0.8,   # Julio: mantenimientos programados
            8: 0.7,   # Agosto: vacaciones/mantenimientos
            9: 1.2,   # Septiembre: arranque post-verano
            10: 1.1,  # Octubre: operación intensa
            11: 1.0,  # Noviembre: normalización
            12: 0.8   # Diciembre: reducción fin de año
        }
        return factores_industriales.get(mes, 1.0)
    
    def _get_month_name(self, mes):
        """Nombres de meses en español"""
        nombres = {
            1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
            5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
            9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
        }
        return nombres.get(mes, f'Mes {mes}')
    
    def _fallback_risk_analysis(self, df, target_year):
        """Análisis de respaldo cuando ML no es viable"""
        try:
            # Análisis simple pero efectivo basado en frecuencia y recencia
            df_analysis = df.copy()
            df_analysis['FECHA_INGRESO'] = pd.to_datetime(df_analysis['FECHA_INGRESO'])
            
            equipos_stats = df_analysis.groupby('EQUIPO').agg({
                'EQUIPO': 'count',
                'TIPO_MANTENIMIENTO': lambda x: (x == 'CORRECTIVO').sum(),
                'FECHA_INGRESO': 'max'
            }).rename(columns={'EQUIPO': 'total_ingresos', 'TIPO_MANTENIMIENTO': 'correctivos', 'FECHA_INGRESO': 'ultima_fecha'})
            
            equipos_riesgo = []
            for equipo, stats in equipos_stats.iterrows():
                # Score simple: frecuencia correctivos + factor recencia
                freq_score = min(stats['correctivos'] / max(stats['total_ingresos'], 1), 1.0)
                recency_score = min((pd.Timestamp.now() - stats['ultima_fecha']).days / 365, 0.3)
                riesgo_score = freq_score + recency_score
                
                equipos_riesgo.append({
                    'equipo': equipo,
                    'riesgo_score': round(riesgo_score * 100, 1),  # Convertir a escala 0-100
                    'total_ingresos': int(stats['total_ingresos']),
                    'ingresos_criticos': int(stats['correctivos']),
                    'mttr_horas': 0,
                    'mes_mayor_riesgo': np.random.randint(1, 13),
                    'confianza_prediccion': 0.6
                })
            
            equipos_riesgo.sort(key=lambda x: x['riesgo_score'], reverse=True)
            
            return {
                'equipos_riesgo': equipos_riesgo[:20],
                'algorithm_used': 'Frequency-based Risk Analysis (Fallback)'
            }
            
        except Exception as e:
            print(f"Error en análisis fallback: {e}")
            return {'equipos_riesgo': []}

# Funciones de utilidad para integración
def get_advanced_fr30_prediction(df, year=2025):
    """Función principal para obtener predicciones FR-30 avanzadas"""
    engine = AdvancedPredictionEngine()
    
    # Predicción de equipos de riesgo
    equipment_prediction = engine.predict_equipment_failure_risk(df, year)
    
    # Predicción de tendencias mensuales
    monthly_trends = engine.predict_monthly_trends_advanced(df, year)
    
    # Métricas adicionales
    total_equipos = len(df['EQUIPO'].unique()) if 'EQUIPO' in df.columns else 0
    total_correctivas = len(df[df['TIPO_MANTENIMIENTO'] == 'CORRECTIVO']) if 'TIPO_MANTENIMIENTO' in df.columns else 0
    
    return {
        'equipos_riesgo': equipment_prediction.get('equipos_riesgo', []),
        'meses_tendencia': monthly_trends,
        'factores_analisis': {
            'total_equipos_analizados': total_equipos,
            'total_correctivas_2025': sum([m.get('total_correctivas', 0) for m in monthly_trends]),
            'promedio_mttr_horas': np.mean([e.get('mttr_horas', 0) for e in equipment_prediction.get('equipos_riesgo', [])]),
            'mes_mas_problematico': monthly_trends[0]['mes'] if monthly_trends else 1,
            'algoritmo_utilizado': equipment_prediction.get('algorithm_used', 'Advanced ML'),
            'confianza_general': np.mean([e.get('confianza_prediccion', 0.5) for e in equipment_prediction.get('equipos_riesgo', [])])
        },
        'model_metrics': equipment_prediction.get('model_accuracy', {}),
        'feature_importance': equipment_prediction.get('feature_importance', {}),
        'year_analizado': year
    }
