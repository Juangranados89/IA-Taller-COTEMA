"""
Modelo de Detección de Anomalías Operativas
Detecta patrones atípicos que preceden fallas usando Isolation Forest
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class AnomalyModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        self.contamination = 0.1  # Porcentaje esperado de anomalías
        
    def prepare_anomaly_features(self, df):
        """
        Prepara features para detección de anomalías
        """
        # Crear ventanas de análisis por equipo
        anomaly_features = []
        
        for equipo in df['CODIGO'].unique():
            equipo_data = df[df['CODIGO'] == equipo].sort_values('FECHA_IN')
            
            if len(equipo_data) < 5:  # Muy pocos datos
                continue
            
            # Calcular features de ventana reciente para cada punto en el tiempo
            for i in range(4, len(equipo_data)):  # Necesitamos al menos 5 puntos
                current_date = equipo_data.iloc[i]['FECHA_IN']
                
                # Ventana de 30 días hacia atrás
                window_30d = equipo_data[
                    (equipo_data['FECHA_IN'] <= current_date) &
                    (equipo_data['FECHA_IN'] >= current_date - pd.Timedelta(days=30))
                ]
                
                # Ventana de 60 días hacia atrás
                window_60d = equipo_data[
                    (equipo_data['FECHA_IN'] <= current_date) &
                    (equipo_data['FECHA_IN'] >= current_date - pd.Timedelta(days=60))
                ]
                
                if len(window_30d) == 0 or len(window_60d) == 0:
                    continue
                
                # Features de anomalía
                features = self._calculate_window_features(window_30d, window_60d, equipo_data.iloc[i])
                features['CODIGO'] = equipo
                features['fecha'] = current_date
                features['periodo'] = self._get_period_label(current_date, equipo_data)
                
                anomaly_features.append(features)
        
        if not anomaly_features:
            raise ValueError("No se pudieron extraer features de anomalía")
        
        return pd.DataFrame(anomaly_features)
    
    def _calculate_window_features(self, window_30d, window_60d, current_record):
        """
        Calcula features de ventana para detección de anomalías
        """
        features = {}
        
        # 1. Ingresos en ventanas
        features['ingresos_30d'] = len(window_30d)
        features['ingresos_60d'] = len(window_60d)
        
        # 2. TBF estadísticos
        tbf_30d = window_30d['tbf_dias'].dropna()
        if len(tbf_30d) > 0:
            features['tbf_med_30d'] = tbf_30d.median()
            features['tbf_std_30d'] = tbf_30d.std()
            features['tbf_min_30d'] = tbf_30d.min()
            features['tbf_max_30d'] = tbf_30d.max()
        else:
            features['tbf_med_30d'] = 0
            features['tbf_std_30d'] = 0
            features['tbf_min_30d'] = 0
            features['tbf_max_30d'] = 0
        
        tbf_60d = window_60d['tbf_dias'].dropna()
        if len(tbf_60d) > 0:
            features['tbf_med_60d'] = tbf_60d.median()
            # Tendencia TBF (slope)
            if len(tbf_60d) >= 3:
                x = np.arange(len(tbf_60d))
                features['tbf_trend'] = np.polyfit(x, tbf_60d.values, 1)[0]
            else:
                features['tbf_trend'] = 0
        else:
            features['tbf_med_60d'] = 0
            features['tbf_trend'] = 0
        
        # 3. Sistemas afectados
        sistemas_30d = window_30d['SISTEMA_AFECTADO'].value_counts()
        features['sistemas_unicos_30d'] = len(sistemas_30d)
        features['sistema_dominante_30d'] = sistemas_30d.iloc[0] if len(sistemas_30d) > 0 else 0
        features['diversidad_sistemas_30d'] = 1 - (features['sistema_dominante_30d'] / max(1, features['ingresos_30d']))
        
        sistemas_60d = window_60d['SISTEMA_AFECTADO'].value_counts()
        features['sistemas_unicos_60d'] = len(sistemas_60d)
        
        # 4. MTTR estadísticos
        mttr_30d = window_30d['ciclo_horas'].dropna()
        if len(mttr_30d) > 0:
            features['mttr_med_30d'] = mttr_30d.median()
            features['mttr_std_30d'] = mttr_30d.std()
            features['mttr_max_30d'] = mttr_30d.max()
        else:
            features['mttr_med_30d'] = 24
            features['mttr_std_30d'] = 0
            features['mttr_max_30d'] = 24
        
        mttr_90d = window_60d['ciclo_horas'].dropna()  # Usamos 60d como aproximación a 90d
        if len(mttr_90d) > 0:
            features['mttr_med_90d'] = mttr_90d.median()
        else:
            features['mttr_med_90d'] = 24
        
        # 5. Patrones temporales
        features['dia_semana'] = current_record['FECHA_IN'].dayofweek
        features['mes'] = current_record['FECHA_IN'].month
        features['es_fin_semana'] = 1 if features['dia_semana'] >= 5 else 0
        
        # 6. Ratios y cambios
        # Ratio de actividad reciente vs histórica
        features['ratio_actividad_30_60'] = (
            features['ingresos_30d'] / max(1, features['ingresos_60d'] - features['ingresos_30d'])
        )
        
        # Cambio en TBF
        if features['tbf_med_60d'] > 0:
            features['cambio_tbf'] = features['tbf_med_30d'] / features['tbf_med_60d']
        else:
            features['cambio_tbf'] = 1.0
        
        # Cambio en MTTR
        if features['mttr_med_90d'] > 0:
            features['cambio_mttr'] = features['mttr_med_30d'] / features['mttr_med_90d']
        else:
            features['cambio_mttr'] = 1.0
        
        return features
    
    def _get_period_label(self, fecha, equipo_data):
        """
        Etiqueta el período como normal o pre-falla
        """
        # Buscar si hay una falla significativa en los próximos 30 días
        future_window = equipo_data[
            (equipo_data['FECHA_IN'] > fecha) &
            (equipo_data['FECHA_IN'] <= fecha + pd.Timedelta(days=30))
        ]
        
        # Considerar como pre-falla si hay múltiples ingresos en poco tiempo
        if len(future_window) >= 2:
            return 'pre_falla'
        else:
            return 'normal'
    
    def train_isolation_forest(self, features_df):
        """
        Entrena el modelo Isolation Forest
        """
        # Filtrar features numéricas
        numeric_features = features_df.select_dtypes(include=[np.number])
        
        # Remover columnas no feature
        exclude_cols = ['fecha', 'CODIGO']
        feature_cols = [col for col in numeric_features.columns if col not in exclude_cols]
        
        X = numeric_features[feature_cols].fillna(0)
        self.feature_names = feature_cols
        
        # Escalar features
        X_scaled = self.scaler.fit_transform(X)
        
        # Entrenar Isolation Forest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            bootstrap=False
        )
        
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Calcular scores de entrenamiento
        anomaly_scores = self.model.decision_function(X_scaled)
        anomaly_labels = self.model.predict(X_scaled)
        
        # Normalizar scores a 0-1
        scores_normalized = self._normalize_scores(anomaly_scores)
        
        return scores_normalized, anomaly_labels
    
    def _normalize_scores(self, scores):
        """
        Normaliza scores de anomalía a rango 0-1
        """
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return np.zeros_like(scores)
        
        # Invertir para que 1 = más anómalo
        normalized = 1 - (scores - min_score) / (max_score - min_score)
        return normalized
    
    def detect_anomalies(self, df):
        """
        Detecta anomalías en el dataset actual
        """
        try:
            # Preparar features
            features_df = self.prepare_anomaly_features(df)
            
            if len(features_df) < 10:
                return self._simple_anomaly_fallback(df)
            
            # Entrenar modelo
            scores, labels = self.train_isolation_forest(features_df)
            
            # Calcular anomalías por equipo (estado actual)
            anomaly_results = self._calculate_current_anomalies(df, features_df, scores)
            
            # Guardar modelo
            self._save_model()
            
            return anomaly_results
            
        except Exception as e:
            print(f"Error en AnomalyModel: {str(e)}")
            return self._simple_anomaly_fallback(df)
    
    def _calculate_current_anomalies(self, df, features_df, scores):
        """
        Calcula scores de anomalía para el estado actual de cada equipo
        """
        anomaly_results = {}
        
        for equipo in df['CODIGO'].unique():
            try:
                # Obtener registros más recientes del equipo
                equipo_features = features_df[features_df['CODIGO'] == equipo]
                
                if len(equipo_features) == 0:
                    anomaly_results[equipo] = {
                        'anomaly_score': 0.0,
                        'banda': "⚪ SIN DATO",
                        'features': {}
                    }
                    continue
                
                # Usar los features más recientes
                latest_idx = equipo_features['fecha'].idxmax()
                latest_features = equipo_features.loc[latest_idx]
                
                # Buscar el score correspondiente
                feature_idx = features_df.index.get_loc(latest_idx)
                anomaly_score = scores[feature_idx]
                
                # Clasificar en banda
                if anomaly_score >= 0.8:
                    banda = "🔴 ALTO (≥80%)"
                elif anomaly_score >= 0.6:
                    banda = "🟠 MEDIO (60-79%)"
                else:
                    banda = "🟢 BAJO (<60%)"
                
                # Extraer features principales
                key_features = {
                    'ingresos_30d': latest_features.get('ingresos_30d', 0),
                    'ingresos_60d': latest_features.get('ingresos_60d', 0),
                    'tbf_med_30d': latest_features.get('tbf_med_30d', 0),
                    'tbf_trend': latest_features.get('tbf_trend', 0),
                    'sistemas_unicos_30d': latest_features.get('sistemas_unicos_30d', 0),
                    'mttr_med_30d': latest_features.get('mttr_med_30d', 0),
                    'ratio_actividad_30_60': latest_features.get('ratio_actividad_30_60', 1),
                    'cambio_tbf': latest_features.get('cambio_tbf', 1),
                    'cambio_mttr': latest_features.get('cambio_mttr', 1)
                }
                
                anomaly_results[equipo] = {
                    'anomaly_score': float(anomaly_score),
                    'banda': banda,
                    'features': key_features,
                    'fecha_analisis': latest_features['fecha'].strftime('%Y-%m-%d')
                }
                
            except Exception as e:
                print(f"Error calculando anomalía para equipo {equipo}: {str(e)}")
                anomaly_results[equipo] = {
                    'anomaly_score': 0.0,
                    'banda': "⚪ ERROR",
                    'features': {}
                }
        
        return anomaly_results
    
    def _simple_anomaly_fallback(self, df):
        """
        Detección simple de anomalías cuando no se puede usar Isolation Forest
        """
        anomaly_results = {}
        
        for equipo in df['CODIGO'].unique():
            equipo_data = df[df['CODIGO'] == equipo].sort_values('FECHA_IN')
            
            if len(equipo_data) == 0:
                continue
            
            # Calcular métricas simples de anomalía
            # 1. Frecuencia reciente vs histórica
            fecha_limite = equipo_data['FECHA_IN'].max() - pd.Timedelta(days=30)
            ingresos_recientes = len(equipo_data[equipo_data['FECHA_IN'] >= fecha_limite])
            ingresos_promedio_mensual = len(equipo_data) / max(1, 
                (equipo_data['FECHA_IN'].max() - equipo_data['FECHA_IN'].min()).days / 30
            )
            
            # 2. TBF reciente vs histórico
            tbf_reciente = equipo_data['tbf_dias'].tail(3).median()
            tbf_historico = equipo_data['tbf_dias'].median()
            
            # Score simple
            score = 0.0
            
            # Aumentar score por alta frecuencia
            if ingresos_recientes > ingresos_promedio_mensual * 2:
                score += 0.4
            
            # Aumentar score por TBF bajo
            if pd.notna(tbf_reciente) and pd.notna(tbf_historico):
                if tbf_reciente < tbf_historico * 0.5:
                    score += 0.3
            
            # Aumentar score por múltiples sistemas
            sistemas_recientes = equipo_data[equipo_data['FECHA_IN'] >= fecha_limite]['SISTEMA_AFECTADO'].nunique()
            if sistemas_recientes > 2:
                score += 0.2
            
            # MTTR alto
            mttr_reciente = equipo_data[equipo_data['FECHA_IN'] >= fecha_limite]['ciclo_horas'].median()
            mttr_historico = equipo_data['ciclo_horas'].median()
            if pd.notna(mttr_reciente) and pd.notna(mttr_historico):
                if mttr_reciente > mttr_historico * 1.5:
                    score += 0.1
            
            # Clasificar
            if score >= 0.6:
                banda = "🔴 ALTO (≥60%)"
            elif score >= 0.4:
                banda = "🟠 MEDIO (40-59%)"
            else:
                banda = "🟢 BAJO (<40%)"
            
            anomaly_results[equipo] = {
                'anomaly_score': float(min(1.0, score)),
                'banda': banda,
                'features': {
                    'ingresos_recientes': ingresos_recientes,
                    'ingresos_promedio': float(ingresos_promedio_mensual),
                    'tbf_reciente': float(tbf_reciente) if pd.notna(tbf_reciente) else 0,
                    'sistemas_recientes': sistemas_recientes,
                    'modelo': 'simple'
                }
            }
        
        return anomaly_results
    
    def get_anomaly_explanation(self, equipo_anomaly):
        """
        Genera explicación de la anomalía detectada
        """
        score = equipo_anomaly.get('anomaly_score', 0)
        features = equipo_anomaly.get('features', {})
        
        explicacion = []
        
        if features.get('ingresos_30d', 0) > 3:
            explicacion.append(f"Alta frecuencia de ingresos: {features['ingresos_30d']} en 30 días")
        
        if features.get('tbf_trend', 0) < -1:
            explicacion.append("Tendencia decreciente en tiempo entre fallas")
        
        if features.get('sistemas_unicos_30d', 0) > 2:
            explicacion.append(f"Múltiples sistemas afectados: {features['sistemas_unicos_30d']}")
        
        if features.get('cambio_mttr', 1) > 1.5:
            explicacion.append("Aumento significativo en tiempo de reparación")
        
        if not explicacion:
            explicacion.append("Patrones operativos dentro de rangos normales")
        
        return explicacion
    
    def _save_model(self):
        """
        Guarda el modelo entrenado
        """
        if self.is_trained:
            try:
                model_data = {
                    'model': self.model,
                    'scaler': self.scaler,
                    'feature_names': self.feature_names,
                    'contamination': self.contamination
                }
                joblib.dump(model_data, 'models/anomaly_model.pkl')
                print("Modelo Anomaly guardado exitosamente")
            except Exception as e:
                print(f"Error guardando modelo Anomaly: {str(e)}")
    
    def load_model(self, model_path='models/anomaly_model.pkl'):
        """
        Carga un modelo previamente entrenado
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.contamination = model_data['contamination']
            self.is_trained = True
            print("Modelo Anomaly cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo Anomaly: {str(e)}")
            return False
