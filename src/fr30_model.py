"""
Modelo FR-30: Riesgo de falla en 30 días
Implementa clasificador con HistGradientBoostingClassifier + calibración
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

class FR30Model:
    def __init__(self):
        self.model = None
        self.calibrated_model = None
        self.label_encoders = {}
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, df):
        """
        Prepara las features para el modelo FR-30
        """
        # Crear copia para no modificar el original
        data = df.copy()
        
        # Ordenar por equipo y fecha
        data = data.sort_values(['CODIGO', 'FECHA_IN'])
        
        # Crear variable objetivo (y)
        data['next_fecha_in'] = data.groupby('CODIGO')['FECHA_IN'].shift(-1)
        data['days_to_next'] = (data['next_fecha_in'] - data['FECHA_OUT'].fillna(data['FECHA_IN'])).dt.days
        data['y'] = (data['days_to_next'] <= 30).astype(int)
        
        # Solo mantener registros con ciclo cerrado (que tienen siguiente ingreso)
        data = data.dropna(subset=['days_to_next'])
        
        if len(data) == 0:
            raise ValueError("No hay suficientes ciclos cerrados para entrenar el modelo")
        
        # Features temporales
        features_df = pd.DataFrame(index=data.index)
        
        # TBF días
        features_df['tbf_dias'] = data['tbf_dias'].fillna(0)
        
        # Días desde última salida
        features_df['dias_desde_ultima_salida'] = data['dias_desde_ultima_salida'].fillna(0)
        
        # Ingresos rolling
        features_df['ingresos_30d'] = data['ingresos_30d'].fillna(0)
        features_df['ingresos_60d'] = data['ingresos_60d'].fillna(0)
        features_df['ingresos_90d'] = data['ingresos_90d'].fillna(0)
        
        # Reincidencia por sistema
        features_df['sistema_count_60d'] = data['sistema_count_60d'].fillna(0)
        
        # Severidad (MTTR)
        features_df['mttr_eq'] = data['mttr_eq'].fillna(24)
        
        # Features de contexto (categóricas)
        if 'FLOTA' in data.columns:
            features_df['flota'] = data['FLOTA'].fillna('UNKNOWN')
        else:
            features_df['flota'] = 'GENERAL'
            
        if 'CLASE' in data.columns:
            features_df['clase'] = data['CLASE'].fillna('UNKNOWN')
        else:
            features_df['clase'] = 'GENERAL'
        
        # Features adicionales
        features_df['mes'] = data['FECHA_IN'].dt.month
        features_df['dia_semana'] = data['FECHA_IN'].dt.dayofweek
        features_df['hora_ingreso'] = data['FECHA_IN'].dt.hour
        
        # TBF estadísticos
        features_df['tbf_med_60d'] = data['tbf_med_60d'].fillna(features_df['tbf_dias'])
        features_df['tbf_std_60d'] = data['tbf_std_60d'].fillna(0)
        
        # Ratio de TBF actual vs histórico
        features_df['tbf_ratio'] = np.where(
            features_df['tbf_med_60d'] > 0,
            features_df['tbf_dias'] / features_df['tbf_med_60d'],
            1.0
        )
        
        # Variables objetivo
        y = data['y']
        
        return features_df, y, data['CODIGO']
    
    def encode_categorical_features(self, features_df, fit=True):
        """
        Codifica features categóricas
        """
        categorical_cols = ['flota', 'clase']
        
        for col in categorical_cols:
            if col in features_df.columns:
                if fit:
                    le = LabelEncoder()
                    features_df[col] = le.fit_transform(features_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        # Manejar categorías no vistas
                        unique_vals = features_df[col].unique()
                        for val in unique_vals:
                            if val not in le.classes_:
                                # Agregar nueva categoría
                                le.classes_ = np.append(le.classes_, val)
                        features_df[col] = le.transform(features_df[col].astype(str))
                    else:
                        features_df[col] = 0
        
        return features_df
    
    def train_and_predict(self, df):
        """
        Entrena el modelo y genera predicciones para todos los equipos
        """
        try:
            # Preparar features
            X, y, equipos = self.prepare_features(df)
            
            if len(X) < 10:
                # Si hay muy pocos datos, usar modelo simple
                return self._simple_model_fallback(df)
            
            # Codificar features categóricas
            X = self.encode_categorical_features(X, fit=True)
            self.feature_names = list(X.columns)
            
            # Split temporal para validación
            tscv = TimeSeriesSplit(n_splits=3)
            
            # Entrenar modelo base
            self.model = HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=6,
                min_samples_leaf=20,
                random_state=42
            )
            
            # Calibrar modelo para obtener probabilidades bien calibradas
            self.calibrated_model = CalibratedClassifierCV(
                self.model, 
                method='isotonic', 
                cv=tscv
            )
            
            # Entrenar
            self.calibrated_model.fit(X, y)
            self.is_trained = True
            
            # Predecir probabilidades para el estado actual de cada equipo
            current_state = self._get_current_equipment_state(df)
            predictions = self._predict_current_risk(current_state)
            
            # Guardar modelo
            self._save_model()
            
            return predictions
            
        except Exception as e:
            print(f"Error en FR30Model: {str(e)}")
            return self._simple_model_fallback(df)
    
    def _get_current_equipment_state(self, df):
        """
        Obtiene el estado actual de cada equipo (último registro)
        """
        # Obtener último registro por equipo
        latest_records = df.sort_values('FECHA_IN').groupby('CODIGO').last()
        
        # Preparar features para el estado actual
        current_features = []
        
        for codigo in latest_records.index:
            equipo_data = df[df['CODIGO'] == codigo].sort_values('FECHA_IN')
            
            if len(equipo_data) == 0:
                continue
                
            last_record = equipo_data.iloc[-1]
            
            # Calcular features actuales
            features = {
                'tbf_dias': last_record.get('tbf_dias', 0) or 0,
                'dias_desde_ultima_salida': last_record.get('dias_desde_ultima_salida', 0) or 0,
                'ingresos_30d': last_record.get('ingresos_30d', 0) or 0,
                'ingresos_60d': last_record.get('ingresos_60d', 0) or 0,
                'ingresos_90d': last_record.get('ingresos_90d', 0) or 0,
                'sistema_count_60d': last_record.get('sistema_count_60d', 0) or 0,
                'mttr_eq': last_record.get('mttr_eq', 24) or 24,
                'flota': last_record.get('FLOTA', 'GENERAL') or 'GENERAL',
                'clase': last_record.get('CLASE', 'GENERAL') or 'GENERAL',
                'mes': last_record['FECHA_IN'].month,
                'dia_semana': last_record['FECHA_IN'].dayofweek,
                'hora_ingreso': last_record['FECHA_IN'].hour,
                'tbf_med_60d': last_record.get('tbf_med_60d', last_record.get('tbf_dias', 0)) or 0,
                'tbf_std_60d': last_record.get('tbf_std_60d', 0) or 0
            }
            
            # Calcular ratio TBF
            features['tbf_ratio'] = (
                features['tbf_dias'] / features['tbf_med_60d'] 
                if features['tbf_med_60d'] > 0 else 1.0
            )
            
            current_features.append((codigo, features))
        
        return current_features
    
    def _predict_current_risk(self, current_state):
        """
        Predice el riesgo actual para cada equipo
        """
        if not self.is_trained or not current_state:
            return {}
        
        predictions = {}
        
        for codigo, features in current_state:
            try:
                # Convertir a DataFrame
                X_current = pd.DataFrame([features])
                
                # Asegurar que tenga todas las features
                for feature in self.feature_names:
                    if feature not in X_current.columns:
                        X_current[feature] = 0
                
                # Reordenar columnas
                X_current = X_current[self.feature_names]
                
                # Codificar categóricas
                X_current = self.encode_categorical_features(X_current, fit=False)
                
                # Predecir
                risk_30d = self.calibrated_model.predict_proba(X_current)[0][1]
                
                # Clasificar en banda
                if risk_30d >= 0.50:
                    banda = "🔴 ALTO (≥50%)"
                elif risk_30d >= 0.30:
                    banda = "🟠 MEDIO (30-49%)"
                else:
                    banda = "🟢 BAJO (<30%)"
                
                predictions[codigo] = {
                    'risk_30d': float(risk_30d),
                    'banda': banda,
                    'features': features
                }
                
            except Exception as e:
                print(f"Error prediciendo para equipo {codigo}: {str(e)}")
                predictions[codigo] = {
                    'risk_30d': 0.0,
                    'banda': "⚪ SIN DATO",
                    'features': features
                }
        
        return predictions
    
    def _simple_model_fallback(self, df):
        """
        Modelo simple cuando no hay suficientes datos
        """
        equipos = df['CODIGO'].unique()
        predictions = {}
        
        for equipo in equipos:
            equipo_data = df[df['CODIGO'] == equipo]
            
            # Modelo simple basado en frecuencia de ingresos
            ingresos_30d = len(equipo_data[
                equipo_data['FECHA_IN'] >= (equipo_data['FECHA_IN'].max() - pd.Timedelta(days=30))
            ])
            
            # Risk simple basado en ingresos recientes
            if ingresos_30d >= 3:
                risk = 0.7
            elif ingresos_30d == 2:
                risk = 0.4
            elif ingresos_30d == 1:
                risk = 0.2
            else:
                risk = 0.1
            
            # Ajustar por TBF promedio
            tbf_avg = equipo_data['tbf_dias'].mean()
            if pd.notna(tbf_avg) and tbf_avg < 15:
                risk = min(0.9, risk * 1.5)
            
            # Banda
            if risk >= 0.50:
                banda = "🔴 ALTO (≥50%)"
            elif risk >= 0.30:
                banda = "🟠 MEDIO (30-49%)"
            else:
                banda = "🟢 BAJO (<30%)"
            
            predictions[equipo] = {
                'risk_30d': float(risk),
                'banda': banda,
                'features': {
                    'ingresos_30d': ingresos_30d,
                    'tbf_promedio': float(tbf_avg) if pd.notna(tbf_avg) else 0,
                    'modelo': 'simple'
                }
            }
        
        return predictions
    
    def get_equipment_details(self, equipo):
        """
        Obtiene detalles del modelo para un equipo específico
        """
        # Esta función se implementaría para obtener detalles específicos
        # del modelo para un equipo dado
        return {
            'risk_30d': 0.0,
            'features': {},
            'model_type': 'FR30'
        }
    
    def _save_model(self):
        """
        Guarda el modelo entrenado
        """
        if self.is_trained:
            try:
                model_data = {
                    'calibrated_model': self.calibrated_model,
                    'label_encoders': self.label_encoders,
                    'feature_names': self.feature_names
                }
                joblib.dump(model_data, 'models/fr30_model.pkl')
                print("Modelo FR-30 guardado exitosamente")
            except Exception as e:
                print(f"Error guardando modelo FR-30: {str(e)}")
    
    def load_model(self, model_path='models/fr30_model.pkl'):
        """
        Carga un modelo previamente entrenado
        """
        try:
            model_data = joblib.load(model_path)
            self.calibrated_model = model_data['calibrated_model']
            self.label_encoders = model_data['label_encoders']
            self.feature_names = model_data['feature_names']
            self.is_trained = True
            print("Modelo FR-30 cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error cargando modelo FR-30: {str(e)}")
            return False
