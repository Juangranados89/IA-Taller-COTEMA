"""
Motor de Machine Learning REAL para COTEMA Analytics
Implementa algoritmos que aprenden automáticamente de los datos históricos
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, classification_report, silhouette_score
    from sklearn.feature_selection import SelectKBest, f_regression
    import joblib
    ML_AVAILABLE = True
    print("✅ Scikit-learn libraries loaded for REAL ML")
except ImportError as e:
    print(f"⚠️  ML libraries not available: {e}")
    ML_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print("✅ XGBoost available")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️  XGBoost not available")


class RealCOTEMAMLEngine:
    """Motor de Machine Learning REAL que aprende automáticamente de datos COTEMA"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.is_trained = False
        self.training_history = []
        self.feature_importance = {}
        self.model_performance = {}
        
    def automatic_feature_engineering(self, df):
        """Feature Engineering automático - Deja que el ML descubra qué es importante"""
        try:
            print("🔬 Iniciando Feature Engineering automático...")
            
            features_df = pd.DataFrame()
            original_df = df.copy()
            
            # ===== 1. DETECCIÓN AUTOMÁTICA DE COLUMNAS =====
            date_cols = []
            categorical_cols = []
            numeric_cols = []
            equipment_col = None
            
            for col in df.columns:
                col_name = str(col).lower()
                
                # Detectar columna de equipos
                if any(keyword in col_name for keyword in ['codigo', 'equipo', 'id']):
                    equipment_col = col
                    continue
                
                # Detectar fechas
                if 'fecha' in col_name or df[col].dtype == 'datetime64[ns]':
                    date_cols.append(col)
                    continue
                
                # Detectar categóricas
                if df[col].dtype == 'object' or df[col].nunique() < 20:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            
            print(f"📊 Detectado: {len(date_cols)} fechas, {len(categorical_cols)} categóricas, {len(numeric_cols)} numéricas")
            
            # ===== 2. FEATURES TEMPORALES AUTOMÁTICAS =====
            for date_col in date_cols:
                try:
                    dates = pd.to_datetime(df[date_col], errors='coerce')
                    prefix = date_col.replace('fecha_', '').replace('_', '')
                    
                    # Features temporales básicas
                    features_df[f'{prefix}_año'] = dates.dt.year
                    features_df[f'{prefix}_mes'] = dates.dt.month
                    features_df[f'{prefix}_dia_semana'] = dates.dt.dayofweek
                    features_df[f'{prefix}_trimestre'] = dates.dt.quarter
                    features_df[f'{prefix}_es_fin_semana'] = (dates.dt.dayofweek >= 5).astype(int)
                    
                    # Features cíclicas (para capturar estacionalidad)
                    features_df[f'{prefix}_mes_sin'] = np.sin(2 * np.pi * dates.dt.month / 12)
                    features_df[f'{prefix}_mes_cos'] = np.cos(2 * np.pi * dates.dt.month / 12)
                    
                    # Días desde época (para trends temporales)
                    epoch = datetime(2020, 1, 1)
                    features_df[f'{prefix}_dias_desde_epoca'] = (dates - epoch).dt.days
                    
                except Exception as e:
                    print(f"⚠️  Error procesando fecha {date_col}: {e}")
            
            # ===== 3. FEATURES DE EQUIPOS AUTOMÁTICAS =====
            if equipment_col:
                # Extraer tipo de equipo (ej: CG-TC06 -> CG)
                equipment_types = df[equipment_col].str.extract(r'([A-Z]+)-').fillna('OTHER')
                features_df['equipo_tipo'] = equipment_types[0]
                
                # One-hot encoding automático para tipos
                if len(equipment_types[0].unique()) < 15:  # Solo si no hay demasiados tipos
                    type_dummies = pd.get_dummies(equipment_types[0], prefix='tipo')
                    features_df = pd.concat([features_df, type_dummies], axis=1)
                
                # Extraer número de equipo si existe
                equipment_numbers = df[equipment_col].str.extract(r'-([A-Z]*\d+)').fillna('0')
                features_df['equipo_numero'] = pd.to_numeric(equipment_numbers[0].str.extract(r'(\d+)')[0], errors='coerce').fillna(0)
            
            # ===== 4. FEATURES CATEGÓRICAS AUTOMÁTICAS =====
            for col in categorical_cols:
                try:
                    # One-hot encoding automático
                    unique_vals = df[col].nunique()
                    if unique_vals < 10:  # Solo para variables con pocas categorías
                        dummies = pd.get_dummies(df[col], prefix=col[:10], dummy_na=True)
                        features_df = pd.concat([features_df, dummies], axis=1)
                    else:
                        # Label encoding para variables con muchas categorías
                        le = LabelEncoder()
                        features_df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                        self.encoders[col] = le
                        
                except Exception as e:
                    print(f"⚠️  Error procesando categórica {col}: {e}")
            
            # ===== 5. FEATURES NUMÉRICAS AUTOMÁTICAS =====
            for col in numeric_cols:
                try:
                    values = pd.to_numeric(df[col], errors='coerce')
                    
                    # Features básicas
                    features_df[f'{col}_valor'] = values
                    
                    # Features derivadas automáticas
                    if values.std() > 0:  # Solo si hay variación
                        features_df[f'{col}_log'] = np.log1p(np.abs(values))
                        features_df[f'{col}_sqrt'] = np.sqrt(np.abs(values))
                        features_df[f'{col}_normalizado'] = (values - values.mean()) / (values.std() + 1e-8)
                        
                        # Binning automático
                        features_df[f'{col}_quartil'] = pd.qcut(values, q=4, labels=False, duplicates='drop')
                        
                except Exception as e:
                    print(f"⚠️  Error procesando numérica {col}: {e}")
            
            # ===== 6. FEATURES DE INTERACCIÓN AUTOMÁTICAS =====
            # Solo entre fechas (para calcular duraciones)
            if len(date_cols) >= 2:
                try:
                    date1 = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    date2 = pd.to_datetime(df[date_cols[1]], errors='coerce')
                    
                    # Duración en días
                    duracion = (date2 - date1).dt.days
                    features_df['duracion_dias'] = duracion
                    features_df['duracion_log'] = np.log1p(np.abs(duracion))
                    features_df['duracion_es_largo'] = (duracion > duracion.quantile(0.75)).astype(int)
                    
                except Exception as e:
                    print(f"⚠️  Error calculando duraciones: {e}")
            
            # ===== 7. FEATURES AGREGADAS POR EQUIPO =====
            if equipment_col:
                try:
                    # Estadísticas históricas por equipo
                    equipo_stats = df.groupby(equipment_col).agg({
                        col: ['count', 'nunique'] for col in categorical_cols + numeric_cols if col in df.columns
                    }).fillna(0)
                    
                    # Aplanar columnas multinivel
                    equipo_stats.columns = [f'hist_{col}_{stat}' for col, stat in equipo_stats.columns]
                    
                    # Merge con datos principales
                    features_df = features_df.merge(
                        equipo_stats.reset_index(), 
                        left_on=features_df.index.map(lambda x: df.iloc[x][equipment_col] if x < len(df) else None),
                        right_on=equipment_col,
                        how='left'
                    ).fillna(0)
                    
                except Exception as e:
                    print(f"⚠️  Error calculando features históricas: {e}")
            
            # ===== 8. LIMPIEZA FINAL =====
            # Eliminar columnas con varianza cero
            features_df = features_df.select_dtypes(include=[np.number])
            features_df = features_df.loc[:, features_df.var() > 1e-8]
            
            # Rellenar NaN
            features_df = features_df.fillna(features_df.median())
            
            # Guardar nombres de features
            self.feature_names = list(features_df.columns)
            
            print(f"✅ Feature Engineering completado: {len(features_df.columns)} features generadas")
            print(f"📋 Features principales: {self.feature_names[:10]}...")
            
            return features_df, original_df
            
        except Exception as e:
            print(f"❌ Error en Feature Engineering: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(), df
    
    def create_ml_targets(self, df, features_df):
        """Crear targets automáticamente basados en los datos"""
        try:
            targets = {}
            
            # ===== TARGET 1: TIEMPO HASTA PRÓXIMO EVENTO =====
            # Buscar columnas de fecha para calcular tiempo entre eventos
            date_cols = [col for col in df.columns if 'fecha' in str(col).lower()]
            
            if len(date_cols) >= 2:
                try:
                    fecha_in = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    fecha_out = pd.to_datetime(df[date_cols[1]], errors='coerce')
                    
                    # Target: días en taller (tiempo de reparación)
                    dias_reparacion = (fecha_out - fecha_in).dt.days
                    targets['tiempo_reparacion'] = np.clip(dias_reparacion.fillna(30), 1, 365)
                    
                    print(f"✅ Target 'tiempo_reparacion' creado: {targets['tiempo_reparacion'].describe()}")
                    
                except Exception as e:
                    print(f"⚠️  Error creando target temporal: {e}")
            
            # ===== TARGET 2: CLASIFICACIÓN DE CRITICIDAD =====
            # Buscar columna de tipo de atención
            atencion_col = None
            for col in df.columns:
                if 'atencion' in str(col).lower() or 'tipo' in str(col).lower():
                    atencion_col = col
                    break
            
            if atencion_col:
                try:
                    atencion_values = df[atencion_col].fillna('UNKNOWN')
                    
                    # Mapeo automático a criticidad
                    criticidad_map = {}
                    for val in atencion_values.unique():
                        val_str = str(val).upper()
                        if 'CORRECTIVA' in val_str:
                            criticidad_map[val] = 2  # Alta criticidad
                        elif 'PREVENTIVA' in val_str:
                            criticidad_map[val] = 0  # Baja criticidad
                        else:
                            criticidad_map[val] = 1  # Media criticidad
                    
                    targets['criticidad'] = atencion_values.map(criticidad_map).fillna(1)
                    
                    print(f"✅ Target 'criticidad' creado: {targets['criticidad'].value_counts()}")
                    
                except Exception as e:
                    print(f"⚠️  Error creando target de criticidad: {e}")
            
            # ===== TARGET 3: PROBABILIDAD DE REINCIDENCIA =====
            # Calcular si el equipo volverá pronto basado en historial
            equipment_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['codigo', 'equipo']):
                    equipment_col = col
                    break
            
            if equipment_col and len(date_cols) >= 1:
                try:
                    fecha_evento = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    
                    # Para cada registro, ver si hay un evento futuro del mismo equipo en 90 días
                    reincidencia = []
                    for idx, row in df.iterrows():
                        equipo = row[equipment_col]
                        fecha_actual = fecha_evento.iloc[idx]
                        
                        if pd.isna(fecha_actual):
                            reincidencia.append(0)
                            continue
                        
                        # Buscar eventos futuros del mismo equipo
                        eventos_futuros = df[
                            (df[equipment_col] == equipo) & 
                            (fecha_evento > fecha_actual) &
                            (fecha_evento <= fecha_actual + timedelta(days=90))
                        ]
                        
                        reincidencia.append(1 if len(eventos_futuros) > 0 else 0)
                    
                    targets['reincidencia_90d'] = np.array(reincidencia)
                    
                    print(f"✅ Target 'reincidencia_90d' creado: {np.mean(targets['reincidencia_90d']):.1%} tasa de reincidencia")
                    
                except Exception as e:
                    print(f"⚠️  Error creando target de reincidencia: {e}")
            
            # ===== TARGET 4: ANOMALÍA BASADA EN DURACIÓN =====
            if 'tiempo_reparacion' in targets:
                try:
                    tiempo_rep = targets['tiempo_reparacion']
                    
                    # Definir anomalías como outliers en tiempo de reparación
                    q75 = tiempo_rep.quantile(0.75)
                    q25 = tiempo_rep.quantile(0.25)
                    iqr = q75 - q25
                    upper_bound = q75 + 1.5 * iqr
                    
                    targets['es_anomalia'] = (tiempo_rep > upper_bound).astype(int)
                    
                    print(f"✅ Target 'es_anomalia' creado: {np.mean(targets['es_anomalia']):.1%} casos anómalos")
                    
                except Exception as e:
                    print(f"⚠️  Error creando target de anomalía: {e}")
            
            return targets
            
        except Exception as e:
            print(f"❌ Error creando targets: {e}")
            return {}
    
    def train_real_ml_models(self, df):
        """
        Entrena todos los modelos de ML (FR-30, RUL, etc.) usando los datos proporcionados.
        """
        if not ML_AVAILABLE:
            print("❌ ML libraries not available. Cannot train models.")
            return False
        
        try:
            print("🚀 Iniciando ciclo de entrenamiento de ML Real...")
            self.is_trained = False
            
            # 1. Feature Engineering Automático
            features_df, original_df = self.automatic_feature_engineering(df)
            if features_df.empty:
                print("❌ Falló el Feature Engineering. Abortando entrenamiento.")
                return False

            # 2. Creación de Targets de ML
            targets = self.create_ml_targets(original_df, features_df)
            
            # ===== ENTRENAMIENTO DEL MODELO FR-30 (RIESGO DE FALLA A 30 DÍAS) =====
            # Usaremos 'criticidad' como nuestro target proxy para el riesgo de falla.
            # 2 = Correctiva (falla), 1 = Media, 0 = Preventiva (no falla)
            if 'criticidad' in targets:
                print("🎯 Target 'criticidad' encontrado. Entrenando modelo FR-30...")
                
                y = targets['criticidad']
                X = features_df
                
                # Asegurarse que X y y tienen el mismo índice
                y = y.loc[X.index]

                # Normalizar features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                self.scalers['fr30'] = scaler
                self.feature_names = X.columns.tolist() # Guardar nombres de features

                # Entrenar modelo de clasificación para predecir la criticidad
                # Usamos un clasificador porque el target es categórico (0, 1, 2)
                if XGB_AVAILABLE:
                    print("⚡️ Usando XGBoost para FR-30.")
                    model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', use_label_encoder=False, random_state=42)
                else:
                    print("🌳 Usando RandomForest para FR-30.")
                    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
                
                model.fit(X_scaled, y)
                self.models['fr30'] = model
                
                print("✅ Modelo FR-30 (Criticidad) entrenado.")
                
                # Guardar importancia de features
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feature_importance = sorted(zip(self.feature_names, importances), key=lambda x: x[1], reverse=True)
                    self.feature_importance['fr30'] = feature_importance
                    print(f"🔝 Top 5 features para FR-30: {feature_importance[:5]}")

            else:
                print("⚠️ No se pudo encontrar un target adecuado para FR-30. Modelo no entrenado.")

            # (Aquí se podrían entrenar otros modelos como RUL si hubiera un target)

            self.is_trained = 'fr30' in self.models
            if self.is_trained:
                print("✅ Ciclo de entrenamiento completado. El motor de ML está listo.")
            else:
                print("❌ No se pudo entrenar ningún modelo principal.")
                
            return self.is_trained

        except Exception as e:
            print(f"❌ Error catastrófico durante el entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            self.is_trained = False
            return False

    def predict_fr30_for_all_equipment(self, df):
        """
        Predice el riesgo de falla (FR-30) para todos los equipos en el DataFrame.
        """
        if not self.is_trained or 'fr30' not in self.models:
            print("❌ Modelo FR-30 no entrenado. No se pueden hacer predicciones.")
            # Intentar entrenar sobre la marcha
            print("🔧 Intentando entrenar modelo sobre la marcha...")
            if not self.train_real_ml_models(df):
                 return None
            
        try:
            print("🔮 Iniciando predicción de riesgo FR-30 para todos los equipos...")
            
            # 1. Aplicar el mismo Feature Engineering
            features_df, _ = self.automatic_feature_engineering(df)
            if features_df.empty:
                print("❌ Falló el Feature Engineering para los datos de predicción.")
                return None

            # 2. Alinear columnas con las del entrenamiento
            # Asegurarse de que el df de predicción tenga las mismas columnas que el de entrenamiento
            missing_cols = set(self.feature_names) - set(features_df.columns)
            for c in missing_cols:
                features_df[c] = 0 # Añadir columnas faltantes con 0
            
            # Asegurarse de que el orden de las columnas es el mismo
            features_df = features_df[self.feature_names]

            # 3. Escalar los datos con el scaler guardado
            scaler = self.scalers['fr30']
            X_scaled = scaler.transform(features_df)

            # 4. Predecir probabilidades
            model = self.models['fr30']
            # Esto devuelve la probabilidad para cada clase [clase_0, clase_1, clase_2]
            probabilities = model.predict_proba(X_scaled)

            # El riesgo es la probabilidad de las clases de mayor criticidad (1 y 2)
            # Clase 0: Preventiva, Clase 1: Media, Clase 2: Correctiva
            fr30_risk = probabilities[:, 1] + probabilities[:, 2] # Suma de P(Media) y P(Alta)

            # 5. Crear DataFrame de resultados
            # Necesitamos la columna de código de equipo del df original
            equipment_col = None
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['codigo', 'equipo', 'id']):
                    equipment_col = col
                    break
            
            if not equipment_col:
                print("❌ No se encontró la columna de código de equipo en los datos originales.")
                return None

            results_df = pd.DataFrame({
                'codigo': df[equipment_col],
                'fr30_risk': fr30_risk
            })

            # Agrupar por equipo y tomar el riesgo máximo (un equipo puede tener varios registros)
            final_predictions = results_df.groupby('codigo').agg(
                fr30_risk=('fr30_risk', 'max')
            ).reset_index()

            print(f"✅ Predicción completada para {len(final_predictions)} equipos únicos.")
            
            return final_predictions

        except Exception as e:
            print(f"❌ Error durante la predicción masiva: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def train(self, df):
        """Método simplificado para entrenar el motor REAL con datos nuevos"""
        return self.train_real_ml_models(df)
    
    def predict(self, df):
        """Método simplificado para predecir con el motor REAL"""
        return self.predict_fr30_for_all_equipment(df)
    
    def get_model_insights(self):
        """Obtener insights automáticos del modelo"""
        if not self.is_trained:
            return {"error": "Modelos no entrenados"}
        
        insights = {
            'feature_importance': {},
            'model_performance': self.model_performance,
            'training_history': self.training_history,
            'recommendations': []
        }
        
        # Feature importance por modelo
        for model_name, importance_df in self.feature_importance.items():
            if not importance_df.empty:
                insights['feature_importance'][model_name] = importance_df.head(10).to_dict('records')
        
        # Generar recomendaciones automáticas
        for model_name, perf in self.model_performance.items():
            if perf['type'] == 'classification' and perf.get('cv_mean', 0) > 0.8:
                insights['recommendations'].append(f"Modelo {model_name} tiene excelente precisión ({perf['cv_mean']:.1%})")
            elif perf['type'] == 'regression' and perf.get('cv_r2_mean', 0) > 0.7:
                insights['recommendations'].append(f"Modelo {model_name} explica bien la varianza (R²={perf['cv_r2_mean']:.1%})")
        
        return insights
