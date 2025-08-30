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
        """Entrenar modelos de ML REALES que aprenden de los datos"""
        if not ML_AVAILABLE:
            print("❌ ML libraries not available")
            return False
        
        try:
            print("🤖 Iniciando entrenamiento de Machine Learning REAL...")
            
            # 1. Feature Engineering automático
            features_df, original_df = self.automatic_feature_engineering(df)
            
            if features_df.empty:
                print("❌ No se pudieron generar features")
                return False
            
            # 2. Crear targets automáticamente
            targets = self.create_ml_targets(original_df, features_df)
            
            if not targets:
                print("❌ No se pudieron generar targets")
                return False
            
            # 3. Preparar datos para entrenamiento
            X = features_df.values
            
            # Normalizar features
            self.scalers['main'] = StandardScaler()
            X_scaled = self.scalers['main'].fit_transform(X)
            
            print(f"📊 Datos preparados: {X_scaled.shape[0]} muestras, {X_scaled.shape[1]} features")
            
            # 4. Entrenar múltiples modelos para cada target
            for target_name, y in targets.items():
                print(f"\n🔄 Entrenando modelos para target: {target_name}")
                
                try:
                    # Filtrar datos válidos
                    valid_mask = ~(pd.isna(y) | pd.isna(X_scaled).any(axis=1))
                    X_valid = X_scaled[valid_mask]
                    y_valid = y[valid_mask]
                    
                    if len(y_valid) < 10:
                        print(f"⚠️  Pocos datos válidos para {target_name}: {len(y_valid)}")
                        continue
                    
                    # Decidir tipo de problema (regresión vs clasificación)
                    is_classification = len(np.unique(y_valid)) <= 10 and target_name in ['criticidad', 'reincidencia_90d', 'es_anomalia']
                    
                    if is_classification:
                        print(f"   📋 Problema de clasificación detectado")
                        
                        # Split de datos
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_valid, y_valid, test_size=0.2, random_state=42, stratify=y_valid
                        )
                        
                        # Random Forest Classifier
                        rf_model = RandomForestClassifier(
                            n_estimators=100, 
                            max_depth=10, 
                            random_state=42,
                            n_jobs=-1
                        )
                        rf_model.fit(X_train, y_train)
                        
                        # Validación cruzada
                        cv_scores = cross_val_score(rf_model, X_valid, y_valid, cv=5, scoring='accuracy')
                        
                        # Feature importance
                        feature_importance = pd.DataFrame({
                            'feature': self.feature_names,
                            'importance': rf_model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        
                        self.models[target_name] = rf_model
                        self.feature_importance[target_name] = feature_importance
                        self.model_performance[target_name] = {
                            'type': 'classification',
                            'cv_mean': cv_scores.mean(),
                            'cv_std': cv_scores.std(),
                            'test_accuracy': rf_model.score(X_test, y_test),
                            'n_samples': len(y_valid)
                        }
                        
                        print(f"   ✅ Accuracy: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                        print(f"   🔝 Top features: {feature_importance.head(3)['feature'].tolist()}")
                        
                    else:
                        print(f"   📈 Problema de regresión detectado")
                        
                        # Split de datos
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_valid, y_valid, test_size=0.2, random_state=42
                        )
                        
                        # Random Forest Regressor
                        rf_model = RandomForestRegressor(
                            n_estimators=100, 
                            max_depth=10, 
                            random_state=42,
                            n_jobs=-1
                        )
                        rf_model.fit(X_train, y_train)
                        
                        # XGBoost si está disponible
                        if XGB_AVAILABLE:
                            xgb_model = xgb.XGBRegressor(
                                n_estimators=100,
                                max_depth=6,
                                random_state=42,
                                n_jobs=-1
                            )
                            xgb_model.fit(X_train, y_train)
                            
                            # Comparar modelos
                            rf_score = rf_model.score(X_test, y_test)
                            xgb_score = xgb_model.score(X_test, y_test)
                            
                            if xgb_score > rf_score:
                                best_model = xgb_model
                                model_type = 'XGBoost'
                            else:
                                best_model = rf_model
                                model_type = 'RandomForest'
                        else:
                            best_model = rf_model
                            model_type = 'RandomForest'
                        
                        # Validación cruzada
                        cv_scores = cross_val_score(best_model, X_valid, y_valid, cv=5, scoring='r2')
                        
                        # Feature importance
                        if hasattr(best_model, 'feature_importances_'):
                            feature_importance = pd.DataFrame({
                                'feature': self.feature_names,
                                'importance': best_model.feature_importances_
                            }).sort_values('importance', ascending=False)
                        else:
                            feature_importance = pd.DataFrame()
                        
                        # Métricas
                        y_pred = best_model.predict(X_test)
                        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                        
                        self.models[target_name] = best_model
                        self.feature_importance[target_name] = feature_importance
                        self.model_performance[target_name] = {
                            'type': 'regression',
                            'model_used': model_type,
                            'cv_r2_mean': cv_scores.mean(),
                            'cv_r2_std': cv_scores.std(),
                            'test_r2': best_model.score(X_test, y_test),
                            'rmse': rmse,
                            'n_samples': len(y_valid)
                        }
                        
                        print(f"   ✅ R² Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f}) - {model_type}")
                        print(f"   📊 RMSE: {rmse:.2f}")
                        if not feature_importance.empty:
                            print(f"   🔝 Top features: {feature_importance.head(3)['feature'].tolist()}")
                
                except Exception as e:
                    print(f"   ❌ Error entrenando {target_name}: {e}")
                    continue
            
            # 5. Entrenar modelo de clustering
            try:
                print(f"\n🔄 Entrenando modelo de clustering...")
                
                # K-means para agrupar equipos por comportamiento
                n_clusters = min(8, len(np.unique(original_df.iloc[:, 0])) // 3)  # Ajustar número de clusters
                
                if n_clusters >= 2:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    
                    # Calcular silhouette score
                    sil_score = silhouette_score(X_scaled, clusters)
                    
                    self.models['clustering'] = kmeans
                    self.model_performance['clustering'] = {
                        'type': 'clustering',
                        'n_clusters': n_clusters,
                        'silhouette_score': sil_score,
                        'n_samples': len(X_scaled)
                    }
                    
                    print(f"   ✅ Clustering completado: {n_clusters} clusters, Silhouette: {sil_score:.3f}")
                
            except Exception as e:
                print(f"   ❌ Error en clustering: {e}")
            
            # 6. Guardar historia de entrenamiento
            self.training_history.append({
                'timestamp': datetime.now(),
                'n_samples': len(df),
                'n_features': len(self.feature_names),
                'models_trained': list(self.models.keys()),
                'performance': self.model_performance.copy()
            })
            
            self.is_trained = True
            
            print(f"\n🎉 Entrenamiento completado exitosamente!")
            print(f"📊 Modelos entrenados: {list(self.models.keys())}")
            print(f"🔬 Features generadas: {len(self.feature_names)}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error en entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict_with_real_ml(self, equipment_data=None, df=None):
        """Hacer predicciones usando los modelos ML reales entrenados"""
        if not self.is_trained:
            return {"error": "Modelos no entrenados"}
        
        try:
            predictions = {}
            
            # Si se proporciona un equipo específico, hacer predicción individual
            if equipment_data:
                # TODO: Implementar predicción individual
                predictions['individual'] = "Funcionalidad en desarrollo"
            
            # Si se proporciona DataFrame, hacer predicciones en lote
            if df is not None:
                # Generar features de la misma manera que en entrenamiento
                features_df, _ = self.automatic_feature_engineering(df)
                
                if not features_df.empty:
                    X = features_df.values
                    X_scaled = self.scalers['main'].transform(X)
                    
                    # Hacer predicciones con cada modelo
                    for model_name, model in self.models.items():
                        if model_name == 'clustering':
                            clusters = model.predict(X_scaled)
                            predictions[model_name] = {
                                'clusters': clusters.tolist(),
                                'cluster_centers': model.cluster_centers_.tolist()
                            }
                        else:
                            pred = model.predict(X_scaled)
                            
                            # Agregar intervalos de confianza para regresión
                            if hasattr(model, 'predict_proba'):
                                # Clasificación - probabilidades
                                probas = model.predict_proba(X_scaled)
                                predictions[model_name] = {
                                    'predictions': pred.tolist(),
                                    'probabilities': probas.tolist(),
                                    'confidence': np.max(probas, axis=1).tolist()
                                }
                            else:
                                # Regresión - estimaciones de incertidumbre
                                predictions[model_name] = {
                                    'predictions': pred.tolist(),
                                    'confidence': 'high'  # TODO: Implementar incertidumbre real
                                }
            
            # Agregar información del modelo
            predictions['model_info'] = {
                'trained_at': self.training_history[-1]['timestamp'].isoformat() if self.training_history else None,
                'models_available': list(self.models.keys()),
                'feature_count': len(self.feature_names),
                'performance': self.model_performance
            }
            
            return predictions
            
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
            return {"error": str(e)}
    
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
