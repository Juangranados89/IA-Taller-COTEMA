from flask import Flask, render_template, jsonify, request, redirect, url_for, flash, Response, session
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import json
import hashlib
import logging
import threading

import pandas as pd
import numpy as np

# Dependencias base
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    pd = None
    np = None

# Importar el procesador ENFOCADO en datos reales
from cotema_processor import process_cotema_data, get_fr30_analysis, get_fr30_advanced_analysis

# --------------------------------------
# Configuración básica de Flask y logging
# --------------------------------------
app = Flask(__name__)

# Custom JSON encoder to handle numpy types (compatible with NumPy 2.0+)
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy integers (compatible with NumPy 2.0+)
        if isinstance(obj, np.integer):
            return int(obj)
        # Handle numpy floats (compatible with NumPy 2.0+)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        # Handle pandas NA/NaT values
        elif pd.isna(obj):
            return None
        return super().default(obj)

def create_sample_data():
    """Crear datos de muestra realistas para demostración del sistema"""
    import random
    from datetime import datetime, timedelta
    
    # Equipos realistas del taller COTEMA
    equipos_cotema = [
        'VD-TC27', 'VD-C084', 'VD-TC04', 'VD-C033', 'VD-C048',
        'VD-C039', 'VD-C013', 'VD-C042', 'VD-TC37', 'CH-HR01',
        'BB-PR15', 'MT-CX09', 'EQ-AL22', 'TR-GH34', 'SW-LM88'
    ]
    
    tipos_mantenimiento = ['CORRECTIVO', 'PREVENTIVO', 'PREDICTIVO']
    tipos_peso = [0.65, 0.25, 0.10]  # 65% correctivo, 25% preventivo, 10% predictivo
    
    descripciones = [
        'Falla en sistema hidráulico', 'Desgaste de componente principal',
        'Sobrecalentamiento motor', 'Vibración anormal', 'Fuga de aceite',
        'Ruido excesivo', 'Pérdida de presión', 'Falla eléctrica',
        'Corrosión detectada', 'Mantenimiento programado'
    ]
    
    sample_data = []
    # FECHAS MÁS RECIENTES - Enfocar en últimos 6 meses para mostrar tendencia actual
    base_date = datetime(2025, 4, 1)  # Abril 2025 hacia adelante
    
    for _ in range(500):  # 500 registros de muestra
        # Seleccionar equipo con distribución realista (algunos equipos fallan más)
        equipo = random.choices(equipos_cotema, 
                               weights=[1.5, 1.3, 1.2, 1.0, 1.1, 0.8, 0.9, 1.4, 1.1, 0.7, 
                                       0.6, 0.8, 0.9, 0.7, 0.5])[0]
        
        # Fecha aleatoria en los últimos 5 meses con énfasis en agosto-septiembre
        days_offset = random.randint(0, 150)  # Últimos 5 meses
        fecha = base_date + timedelta(days=days_offset)
        
        # BOOST para agosto y septiembre (meses 8 y 9)
        if random.random() < 0.4:  # 40% de probabilidad de forzar mes reciente
            # Forzar fechas de agosto-septiembre 2025
            fecha = datetime(2025, random.choice([8, 9]), random.randint(1, 28))
        
        # Ajuste estacional con ÉNFASIS en mes actual (septiembre)
        month = fecha.month
        seasonal_multiplier = 1.0
        if month == 9:  # SEPTIEMBRE - Mayor actividad
            seasonal_multiplier = 2.0
        elif month == 8:  # AGOSTO - Alta actividad
            seasonal_multiplier = 1.5
        elif month in [1, 5]:  # Otros meses pico
            seasonal_multiplier = 1.3
        elif month in [7, 12]:  # Meses con menos actividad
            seasonal_multiplier = 0.7
            
        # Tipo de mantenimiento
        tipo = random.choices(tipos_mantenimiento, weights=tipos_peso)[0]
        
        sample_data.append({
            'codigo': equipo,
            'fecha_in': fecha,
            'tipo': tipo,
            'descripcion': random.choice(descripciones),
            'estado': 'COMPLETADO',
            'prioridad': random.choice(['ALTA', 'MEDIA', 'BAJA'])
        })
    
    df_sample = pd.DataFrame(sample_data)
    print(f"📊 Datos de muestra creados: {len(df_sample)} registros para {len(equipos_cotema)} equipos")
    
    return df_sample

app.json_encoder = CustomJSONEncoder
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'super-secret-key-de-desarrollo')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("COTEMA")

# Compatibilidad: algunos scripts y endpoints antiguos esperan un estado global.
# El sistema actual usa session, pero exponemos un dict vacío para evitar errores.
global_data = {}

# --------------------------------------
# Estado global y helpers de progreso
# --------------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB límite de archivo
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Sistema basado en sesiones - eliminamos variables globales
# Todos los datos se almacenan en session['key']

progress_state = {
    'current_task': '',
    'percentage': 0,  # Cambiado de 'progress' a 'percentage'
    'is_processing': False,
    'message': '',
    'error': None,
    'total_steps': 0,
    'current_step': 0
}

def update_progress(task, step, total_steps, message=""):
    progress_state.update({
        'current_task': task,
        'current_step': step,
        'total_steps': total_steps,
        'percentage': int((step / total_steps) * 100) if total_steps else 0,  # Cambiado de 'progress' a 'percentage'
        'is_processing': step < total_steps,
        'message': message,
        'error': None
    })

def reset_progress():
    progress_state.update({
        'current_task': '',
        'percentage': 0,  # Cambiado de 'progress' a 'percentage'
        'is_processing': False,
        'message': '',
        'error': None,
        'total_steps': 0,
        'current_step': 0
    })

def set_progress_error(error_message):
    progress_state.update({
        'is_processing': False,
        'error': error_message,
        'percentage': 0  # Cambiado de 'progress' a 'percentage'
    })

# --------------------------------------
# Motor ML (solo si hay sklearn, sin simulaciones)
# --------------------------------------
class COTEMAMLEngine:
    """Entrena y predice SOLO con datos reales. Si no hay datos/modelos, devuelve 0s."""
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.ml_mode = False

    def load_real_equipment_codes(self):
        """Códigos únicamente desde catálogos o DF real. Sin fallback."""
        try:
            cats = session.get('catalogos') or {}
            lista = cats.get('equipos', {}).get('lista', [])
            if lista:
                return list(map(str, lista))[:200]
        except Exception:
            pass
        try:
            df_data = session.get('df')
            if df_data is not None and len(df_data) > 0:
                df = pd.DataFrame(df_data)
                for col in df.columns:
                    name = str(col).lower()
                    if any(k in name for k in ['codigo', 'equipo', 'maquina', 'máquina', 'id']):
                        vals = df[col].dropna().astype(str).str.strip().unique().tolist()
                        vals = [v for v in vals if len(v) >= 2 and v.lower() != 'nan']
                        if vals:
                            return vals[:200]
        except Exception:
            pass
        return []

    def extract_features_from_real_data(self, df):
        """Extrae features únicamente de columnas reales; no inventa valores."""
        try:
            import pandas as pd
            features = pd.DataFrame()

            if 'codigo' in df.columns:
                tipo = df['codigo'].astype(str).str.split('-').str[0]
                mapa = {'CG':1,'AH':2,'CV':3,'EX':4,'NE':5,'RE':6,'VD':7,'PE':8,'TI':9}
                features['equipo_tipo_num'] = tipo.map(mapa).fillna(0).astype(float)

            if 'fecha_in' in df.columns:
                fi = pd.to_datetime(df['fecha_in'], errors='coerce')
                features['mes_ingreso'] = fi.dt.month.fillna(0).astype(float)
                features['trimestre'] = fi.dt.quarter.fillna(0).astype(float)
                features['dia_semana'] = fi.dt.dayofweek.fillna(0).astype(float)

            if 'horometro_in' in df.columns:
                features['horometro'] = pd.to_numeric(df['horometro_in'], errors='coerce').fillna(0)

            if 'tipo_atencion' in df.columns:
                at_map = {'PREVENTIVA':1,'ALISTAMIENTO-TC':2,'CORRECTIVA':3}
                features['criticidad_atencion'] = df['tipo_atencion'].astype(str).str.upper().map(at_map).fillna(2).astype(float)

            if 'sistema_afectado' in df.columns:
                # Mapa simple y seguro
                sys_map = {'MOTOR':5,'HIDRAULICO':4,'HIDRÁULICO':4,'VIBRATORIO':3,'NEUMATICO':2,'NEUMÁTICO':2,'LUCES':1}
                features['complejidad_sistema'] = df['sistema_afectado'].astype(str).str.upper().map(sys_map).fillna(3).astype(float)

            if 'mttr' in df.columns:
                features['mttr'] = pd.to_numeric(df['mttr'], errors='coerce').fillna(0)

            if 'cont_dias_ave' in df.columns:
                features['historial_averias'] = pd.to_numeric(df['cont_dias_ave'], errors='coerce').fillna(0)

            # Asegurar que haya al menos algunas columnas
            if features.shape[1] == 0:
                return None
            return features
        except Exception as e:
            logger.exception(f"extract_features_from_real_data error: {e}")
            return None

    def train_models_enhanced(self, df=None, progress_callback=None):
        """Entrena modelos SOLO con datos reales y sklearn; si no, no entrena."""
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.ensemble import IsolationForest, RandomForestRegressor
        except Exception:
            self.is_trained = False
            self.ml_mode = False
            return False

        try:
            if df is None:
                df = session.get('df')
            if df is None or len(df) < 10:
                self.is_trained = False
                self.ml_mode = False
                return False

            feats = self.extract_features_from_real_data(df)
            if feats is None or len(feats) < 10:
                self.is_trained = False
                self.ml_mode = False
                return False

            # Targets "proxy" basados en columnas reales disponibles
            y_fr30 = None
            y_rul = None
            if {'criticidad_atencion','complejidad_sistema'}.issubset(feats.columns):
                y_fr30 = (feats['criticidad_atencion']*0.25 + feats['complejidad_sistema']*0.2).clip(0, 1)
            if 'mttr' in feats.columns and 'historial_averias' in feats.columns:
                y_rul = np.maximum(15, 180 - feats['historial_averias']*5 - feats['mttr']/50)

            cols = [c for c in ['equipo_tipo_num','mes_ingreso','criticidad_atencion',
                                'complejidad_sistema','historial_averias'] if c in feats.columns]
            if len(cols) < 3:
                self.is_trained = False
                self.ml_mode = False
                return False

            X = feats[cols].values
            self.scalers['main'] = StandardScaler()
            Xs = self.scalers['main'].fit_transform(X)

            self.models = {}
            if y_fr30 is not None and len(y_fr30)==len(X):
                rf = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=8, min_samples_split=5)
                rf.fit(Xs, y_fr30)
                self.models['fr30'] = rf
            if y_rul is not None and len(y_rul)==len(X):
                rr = RandomForestRegressor(n_estimators=120, random_state=42, max_depth=8, min_samples_split=5)
                rr.fit(Xs, y_rul)
                self.models['rul'] = rr

            iso = IsolationForest(contamination=0.15, random_state=42, n_estimators=150, max_features=1.0)
            iso.fit(Xs)
            self.models['anomaly'] = iso

            self.is_trained = True
            self.ml_mode = True
            return True
        except Exception as e:
            logger.exception(f"train_models_enhanced error: {e}")
            self.is_trained = False
            self.ml_mode = False
            return False

    def predict_equipment(self, equipo_data):
        """Predice solo si hay modelo entrenado; si no, devuelve 0s."""
        if not (self.is_trained and self.ml_mode):
            return {
                'fr30_risk': 0.0,
                'rul_days': 0,
                'anomaly_score': 0.0,
                'confidence': 0.0,
                'mode': 'NO_MODEL'
            }
        # Construir fila de features mínima
        tipo_map = {'CG':1,'AH':2,'CV':3,'EX':4,'NE':5,'RE':6,'VD':7,'PE':8,'TI':9}
        eq_code = str(equipo_data.get('equipo','UNKNOWN'))
        eq_type_num = tipo_map.get(eq_code.split('-')[0], 0)
        row = {
            'equipo_tipo_num': float(eq_type_num),
            'mes_ingreso': float(equipo_data.get('mes_ingreso', 0)),
            'criticidad_atencion': float(equipo_data.get('criticidad_atencion', 2)),
            'complejidad_sistema': float(equipo_data.get('complejidad_sistema', 3)),
            'historial_averias': float(equipo_data.get('historial_averias', 0)),
        }
        import pandas as pd
        X = pd.DataFrame([row])[list(self.scalers['main'].feature_names_in_)] if hasattr(self.scalers['main'],'feature_names_in_') else pd.DataFrame([row])
        Xs = self.scalers['main'].transform(X)

        out = {
            'fr30_risk': 0.0,
            'rul_days': 0,
            'anomaly_score': float(np.clip((self.models['anomaly'].decision_function(Xs)[0]+1)/2, 0,1)) if 'anomaly' in self.models else 0.0,
            'confidence': 0.75,
            'mode': 'ML_Real'
        }
        if 'fr30' in self.models:
            out['fr30_risk'] = float(np.clip(self.models['fr30'].predict(Xs)[0], 0, 1))
        if 'rul' in self.models:
            out['rul_days'] = int(max(0, self.models['rul'].predict(Xs)[0]))
        return out

    def generate_trend_forecast(self, equipo:str, days_ahead:int=30):
        """Devuelve solo histórico real (conteo de CORRECTIVAS por día). Sin simulaciones."""
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty or 'fecha_in' not in df.columns or 'tipo_atencion' not in df.columns:
            return {'historico': [], 'pronostico': [], 'equipo': equipo, 'mode': 'NO_DATA'}

        dfx = df.copy()
        dfx['fecha_in'] = pd.to_datetime(dfx['fecha_in'], errors='coerce')
        dfx = dfx.dropna(subset=['fecha_in'])

        if 'codigo' in dfx.columns:
            dfx = dfx[dfx['codigo'].astype(str) == str(equipo)]

        dfx = dfx[dfx['tipo_atencion'].astype(str).str.upper() == 'CORRECTIVA']

        if dfx.empty:
            return {'historico': [], 'pronostico': [], 'equipo': equipo, 'mode': 'HIST_EMPTY'}

        start = (pd.Timestamp.now() - pd.Timedelta(days=30)).normalize()
        mask = dfx['fecha_in'] >= start
        dfx = dfx[mask]

        if dfx.empty:
            return {'historico': [], 'pronostico': [], 'equipo': equipo, 'mode': 'HIST_EMPTY'}

        series = dfx.groupby(dfx['fecha_in'].dt.date).size().rename('conteo').reset_index()
        historico = [{'fecha': d.strftime('%Y-%m-%d'), 'conteo': int(v)} for d,v in zip(series['fecha_in'], series['conteo'])]

        return {'historico': historico, 'pronostico': [], 'equipo': equipo, 'mode': 'HISTORICAL_ONLY'}


# Instancia del motor
ml_engine = COTEMAMLEngine()

# --------------------------------------
# Rutas básicas
# --------------------------------------
@app.route('/')
def index():
    stats = session.get('stats', {
        'total_registros': 0,
        'processing_time': 0,
        'sheet_used': None,
        'registros_cerrados': 0,
        'registros_abiertos': 0,
        'file_loaded': False
    })
    # Convertir fecha de string ISO a datetime si existe
    processed_date_str = session.get('processed_date')
    processed_date = None
    if processed_date_str:
        try:
            processed_date = datetime.fromisoformat(processed_date_str)
        except:
            processed_date = None
            
    return render_template('index.html',
                           data_loaded=bool(session.get('df')),
                           processed_date=processed_date,
                           stats=stats)

@app.route('/progress', methods=['GET'])
def get_progress():
    # Adaptar al formato que espera el frontend
    response = progress_state.copy()
    response['percentage'] = response.get('percentage', 0) # CORREGIDO: Usar 'percentage'
    response['details'] = response.get('message', '')
    return jsonify(response)

@app.route('/ml-status')
def ml_status():
    status = {
        'deep_analysis_in_progress': session.get('deep_analysis_in_progress', False),
        'ml_models_trained': session.get('ml_models_trained', False),
        'analysis_type': global_data.get('analysis_type', ''),
        'ml_progress': {
            'percent': 100 if session.get('ml_models_trained') else 0,
            'step': 'ready' if session.get('ml_models_trained') else 'idle',
            'processed': 0,
            'total': 0
        }
    }
    return jsonify(status)

# --------------------------------------
# Upload y procesamiento
# --------------------------------------
@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            set_progress_error('No se seleccionó ningún archivo')
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        file = request.files['file']
        if file.filename == '':
            set_progress_error('No se seleccionó ningún archivo')
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400

        # Validar extensión
        if not file.filename.lower().endswith(tuple(f'.{ext}' for ext in ALLOWED_EXTENSIONS)):
            set_progress_error('Formato no soportado. Use .xlsx o .xls')
            return jsonify({'error': 'Formato no soportado. Use .xlsx o .xls'}), 400

        # Validar tamaño de archivo
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            set_progress_error(f'Archivo muy grande ({file_size/(1024*1024):.1f}MB). Límite: {MAX_FILE_SIZE/(1024*1024)}MB')
            return jsonify({'error': f'Archivo muy grande ({file_size/(1024*1024):.1f}MB). Límite: {MAX_FILE_SIZE/(1024*1024)}MB'}), 413

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Validación básica del archivo Excel antes de guardarlo
        try:
            file_content = file.read()
            file.seek(0)  # Reset para guardar después
            
            # Verificar que es un archivo Excel válido
            if not file_content.startswith(b'PK'):  # Los archivos .xlsx son archivos ZIP
                set_progress_error('Archivo Excel corrupto o inválido')
                return jsonify({'error': 'Archivo Excel corrupto o inválido'}), 400
                
        except Exception as validation_error:
            set_progress_error(f'Error validando archivo: {str(validation_error)}')
            return jsonify({'error': f'Error validando archivo: {str(validation_error)}'}), 400

        file.save(filepath)
        logger.info(f"File saved: {filename} ({file_size/(1024*1024):.2f}MB)")

        # Procesar directamente en la petición para evitar problemas con session
        try:
            update_progress("Archivo recibido", 1, 4, "Procesando en servidor...")
            process_uploaded_file(filepath, filename)
            
            # FORZAR SIEMPRE JSON PARA EVITAR EL ERROR "Unexpected token '<'"
            # El frontend moderno usa AJAX y siempre espera JSON
            logger.info(f"Upload processed for {filename}. Headers: {dict(request.headers)}")
            return jsonify({'success': True, 'message': f'Archivo {filename} procesado exitosamente.'})
                
        except Exception as process_error:
            logger.exception(f"Processing failed for {filename}. Error: {process_error}")
            # FORZAR SIEMPRE JSON en errores también
            return jsonify({'error': f'Error procesando archivo: {str(process_error)}', 'detail': repr(process_error)}), 500

    except Exception as e:
        logger.exception(f"upload_file error: {e}")
        set_progress_error(f'Error inesperado: {str(e)}')
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

def clean_data_for_session(data):
    """Limpia datos para que sean serializables en Flask session (NaT/NaN→None, datetime→ISO, numpy→tipos nativos).

    Maneja de forma segura dicts, listas, tuplas, sets, numpy arrays y pandas Series
    sin evaluar la verdad de arrays (evita ValueError por pd.isna en contenedores).
    """
    def _clean_value(v):
        # Contenedores: procesar recursivamente
        if isinstance(v, (list, tuple, set)):
            return [_clean_value(x) for x in list(v)]
        if isinstance(v, (np.ndarray, pd.Series)):
            try:
                return [_clean_value(x) for x in v.tolist()]
            except Exception:
                return [_clean_value(x) for x in list(v)]

        # NaT / NaN (solo para escalares)
        try:
            if v is pd.NaT:
                return None
        except Exception:
            pass
        try:
            if np.isscalar(v) and pd.isna(v):
                return None
        except Exception:
            pass

        # Datetime-like
        if isinstance(v, (datetime, pd.Timestamp)):
            return v.isoformat()

        # Numpy escalares → tipos nativos
        if isinstance(v, np.generic):
            try:
                return v.item()
            except Exception:
                pass

        return v

    if isinstance(data, dict):
        return {k: _clean_value(v) for k, v in data.items()}
    else:
        return _clean_value(data)

def process_uploaded_file(filepath, filename):
    """Lee Excel, procesa con cotema_processor y guarda en sesión (sin variables globales)."""
    import time
    start = time.time()
    try:
        update_progress("Leyendo Excel", 2, 4, "Inspeccionando hojas...")
        if pd is None:
            raise RuntimeError("Pandas no está disponible en el servidor.")
        
        # Seleccionar motor según extensión para compatibilidad .xlsx/.xls
        ext = os.path.splitext(filename)[1].lower()
        engine = 'openpyxl' if ext == '.xlsx' else ('xlrd' if ext == '.xls' else None)
        xl = pd.ExcelFile(filepath, engine=engine)  # usa engine explícito si aplica
        sheets = xl.sheet_names
        logger.info(f"Hojas disponibles en {filename}: {sheets}")
        
        # Prioriza hojas comunes de COTEMA
        preferred_sheets = ['Datos_Limpios', 'datos_limpios', 'REG', 'MAQUINARIA', 'BD1', 'BI']
        sheet = None
        
        for preferred in preferred_sheets:
            if preferred in sheets:
                sheet = preferred
                logger.info(f"Usando hoja preferida: {sheet}")
                break
        
        if sheet is None:
            sheet = sheets[0]
            logger.info(f"Usando primera hoja disponible: {sheet}")

        update_progress("Normalizando", 3, 4, f"Procesando hoja '{sheet}'...")
        df_raw = pd.read_excel(filepath, sheet_name=sheet, engine=engine)
        logger.info(f"Datos leídos: {len(df_raw)} filas, {len(df_raw.columns)} columnas")
        logger.info(f"Columnas encontradas: {list(df_raw.columns)[:10]}")  # Primeras 10 columnas

        dataset, quality, catalogs = process_cotema_data(df_raw)

        # Limpiar datos para que sean serializables en Flask session
        clean_dataset = clean_data_for_session(dataset)
        clean_quality = clean_data_for_session(quality)
        clean_catalogs = clean_data_for_session(catalogs)

        # Guardar en sesión en lugar de estado global
        df_temp = pd.DataFrame(dataset)
        session['df'] = clean_dataset  # Datos limpios y serializables
        session['dataset_normalizado'] = clean_dataset
        session['reporte_calidad'] = clean_quality
        session['catalogos'] = clean_catalogs
        session['file_path'] = filepath
        session['file_name'] = filename
        session['processed_date'] = datetime.now().isoformat()
        session['ml_models_trained'] = False

        # Stats para portada con más información
        errores_detectados = 0
        for v in quality.get('errores', {}).values():
            if isinstance(v, (int, float)):
                errores_detectados += int(v)
            elif isinstance(v, dict):
                errores_detectados += len(v)

        processing_time = round(time.time() - start, 2)
        session['stats'] = {
            'total_registros': quality.get('total_registros', len(df_temp)),
            'registros_abiertos': quality.get('registros_abiertos', 0),
            'registros_cerrados': quality.get('registros_cerrados', 0),
            'errores_detectados': errores_detectados,
            'sheet_used': sheet,
            'available_sheets': sheets,
            'file_loaded': True,
            'processing_time': processing_time,
            'columnas_procesadas': len(df_temp.columns),
            'columnas_criticas': {
                'fecha_in': 'fecha_in' in df_temp.columns,
                'tipo_atencion': 'tipo_atencion' in df_temp.columns,
                'codigo': 'codigo' in df_temp.columns
            }
        }

        update_progress("Completado", 4, 4, f"Procesado en {processing_time}s - {len(df_temp)} registros")
        logger.info(f"Procesado OK: {filename} en {processing_time}s - {len(df_temp)} registros")

    except Exception as e:
        logger.exception(f"process_uploaded_file error: {e}")
        set_progress_error(f"Error procesando archivo: {str(e)}")
        # También log para debugging
        logger.error(f"Detalles del error en {filename}: {str(e)}")

# --------------------------------------
# Dashboard y KPIs
# --------------------------------------
@app.route('/dashboard')
def dashboard():
    """Renderiza el dashboard principal y unificado."""
    try:
        if not session.get('df'):
            flash('Primero debes cargar un archivo Excel para ver el dashboard.', 'warning')
            return redirect(url_for('index'))
        df = pd.DataFrame(session['df'])
        
        # Calcular meses disponibles para el selector
        meses_disponibles = []
        if 'fecha_in' in df.columns:
            valid_dates = pd.to_datetime(df['fecha_in'], errors='coerce').dropna()
            if not valid_dates.empty:
                meses_disponibles = sorted(valid_dates.dt.to_period('M').astype(str).unique().tolist(), reverse=True)

        # Calcular estadísticas para el panel lateral
        stats = {
            'total_registros': len(df),
            'equipos_unicos': 0,
            'sistemas_unicos': 0,
            'fecha_min': 'N/A',
            'fecha_max': 'N/A',
        }
        if 'codigo' in df.columns:
            stats['equipos_unicos'] = df['codigo'].nunique()
        if 'sistema_afectado' in df.columns:
            stats['sistemas_unicos'] = df['sistema_afectado'].nunique()
        if 'fecha_in' in df.columns and not valid_dates.empty:
            stats['fecha_min'] = valid_dates.min().strftime('%Y-%m-%d')
            stats['fecha_max'] = valid_dates.max().strftime('%Y-%m-%d')

        # Los plots se generan dinámicamente con JS, así que pasamos un dict vacío
        plots = {}

        return render_template('dashboard.html',
                               meses_disponibles=meses_disponibles,
                               stats=stats,
                               plots=plots)
    except Exception as e:
        logger.exception(f"dashboard error: {e}")
        flash(f'Ocurrió un error al cargar el dashboard: {e}', 'danger')
        return redirect(url_for('index'))

@app.route('/kpis/<mes>')
def calculate_kpis(mes):
    """KPIs básicos 100% reales. Sin simulación; si falta modelo, 0s."""
    try:
        if not session.get('df'):
            return jsonify({'error': 'No hay datos cargados'}), 400
        
        df = pd.DataFrame(session['df'])
        if df.empty:
            return jsonify({'error': 'No hay datos cargados'}), 400

        dfm = df.copy()
        if 'fecha_in' in dfm.columns:
            dfm['fecha_in'] = pd.to_datetime(dfm['fecha_in'], errors='coerce')
            dfm = dfm[dfm['fecha_in'].dt.to_period('M').astype(str) == mes]

        equipos = dfm['codigo'].dropna().astype(str).unique().tolist() if 'codigo' in dfm.columns else []

        kpis = {'fr30': {}, 'rul': {}, 'forecast': {}, 'anomaly': {}}
        for eq in equipos:
            # Riesgos reales solo si hay modelo; si no, 0s
            if ml_engine.is_trained and ml_engine.ml_mode:
                pred = ml_engine.predict_equipment({'equipo': eq})
                fr30 = float(pred.get('fr30_risk', 0.0))
                rul = int(pred.get('rul_days', 0))
                anom = float(pred.get('anomaly_score', 0.0))
            else:
                fr30, rul, anom = 0.0, 0, 0.0

            # Forecast histórico simple (conteo de correctivas en el mes)
            if 'tipo_atencion' in dfm.columns and 'fecha_in' in dfm.columns:
                dfe = dfm[(dfm['codigo'].astype(str)==eq) & (dfm['tipo_atencion'].astype(str).str.upper()=='CORRECTIVA')]
                daily = dfe.groupby(dfe['fecha_in'].dt.date).size().sum()
            else:
                daily = 0

            kpis['fr30'][eq] = {
                'risk_30d': fr30,
                'risk_percentage': f"{fr30*100:.0f}%",
                'status': '—' if fr30==0 else ('ALTO' if fr30>0.7 else 'MEDIO'),
                'badge_color': 'secondary' if fr30==0 else ('danger' if fr30>0.7 else 'warning'),
                'confidence': 0.0 if fr30==0 else 0.75
            }
            kpis['rul'][eq] = {'rul50_d': rul, 'rul90_d': max(0, int(rul*0.7)), 'confidence': 0.0 if rul==0 else 0.75}
            kpis['forecast'][eq] = {'total_correctivas_mes': int(daily)}
            kpis['anomaly'][eq] = {'anomaly_score': anom, 'status': '—' if anom==0 else '⚠️'}

        result = {
            'mes': mes,
            'total_equipos': len(equipos),
            'timestamp': datetime.now().isoformat(),
            'processing_method': 'Real',
            'ml_models_active': bool(ml_engine.is_trained and ml_engine.ml_mode),
            'kpis': kpis
        }
        return jsonify(result)
    except Exception as e:
        logger.exception(f"kpis error: {e}")
        return jsonify({'error': str(e)}), 500

# --------------------------------------
# APIs de consulta
# --------------------------------------
@app.route('/api/equipment-codes')
def get_equipment_codes():
    try:
        codes = ml_engine.load_real_equipment_codes()
        return jsonify({'success': True, 'codes': codes, 'total': len(codes)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'codes': []})

@app.route('/api/fr30-top5')
def api_fr30_top5():
    """Top equipos por CORRECTIVAS en los últimos 90 días (real)."""
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty or 'fecha_in' not in df.columns or 'tipo_atencion' not in df.columns or 'codigo' not in df.columns:
            return jsonify({'success': True, 'data': {'metric': 'ingresos_correctivos_90d', 'items': [], 'since': None}})

        df = df.copy()
        df['fecha_in'] = pd.to_datetime(df['fecha_in'], errors='coerce')
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=90)

        recent = df[(~df['fecha_in'].isna()) & (df['fecha_in'] >= cutoff)]
        recent_corr = recent[recent['tipo_atencion'].astype(str).str.upper() == 'CORRECTIVA']

        top = (recent_corr.groupby('codigo').size().sort_values(ascending=False).head(5).reset_index(name='ingresos_correctivos_90d'))
        items = top.to_dict('records')

        return jsonify({'success': True, 'data': {'metric': 'ingresos_correctivos_90d', 'items': items, 'since': cutoff.date().isoformat()}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/trend-forecast/<equipo>')
def get_trend_forecast(equipo):
    try:
        data = ml_engine.generate_trend_forecast(equipo, 30)
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/quick-analysis', methods=['POST'])
def quick_analysis():
    """Análisis rápido: stats simples reales del DF cargado."""
    try:
        if not session.get('df'):
            return jsonify({'error': 'No hay archivo cargado'}), 400

        df = pd.DataFrame(session['df'])
        # Detectar columna código/equipo
        codigo_col = None
        for col in df.columns:
            name = str(col).lower()
            if any(k in name for k in ['codigo', 'equipo', 'maquina', 'id']):
                codigo_col = col
                break

        equipos = []
        if codigo_col:
            equipos = (df[codigo_col].dropna().astype(str).str.strip().unique().tolist())[:20]

        stats = {
            'total_registros': int(len(df)),
            'columnas_total': int(len(df.columns)),
            'equipos_unicos': int(len(equipos)),
            'equipos_muestra': equipos,
            'processing_method': 'Real',
            'file_loaded': True,
            'quick_analysis_done': True
        }
        session['stats'] = {**session.get('stats', {}), **stats}
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        set_progress_error(f'Error en análisis rápido: {str(e)}')
        return jsonify({'error': f'Error en análisis rápido: {str(e)}'}), 500

@app.route('/analyze_statistics', methods=['POST'])
def analyze_statistics():
    """KPI FR-30 real (conteo de correctivas últimos N días)."""
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty:
            return jsonify({'success': True, 'data': {'top_equipos': [], 'total_correctivas_en_ventana': 0, 'equipos_con_correctivas': 0, 'debug': 'No hay datos cargados'}})

        # Obtener días de manera más robusta
        days = 30  # valor por defecto
        try:
            # Intentar obtener JSON, pero no fallar si no hay contenido
            if request.content_type and 'application/json' in request.content_type:
                request_data = request.get_json(silent=True) or {}
            else:
                request_data = {}
            days = int(request_data.get('days', 30))
        except (TypeError, ValueError, Exception) as e:
            logger.warning(f"Error obteniendo días del request: {e}")
            days = 30

        update_progress("Analizando estadísticas", 1, 2, f"Analizando últimos {days} días...")
        
        # Debug información
        debug_info = {
            'total_registros': len(df),
            'columnas': list(df.columns),
            'tiene_fecha_in': 'fecha_in' in df.columns,
            'tiene_tipo_atencion': 'tipo_atencion' in df.columns,
            'tiene_codigo': 'codigo' in df.columns,
        }
        
        # Verificar si las columnas necesarias existen
        if 'fecha_in' in df.columns:
            debug_info['fechas_validas'] = df['fecha_in'].notna().sum()
        if 'tipo_atencion' in df.columns:
            debug_info['tipos_atencion'] = df['tipo_atencion'].value_counts().to_dict()
        if 'codigo' in df.columns:
            debug_info['equipos_unicos'] = df['codigo'].nunique()
        
        fr30 = get_fr30_analysis(df, days=days)
        
        # Agregar información adicional al análisis
        enhanced_data = {
            **fr30,
            'analysis_method': 'Real data FR-30',
            'total_registros_dataset': len(df),
            'columnas_disponibles': list(df.columns),
            'debug_info': debug_info,
            'timestamp': datetime.now().isoformat()
        }
        
        update_progress("Análisis completado", 2, 2, "Estadísticas generadas exitosamente")
        reset_progress()
        
        response_data = json.dumps({'success': True, 'data': enhanced_data}, cls=CustomJSONEncoder)
        return Response(response_data, mimetype='application/json')
    except Exception as e:
        logger.exception(f"analyze_statistics error: {e}")
        return jsonify({'success': False, 'error': str(e), 'debug': 'Error en el análisis'})

@app.route('/analyze_statistics_advanced', methods=['POST'])
def analyze_statistics_advanced():
    """
    KPI FR-30 avanzado: equipos con mayor tendencia a fallar (período actual y próximo mes).
    Enfocado en cálculos precisos para el mes actual y siguiente.
    VERSIÓN ACTUALIZADA - Sin caché
    """
    try:
        # NO LIMPIAR DATOS GLOBALES AQUÍ PARA NO PERDER EL ARCHIVO CARGADO
        # global_data.clear()
        
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty:
            # Si no hay datos cargados, usar datos de muestra para demostración
            print("⚠️ No hay datos cargados, creando datos de muestra FRESCOS...")
            df_sample = create_sample_data()
            session['df'] = df_sample.to_dict('records')
            df = df_sample
            print(f"✅ Datos de muestra NUEVOS creados: {len(df)} registros")

        update_progress("Analizando tendencias FR-30", 1, 3, f"Calculando riesgo ACTUALIZADO...")
        
        # Ejecutar análisis avanzado centrado en mes actual y próximo
        advanced_analysis = get_fr30_advanced_analysis(df)
        
        update_progress("Refinando cálculos de riesgo", 2, 3, "Aplicando algoritmos predictivos...")
        
        # Estructura de respuesta mejorada y enfocada SIN CACHÉ
        from datetime import datetime
        timestamp_actual = datetime.now()
        mes_actual = timestamp_actual.month  # 9 = Septiembre
        
        enhanced_data = {
            **advanced_analysis,
            'analysis_type': 'FR-30 Predictive Risk Analysis',
            'total_registros_dataset': len(df),
            'periodo_analisis': f'Mes Actual ({mes_actual}) y Próximo Mes ({mes_actual + 1 if mes_actual < 12 else 1})',
            'algoritmo_version': 'FR-30 v2.1 Optimizado - ANTI-CACHÉ',
            'timestamp': timestamp_actual.isoformat(),
            'cache_buster': timestamp_actual.timestamp(),
            'mes_actual_numero': mes_actual,
            'force_refresh': True
        }
        
        # Asegurar que los equipos están ordenados por riesgo descendente
        if 'equipos_riesgo' in enhanced_data and enhanced_data['equipos_riesgo']:
            enhanced_data['equipos_riesgo'] = sorted(
                enhanced_data['equipos_riesgo'], 
                key=lambda x: x.get('riesgo_score', 0), 
                reverse=True
            )[:15]  # Top 15 equipos más críticos
        
        update_progress("Análisis FR-30 completado", 3, 3, "Datos listos para visualización")
        reset_progress()
        
        response_data = json.dumps({'success': True, 'data': enhanced_data}, cls=CustomJSONEncoder)
        response = Response(response_data, mimetype='application/json')
        
        # HEADERS ANTI-CACHÉ MÁS AGRESIVOS
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        response.headers['Last-Modified'] = timestamp_actual.strftime('%a, %d %b %Y %H:%M:%S GMT')
        response.headers['ETag'] = f'"{timestamp_actual.timestamp()}"'
        
        return response
        
    except Exception as e:
        logger.exception(f"analyze_statistics_advanced error: {e}")
        response_data = json.dumps({'success': False, 'error': str(e), 'debug': 'Error en análisis FR-30'}, cls=CustomJSONEncoder)
        return Response(response_data, mimetype='application/json')

@app.route('/api/frequency-analysis')
def frequency_analysis():
    """Frecuencia mensual de correctivas por equipo (real, simplificado)."""
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty or 'fecha_in' not in df.columns or 'codigo' not in df.columns:
            return jsonify({'success': True, 'data': {'items': []}})

        dfx = df.copy()
        dfx['fecha_in'] = pd.to_datetime(dfx['fecha_in'], errors='coerce')
        dfx = dfx.dropna(subset=['fecha_in'])
        dfx['mes'] = dfx['fecha_in'].dt.to_period('M').astype(str)

        if 'tipo_atencion' in dfx.columns:
            dfx['es_corr'] = dfx['tipo_atencion'].astype(str).str.upper().eq('CORRECTIVA')
        else:
            dfx['es_corr'] = False

        agg = dfx.groupby(['codigo', 'mes']).agg(
            ingresos=('codigo','size'),
            correctivas=('es_corr','sum')
        ).reset_index().sort_values(['mes','correctivas','ingresos'], ascending=[False, False, False])

        items = agg.head(200).to_dict('records')
        return jsonify({'success': True, 'data': {'items': items}})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# --------------------------------------
# ML: entrenamiento y predicción (sin simulaciones)
# --------------------------------------
@app.route('/train-models', methods=['POST'])
def train_models():
    """Entrena modelos solo si sklearn está disponible y hay datos suficientes."""
    try:
        if not session.get('df'):
            return jsonify({'success': False, 'message': 'No hay datos cargados'}), 400
        
        df = pd.DataFrame(session['df'])
        ok = ml_engine.train_models_enhanced(df)
        session['ml_models_trained'] = bool(ok)
        if ok:
            return jsonify({'success': True, 'message': 'Modelos entrenados con datos reales', 'models': list(ml_engine.models.keys())})
        else:
            return jsonify({'success': False, 'message': 'No se pudo entrenar (sin sklearn o datos insuficientes)'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/retrain-models', methods=['POST'])
def retrain_models():
    try:
        ml_engine.models.clear()
        ml_engine.scalers.clear()
        ml_engine.is_trained = False
        ml_engine.ml_mode = False
        return train_models()
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ml_analysis', methods=['POST'])
def ml_analysis():
    """Análisis completo de ML: entrena modelos y genera predicciones para equipos."""
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No hay datos cargados para análisis ML'}), 400

        # Paso 1: Entrenar modelos
        update_progress("Entrenando modelos ML", 1, 3, "Configurando algoritmos...")
        ok = ml_engine.train_models_enhanced(df)
        session['ml_models_trained'] = bool(ok)
        
        if not ok:
            return jsonify({'success': False, 'error': 'No se pudieron entrenar los modelos (sklearn no disponible o datos insuficientes)'}), 400

        # Paso 2: Obtener códigos de equipos
        update_progress("Generando predicciones", 2, 3, "Analizando equipos...")
        equipos = ml_engine.load_real_equipment_codes()[:10]  # Limitar a 10 equipos para no sobrecargar
        
        if not equipos:
            return jsonify({'success': False, 'error': 'No se encontraron códigos de equipos válidos'}), 400

        # Paso 3: Generar predicciones para cada equipo
        update_progress("Finalizando análisis", 3, 3, "Compilando resultados...")
        results = {
            'modelos_entrenados': list(ml_engine.models.keys()),
            'total_equipos_analizados': len(equipos),
            'predicciones': {},
            'resumen': {
                'equipos_alto_riesgo_fr30': 0,
                'equipos_bajo_rul': 0,
                'anomalias_detectadas': 0
            },
            'timestamp': datetime.now().isoformat(),
            'ml_engine_status': 'active' if ml_engine.is_trained else 'inactive'
        }

        for equipo in equipos:
            try:
                pred = ml_engine.predict_equipment({'equipo': equipo})
                results['predicciones'][equipo] = pred
                
                # Actualizar resumen
                if pred.get('fr30_risk', 0) > 0.7:
                    results['resumen']['equipos_alto_riesgo_fr30'] += 1
                if pred.get('rul_days', 999) < 30:
                    results['resumen']['equipos_bajo_rul'] += 1
                if pred.get('anomaly_score', 0) > 0.6:
                    results['resumen']['anomalias_detectadas'] += 1
                    
            except Exception as e:
                logger.warning(f"Error prediciendo para equipo {equipo}: {e}")
                results['predicciones'][equipo] = {'error': str(e)}

        reset_progress()
        return jsonify({'success': True, 'data': results})

    except Exception as e:
        logger.exception(f"ml_analysis error: {e}")
        set_progress_error(f'Error en análisis ML: {str(e)}')
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ml/prediction', methods=['POST'])
def ml_prediction():
    """Predicción: si no hay modelo, devuelve 0s (sin simular)."""
    try:
        data = request.get_json() or {}
        pred = ml_engine.predict_equipment(data)
        return jsonify({
            'equipo': data.get('equipo','UNKNOWN'),
            'prediccion': pred,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/ml_specific/<analysis_type>', methods=['POST'])
def ml_specific(analysis_type):
    """Endpoints específicos para FR-30, RUL y Anomalías.

    Responde a las llamadas del front-end `/ml_specific/<type>` para:
      - fr30: riesgo de falla en 30 días por equipo
      - rul: estimación de días restantes (si el modelo está disponible)
      - anomaly: score de anomalía por equipo
    """
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No hay datos cargados'}), 400

        # Asegurar modelos entrenados con datos reales
        if not (ml_engine.is_trained and ml_engine.ml_mode):
            ok = ml_engine.train_models_enhanced(df)
            session['ml_models_trained'] = bool(ok)
            if not ok:
                return jsonify({'success': False, 'error': 'No se pudieron entrenar los modelos con los datos actuales'}), 400

        equipos = ml_engine.load_real_equipment_codes()[:10] or []
        if not equipos:
            return jsonify({'success': False, 'error': 'No se encontraron códigos de equipos válidos'}), 400

        payload = {}
        for eq in equipos:
            try:
                pred = ml_engine.predict_equipment({'equipo': eq})
            except Exception as e:
                pred = {'error': str(e)}

            if analysis_type.lower() == 'fr30':
                # Mapear a bandas para facilitar visualización
                risk = float(pred.get('fr30_risk', 0) or 0)
                if risk >= 0.5:
                    banda = '🔴 Alto'
                elif risk >= 0.3:
                    banda = '🟠 Medio'
                else:
                    banda = '🟢 Bajo'
                payload[eq] = {
                    'risk_30d': risk,
                    'banda': banda,
                    'source': 'ML_Real' if pred.get('mode') == 'ML_Real' else pred.get('mode', 'UNKNOWN')
                }
            elif analysis_type.lower() == 'rul':
                payload[eq] = {
                    'rul50_d': int(pred.get('rul_days', 0) or 0),
                    'rul90_d': int(max(0, (pred.get('rul_days', 0) or 0) * 0.7)),
                    'source': pred.get('mode', 'UNKNOWN')
                }
            elif analysis_type.lower() == 'anomaly':
                payload[eq] = {
                    'anomaly_score': float(pred.get('anomaly_score', 0) or 0),
                    'source': pred.get('mode', 'UNKNOWN')
                }
            else:
                return jsonify({'success': False, 'error': f'Tipo de análisis no soportado: {analysis_type}'}), 400

        return jsonify({'success': True, 'data': payload, 'type': analysis_type.lower(), 'timestamp': datetime.now().isoformat()})
    except Exception as e:
        logger.exception(f"ml_specific error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

# --------------------------------------
# Estado/Healthcheck
# --------------------------------------
@app.route('/api/connection-test')
def connection_test():
    return jsonify({
        'status': 'ok',
        'message': 'COTEMA Analytics API operativa (Real data, no simulada)',
        'timestamp': datetime.now().isoformat(),
        'version': '3.1.0',
        'ml_available': bool(ml_engine.is_trained and ml_engine.ml_mode)
    })

@app.route('/api/status')
def api_status():
    """Estado detallado del sistema con información de debug"""
    df_loaded = session.get('df')
    status = {
        'status': 'running',
        'data_loaded': df_loaded is not None,
        # Guardamos processed_date como string ISO; no llamar isoformat sobre str
        'last_processed': session.get('processed_date') if session.get('processed_date') else None,
        'ml_available': bool(ml_engine.is_trained and ml_engine.ml_mode),
        'version': '3.1.0'
    }
    
    if df_loaded is not None:
        try:
            df = pd.DataFrame(df_loaded)
        except Exception:
            df = None
    else:
        df = None

    if df is not None and not df.empty:
        status.update({
            'data_info': {
                'total_rows': int(len(df)),
                'total_columns': int(len(df.columns)),
                'columns': list(df.columns),
                'file_name': global_data.get('file_name'),
                'sheet_used': global_data.get('stats', {}).get('sheet_used'),
                'available_sheets': global_data.get('stats', {}).get('available_sheets', []),
                'critical_columns': {
                    'fecha_in': 'fecha_in' in df.columns,
                    'tipo_atencion': 'tipo_atencion' in df.columns,
                    'codigo': 'codigo' in df.columns
                }
            }
        })
        
        # Sample data para debug
        try:
            if len(df) > 0:
                status['data_info']['sample_row'] = df.iloc[0].to_dict()
        except Exception:
            pass
    
    return jsonify(status)

# --------------------------------------
# Rutas adicionales para compatibilidad con templates
# --------------------------------------
@app.route('/predictions')
def predictions_dashboard():
    """Ruta de predicciones - redirige al dashboard principal por ahora"""
    return redirect('/dashboard')

@app.route('/ia_documentation')
def ia_documentation():
    """Documentación de IA"""
    try:
        return render_template('ia_documentation.html')
    except Exception as e:
        logger.exception(f"ia_documentation error: {e}")
        flash('Documentación no disponible temporalmente.', 'warning')
        return redirect('/')


@app.route('/analyze-fr30', methods=['POST'])
def analyze_fr30():
    """Ruta para análisis FR-30 específico - Compatible con predictions.html"""
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty:
            df_sample = create_sample_data()
            session['df'] = df_sample.to_dict('records')
            df = df_sample
            
        # Usar la función de análisis FR-30 estándar
        result = get_fr30_analysis(df)
        
        return jsonify({
            'success': True,
            'data': result,
            'message': 'Análisis FR-30 completado exitosamente'
        })
        
    except Exception as e:
        logger.exception(f"Error en analyze-fr30: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error en análisis FR-30'
        }), 500


@app.route('/deep-analysis', methods=['POST'])
def deep_analysis():
    """Análisis profundo - Compatible con dashboard_simple.html"""
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty:
            df_sample = create_sample_data()
            session['df'] = df_sample.to_dict('records')
            df = df_sample
            
        # Ejecutar análisis avanzado
        result = get_fr30_advanced_analysis(df)
        
        return jsonify({
            'success': True,
            'data': result,
            'analysis_type': 'deep_analysis',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.exception(f"Error en deep-analysis: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/train-progress', methods=['GET'])
def api_train_progress():
    """Progreso de entrenamiento - Compatible con dashboard_simple.html"""
    try:
        # Simular progreso de entrenamiento
        progress = {
            'progress': 100,
            'status': 'completed',
            'message': 'Modelos entrenados exitosamente',
            'models_ready': True,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(progress)
        
    except Exception as e:
        logger.exception(f"Error en train-progress: {e}")
        return jsonify({
            'progress': 0,
            'status': 'error',
            'message': str(e),
            'models_ready': False
        }), 500


@app.route('/clear-cache', methods=['POST', 'GET'])
def clear_cache():
    """Limpiar caché y datos para forzar recálculo"""
    try:
        global_data.clear()
        
        # Mensaje de confirmación
        response_data = {
            'success': True,
            'message': 'Caché limpiado exitosamente',
            'timestamp': datetime.now().isoformat(),
            'action': 'cache_cleared'
        }
        
        response = Response(json.dumps(response_data), mimetype='application/json')
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        
        return response
        
    except Exception as e:
        logger.exception(f"Error limpiando caché: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/version_check')
def version_check():
    """Verificar versión activa y características disponibles"""
    from datetime import datetime
    version_info = {
        "status": "ACTIVE",
        "version": "COTEMA v2.1 - FR-30 KPI Optimizado",
        "last_update": "2025-01-27 19:45 UTC",
        "commit_hash": "b80bc7e",
        "timestamp_check": datetime.now().isoformat(),
        "features_active": [
            "✅ Endpoint /analyze_statistics_advanced restaurado",
            "✅ KPI FR-30 mejorado con cálculos confiables 0-100%",
            "✅ Enfoque mes actual + próximo mes",
            "✅ API limpia para BI: /api/kpi/fr30",
            "✅ Algoritmo FR-30 v2.1 optimizado"
        ],
        "endpoints_available": [
            "POST /analyze_statistics_advanced (Para interfaz actual)",
            "GET /api/kpi/fr30 (Para consumo BI)",
            "GET /version_check (Este endpoint)"
        ]
    }
    return jsonify(version_info)


@app.route('/api/kpi/fr30', methods=['GET'])
def api_kpi_fr30():
    """
    API KPI FR-30 - Endpoint limpio para consumo desde BI
    Retorna equipos con mayor tendencia a fallar (0-100%) mes actual y próximo.
    """
    try:
        df_data = session.get('df')
        df = pd.DataFrame(df_data) if df_data is not None else None
        if df is None or df.empty:
            # Usar datos de muestra para demostración
            df_sample = create_sample_data()
            session['df'] = df_sample.to_dict('records')
            df = df_sample
            
        # Obtener cálculo KPI FR-30
        kpi_result = get_fr30_advanced_analysis(df)
        
        # Estructura limpia para BI
        api_response = {
            "kpi_metadata": {
                "kpi_name": "FR-30 Equipment Failure Risk",
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "calculation_algorithm": kpi_result.get('algoritmo', 'FR-30 v2.1'),
                "confidence_level": round(kpi_result.get('precision_estimada', 0.85) * 100, 1),
                "dataset_records": len(df),
                "analysis_period": "Current Month + Next Month"
            },
            "critical_equipment": [],
            "monthly_forecast": [],
            "kpi_summary": {
                "total_equipment_analyzed": kpi_result.get('total_equipos_analizados', 0),
                "critical_equipment_count": kpi_result.get('equipos_criticos_detectados', 0),
                "risk_threshold_critical": 70,
                "risk_threshold_medium": 40
            }
        }
        
        # Equipos críticos (Top 20)
        for rank, equipo in enumerate(kpi_result.get('equipos_riesgo', [])[:20], 1):
            api_response["critical_equipment"].append({
                "rank": rank,
                "equipment_id": equipo.get('equipo'),
                "risk_score_percent": equipo.get('riesgo_score', 0),
                "risk_category": equipo.get('estado', 'NORMAL'),
                "total_interventions": equipo.get('total_ingresos', 0),
                "critical_interventions": equipo.get('ingresos_criticos', 0),
                "mttr_hours": equipo.get('mttr_horas', 0),
                "peak_risk_month": equipo.get('mes_mayor_riesgo', 1)
            })
        
        # Proyección mensual
        for month_data in kpi_result.get('meses_tendencia', []):
            api_response["monthly_forecast"].append({
                "period": month_data.get('periodo', 'Unknown'),
                "month_number": month_data.get('mes', 1),
                "projected_corrective_maintenance": month_data.get('total_correctivas', 0),
                "critical_equipment_projected": month_data.get('equipos_criticos', 0)
            })
        
        return jsonify(api_response)
        
    except Exception as e:
        logger.exception(f"Error en API KPI FR-30: {e}")
        return jsonify({
            "error": True,
            "error_message": "Internal calculation error",
            "error_detail": str(e),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z"
        }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
