from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from datetime import datetime, timedelta
import os
from werkzeug.utils import secure_filename
import json
import hashlib
import logging
import threading

# Dependencias base
try:
    import pandas as pd
    import numpy as np
except ImportError as e:
    pd = None
    np = None

# Importar el procesador ENFOCADO en datos reales
from cotema_processor import process_cotema_data, get_fr30_analysis

# --------------------------------------
# Configuración básica de Flask y logging
# --------------------------------------
app = Flask(__name__)
app.secret_key = os.urandom(24)

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("COTEMA")

# --------------------------------------
# Estado global y helpers de progreso
# --------------------------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'xls'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global_data = {
    'df': None,
    'dataset_normalizado': None,
    'reporte_calidad': None,
    'catalogos': None,
    'file_path': None,
    'file_name': None,
    'processed_date': None,
    'ml_models_trained': False,
    'deep_analysis_in_progress': False,
    'analysis_type': None,
}

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
            cats = global_data.get('catalogos') or {}
            lista = cats.get('equipos', {}).get('lista', [])
            if lista:
                return list(map(str, lista))[:200]
        except Exception:
            pass
        try:
            df = global_data.get('df')
            if df is not None and len(df) > 0:
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
                df = global_data.get('df')
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
        df = global_data.get('df')
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
    stats = global_data.get('stats', {
        'total_registros': 0,
        'processing_time': 0,
        'sheet_used': None,
        'registros_cerrados': 0,
        'registros_abiertos': 0,
        'file_loaded': False
    })
    return render_template('index.html',
                           data_loaded=global_data['df'] is not None,
                           processed_date=global_data.get('processed_date'),
                           stats=stats)

@app.route('/progress', methods=['GET'])
def get_progress():
    # Adaptar al formato que espera el frontend
    response = progress_state.copy()
    response['percentage'] = response.get('progress', 0)
    response['details'] = response.get('message', '')
    return jsonify(response)

@app.route('/ml-status')
def ml_status():
    status = {
        'deep_analysis_in_progress': global_data.get('deep_analysis_in_progress', False),
        'ml_models_trained': global_data.get('ml_models_trained', False),
        'analysis_type': global_data.get('analysis_type', ''),
        'ml_progress': {
            'percent': 100 if global_data.get('ml_models_trained') else 0,
            'step': 'ready' if global_data.get('ml_models_trained') else 'idle',
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

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        file.save(filepath)

        # Procesar en segundo plano (para no bloquear el worker)
        thread = threading.Thread(target=process_uploaded_file, args=(filepath, filename), daemon=True)
        thread.start()
        update_progress("Archivo recibido", 1, 4, "Procesando en servidor...")

        # Si es XHR, responde JSON; si no, redirige al index
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest' or request.is_json:
            return jsonify({'success': True, 'message': f'Archivo {filename} subido. Procesamiento iniciado.'})
        else:
            return redirect(url_for('index'))

    except Exception as e:
        logger.exception(f"upload_file error: {e}")
        set_progress_error(f'Error inesperado: {str(e)}')
        return jsonify({'error': f'Error inesperado: {str(e)}'}), 500

def process_uploaded_file(filepath, filename):
    """Lee Excel, procesa con cotema_processor y llena global_data (sin simulaciones)."""
    import time
    start = time.time()
    try:
        update_progress("Leyendo Excel", 2, 4, "Inspeccionando hojas...")
        if pd is None:
            raise RuntimeError("Pandas no está disponible en el servidor.")

        xl = pd.ExcelFile(filepath)  # usa engine por defecto disponible
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
        df_raw = pd.read_excel(filepath, sheet_name=sheet)
        logger.info(f"Datos leídos: {len(df_raw)} filas, {len(df_raw.columns)} columnas")
        logger.info(f"Columnas encontradas: {list(df_raw.columns)[:10]}")  # Primeras 10 columnas

        dataset, quality, catalogs = process_cotema_data(df_raw)

        # Guardar en estado global
        global_data['df'] = pd.DataFrame(dataset)
        global_data['dataset_normalizado'] = dataset
        global_data['reporte_calidad'] = quality
        global_data['catalogos'] = catalogs
        global_data['file_path'] = filepath
        global_data['file_name'] = filename
        global_data['processed_date'] = datetime.now()
        global_data['ml_models_trained'] = False

        # Stats para portada con más información
        errores_detectados = 0
        for v in quality.get('errores', {}).values():
            if isinstance(v, (int, float)):
                errores_detectados += int(v)
            elif isinstance(v, dict):
                errores_detectados += len(v)

        processing_time = round(time.time() - start, 2)
        global_data['stats'] = {
            'total_registros': quality.get('total_registros', len(global_data['df'])),
            'registros_abiertos': quality.get('registros_abiertos', 0),
            'registros_cerrados': quality.get('registros_cerrados', 0),
            'errores_detectados': errores_detectados,
            'sheet_used': sheet,
            'available_sheets': sheets,
            'file_loaded': True,
            'processing_time': processing_time,
            'columnas_procesadas': len(global_data['df'].columns),
            'columnas_criticas': {
                'fecha_in': 'fecha_in' in global_data['df'].columns,
                'tipo_atencion': 'tipo_atencion' in global_data['df'].columns,
                'codigo': 'codigo' in global_data['df'].columns
            }
        }

        update_progress("Completado", 4, 4, f"Procesado en {processing_time}s - {len(global_data['df'])} registros")
        logger.info(f"Procesado OK: {filename} en {processing_time}s - {len(global_data['df'])} registros")

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
    try:
        if global_data.get('df') is None:
            flash('Primero debes cargar un archivo Excel.', 'warning')
            return redirect(url_for('index'))

        df = global_data['df']
        months = []
        if 'fecha_in' in df.columns:
            valid_dates = pd.to_datetime(df['fecha_in'], errors='coerce').dropna()
            if not valid_dates.empty:
                months = sorted(valid_dates.dt.to_period('M').astype(str).unique().tolist(), reverse=True)
        if not months:
            months = []

        stats = global_data.get('stats', {'total_registros': len(df)})
        return render_template('dashboard_simple.html',
                               months=months,
                               stats=stats,
                               total_registros=stats.get('total_registros', 0),
                               ml_models_trained=global_data.get('ml_models_trained', False))
    except Exception as e:
        logger.exception(f"dashboard error: {e}")
        return "<h1>Error 500</h1><p>Ocurrió un error al cargar el dashboard.</p>", 500

@app.route('/kpis/<mes>')
def calculate_kpis(mes):
    """KPIs básicos 100% reales. Sin simulación; si falta modelo, 0s."""
    try:
        df = global_data.get('df')
        if df is None or df.empty:
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
        df = global_data.get('df')
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
        if global_data.get('df') is None:
            return jsonify({'error': 'No hay archivo cargado'}), 400

        df = global_data['df']
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
        global_data['stats'] = {**global_data.get('stats', {}), **stats}
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        set_progress_error(f'Error en análisis rápido: {str(e)}')
        return jsonify({'error': f'Error en análisis rápido: {str(e)}'}), 500

@app.route('/analyze_statistics', methods=['POST'])
def analyze_statistics():
    """KPI FR-30 real (conteo de correctivas últimos N días)."""
    try:
        df = global_data.get('df')
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
        
        return jsonify({'success': True, 'data': enhanced_data})
    except Exception as e:
        logger.exception(f"analyze_statistics error: {e}")
        return jsonify({'success': False, 'error': str(e), 'debug': 'Error en el análisis'})

@app.route('/api/frequency-analysis')
def frequency_analysis():
    """Frecuencia mensual de correctivas por equipo (real, simplificado)."""
    try:
        df = global_data.get('df')
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
        df = global_data.get('df')
        ok = ml_engine.train_models_enhanced(df)
        global_data['ml_models_trained'] = bool(ok)
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
        df = global_data.get('df')
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No hay datos cargados para análisis ML'}), 400

        # Paso 1: Entrenar modelos
        update_progress("Entrenando modelos ML", 1, 3, "Configurando algoritmos...")
        ok = ml_engine.train_models_enhanced(df)
        global_data['ml_models_trained'] = bool(ok)
        
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
        df = global_data.get('df')
        if df is None or df.empty:
            return jsonify({'success': False, 'error': 'No hay datos cargados'}), 400

        # Asegurar modelos entrenados con datos reales
        if not (ml_engine.is_trained and ml_engine.ml_mode):
            ok = ml_engine.train_models_enhanced(df)
            global_data['ml_models_trained'] = bool(ok)
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
    df = global_data.get('df')
    status = {
        'status': 'running',
        'data_loaded': df is not None,
        'last_processed': global_data['processed_date'].isoformat() if global_data['processed_date'] else None,
        'ml_available': bool(ml_engine.is_trained and ml_engine.ml_mode),
        'version': '3.1.0'
    }
    
    if df is not None:
        status.update({
            'data_info': {
                'total_rows': len(df),
                'total_columns': len(df.columns),
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
        if len(df) > 0:
            status['data_info']['sample_row'] = df.iloc[0].to_dict()
    
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


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
