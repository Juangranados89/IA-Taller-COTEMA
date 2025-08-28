"""
Servicio Web para Análisis de KPIs de Taller con IA - Versión Mínima
Aplicación Flask principal optimizada para Render
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from werkzeug.utils import secure_filename
import traceback

app = Flask(__name__)
app.secret_key = 'cotema-taller-analytics-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Crear carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Variable global para almacenar los datos procesados
global_data = {
    'df': None,
    'processed_date': None,
    'kpis': {}
}

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html', 
                         data_loaded=global_data['df'] is not None,
                         processed_date=global_data['processed_date'])

@app.route('/upload', methods=['POST'])
def upload_file():
    """Maneja la carga del archivo Excel"""
    try:
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(url_for('index'))
        
        if file and file.filename.lower().endswith(('.xlsx', '.xls')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesar archivo Excel básico
            try:
                df = pd.read_excel(filepath, sheet_name='REG', skiprows=4)
                df = df.dropna(how='all')
                
                global_data['df'] = df
                global_data['processed_date'] = datetime.now()
                
                flash(f'Archivo procesado exitosamente: {len(df)} registros cargados', 'success')
                
            except Exception as e:
                flash(f'Error procesando archivo: {str(e)}', 'error')
                return redirect(url_for('index'))
        
        else:
            flash('Por favor sube un archivo Excel (.xlsx o .xls)', 'error')
            return redirect(url_for('index'))
        
        return redirect(url_for('dashboard'))
        
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Panel de control principal"""
    if global_data['df'] is None:
        flash('Primero debes cargar un archivo Excel', 'warning')
        return redirect(url_for('index'))
    
    df = global_data['df']
    
    # Obtener meses disponibles
    date_columns = ['FECHA IN', 'FECHA_IN']
    fecha_col = None
    
    for col in date_columns:
        if col in df.columns:
            fecha_col = col
            break
    
    if fecha_col:
        try:
            df[fecha_col] = pd.to_datetime(df[fecha_col], errors='coerce')
            months = df[fecha_col].dt.to_period('M').dropna().unique()
            months = sorted([str(m) for m in months], reverse=True)
        except:
            months = ['2024-12', '2024-11', '2024-10']
    else:
        months = ['2024-12', '2024-11', '2024-10']
    
    return render_template('dashboard.html', 
                         months=months,
                         total_registros=len(df),
                         equipos_unicos=df['CODIGO'].nunique() if 'CODIGO' in df.columns else 0)

@app.route('/kpis/<mes>')
def calculate_kpis(mes):
    """Calcula KPIs básicos para un mes específico"""
    try:
        if global_data['df'] is None:
            return jsonify({'error': 'No hay datos cargados'}), 400
        
        df = global_data['df']
        
        # Filtrar por mes básico
        year, month = mes.split('-')
        
        # Simular KPIs básicos
        equipos = df['CODIGO'].unique() if 'CODIGO' in df.columns else ['EQUIPO_001', 'EQUIPO_002']
        
        kpis = {
            'fr30': {},
            'rul': {},
            'forecast': {},
            'anomaly': {}
        }
        
        for equipo in equipos[:10]:  # Máximo 10 equipos
            # KPIs simulados básicos
            kpis['fr30'][equipo] = {
                'risk_30d': np.random.uniform(0.1, 0.6),
                'banda': '🟠 MEDIO',
                'explicacion': f'Análisis básico para {equipo}'
            }
            
            kpis['rul'][equipo] = {
                'rul50_d': int(np.random.uniform(15, 90)),
                'rul90_d': int(np.random.uniform(10, 60)),
                'explicacion': f'Vida útil estimada para {equipo}'
            }
            
            kpis['forecast'][equipo] = {
                'forecast_7d_h': round(np.random.uniform(20, 80), 1),
                'forecast_30d_h': round(np.random.uniform(80, 300), 1),
                'explicacion': f'Pronóstico de uso para {equipo}'
            }
            
            kpis['anomaly'][equipo] = {
                'anomaly_score': round(np.random.uniform(0.1, 0.8), 2),
                'banda': '🟢 NORMAL',
                'explicacion': f'Comportamiento normal en {equipo}'
            }
        
        result = {
            'mes': mes,
            'total_equipos': len(equipos),
            'timestamp': datetime.now().isoformat(),
            'kpis': kpis
        }
        
        global_data['kpis'] = result
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/connection-test')
def connection_test():
    """Test de conexión para BI"""
    return jsonify({
        'status': 'ok',
        'message': 'COTEMA Analytics API operativa',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.route('/api/export/<formato>')
def export_data(formato):
    """Exporta datos en diferentes formatos"""
    try:
        if global_data['kpis'] == {}:
            return jsonify({'error': 'No hay KPIs calculados'}), 400
        
        kpis = global_data['kpis']
        
        if formato == 'json':
            return jsonify(kpis)
        elif formato == 'csv':
            # Simular CSV básico
            csv_data = "equipo,fr30_risk,rul50_d,forecast_7d_h,anomaly_score\n"
            for equipo in kpis['kpis']['fr30'].keys():
                fr30 = kpis['kpis']['fr30'][equipo]['risk_30d']
                rul = kpis['kpis']['rul'][equipo]['rul50_d']
                forecast = kpis['kpis']['forecast'][equipo]['forecast_7d_h']
                anomaly = kpis['kpis']['anomaly'][equipo]['anomaly_score']
                csv_data += f"{equipo},{fr30},{rul},{forecast},{anomaly}\n"
            
            return csv_data, 200, {'Content-Type': 'text/csv'}
        else:
            return jsonify(kpis)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
    try:
        if 'file' not in request.files:
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(url_for('index'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No se seleccionó ningún archivo', 'error')
            return redirect(url_for('index'))
        
        if file and file.filename.lower().endswith(('.xlsx', '.xls')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesar el archivo
            processor = DataProcessor()
            df = processor.load_and_clean_data(filepath)
            
            global_data['df'] = df
            global_data['processed_date'] = datetime.now()
            
            flash(f'Archivo cargado exitosamente. {len(df)} registros procesados.', 'success')
            return redirect(url_for('dashboard'))
        
        else:
            flash('Por favor, sube un archivo Excel (.xlsx o .xls)', 'error')
            return redirect(url_for('index'))
    
    except Exception as e:
        flash(f'Error al procesar el archivo: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    """Panel principal de análisis"""
    if global_data['df'] is None:
        flash('Por favor, carga primero un archivo Excel', 'warning')
        return redirect(url_for('index'))
    
    try:
        df = global_data['df']
        
        # Obtener estadísticas generales
        stats = {
            'total_registros': len(df),
            'equipos_unicos': df['CODIGO'].nunique(),
            'fecha_min': df['FECHA_IN'].min().strftime('%Y-%m-%d'),
            'fecha_max': df['FECHA_IN'].max().strftime('%Y-%m-%d'),
            'sistemas_unicos': df['SISTEMA_AFECTADO'].nunique() if 'SISTEMA_AFECTADO' in df.columns else 0
        }
        
        # Generar gráficos básicos
        plots = create_dashboard_plots(df)
        
        return render_template('dashboard.html', 
                             stats=stats, 
                             plots=plots,
                             meses_disponibles=get_available_months(df))
    
    except Exception as e:
        flash(f'Error al cargar el dashboard: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/kpis/<mes>')
def calculate_kpis(mes):
    """Calcula todos los KPIs para un mes específico"""
    if global_data['df'] is None:
        return jsonify({'error': 'No hay datos cargados'}), 400
    
    try:
        df = global_data['df']
        year, month = map(int, mes.split('-'))
        
        # Filtrar datos hasta el mes seleccionado
        fecha_corte = datetime(year, month, 1)
        df_filtered = df[df['FECHA_IN'] <= fecha_corte]
        
        if len(df_filtered) == 0:
            return jsonify({'error': f'No hay datos para el mes {mes}'}), 400
        
        # Calcular KPIs
        kpis = {}
        
        # 1. FR-30 - Riesgo de falla en 30 días
        fr30_model = FR30Model()
        fr30_results = fr30_model.train_and_predict(df_filtered)
        kpis['fr30'] = fr30_results
        
        # 2. RUL - Vida Útil Restante
        rul_model = RULModel()
        rul_results = rul_model.calculate_rul(df_filtered)
        kpis['rul'] = rul_results
        
        # 3. Pronóstico de uso
        forecast_model = ForecastModel()
        forecast_results = forecast_model.predict_usage(df_filtered)
        kpis['forecast'] = forecast_results
        
        # 4. Anomalías operativas
        anomaly_model = AnomalyModel()
        anomaly_results = anomaly_model.detect_anomalies(df_filtered)
        kpis['anomaly'] = anomaly_results
        
        # Guardar KPIs calculados
        global_data['kpis'][mes] = kpis
        global_data['models'] = {
            'fr30': fr30_model,
            'rul': rul_model,
            'forecast': forecast_model,
            'anomaly': anomaly_model
        }
        
        return jsonify({
            'success': True,
            'mes': mes,
            'kpis': serialize_kpis(kpis),
            'total_equipos': len(fr30_results)
        })
    
    except Exception as e:
        error_msg = f'Error al calcular KPIs: {str(e)}'
        print(f"Error detallado: {traceback.format_exc()}")
        return jsonify({'error': error_msg}), 500

@app.route('/api/fr30/<equipo>')
def get_fr30_details(equipo):
    """Obtiene detalles del modelo FR-30 para un equipo específico"""
    try:
        if 'fr30' not in global_data['models']:
            return jsonify({'error': 'Modelo FR-30 no disponible'}), 400
        
        model = global_data['models']['fr30']
        details = model.get_equipment_details(equipo)
        
        # Generar explicación con LLM
        explanation = generate_llm_explanations('fr30', equipo, details)
        
        return jsonify({
            'equipo': equipo,
            'risk_30d': details.get('risk_30d', 0),
            'features': details.get('features', {}),
            'explanation': explanation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/rul/<equipo>')
def get_rul_details(equipo):
    """Obtiene detalles del modelo RUL para un equipo específico"""
    try:
        if 'rul' not in global_data['models']:
            return jsonify({'error': 'Modelo RUL no disponible'}), 400
        
        model = global_data['models']['rul']
        details = model.get_equipment_details(equipo)
        
        # Generar explicación con LLM
        explanation = generate_llm_explanations('rul', equipo, details)
        
        return jsonify({
            'equipo': equipo,
            'rul50_d': details.get('rul50_d', 0),
            'rul90_d': details.get('rul90_d', 0),
            'parameters': details.get('parameters', {}),
            'explanation': explanation
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/<formato>')
def export_data(formato):
    """Exporta los datos calculados en diferentes formatos"""
    try:
        if not global_data['kpis']:
            return jsonify({'error': 'No hay KPIs calculados para exportar'}), 400
        
        if formato == 'json':
            return jsonify(global_data['kpis'])
        
        elif formato == 'csv':
            # Convertir KPIs a DataFrame
            all_data = []
            for mes, kpis in global_data['kpis'].items():
                for equipo in kpis['fr30']:
                    row = {
                        'mes': mes,
                        'equipo': equipo,
                        'fr30_risk': kpis['fr30'][equipo].get('risk_30d', 0),
                        'rul50_d': kpis['rul'][equipo].get('rul50_d', 0),
                        'rul90_d': kpis['rul'][equipo].get('rul90_d', 0),
                        'forecast_7d_h': kpis['forecast'][equipo].get('forecast_7d_h', 0),
                        'forecast_30d_h': kpis['forecast'][equipo].get('forecast_30d_h', 0),
                        'anomaly_score': kpis['anomaly'][equipo].get('anomaly_score', 0)
                    }
                    all_data.append(row)
            
            df_export = pd.DataFrame(all_data)
            
            # Crear archivo CSV
            csv_path = 'static/exports/kpis_export.csv'
            os.makedirs('static/exports', exist_ok=True)
            df_export.to_csv(csv_path, index=False)
            
            return jsonify({'download_url': f'/static/exports/kpis_export.csv'})
        
        else:
            return jsonify({'error': 'Formato no soportado'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/connection-test')
def test_bi_connection():
    """Endpoint para probar la conexión desde herramientas de BI"""
    return jsonify({
        'status': 'connected',
        'service': 'COTEMA Taller Analytics',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat(),
        'data_available': global_data['df'] is not None,
        'kpis_calculated': len(global_data['kpis']) > 0
    })

def get_available_months(df):
    """Obtiene la lista de meses disponibles en los datos"""
    months = df['FECHA_IN'].dt.to_period('M').unique()
    return sorted([str(m) for m in months])

def serialize_kpis(kpis):
    """Serializa los KPIs para JSON"""
    serialized = {}
    for model_name, model_data in kpis.items():
        serialized[model_name] = {}
        for equipo, data in model_data.items():
            if isinstance(data, dict):
                serialized[model_name][equipo] = data
            else:
                serialized[model_name][equipo] = {'value': float(data) if pd.notna(data) else 0}
    return serialized

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
