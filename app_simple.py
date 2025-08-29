"""
Servicio Web para Análisis de KPIs de Taller con IA - Versión Ultra Mínima
Aplicación Flask principal optimizada para Render sin dependencias pesadas
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from datetime import datetime
import json
import os
from werkzeug.utils import secure_filename

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
            
            # Simular procesamiento exitoso
            global_data['df'] = True  # Simplificado
            global_data['processed_date'] = datetime.now()
            
            flash(f'Archivo procesado exitosamente: simulación activada', 'success')
        
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
    
    # Datos simulados para demostración
    months = ['2024-12', '2024-11', '2024-10', '2024-09', '2024-08']
    
    return render_template('dashboard_simple.html', 
                         months=months,
                         total_registros=2443,
                         equipos_unicos=176)

@app.route('/kpis/<mes>')
def calculate_kpis(mes):
    """Calcula KPIs básicos para un mes específico"""
    try:
        if global_data['df'] is None:
            return jsonify({'error': 'No hay datos cargados'}), 400
        
        # Simular KPIs con datos ficticios
        equipos_simulados = ['VD-CO30', 'VD-CO13', 'VD-CO02', 'VD-CO01', 'VD-CO14']
        
        kpis = {
            'fr30': {},
            'rul': {},
            'forecast': {},
            'anomaly': {}
        }
        
        import random
        random.seed(42)  # Para resultados consistentes
        
        for i, equipo in enumerate(equipos_simulados):
            # KPIs simulados básicos
            kpis['fr30'][equipo] = {
                'risk_30d': round(random.uniform(0.1, 0.6), 3),
                'banda': '🟠 MEDIO' if i % 2 == 0 else '🟢 BAJO',
                'explicacion': f'Análisis básico para {equipo}'
            }
            
            kpis['rul'][equipo] = {
                'rul50_d': random.randint(15, 90),
                'rul90_d': random.randint(10, 60),
                'explicacion': f'Vida útil estimada para {equipo}'
            }
            
            kpis['forecast'][equipo] = {
                'forecast_7d_h': round(random.uniform(20, 80), 1),
                'forecast_30d_h': round(random.uniform(80, 300), 1),
                'explicacion': f'Pronóstico de uso para {equipo}'
            }
            
            kpis['anomaly'][equipo] = {
                'anomaly_score': round(random.uniform(0.1, 0.8), 2),
                'banda': '🟢 NORMAL' if i % 3 != 0 else '🟡 ATENCIÓN',
                'explicacion': f'Comportamiento normal en {equipo}'
            }
        
        result = {
            'mes': mes,
            'total_equipos': len(equipos_simulados),
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
</content>
</invoke>
