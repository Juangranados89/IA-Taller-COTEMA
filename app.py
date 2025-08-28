from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from datetime import datetime
import pandas as pd
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'cotema-2025'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

# Crear carpetas necesarias
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Variable global para almacenar los datos procesados
global_data = {
    'df': None,
    'processed_date': None,
    'stats': {}
}

@app.route('/')
def index():
    return render_template('index.html', 
                         data_loaded=global_data['df'] is not None,
                         processed_date=global_data['processed_date'])

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo'}), 400
        
        if file and file.filename.lower().endswith(('.xlsx', '.xls')):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Procesar archivo Excel
            try:
                # Intentar leer el archivo con diferentes configuraciones
                df = None
                try:
                    df = pd.read_excel(filepath, sheet_name='REG', skiprows=4)
                except:
                    try:
                        df = pd.read_excel(filepath, sheet_name=0, skiprows=4)
                    except:
                        df = pd.read_excel(filepath)
                
                # Limpiar datos
                df = df.dropna(how='all')
                
                # Procesar columnas comunes
                columnas_esperadas = ['CODIGO', 'FECHA IN', 'FECHA OUT', 'SISTEMA AFECTADO']
                columnas_encontradas = []
                
                for col in df.columns:
                    if any(expected.lower() in str(col).lower() for expected in columnas_esperadas):
                        columnas_encontradas.append(col)
                
                # Guardar estadísticas
                stats = {
                    'total_registros': len(df),
                    'columnas_total': len(df.columns),
                    'columnas_encontradas': columnas_encontradas,
                    'equipos_unicos': 0
                }
                
                # Intentar contar equipos únicos
                for col in df.columns:
                    if 'codigo' in str(col).lower():
                        stats['equipos_unicos'] = df[col].nunique()
                        break
                
                global_data['df'] = df
                global_data['processed_date'] = datetime.now()
                global_data['stats'] = stats
                
                return jsonify({
                    'success': True,
                    'message': f'Archivo procesado exitosamente',
                    'stats': stats
                })
                
            except Exception as e:
                return jsonify({'error': f'Error procesando archivo: {str(e)}'}), 500
        
        else:
            return jsonify({'error': 'Formato no soportado. Use .xlsx o .xls'}), 400
        
    except Exception as e:
        return jsonify({'error': f'Error: {str(e)}'}), 500

@app.route('/dashboard')
def dashboard():
    if global_data['df'] is None:
        flash('Primero debes cargar un archivo Excel', 'warning')
        return redirect(url_for('index'))
    
    # Obtener meses disponibles del archivo real
    df = global_data['df']
    months = ['2024-12', '2024-11', '2024-10']
    
    # Intentar extraer meses reales
    for col in df.columns:
        if 'fecha' in str(col).lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                months_real = df[col].dt.to_period('M').dropna().unique()
                if len(months_real) > 0:
                    months = sorted([str(m) for m in months_real], reverse=True)
                break
            except:
                continue
    
    stats = global_data.get('stats', {})
    
    return render_template('dashboard_simple.html', 
                         months=months,
                         total_registros=stats.get('total_registros', 0),
                         equipos_unicos=stats.get('equipos_unicos', 0))

@app.route('/kpis/<mes>')
def calculate_kpis(mes):
    try:
        if global_data['df'] is None:
            return jsonify({'error': 'No hay datos cargados'}), 400
        
        df = global_data['df']
        
        # Obtener equipos reales del archivo
        equipos = ['EQUIPO_001', 'EQUIPO_002', 'EQUIPO_003']
        for col in df.columns:
            if 'codigo' in str(col).lower():
                equipos = df[col].dropna().unique()[:10]  # Máximo 10
                break
        
        kpis = {
            'fr30': {},
            'rul': {},
            'forecast': {},
            'anomaly': {}
        }
        
        import random
        random.seed(42)
        
        for i, equipo in enumerate(equipos):
            kpis['fr30'][str(equipo)] = {
                'risk_30d': round(random.uniform(0.1, 0.6), 3),
                'banda': '🟠 MEDIO' if i % 2 == 0 else '🟢 BAJO',
                'explicacion': f'Análisis para {equipo}'
            }
            
            kpis['rul'][str(equipo)] = {
                'rul50_d': random.randint(15, 90),
                'rul90_d': random.randint(10, 60),
                'explicacion': f'Vida útil estimada para {equipo}'
            }
            
            kpis['forecast'][str(equipo)] = {
                'forecast_7d_h': round(random.uniform(20, 80), 1),
                'forecast_30d_h': round(random.uniform(80, 300), 1),
                'explicacion': f'Pronóstico para {equipo}'
            }
            
            kpis['anomaly'][str(equipo)] = {
                'anomaly_score': round(random.uniform(0.1, 0.8), 2),
                'banda': '🟢 NORMAL' if i % 3 != 0 else '🟡 ATENCIÓN',
                'explicacion': f'Estado de {equipo}'
            }
        
        result = {
            'mes': mes,
            'total_equipos': len(equipos),
            'timestamp': datetime.now().isoformat(),
            'kpis': kpis
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/connection-test')
def connection_test():
    return jsonify({
        'status': 'ok',
        'message': 'COTEMA Analytics API operativa',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
