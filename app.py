from flask import Flask, render_template, jsonify, request, redirect, url_for, flash
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'cotema-2025'

@app.route('/')
def index():
    return render_template('index.html', data_loaded=False, processed_date=None)

@app.route('/upload', methods=['POST'])
def upload_file():
    flash('Funcionalidad de carga en desarrollo', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard_simple.html', 
                         months=['2024-12', '2024-11', '2024-10'],
                         total_registros=0,
                         equipos_unicos=0)

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
