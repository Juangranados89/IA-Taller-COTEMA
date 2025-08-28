#!/bin/bash

# COTEMA Analytics - Script de Configuración para Ubuntu 24.04
# Este script configura el entorno de desarrollo para el servicio de análisis

set -e  # Salir en caso de error

echo "🚀 Configurando COTEMA Analytics para Ubuntu 24.04..."

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para mostrar mensajes
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Verificar si estamos en Ubuntu
if ! grep -q "Ubuntu" /etc/os-release; then
    log_warning "Este script está optimizado para Ubuntu 24.04"
fi

# Actualizar sistema
log_info "Actualizando paquetes del sistema..."
sudo apt update && sudo apt upgrade -y

# Instalar dependencias del sistema
log_info "Instalando dependencias del sistema..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    curl \
    build-essential \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    liblzma-dev

# Verificar versión de Python
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
log_info "Versión de Python: $PYTHON_VERSION"

if [ "$(printf '%s\n' "3.8" "$PYTHON_VERSION" | sort -V | head -n1)" != "3.8" ]; then
    log_error "Se requiere Python 3.8 o superior"
    exit 1
fi

# Crear directorio del proyecto si no existe
PROJECT_DIR="/workspaces/IA-Taller-COTEMA"
if [ ! -d "$PROJECT_DIR" ]; then
    log_info "Creando directorio del proyecto..."
    mkdir -p "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"

# Crear entorno virtual
log_info "Creando entorno virtual de Python..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    log_success "Entorno virtual creado"
else
    log_info "Entorno virtual ya existe"
fi

# Activar entorno virtual
log_info "Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
log_info "Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias de Python
log_info "Instalando dependencias de Python..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    log_success "Dependencias instaladas desde requirements.txt"
else
    log_warning "requirements.txt no encontrado, instalando dependencias básicas..."
    pip install flask==2.3.3 \
                pandas==2.0.3 \
                scikit-learn==1.3.0 \
                plotly==5.17.0 \
                prophet==1.1.4 \
                lifelines==0.27.7 \
                openpyxl==3.1.2 \
                gunicorn==21.2.0
fi

# Verificar instalación
log_info "Verificando instalación..."
python3 -c "
import flask, pandas, sklearn, plotly, prophet, lifelines, openpyxl
print('✅ Todas las dependencias principales instaladas correctamente')
"

# Configurar variables de entorno
log_info "Configurando variables de entorno..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# Configuración COTEMA Analytics
FLASK_APP=app.py
FLASK_ENV=development
FLASK_DEBUG=True
SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Configuración de archivos
UPLOAD_FOLDER=uploads
MAX_CONTENT_LENGTH=52428800

# Configuración de logging
LOG_LEVEL=INFO
LOG_FILE=logs/cotema.log

# Configuración de base de datos (SQLite para desarrollo)
DATABASE_URL=sqlite:///cotema.db

# Configuración de modelos AI
MODEL_CACHE_TTL=3600
MAX_EQUIPOS_BATCH=100

# Configuración de gráficos
PLOTLY_THEME=plotly_white
CHART_HEIGHT=400
CHART_WIDTH=600
EOF
    log_success "Archivo .env creado"
fi

# Crear directorios necesarios
log_info "Creando estructura de directorios..."
mkdir -p uploads logs data/models data/cache static/exports

# Configurar permisos
chmod 755 uploads logs data/models data/cache static/exports

# Crear archivo de servicio systemd (opcional)
log_info "¿Deseas instalar COTEMA como servicio del sistema? (y/N)"
read -r INSTALL_SERVICE

if [[ $INSTALL_SERVICE =~ ^[Yy]$ ]]; then
    log_info "Creando servicio systemd..."
    
    sudo tee /etc/systemd/system/cotema-analytics.service > /dev/null << EOF
[Unit]
Description=COTEMA Analytics Web Service
After=network.target

[Service]
Type=exec
User=$USER
Group=$USER
WorkingDirectory=$PROJECT_DIR
Environment=PATH=$PROJECT_DIR/venv/bin
ExecStart=$PROJECT_DIR/venv/bin/gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
ExecReload=/bin/kill -s HUP \$MAINPID
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable cotema-analytics
    log_success "Servicio systemd configurado"
    log_info "Para iniciar el servicio: sudo systemctl start cotema-analytics"
    log_info "Para ver logs: sudo journalctl -u cotema-analytics -f"
fi

# Configurar firewall (si está habilitado)
if command -v ufw &> /dev/null && ufw status | grep -q "Status: active"; then
    log_info "Configurando firewall..."
    sudo ufw allow 5000/tcp
    log_success "Puerto 5000 habilitado en firewall"
fi

# Crear script de inicio rápido
log_info "Creando scripts de utilidad..."

cat > start_dev.sh << 'EOF'
#!/bin/bash
# Script para iniciar COTEMA Analytics en modo desarrollo

cd "$(dirname "$0")"
source venv/bin/activate
export FLASK_ENV=development
export FLASK_DEBUG=True

echo "🚀 Iniciando COTEMA Analytics en modo desarrollo..."
echo "Servidor disponible en: http://localhost:5000"
echo "Presiona Ctrl+C para detener"

python app.py
EOF

cat > start_prod.sh << 'EOF'
#!/bin/bash
# Script para iniciar COTEMA Analytics en modo producción

cd "$(dirname "$0")"
source venv/bin/activate

echo "🚀 Iniciando COTEMA Analytics en modo producción..."
echo "Servidor disponible en: http://localhost:5000"
echo "Presiona Ctrl+C para detener"

gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 app:app
EOF

cat > test_app.sh << 'EOF'
#!/bin/bash
# Script para ejecutar tests

cd "$(dirname "$0")"
source venv/bin/activate

echo "🧪 Ejecutando tests de COTEMA Analytics..."

# Test básico de importaciones
python3 -c "
import sys
sys.path.append('.')

try:
    from src.data_processor import DataProcessor
    from src.fr30_model import FR30Model
    from src.rul_model import RULModel
    from src.forecast_model import ForecastModel
    from src.anomaly_model import AnomalyModel
    from src.visualizations import create_visualizations
    from src.utils import generate_explanations
    print('✅ Todos los módulos importados correctamente')
except ImportError as e:
    print(f'❌ Error de importación: {e}')
    sys.exit(1)

# Test de Flask
try:
    from app import app
    with app.test_client() as client:
        response = client.get('/')
        if response.status_code == 200:
            print('✅ Flask app funciona correctamente')
        else:
            print(f'❌ Error en Flask app: {response.status_code}')
            sys.exit(1)
except Exception as e:
    print(f'❌ Error en Flask test: {e}')
    sys.exit(1)

print('🎉 Todos los tests pasaron exitosamente')
"
EOF

chmod +x start_dev.sh start_prod.sh test_app.sh

# Verificar configuración final
log_info "Ejecutando verificación final..."
./test_app.sh

if [ $? -eq 0 ]; then
    log_success "🎉 Configuración completada exitosamente!"
    echo ""
    echo "======================================"
    echo "  COTEMA Analytics - Listo para usar  "
    echo "======================================"
    echo ""
    echo "Para iniciar en modo desarrollo:"
    echo "  ./start_dev.sh"
    echo ""
    echo "Para iniciar en modo producción:"
    echo "  ./start_prod.sh"
    echo ""
    echo "Para ejecutar tests:"
    echo "  ./test_app.sh"
    echo ""
    echo "Directorios creados:"
    echo "  📁 uploads/     - Archivos Excel"
    echo "  📁 logs/        - Logs del sistema"
    echo "  📁 data/models/ - Modelos entrenados"
    echo "  📁 data/cache/  - Cache de datos"
    echo "  📁 static/exports/ - Exportaciones"
    echo ""
    echo "Variables de entorno configuradas en .env"
    
    if [[ $INSTALL_SERVICE =~ ^[Yy]$ ]]; then
        echo ""
        echo "Servicio systemd configurado:"
        echo "  sudo systemctl start cotema-analytics"
        echo "  sudo systemctl stop cotema-analytics"
        echo "  sudo systemctl status cotema-analytics"
    fi
    
    echo ""
    echo "🔗 Documentación: README.md"
    echo "🌐 Interfaz web: http://localhost:5000"
    echo ""
else
    log_error "Error en la configuración. Revisa los mensajes anteriores."
    exit 1
fi
