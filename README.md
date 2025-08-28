# COTEMA - Análisis Predictivo de Taller

Servicio web para análisis de KPIs predictivos en talleres de mantenimiento usando inteligencia artificial.

## 🚀 Características

### Modelos de IA Implementados

1. **FR-30 - Riesgo de Falla en 30 días**
   - Clasificador HistGradientBoostingClassifier con calibración
   - Probabilidades calibradas de reingreso al taller
   - Análisis de patrones temporales y reincidencia

2. **RUL - Vida Útil Restante**
   - Modelos de supervivencia Weibull parametrizados
   - RUL-50 (mediana) y RUL-90 (conservador)
   - Recomendaciones de ventanas de mantenimiento

3. **Pronóstico de Uso**
   - Predicción de horas operativas con Prophet
   - Proyecciones a 7 y 30 días
   - Análisis de estacionalidad

4. **Detección de Anomalías**
   - Isolation Forest para early warning
   - Detección de patrones atípicos pre-falla
   - Análisis multidimensional

### Funcionalidades Web

- ✅ Carga y procesamiento automático de archivos Excel
- ✅ Dashboard interactivo con visualizaciones Plotly
- ✅ Cálculo de KPIs por período seleccionado
- ✅ Explicaciones generadas por IA para cada equipo
- ✅ APIs REST para integración con herramientas de BI
- ✅ Exportación en formatos CSV y JSON
- ✅ Interfaz responsive y moderna

## 📋 Requisitos de Datos

### Columnas Requeridas
- `CODIGO` - Código único del equipo
- `FECHA_IN` - Fecha de ingreso al taller
- `FECHA_OUT` - Fecha de salida del taller

### Columnas Opcionales
- `SISTEMA_AFECTADO` - Sistema que presentó la falla
- `FLOTA` - Tipo de flota del equipo
- `CLASE` - Clasificación del equipo

## 🛠️ Instalación Local

### Prerrequisitos
- Python 3.8+
- pip

### Pasos de Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/Juangranados89/IA-Taller-COTEMA.git
cd IA-Taller-COTEMA
```

2. **Crear entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

4. **Ejecutar la aplicación**
```bash
python app.py
```

5. **Acceder a la aplicación**
- Abrir navegador en: `http://localhost:5000`

## 🌐 Despliegue en Render

### Opción 1: Despliegue Automático
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Juangranados89/IA-Taller-COTEMA)

### Opción 2: Despliegue Manual

1. **Crear cuenta en Render.com**

2. **Conectar repositorio de GitHub**

3. **Configurar el servicio web:**
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Environment:** `Python 3`
   - **Plan:** Free (o superior según necesidades)

4. **Variables de entorno (opcional):**
   - `FLASK_ENV=production`
   - `SECRET_KEY=tu_clave_secreta_aqui`

## 📚 Uso de la Aplicación

### 1. Carga de Datos
1. Acceder a la página principal
2. Seleccionar archivo Excel con los datos del taller
3. Hacer clic en "Cargar y Procesar Datos"
4. Esperar confirmación de procesamiento

### 2. Análisis de Datos
1. Ir al Dashboard
2. Explorar las visualizaciones en la pestaña "Vista General"
3. Seleccionar un mes en el panel lateral
4. Hacer clic en "Calcular KPIs"
5. Revisar resultados en la pestaña "KPIs Predictivos"

### 3. Interpretación de Resultados

#### FR-30 (Riesgo de Falla)
- 🔴 **Alto (≥50%)**: Atención inmediata requerida
- 🟠 **Medio (30-49%)**: Monitoreo cercano recomendado
- 🟢 **Bajo (<30%)**: Operación normal

#### RUL (Vida Útil Restante)
- **RUL-50**: Tiempo mediano hasta próxima falla
- **RUL-90**: Tiempo conservador (90% confianza)
- **Urgencia**: Basada en RUL-90 < 7 días (crítica)

#### Anomalías
- 🔴 **Alto (≥80%)**: Patrones muy atípicos detectados
- 🟠 **Medio (60-79%)**: Algunos patrones anómalos
- 🟢 **Bajo (<60%)**: Operación normal

## 🔌 Integración con BI

### Endpoints Disponibles

#### Obtener KPIs
```
GET /kpis/{mes}
```
Ejemplo: `/kpis/2024-03`

#### Exportar Datos
```
GET /api/export/csv
GET /api/export/json
```

#### Test de Conectividad
```
GET /api/connection-test
```

### Ejemplo de Integración con Power BI

1. **Fuente de Datos Web**
2. **URL**: `https://tu-app.render.com/api/export/json`
3. **Tipo**: JSON
4. **Actualización**: Programada según necesidades

## 🏗️ Arquitectura

```
IA-Taller-COTEMA/
├── app.py                 # Aplicación Flask principal
├── requirements.txt       # Dependencias Python
├── src/                   # Módulos de análisis
│   ├── data_processor.py  # Procesamiento de datos
│   ├── fr30_model.py      # Modelo FR-30
│   ├── rul_model.py       # Modelo RUL
│   ├── forecast_model.py  # Modelo de pronóstico
│   ├── anomaly_model.py   # Detección de anomalías
│   ├── visualizations.py # Generación de gráficos
│   └── utils.py           # Utilidades y LLM
├── templates/             # Templates HTML
├── static/                # Archivos estáticos
├── uploads/               # Archivos cargados
└── models/                # Modelos entrenados
```

## 📊 Ejemplos de Visualizaciones

- **Evolución Temporal**: Tendencias de ingresos al taller
- **Top Equipos**: Ranking por frecuencia de fallas
- **Heatmap de Actividad**: Patrones por día/hora
- **Análisis TBF**: Distribución de tiempo entre fallas
- **Scatter Multidimensional**: RUL vs Riesgo vs Anomalías

## 🔧 Configuración Avanzada

### Variables de Entorno
```bash
FLASK_ENV=production
SECRET_KEY=clave_secreta_segura
MAX_CONTENT_LENGTH=52428800  # 50MB
```

### Personalización de Modelos
- Modificar parámetros en cada modelo (`src/*.py`)
- Ajustar umbrales de clasificación
- Configurar ventanas temporales de análisis

## 🤝 Contribución

1. Fork del repositorio
2. Crear rama para nueva funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## 🆘 Soporte

- **Issues**: Reportar problemas en GitHub Issues
- **Documentación**: Ver documentación en el código
- **Email**: [contacto](mailto:contacto@ejemplo.com)

## 🔄 Actualizaciones

### Versión 1.0.0 (Actual)
- ✅ Implementación completa de 4 modelos de IA
- ✅ Dashboard web interactivo
- ✅ APIs REST para integración
- ✅ Despliegue en Render
- ✅ Exportación de datos

### Próximas Versiones
- 🔄 Integración con bases de datos
- 🔄 Autenticación y roles de usuario
- 🔄 Alertas automáticas por email
- 🔄 Modelo de clasificación de criticidad
- 🔄 Análisis de causa raíz automatizado

---

**COTEMA Analytics** - Transformando datos de taller en insights predictivos con IA 🤖⚙️
