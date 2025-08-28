// Funciones JavaScript para COTEMA Analytics

// Configuración global
const CONFIG = {
    API_BASE_URL: window.location.origin,
    CHART_COLORS: {
        primary: '#007bff',
        success: '#28a745',
        warning: '#ffc107',
        danger: '#dc3545',
        info: '#17a2b8',
        secondary: '#6c757d'
    },
    CHART_THEMES: {
        plotly: {
            paper_bgcolor: 'white',
            plot_bgcolor: 'white',
            font: { family: 'Segoe UI, sans-serif', size: 12 },
            colorway: ['#007bff', '#28a745', '#ffc107', '#dc3545', '#17a2b8', '#6c757d']
        }
    }
};

// Utilidades generales
const Utils = {
    // Formatear números
    formatNumber: (value, type = 'decimal', decimals = 2) => {
        if (value === null || value === undefined || isNaN(value)) return 'N/A';
        
        switch (type) {
            case 'percentage':
                return `${(value * 100).toFixed(1)}%`;
            case 'currency':
                return new Intl.NumberFormat('es-ES', { 
                    style: 'currency', 
                    currency: 'EUR' 
                }).format(value);
            case 'integer':
                return Math.round(value).toLocaleString('es-ES');
            case 'decimal':
                return value.toFixed(decimals);
            case 'days':
                return `${Math.round(value)} días`;
            case 'hours':
                return `${value.toFixed(1)} hrs`;
            default:
                return value.toString();
        }
    },

    // Mostrar notificaciones
    showNotification: (message, type = 'info', duration = 5000) => {
        const alertClass = {
            'success': 'alert-success',
            'error': 'alert-danger',
            'warning': 'alert-warning',
            'info': 'alert-info'
        }[type] || 'alert-info';

        const icon = {
            'success': 'fa-check-circle',
            'error': 'fa-exclamation-triangle',
            'warning': 'fa-exclamation-circle',
            'info': 'fa-info-circle'
        }[type] || 'fa-info-circle';

        const alertDiv = document.createElement('div');
        alertDiv.className = `alert ${alertClass} alert-dismissible fade show`;
        alertDiv.style.cssText = 'position: fixed; top: 70px; right: 20px; z-index: 1050; min-width: 300px;';
        
        alertDiv.innerHTML = `
            <i class="fas ${icon}"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto-remove después del tiempo especificado
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.remove();
            }
        }, duration);
    },

    // Mostrar loading spinner
    showLoading: (elementId, message = 'Cargando...') => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary mb-3" role="status">
                        <span class="visually-hidden">Cargando...</span>
                    </div>
                    <p class="text-muted">${message}</p>
                </div>
            `;
        }
    },

    // Mostrar error
    showError: (elementId, message) => {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <i class="fas fa-exclamation-triangle"></i>
                    <strong>Error:</strong> ${message}
                </div>
            `;
        }
    },

    // Validar archivo Excel
    validateExcelFile: (file) => {
        const allowedTypes = [
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ];
        
        const allowedExtensions = ['.xls', '.xlsx'];
        const fileExtension = file.name.toLowerCase().substring(file.name.lastIndexOf('.'));
        
        if (!allowedTypes.includes(file.type) && !allowedExtensions.includes(fileExtension)) {
            return { valid: false, error: 'El archivo debe ser un Excel (.xls o .xlsx)' };
        }
        
        if (file.size > 50 * 1024 * 1024) { // 50MB
            return { valid: false, error: 'El archivo no debe superar los 50MB' };
        }
        
        return { valid: true };
    },

    // Copiar al portapapeles
    copyToClipboard: (text) => {
        navigator.clipboard.writeText(text).then(() => {
            Utils.showNotification('Copiado al portapapeles', 'success', 2000);
        }).catch(() => {
            Utils.showNotification('Error al copiar', 'error');
        });
    },

    // Descargar datos como archivo
    downloadData: (data, filename, type = 'json') => {
        let content, mimeType;
        
        if (type === 'json') {
            content = JSON.stringify(data, null, 2);
            mimeType = 'application/json';
        } else if (type === 'csv') {
            content = data; // Asumimos que ya está en formato CSV
            mimeType = 'text/csv';
        }
        
        const blob = new Blob([content], { type: mimeType });
        const url = window.URL.createObjectURL(blob);
        
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        window.URL.revokeObjectURL(url);
    }
};

// API Client
const APIClient = {
    // Método base para hacer peticiones
    request: async (endpoint, options = {}) => {
        const url = `${CONFIG.API_BASE_URL}${endpoint}`;
        const defaultOptions = {
            headers: {
                'Content-Type': 'application/json',
            }
        };
        
        const config = { ...defaultOptions, ...options };
        
        try {
            const response = await fetch(url, config);
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.error || `HTTP ${response.status}`);
            }
            
            return data;
        } catch (error) {
            console.error('API Error:', error);
            throw error;
        }
    },

    // Calcular KPIs
    calculateKPIs: async (month) => {
        return await APIClient.request(`/kpis/${month}`);
    },

    // Obtener detalles de FR-30
    getFR30Details: async (equipo) => {
        return await APIClient.request(`/api/fr30/${equipo}`);
    },

    // Obtener detalles de RUL
    getRULDetails: async (equipo) => {
        return await APIClient.request(`/api/rul/${equipo}`);
    },

    // Test de conexión
    connectionTest: async () => {
        return await APIClient.request('/api/connection-test');
    },

    // Exportar datos
    exportData: async (format) => {
        return await APIClient.request(`/api/export/${format}`);
    }
};

// Generador de gráficos
const ChartGenerator = {
    // Configuración base de Plotly
    getBaseLayout: (title) => ({
        title: { text: title, font: { size: 16, family: 'Segoe UI' } },
        paper_bgcolor: 'white',
        plot_bgcolor: 'white',
        font: { family: 'Segoe UI', size: 12 },
        margin: { t: 50, l: 50, r: 50, b: 50 },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 }
    }),

    // Gráfico de barras para FR-30
    createFR30Chart: (data, containerId) => {
        const equipos = Object.keys(data);
        const risks = equipos.map(eq => data[eq].risk_30d || 0);
        const colors = equipos.map(eq => {
            const banda = data[eq].banda || '';
            if (banda.includes('🔴')) return CONFIG.CHART_COLORS.danger;
            if (banda.includes('🟠')) return CONFIG.CHART_COLORS.warning;
            if (banda.includes('🟢')) return CONFIG.CHART_COLORS.success;
            return CONFIG.CHART_COLORS.secondary;
        });

        const trace = {
            x: equipos,
            y: risks,
            type: 'bar',
            marker: { color: colors },
            text: risks.map(r => Utils.formatNumber(r, 'percentage')),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Riesgo: %{y:.1%}<extra></extra>'
        };

        const layout = {
            ...ChartGenerator.getBaseLayout('FR-30: Riesgo de Falla en 30 Días'),
            xaxis: { title: 'Equipos' },
            yaxis: { title: 'Probabilidad', tickformat: '.0%' },
            shapes: [
                { 
                    type: 'line', x0: 0, x1: 1, xref: 'paper', 
                    y0: 0.5, y1: 0.5, 
                    line: { color: CONFIG.CHART_COLORS.danger, dash: 'dash' } 
                },
                { 
                    type: 'line', x0: 0, x1: 1, xref: 'paper', 
                    y0: 0.3, y1: 0.3, 
                    line: { color: CONFIG.CHART_COLORS.warning, dash: 'dash' } 
                }
            ]
        };

        const config = { responsive: true, displayModeBar: true };
        Plotly.newPlot(containerId, [trace], layout, config);
    },

    // Gráfico de barras agrupadas para RUL
    createRULChart: (data, containerId) => {
        const equipos = Object.keys(data);
        const rul50 = equipos.map(eq => data[eq].rul50_d || 0);
        const rul90 = equipos.map(eq => data[eq].rul90_d || 0);

        const trace1 = {
            x: equipos,
            y: rul50,
            type: 'bar',
            name: 'RUL-50 (Mediana)',
            marker: { color: CONFIG.CHART_COLORS.info },
            text: rul50.map(r => Utils.formatNumber(r, 'days')),
            textposition: 'auto'
        };

        const trace2 = {
            x: equipos,
            y: rul90,
            type: 'bar',
            name: 'RUL-90 (Conservador)',
            marker: { color: CONFIG.CHART_COLORS.primary },
            text: rul90.map(r => Utils.formatNumber(r, 'days')),
            textposition: 'auto'
        };

        const layout = {
            ...ChartGenerator.getBaseLayout('RUL: Vida Útil Restante'),
            xaxis: { title: 'Equipos' },
            yaxis: { title: 'Días Restantes' },
            barmode: 'group',
            shapes: [
                { 
                    type: 'line', x0: 0, x1: 1, xref: 'paper', 
                    y0: 7, y1: 7, 
                    line: { color: CONFIG.CHART_COLORS.danger, dash: 'dash' } 
                }
            ]
        };

        const config = { responsive: true, displayModeBar: true };
        Plotly.newPlot(containerId, [trace1, trace2], layout, config);
    },

    // Scatter plot para análisis multidimensional
    createAnomalyScatter: (kpis, containerId) => {
        const equipos = Object.keys(kpis.fr30);
        const x_values = equipos.map(eq => kpis.rul[eq]?.rul50_d || 0);
        const y_values = equipos.map(eq => kpis.fr30[eq]?.risk_30d || 0);
        const colors = equipos.map(eq => kpis.anomaly[eq]?.anomaly_score || 0);

        const trace = {
            x: x_values,
            y: y_values,
            mode: 'markers',
            type: 'scatter',
            text: equipos,
            marker: {
                size: 12,
                color: colors,
                colorscale: 'Viridis',
                colorbar: { title: 'Score Anomalía' },
                line: { width: 1, color: 'white' }
            },
            hovertemplate: '<b>%{text}</b><br>RUL-50: %{x} días<br>FR-30: %{y:.1%}<br>Anomalía: %{marker.color:.2f}<extra></extra>'
        };

        const layout = {
            ...ChartGenerator.getBaseLayout('Análisis Multidimensional'),
            xaxis: { title: 'RUL-50 (días)' },
            yaxis: { title: 'Riesgo FR-30', tickformat: '.0%' },
            shapes: [
                { type: 'line', x0: 30, x1: 30, y0: 0, y1: 1, yref: 'paper', line: { color: 'gray', dash: 'dash' } },
                { type: 'line', y0: 0.5, y1: 0.5, x0: 0, x1: 1, xref: 'paper', line: { color: 'gray', dash: 'dash' } }
            ]
        };

        const config = { responsive: true, displayModeBar: true };
        Plotly.newPlot(containerId, [trace], layout, config);
    }
};

// Manejador de KPIs
const KPIManager = {
    currentData: null,
    currentMonth: null,

    // Calcular KPIs para un mes
    calculate: async (month) => {
        if (!month) {
            Utils.showNotification('Por favor selecciona un mes', 'warning');
            return;
        }

        try {
            Utils.showLoading('kpiResults', 'Calculando KPIs...');
            
            const data = await APIClient.calculateKPIs(month);
            
            KPIManager.currentData = data.kpis;
            KPIManager.currentMonth = month;
            
            KPIManager.displayResults(data);
            KPIManager.updateSummary(data);
            
            Utils.showNotification('KPIs calculados exitosamente', 'success');
            
        } catch (error) {
            Utils.showError('kpiResults', error.message);
            Utils.showNotification(`Error: ${error.message}`, 'error');
        }
    },

    // Mostrar resultados de KPIs
    displayResults: (data) => {
        const container = document.getElementById('kpiResults');
        
        container.innerHTML = `
            <div class="row mb-4">
                <div class="col-12">
                    <h4><i class="fas fa-brain"></i> KPIs Predictivos - ${data.mes}</h4>
                    <p class="text-muted">Análisis generado para ${data.total_equipos} equipos</p>
                </div>
            </div>
            
            <div class="row">
                <div class="col-lg-6 mb-4">
                    <div class="plot-container">
                        <div id="fr30Chart"></div>
                    </div>
                </div>
                <div class="col-lg-6 mb-4">
                    <div class="plot-container">
                        <div id="rulChart"></div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-lg-6 mb-4">
                    <div class="plot-container">
                        <div id="forecastChart"></div>
                    </div>
                </div>
                <div class="col-lg-6 mb-4">
                    <div class="plot-container">
                        <div id="anomalyScatter"></div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-12">
                    <div class="plot-container">
                        <div id="equipmentSummary"></div>
                    </div>
                </div>
            </div>
        `;

        // Generar gráficos
        setTimeout(() => {
            ChartGenerator.createFR30Chart(data.kpis.fr30, 'fr30Chart');
            ChartGenerator.createRULChart(data.kpis.rul, 'rulChart');
            KPIManager.createForecastChart(data.kpis.forecast, 'forecastChart');
            ChartGenerator.createAnomalyScatter(data.kpis, 'anomalyScatter');
            KPIManager.createSummaryTable(data.kpis, 'equipmentSummary');
        }, 100);

        // Cambiar a la pestaña de KPIs
        const kpiTab = new bootstrap.Tab(document.getElementById('kpis-tab'));
        kpiTab.show();
    },

    // Crear gráfico de pronóstico
    createForecastChart: (data, containerId) => {
        const equipos = Object.keys(data);
        const forecast7d = equipos.map(eq => data[eq].forecast_7d_h || 0);
        const forecast30d = equipos.map(eq => data[eq].forecast_30d_h || 0);

        const trace1 = {
            x: equipos,
            y: forecast7d,
            type: 'bar',
            name: 'Pronóstico 7 días',
            marker: { color: CONFIG.CHART_COLORS.success }
        };

        const trace2 = {
            x: equipos,
            y: forecast30d,
            type: 'bar',
            name: 'Pronóstico 30 días',
            marker: { color: '#1e7e34' }
        };

        const layout = {
            ...ChartGenerator.getBaseLayout('Pronóstico de Uso'),
            xaxis: { title: 'Equipos' },
            yaxis: { title: 'Horas de Operación' },
            barmode: 'group'
        };

        const config = { responsive: true, displayModeBar: true };
        Plotly.newPlot(containerId, [trace1, trace2], layout, config);
    },

    // Crear tabla resumen
    createSummaryTable: (kpis, containerId) => {
        const equipos = Object.keys(kpis.fr30);
        
        const tableData = equipos.map(equipo => [
            equipo,
            Utils.formatNumber(kpis.fr30[equipo]?.risk_30d || 0, 'percentage'),
            kpis.fr30[equipo]?.banda || 'N/A',
            Utils.formatNumber(kpis.rul[equipo]?.rul50_d || 0, 'days'),
            Utils.formatNumber(kpis.rul[equipo]?.rul90_d || 0, 'days'),
            Utils.formatNumber(kpis.forecast[equipo]?.forecast_7d_h || 0, 'hours'),
            Utils.formatNumber(kpis.forecast[equipo]?.forecast_30d_h || 0, 'hours'),
            Utils.formatNumber(kpis.anomaly[equipo]?.anomaly_score || 0, 'decimal'),
            kpis.anomaly[equipo]?.banda || 'N/A'
        ]);

        const trace = {
            type: 'table',
            header: {
                values: ['Equipo', 'FR-30', 'Banda FR-30', 'RUL-50', 'RUL-90', 'Pronóstico 7d', 'Pronóstico 30d', 'Anomalía', 'Banda Anomalía'],
                fill: { color: CONFIG.CHART_COLORS.primary },
                font: { color: 'white', size: 12 },
                align: 'center'
            },
            cells: {
                values: [
                    tableData.map(row => row[0]),
                    tableData.map(row => row[1]),
                    tableData.map(row => row[2]),
                    tableData.map(row => row[3]),
                    tableData.map(row => row[4]),
                    tableData.map(row => row[5]),
                    tableData.map(row => row[6]),
                    tableData.map(row => row[7]),
                    tableData.map(row => row[8])
                ],
                fill: { color: ['#f8f9fa', '#ffffff'] },
                align: 'center',
                font: { size: 11 }
            }
        };

        const layout = {
            title: 'Resumen de KPIs por Equipo',
            height: 600,
            margin: { t: 50, l: 20, r: 20, b: 20 }
        };

        const config = { responsive: true, displayModeBar: true };
        Plotly.newPlot(containerId, [trace], layout, config);
    },

    // Actualizar resumen en sidebar
    updateSummary: (data) => {
        // Implementar lógica de resumen en sidebar
        console.log('Updating KPI summary:', data);
    }
};

// Event listeners y inicialización
document.addEventListener('DOMContentLoaded', function() {
    // Inicializar tooltips de Bootstrap
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Auto-hide alerts después de 5 segundos
    setTimeout(() => {
        const alerts = document.querySelectorAll('.alert-floating .alert');
        alerts.forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);

    // Validación de formulario de upload
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            
            if (fileInput.files.length === 0) {
                e.preventDefault();
                Utils.showNotification('Por favor selecciona un archivo', 'warning');
                return;
            }

            const validation = Utils.validateExcelFile(fileInput.files[0]);
            if (!validation.valid) {
                e.preventDefault();
                Utils.showNotification(validation.error, 'error');
                return;
            }

            // Mostrar loading en el botón
            const uploadBtn = document.getElementById('uploadBtn');
            if (uploadBtn) {
                uploadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Procesando...';
                uploadBtn.disabled = true;
            }
        });
    }

    // Función global para calcular KPIs (llamada desde el HTML)
    window.calculateKPIs = function() {
        const mesSelect = document.getElementById('mesSelect');
        if (mesSelect) {
            KPIManager.calculate(mesSelect.value);
        }
    };

    // Test de conexión inicial
    APIClient.connectionTest()
        .then(data => {
            console.log('Conexión establecida:', data);
        })
        .catch(error => {
            console.warn('Error en test de conexión:', error);
        });
});

// Exportar funciones para uso global
window.Utils = Utils;
window.APIClient = APIClient;
window.ChartGenerator = ChartGenerator;
window.KPIManager = KPIManager;
