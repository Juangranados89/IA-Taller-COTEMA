// Monitoreo de estado ML/análisis profundo
function monitorMLStatus() {
    const statusDiv = document.getElementById('mlStatus');
    if (!statusDiv) return;

    function checkStatus() {
        fetch('/ml-status')
            .then(res => res.json())
            .then(data => {
                if (data.deep_analysis_in_progress) {
                    const prog = data.ml_progress || { percent: 0, step: '', processed: 0, total: 0 };
                    statusDiv.innerHTML = `
                        <div class="alert alert-info">
                            <i class="fas fa-robot fa-spin"></i> Análisis profundo en progreso...
                            <div class="progress my-2" style="height: 20px;">
                                <div class="progress-bar progress-bar-striped progress-bar-animated bg-info" role="progressbar" style="width: ${prog.percent}%" aria-valuenow="${prog.percent}" aria-valuemin="0" aria-valuemax="100">${prog.percent}%</div>
                            </div>
                            <div><strong>Paso:</strong> ${prog.step || 'Preparando...'}</div>
                            <div><strong>Registros procesados:</strong> ${prog.processed} / ${prog.total}</div>
                        </div>`;
                    setTimeout(checkStatus, 1500);
                } else if (data.ml_models_trained) {
                    statusDiv.innerHTML = '<div class="alert alert-success"><i class="fas fa-check-circle"></i> Modelos ML entrenados. Resultados avanzados disponibles.</div>';
                } else {
                    statusDiv.innerHTML = '';
                }
            })
            .catch(() => {
                statusDiv.innerHTML = '<div class="alert alert-warning">No se pudo consultar el estado del análisis profundo.</div>';
            });
    }
    checkStatus();
}

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
    },

    // Análisis de frecuencia mensual
    getFrequencyAnalysis: async () => {
        return await APIClient.request('/api/frequency-analysis');
    },

    // Real ML APIs
    trainRealML: async () => {
        return await APIClient.request('/api/train-real-ml', { method: 'POST' });
    },

    getRealMLPredictions: async () => {
        return await APIClient.request('/api/real-ml-predict', { method: 'POST' });
    },

    getRealMLInsights: async () => {
        return await APIClient.request('/api/real-ml-insights');
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

// Manejador de Machine Learning REAL
const RealMLManager = {
    currentData: null,
    isTraining: false,
    isModelTrained: false,

    // Entrenar modelos de ML real
    trainModels: async () => {
        try {
            if (RealMLManager.isTraining) {
                Utils.showNotification('Ya hay un entrenamiento en progreso', 'warning');
                return;
            }

            RealMLManager.isTraining = true;
            Utils.showLoading('realMLResults', 'Entrenando modelos de Machine Learning REAL...');
            
            const response = await APIClient.trainRealML();
            
            if (!response.success) {
                throw new Error(response.error);
            }
            
            Utils.showNotification('Entrenamiento de ML Real iniciado', 'success');
            
            // Monitorear progreso
            RealMLManager.monitorTraining();
            
        } catch (error) {
            RealMLManager.isTraining = false;
            Utils.showError('realMLResults', error.message);
            Utils.showNotification(`Error: ${error.message}`, 'error');
        }
    },

    // Monitorear progreso de entrenamiento
    monitorTraining: async () => {
        const checkProgress = async () => {
            try {
                const progressResponse = await APIClient.request('/api/progress');
                
                if (progressResponse.training_real_ml) {
                    const progress = progressResponse.training_real_ml;
                    
                    if (progress.status === 'completed') {
                        RealMLManager.isTraining = false;
                        RealMLManager.isModelTrained = true;
                        Utils.showNotification('¡Modelos de ML Real entrenados exitosamente!', 'success');
                        RealMLManager.showTrainingComplete();
                        return;
                    } else if (progress.status === 'error') {
                        RealMLManager.isTraining = false;
                        Utils.showError('realMLResults', progress.message);
                        return;
                    }
                    
                    // Continuar monitoreando
                    setTimeout(checkProgress, 2000);
                }
            } catch (error) {
                console.error('Error monitoring training:', error);
                setTimeout(checkProgress, 3000);
            }
        };
        
        checkProgress();
    },

    // Mostrar entrenamiento completado
    showTrainingComplete: () => {
        const container = document.getElementById('realMLResults');
        
        container.innerHTML = `
            <div class="alert alert-success" role="alert">
                <i class="fas fa-check-circle"></i>
                <strong>¡Entrenamiento Completado!</strong> Los modelos de Machine Learning REAL han sido entrenados exitosamente.
            </div>
            
            <div class="row mt-3">
                <div class="col-md-6">
                    <button class="btn btn-primary btn-lg w-100" onclick="RealMLManager.getPredictions()">
                        <i class="fas fa-magic"></i> Ver Predicciones ML
                    </button>
                </div>
                <div class="col-md-6">
                    <button class="btn btn-info btn-lg w-100" onclick="RealMLManager.getInsights()">
                        <i class="fas fa-lightbulb"></i> Ver Insights del Modelo
                    </button>
                </div>
            </div>
        `;
    },

    // Obtener predicciones
    getPredictions: async () => {
        try {
            Utils.showLoading('realMLResults', 'Generando predicciones con ML Real...');
            
            const response = await APIClient.getRealMLPredictions();
            
            if (!response.success) {
                throw new Error(response.error);
            }
            
            RealMLManager.currentData = response.data;
            RealMLManager.displayPredictions(response.data);
            
            Utils.showNotification('Predicciones ML Real generadas exitosamente', 'success');
            
        } catch (error) {
            Utils.showError('realMLResults', error.message);
            Utils.showNotification(`Error: ${error.message}`, 'error');
        }
    },

    // Mostrar predicciones
    displayPredictions: (data) => {
        const container = document.getElementById('realMLResults');
        
        const modelInfo = data.model_info || {};
        const models = Object.keys(data).filter(key => key !== 'model_info');
        
        container.innerHTML = `
            <div class="row mb-4">
                <div class="col-12">
                    <h4><i class="fas fa-brain"></i> Predicciones con Machine Learning REAL</h4>
                    <p class="text-muted">
                        Modelos entrenados: ${modelInfo.models_available ? modelInfo.models_available.join(', ') : 'N/A'} |
                        Features utilizadas: ${modelInfo.feature_count || 'N/A'}
                    </p>
                </div>
            </div>
            
            <div class="row mb-3">
                <div class="col-md-6">
                    <button class="btn btn-outline-primary" onclick="RealMLManager.getInsights()">
                        <i class="fas fa-chart-bar"></i> Ver Insights del Modelo
                    </button>
                </div>
                <div class="col-md-6">
                    <button class="btn btn-outline-success" onclick="RealMLManager.trainModels()">
                        <i class="fas fa-sync"></i> Reentrenar Modelos
                    </button>
                </div>
            </div>
            
            <div class="row">
                ${models.map(modelName => {
                    const modelData = data[modelName];
                    return `
                        <div class="col-lg-6 mb-4">
                            <div class="card">
                                <div class="card-header bg-primary text-white">
                                    <h5><i class="fas fa-cog"></i> ${modelName.toUpperCase()}</h5>
                                </div>
                                <div class="card-body">
                                    ${RealMLManager.renderModelData(modelName, modelData)}
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    },

    // Renderizar datos de modelo específico
    renderModelData: (modelName, modelData) => {
        if (modelName === 'clustering') {
            const clusters = modelData.clusters || [];
            const uniqueClusters = [...new Set(clusters)];
            
            return `
                <h6>Clustering de Equipos</h6>
                <p><strong>Clusters identificados:</strong> ${uniqueClusters.length}</p>
                <p><strong>Equipos agrupados:</strong> ${clusters.length}</p>
                <small class="text-muted">Los equipos han sido agrupados automáticamente por comportamiento similar</small>
            `;
        } else if (modelData.predictions) {
            const predictions = modelData.predictions;
            const confidence = modelData.confidence;
            
            return `
                <h6>Predicciones del Modelo</h6>
                <p><strong>Predicciones generadas:</strong> ${predictions.length}</p>
                ${Array.isArray(confidence) ? 
                    `<p><strong>Confianza promedio:</strong> ${(confidence.reduce((a,b) => a+b, 0) / confidence.length * 100).toFixed(1)}%</p>` :
                    `<p><strong>Confianza:</strong> ${confidence}</p>`
                }
                <div style="max-height: 200px; overflow-y: auto;">
                    <small class="text-muted">
                        Primeras predicciones: ${predictions.slice(0, 10).map(p => typeof p === 'number' ? p.toFixed(2) : p).join(', ')}
                        ${predictions.length > 10 ? '...' : ''}
                    </small>
                </div>
            `;
        }
        
        return '<p class="text-muted">Datos de modelo no disponibles</p>';
    },

    // Obtener insights del modelo
    getInsights: async () => {
        try {
            Utils.showLoading('realMLResults', 'Obteniendo insights del modelo...');
            
            const response = await APIClient.getRealMLInsights();
            
            if (!response.success) {
                throw new Error(response.error);
            }
            
            RealMLManager.displayInsights(response.data);
            
            Utils.showNotification('Insights obtenidos exitosamente', 'success');
            
        } catch (error) {
            Utils.showError('realMLResults', error.message);
            Utils.showNotification(`Error: ${error.message}`, 'error');
        }
    },

    // Mostrar insights
    displayInsights: (data) => {
        const container = document.getElementById('realMLResults');
        
        const featureImportance = data.feature_importance || {};
        const performance = data.model_performance || {};
        const recommendations = data.recommendations || [];
        
        container.innerHTML = `
            <div class="row mb-4">
                <div class="col-12">
                    <h4><i class="fas fa-lightbulb"></i> Insights del Machine Learning REAL</h4>
                    <p class="text-muted">Análisis automático de lo que los modelos aprendieron de tus datos</p>
                </div>
            </div>
            
            ${recommendations.length > 0 ? `
            <div class="row mb-4">
                <div class="col-12">
                    <div class="alert alert-info">
                        <h6><i class="fas fa-recommendations"></i> Recomendaciones Automáticas:</h6>
                        <ul class="mb-0">
                            ${recommendations.map(rec => `<li>${rec}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
            ` : ''}
            
            <div class="row">
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h5><i class="fas fa-chart-bar"></i> Feature Importance</h5>
                        </div>
                        <div class="card-body">
                            ${Object.keys(featureImportance).length > 0 ? 
                                Object.entries(featureImportance).map(([modelName, features]) => `
                                    <h6>${modelName}</h6>
                                    <div class="mb-3">
                                        ${features.slice(0, 5).map(f => `
                                            <div class="d-flex justify-content-between mb-1">
                                                <small>${f.feature}</small>
                                                <small><strong>${(f.importance * 100).toFixed(1)}%</strong></small>
                                            </div>
                                            <div class="progress mb-2" style="height: 5px;">
                                                <div class="progress-bar" style="width: ${(f.importance * 100).toFixed(1)}%"></div>
                                            </div>
                                        `).join('')}
                                    </div>
                                `).join('') :
                                '<p class="text-muted">No hay datos de importancia de features disponibles</p>'
                            }
                        </div>
                    </div>
                </div>
                
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header bg-info text-white">
                            <h5><i class="fas fa-tachometer-alt"></i> Performance de Modelos</h5>
                        </div>
                        <div class="card-body">
                            ${Object.keys(performance).length > 0 ? 
                                Object.entries(performance).map(([modelName, perf]) => `
                                    <div class="mb-3">
                                        <h6>${modelName}</h6>
                                        <small class="text-muted">Tipo: ${perf.type}</small><br>
                                        ${perf.type === 'classification' ? `
                                            <small>Precisión: <strong>${(perf.cv_mean * 100).toFixed(1)}%</strong></small><br>
                                            <small>Muestras: ${perf.n_samples}</small>
                                        ` : perf.type === 'regression' ? `
                                            <small>R² Score: <strong>${(perf.cv_r2_mean * 100).toFixed(1)}%</strong></small><br>
                                            <small>RMSE: ${perf.rmse ? perf.rmse.toFixed(2) : 'N/A'}</small><br>
                                            <small>Modelo: ${perf.model_used || 'N/A'}</small><br>
                                            <small>Muestras: ${perf.n_samples}</small>
                                        ` : perf.type === 'clustering' ? `
                                            <small>Clusters: <strong>${perf.n_clusters}</strong></small><br>
                                            <small>Silhouette Score: ${perf.silhouette_score ? perf.silhouette_score.toFixed(3) : 'N/A'}</small><br>
                                            <small>Muestras: ${perf.n_samples}</small>
                                        ` : ''}
                                    </div>
                                    <hr>
                                `).join('') :
                                '<p class="text-muted">No hay datos de performance disponibles</p>'
                            }
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-3">
                <div class="col-md-6">
                    <button class="btn btn-primary" onclick="RealMLManager.getPredictions()">
                        <i class="fas fa-arrow-left"></i> Volver a Predicciones
                    </button>
                </div>
                <div class="col-md-6">
                    <button class="btn btn-success" onclick="RealMLManager.trainModels()">
                        <i class="fas fa-sync"></i> Reentrenar Modelos
                    </button>
                </div>
            </div>
        `;
    }
};

// Manejador de análisis de frecuencia
const FrequencyAnalysisManager = {
    currentData: null,

    // Ejecutar análisis de frecuencia
    analyze: async () => {
        try {
            Utils.showLoading('frequencyResults', 'Analizando frecuencia de equipos...');
            
            const response = await APIClient.getFrequencyAnalysis();
            
            if (!response.success) {
                throw new Error(response.error);
            }
            
            FrequencyAnalysisManager.currentData = response.data;
            FrequencyAnalysisManager.displayResults(response.data);
            
            Utils.showNotification('Análisis de frecuencia completado exitosamente', 'success');
            
        } catch (error) {
            Utils.showError('frequencyResults', error.message);
            Utils.showNotification(`Error: ${error.message}`, 'error');
        }
    },

    // Mostrar resultados del análisis
    displayResults: (data) => {
        const container = document.getElementById('frequencyResults');
        
        const metodologia = data.metodologia_mejorada || {};
        
        container.innerHTML = `
            <div class="row mb-4">
                <div class="col-12">
                    <h4><i class="fas fa-chart-line"></i> Análisis de Frecuencia Mensual - Metodología Mejorada</h4>
                    <p class="text-muted">
                        Período: ${data.periodo_analizado.desde} - ${data.periodo_analizado.hasta} 
                        (${data.periodo_analizado.total_registros} registros totales)
                    </p>
                    ${metodologia.enfoque ? `
                    <div class="alert alert-success" role="alert">
                        <i class="fas fa-lightbulb"></i> <strong>Enfoque Mejorado:</strong> ${metodologia.enfoque}
                        <ul class="mt-2 mb-0">
                            ${metodologia.filtros_aplicados ? metodologia.filtros_aplicados.map(filtro => `<li>${filtro}</li>`).join('') : ''}
                        </ul>
                    </div>
                    ` : ''}
                </div>
            </div>
            
            <div class="row mb-4">
                <div class="col-lg-3">
                    <div class="card border-success">
                        <div class="card-body text-center">
                            <h5 class="card-title text-success">Equipos Activos</h5>
                            <h3 class="text-success">${data.periodo_analizado.equipos_activos || 'N/A'}</h3>
                            <small class="text-muted">Con actividad reciente</small>
                        </div>
                    </div>
                </div>
                <div class="col-lg-3">
                    <div class="card border-danger">
                        <div class="card-body text-center">
                            <h5 class="card-title text-danger">Mant. Correctivos</h5>
                            <h3 class="text-danger">${data.periodo_analizado.mantenimientos_correctivos || 'N/A'}</h3>
                            <small class="text-muted">Fallos reales</small>
                        </div>
                    </div>
                </div>
                <div class="col-lg-3">
                    <div class="card border-primary">
                        <div class="card-body text-center">
                            <h5 class="card-title text-primary">Mant. Preventivos</h5>
                            <h3 class="text-primary">${data.periodo_analizado.mantenimientos_preventivos || 'N/A'}</h3>
                            <small class="text-muted">Mantenimientos programados</small>
                        </div>
                    </div>
                </div>
                <div class="col-lg-3">
                    <div class="card border-warning">
                        <div class="card-body text-center">
                            <h5 class="card-title text-warning">Equipos en Riesgo</h5>
                            <h3 class="text-warning">${data.resumen.equipos_activos_riesgo || 0}</h3>
                            <small class="text-muted">Con probabilidad significativa</small>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-lg-6 mb-4">
                    <div class="plot-container">
                        <div id="frequencyChart"></div>
                    </div>
                </div>
                <div class="col-lg-6 mb-4">
                    <div class="plot-container">
                        <div id="riskProjectionChart"></div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h5><i class="fas fa-exclamation-triangle"></i> Equipos Activos con Riesgo Real</h5>
                        </div>
                        <div class="card-body">
                            <div id="frequencyTable" style="height: 400px; overflow-y: auto;"></div>
                        </div>
                    </div>
                </div>
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header bg-warning text-white">
                            <h5><i class="fas fa-calendar-check"></i> Proyección de Fallos Próximo Mes</h5>
                        </div>
                        <div class="card-body">
                            <div id="projectionTable" style="height: 400px; overflow-y: auto;"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Generar gráficos y tablas
        setTimeout(() => {
            FrequencyAnalysisManager.createFrequencyChart(data.equipos_frecuentes, 'frequencyChart');
            FrequencyAnalysisManager.createRiskProjectionChart(data.proyeccion_fallos, 'riskProjectionChart');
            FrequencyAnalysisManager.createFrequencyTable(data.equipos_frecuentes, 'frequencyTable');
            FrequencyAnalysisManager.createProjectionTable(data.proyeccion_fallos, 'projectionTable');
        }, 100);
    },

    // Crear gráfico de frecuencia mensual
    createFrequencyChart: (data, containerId) => {
        const equipos = data.map(item => item.equipo);
        const correctivos = data.map(item => item.promedio_correctivos_mes || 0);
        const totales = data.map(item => item.promedio_total_mes || 0);
        const ratios = data.map(item => item.ratio_correctivos || 0);

        const trace1 = {
            x: equipos,
            y: correctivos,
            type: 'bar',
            name: 'Correctivos/Mes',
            marker: { color: CONFIG.CHART_COLORS.danger },
            text: correctivos.map(p => p.toFixed(1)),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Correctivos: %{y:.1f}/mes<extra></extra>'
        };

        const trace2 = {
            x: equipos,
            y: totales,
            type: 'bar',
            name: 'Total/Mes',
            marker: { color: CONFIG.CHART_COLORS.primary },
            text: totales.map(p => p.toFixed(1)),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Total: %{y:.1f}/mes<extra></extra>'
        };

        const trace3 = {
            x: equipos,
            y: ratios,
            type: 'scatter',
            mode: 'markers',
            name: 'Ratio Correctivos',
            marker: { 
                color: CONFIG.CHART_COLORS.warning,
                size: 10,
                symbol: 'diamond'
            },
            yaxis: 'y2',
            hovertemplate: '<b>%{x}</b><br>Ratio: %{y:.1%}<extra></extra>'
        };

        const layout = {
            ...ChartGenerator.getBaseLayout('Análisis Mejorado: Correctivos vs Total'),
            xaxis: { title: 'Equipos Activos' },
            yaxis: { title: 'Promedio Ingresos/Mes' },
            yaxis2: {
                title: 'Ratio Correctivos',
                overlaying: 'y',
                side: 'right',
                tickformat: '.0%'
            },
            barmode: 'group'
        };

        const config = { responsive: true, displayModeBar: true };
        Plotly.newPlot(containerId, [trace1, trace2, trace3], layout, config);
    },

    // Crear gráfico de proyección de riesgo
    createRiskProjectionChart: (data, containerId) => {
        const equipos = data.map(item => item.equipo);
        const probabilidades = data.map(item => item.probabilidad_fallo_proximo_mes);
        const dias_estimados = data.map(item => item.dias_estimados_proximo_ingreso);

        const colors = probabilidades.map(prob => {
            if (prob >= 0.7) return CONFIG.CHART_COLORS.danger;
            if (prob >= 0.4) return CONFIG.CHART_COLORS.warning;
            return CONFIG.CHART_COLORS.success;
        });

        const trace = {
            x: equipos,
            y: probabilidades,
            type: 'bar',
            marker: { color: colors },
            text: probabilidades.map(p => `${(p * 100).toFixed(1)}%`),
            textposition: 'auto',
            hovertemplate: '<b>%{x}</b><br>Probabilidad: %{y:.1%}<br>Días estimados: ' + 
                          dias_estimados.map(d => d).join(',').split(',').map((d, i) => i === equipos.indexOf('%{x}') ? d : '').filter(Boolean)[0] + '<extra></extra>'
        };

        const layout = {
            ...ChartGenerator.getBaseLayout('Probabilidad de Fallo Próximo Mes'),
            xaxis: { title: 'Equipos' },
            yaxis: { title: 'Probabilidad', tickformat: '.0%' },
            shapes: [
                { 
                    type: 'line', x0: 0, x1: 1, xref: 'paper', 
                    y0: 0.7, y1: 0.7, 
                    line: { color: CONFIG.CHART_COLORS.danger, dash: 'dash' },
                    annotation: { text: 'Alto Riesgo (70%)', x: 0.02, y: 0.72 }
                },
                { 
                    type: 'line', x0: 0, x1: 1, xref: 'paper', 
                    y0: 0.4, y1: 0.4, 
                    line: { color: CONFIG.CHART_COLORS.warning, dash: 'dash' },
                    annotation: { text: 'Riesgo Medio (40%)', x: 0.02, y: 0.42 }
                }
            ]
        };

        const config = { responsive: true, displayModeBar: true };
        Plotly.newPlot(containerId, [trace], layout, config);
    },

    // Crear tabla de frecuencia
    createFrequencyTable: (data, containerId) => {
        const container = document.getElementById(containerId);
        
        const tableHTML = `
            <table class="table table-sm table-hover">
                <thead class="table-success">
                    <tr>
                        <th>Equipo</th>
                        <th>Correctivos/Mes</th>
                        <th>Total/Mes</th>
                        <th>Ratio Correctivos</th>
                        <th>Score Riesgo</th>
                        <th>Estado</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.map(item => `
                        <tr>
                            <td><strong>${item.equipo}</strong></td>
                            <td><span class="badge bg-danger">${item.promedio_correctivos_mes || 0}</span></td>
                            <td><span class="badge bg-primary">${item.promedio_total_mes || 0}</span></td>
                            <td>
                                <span class="badge ${(item.ratio_correctivos || 0) >= 0.5 ? 'bg-danger' : (item.ratio_correctivos || 0) >= 0.3 ? 'bg-warning' : 'bg-success'}">
                                    ${((item.ratio_correctivos || 0) * 100).toFixed(1)}%
                                </span>
                            </td>
                            <td><span class="badge ${item.score_riesgo >= 2 ? 'bg-danger' : item.score_riesgo >= 1 ? 'bg-warning' : 'bg-success'}">${item.score_riesgo}</span></td>
                            <td><i class="fas fa-check-circle text-success" title="Equipo Activo"></i></td>
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        `;
        
        container.innerHTML = tableHTML;
    },

    // Crear tabla de proyección
    createProjectionTable: (data, containerId) => {
        const container = document.getElementById(containerId);
        
        const tableHTML = `
            <table class="table table-sm table-hover">
                <thead class="table-warning">
                    <tr>
                        <th>Equipo</th>
                        <th>Probabilidad</th>
                        <th>Días Estimados</th>
                        <th>Tendencia Correctivos</th>
                        <th>Tipo Riesgo</th>
                    </tr>
                </thead>
                <tbody>
                    ${data.map(item => {
                        const probPercent = (item.probabilidad_fallo_proximo_mes * 100).toFixed(1);
                        const riskClass = item.probabilidad_fallo_proximo_mes >= 0.6 ? 'danger' : 
                                         item.probabilidad_fallo_proximo_mes >= 0.3 ? 'warning' : 'success';
                        const riskIcon = item.probabilidad_fallo_proximo_mes >= 0.6 ? 'fa-exclamation-triangle' : 
                                        item.probabilidad_fallo_proximo_mes >= 0.3 ? 'fa-exclamation-circle' : 'fa-check-circle';
                        
                        const tipoRiesgo = item.probabilidad_fallo_proximo_mes >= 0.6 ? 'Alto' :
                                          item.probabilidad_fallo_proximo_mes >= 0.3 ? 'Medio' : 'Bajo';
                        
                        return `
                            <tr>
                                <td><strong>${item.equipo}</strong></td>
                                <td><span class="badge bg-${riskClass}">${probPercent}%</span></td>
                                <td><span class="badge bg-info">${item.dias_estimados_proximo_ingreso} días</span></td>
                                <td>${item.tendencia_correctivos_reciente ? item.tendencia_correctivos_reciente.toFixed(1) : 'N/A'}</td>
                                <td>
                                    <i class="fas ${riskIcon} text-${riskClass}"></i>
                                    <small class="text-${riskClass}">${tipoRiesgo}</small>
                                </td>
                            </tr>
                        `;
                    }).join('')}
                </tbody>
            </table>
        `;
        
        container.innerHTML = tableHTML;
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
    // Iniciar monitoreo de estado ML si existe el div correspondiente
    monitorMLStatus();
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

    // Validación básica del formulario de upload (sin interferir con el manejo principal)
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm && !uploadForm.hasAttribute('data-enhanced')) {
        uploadForm.setAttribute('data-enhanced', 'true');
        uploadForm.addEventListener('submit', function(e) {
            const fileInput = document.getElementById('file');
            
            if (fileInput && fileInput.files.length === 0) {
                e.preventDefault();
                Utils.showNotification('Por favor selecciona un archivo', 'warning');
                return;
            }

            if (fileInput && fileInput.files.length > 0) {
                const validation = Utils.validateExcelFile(fileInput.files[0]);
                if (!validation.valid) {
                    e.preventDefault();
                    Utils.showNotification(validation.error, 'error');
                    return;
                }
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

    // Función global para análisis de frecuencia (llamada desde el HTML)
    window.analyzeFrequency = function() {
        FrequencyAnalysisManager.analyze();
    };

    // Función global para entrenar ML real (llamada desde el HTML)
    window.trainRealML = function() {
        RealMLManager.trainModels();
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
window.FrequencyAnalysisManager = FrequencyAnalysisManager;
window.RealMLManager = RealMLManager;
