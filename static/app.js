document.addEventListener('DOMContentLoaded', function() {
    const testApiBtn = document.getElementById('test-api-btn');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('validation-results');

    if (testApiBtn) {
        testApiBtn.addEventListener('click', function() {
            loader.style.display = 'block';
            resultsContainer.style.display = 'none';

            // Consumir el endpoint de la API
            fetch('/api/v1/predictive_analysis')
            .then(response => {
                if (!response.ok) {
                    throw new Error(`Error HTTP: ${response.status}`);
                }
                return response.json();
            })
            .then(data => {
                loader.style.display = 'none';
                displayValidationData(data);
                resultsContainer.style.display = 'block';
            })
            .catch(error => {
                loader.style.display = 'none';
                console.error('Error al consumir la API:', error);
                alert('Error al obtener datos de la API: ' + error.message);
            });
        });
    }
});

// Variable para el gráfico
let workloadChart = null;

function displayValidationData(data) {
    // Mostrar equipos de riesgo
    if (data.risk_ranking && data.risk_ranking.length > 0) {
        createRiskTable(data.risk_ranking.slice(0, 10)); // Top 10
    }

    // Mostrar proyección de carga de trabajo
    if (data.workload_forecast && data.workload_forecast.length > 0) {
        createWorkloadChart(data.workload_forecast);
    }
}

function createRiskTable(riskData) {
    const tableBody = document.getElementById('risk-table-body');
    tableBody.innerHTML = '';

    riskData.forEach(equipo => {
        const riskScore = equipo.score_riesgo || 0;
        const row = `
            <tr>
                <td>${equipo.rank}</td>
                <td><strong>${equipo.equipo}</strong></td>
                <td class="text-center">
                    <span class="badge rounded-pill" style="background-color: ${getRiskColor(riskScore)};">
                        ${riskScore}%
                    </span>
                </td>
                <td class="text-center">${equipo.ingresos_criticos || 0}</td>
                <td class="text-center">${equipo.mttr_horas ? equipo.mttr_horas + 'h' : 'N/A'}</td>
            </tr>
        `;
        tableBody.innerHTML += row;
    });
}

function createWorkloadChart(forecastData) {
    const ctx = document.getElementById('workload-chart').getContext('2d');
    
    // Destruir gráfico anterior si existe
    if (workloadChart) {
        workloadChart.destroy();
    }

    workloadChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: forecastData.map(d => d.periodo),
            datasets: [{
                label: 'Correctivos Proyectados',
                data: forecastData.map(d => d.correctivos_proyectados),
                backgroundColor: ['rgba(54, 162, 235, 0.7)', 'rgba(255, 159, 64, 0.7)'],
                borderColor: ['rgba(54, 162, 235, 1)', 'rgba(255, 159, 64, 1)'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    ticks: { color: '#fff' }
                },
                x: { 
                    ticks: { color: '#fff' }
                }
            }
        }
    });
}

function getRiskColor(score) {
    if (score >= 85) return '#dc3545';  // Rojo
    if (score >= 75) return '#ffc107';  // Amarillo
    return '#198754';  // Verde
}
