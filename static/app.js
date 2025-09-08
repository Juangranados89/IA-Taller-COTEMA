document.addEventListener('DOMContentLoaded', function() {
    const testApiBtn = document.getElementById('test-api-btn');
    const loader = document.getElementById('loader');
    const resultsContainer = document.getElementById('validation-results');

    if (testApiBtn) {
        testApiBtn.addEventListener('click', function() {
            loader.style.display = 'block';
            resultsContainer.style.display = 'none';

            // Consumir el endpoint corregido para BI
            fetch('/api/kpi/fr30')
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
    // Mostrar equipos de riesgo - adaptado al nuevo endpoint
    if (data.critical_equipment && data.critical_equipment.length > 0) {
        createRiskTable(data.critical_equipment.slice(0, 10)); // Top 10
    }

    // Mostrar proyección de carga de trabajo - adaptado al nuevo endpoint
    if (data.monthly_forecast && data.monthly_forecast.length > 0) {
        createWorkloadChart(data.monthly_forecast);
    }
}

function createRiskTable(riskData) {
    const tableBody = document.getElementById('risk-table-body');
    tableBody.innerHTML = '';

    riskData.forEach(equipo => {
        const riskScore = equipo.risk_score_percent || 0;
        const row = `
            <tr>
                <td>${equipo.rank}</td>
                <td><strong>${equipo.equipment_id}</strong></td>
                <td class="text-center">
                    <span class="badge rounded-pill" style="background-color: ${getRiskColor(riskScore)};">
                        ${riskScore}%
                    </span>
                </td>
                <td class="text-center">${equipo.critical_interventions || 0}</td>
                <td class="text-center">${equipo.mttr_hours ? equipo.mttr_hours + 'h' : 'N/A'}</td>
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
            labels: forecastData.map(d => d.period),
            datasets: [{
                label: 'Correctivos Proyectados',
                data: forecastData.map(d => d.projected_corrective_maintenance),
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
