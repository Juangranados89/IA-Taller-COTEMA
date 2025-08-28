# 📊 COTEMA Analytics - KPI API Documentation

Esta documentación describe las APIs REST disponibles para integración con herramientas de Business Intelligence como Power BI, Tableau, Qlik Sense, etc.

## 🔗 Base URL
```
https://tu-dominio-render.com/api
```

## 🔑 Autenticación
Las APIs actuales no requieren autenticación, pero se recomienda implementar API keys para producción.

## 📋 Endpoints Disponibles

### 1. Test de Conexión
**GET** `/api/connection-test`

Verifica que el servicio esté operativo.

**Respuesta:**
```json
{
  "status": "ok",
  "message": "COTEMA Analytics API operativa",
  "timestamp": "2024-12-19T10:30:00Z",
  "version": "1.0.0"
}
```

### 2. Obtener KPIs por Mes
**GET** `/kpis/{mes}`

Calcula todos los KPIs para un mes específico.

**Parámetros:**
- `mes` (string): Mes en formato YYYY-MM (ej: "2024-01")

**Respuesta:**
```json
{
  "mes": "2024-01",
  "timestamp": "2024-12-19T10:30:00Z",
  "total_equipos": 15,
  "kpis": {
    "fr30": {
      "EQUIPO_001": {
        "risk_30d": 0.35,
        "banda": "🟠 MEDIO",
        "explicacion": "Riesgo medio basado en patrones históricos..."
      }
    },
    "rul": {
      "EQUIPO_001": {
        "rul50_d": 45,
        "rul90_d": 25,
        "maintenance_window": "2024-01-15 a 2024-01-25",
        "explicacion": "Mantenimiento recomendado en 25-45 días..."
      }
    },
    "forecast": {
      "EQUIPO_001": {
        "forecast_7d_h": 52.3,
        "forecast_30d_h": 210.8,
        "trend": "creciente",
        "explicacion": "Se proyecta un incremento del 15% en el uso..."
      }
    },
    "anomaly": {
      "EQUIPO_001": {
        "anomaly_score": 0.23,
        "banda": "🟢 NORMAL",
        "anomalies_detected": 2,
        "explicacion": "Comportamiento dentro de rangos normales..."
      }
    }
  }
}
```

### 3. Detalles FR-30 por Equipo
**GET** `/api/fr30/{equipo}`

Obtiene análisis detallado de FR-30 para un equipo específico.

**Parámetros:**
- `equipo` (string): ID del equipo

**Respuesta:**
```json
{
  "equipo": "EQUIPO_001",
  "timestamp": "2024-12-19T10:30:00Z",
  "fr30": {
    "risk_30d": 0.35,
    "banda": "🟠 MEDIO",
    "features_importance": {
      "tbf_mean": 0.28,
      "tbf_std": 0.22,
      "failure_rate": 0.19,
      "usage_intensity": 0.15,
      "maintenance_delay": 0.16
    },
    "trend_7d": "creciente",
    "trend_30d": "estable",
    "historical_risks": [
      {"fecha": "2024-01-01", "risk": 0.31},
      {"fecha": "2024-01-08", "risk": 0.33},
      {"fecha": "2024-01-15", "risk": 0.35}
    ],
    "explicacion": "El riesgo ha aumentado un 12% en los últimos 7 días debido a...",
    "recomendaciones": [
      "Programar inspección en próximos 7 días",
      "Revisar logs de mantenimiento preventivo",
      "Monitorear tendencia durante próxima semana"
    ]
  }
}
```

### 4. Detalles RUL por Equipo
**GET** `/api/rul/{equipo}`

Obtiene análisis detallado de RUL para un equipo específico.

**Parámetros:**
- `equipo` (string): ID del equipo

**Respuesta:**
```json
{
  "equipo": "EQUIPO_001",
  "timestamp": "2024-12-19T10:30:00Z",
  "rul": {
    "rul50_d": 45,
    "rul90_d": 25,
    "rul10_d": 65,
    "confidence_interval": [25, 65],
    "maintenance_window": "2024-01-15 a 2024-01-25",
    "weibull_params": {
      "shape": 2.3,
      "scale": 85.2,
      "fit_quality": 0.89
    },
    "survival_probability": [
      {"dias": 10, "probabilidad": 0.95},
      {"dias": 30, "probabilidad": 0.75},
      {"dias": 60, "probabilidad": 0.45}
    ],
    "explicacion": "Basado en análisis de supervivencia Weibull, el equipo tiene...",
    "recomendaciones": [
      "Planificar mantenimiento entre día 25-45",
      "Preparar repuestos críticos",
      "Considerar equipo de respaldo"
    ]
  }
}
```

### 5. Pronóstico por Equipo
**GET** `/api/forecast/{equipo}`

Obtiene pronósticos de uso para un equipo específico.

**Parámetros:**
- `equipo` (string): ID del equipo

**Respuesta:**
```json
{
  "equipo": "EQUIPO_001",
  "timestamp": "2024-12-19T10:30:00Z",
  "forecast": {
    "forecast_7d_h": 52.3,
    "forecast_30d_h": 210.8,
    "forecast_90d_h": 650.2,
    "confidence_intervals": {
      "7d": [48.1, 56.8],
      "30d": [195.2, 228.4],
      "90d": [580.1, 720.3]
    },
    "seasonal_patterns": {
      "weekly": "Picos lunes-miércoles",
      "monthly": "Mayor uso primera quincena",
      "yearly": "Incremento en Q1 y Q3"
    },
    "trend": "creciente",
    "trend_strength": 0.15,
    "seasonality_strength": 0.23,
    "daily_forecast": [
      {"fecha": "2024-01-20", "horas": 7.2, "intervalo": [6.5, 7.9]},
      {"fecha": "2024-01-21", "horas": 8.1, "intervalo": [7.3, 8.8]}
    ],
    "explicacion": "El modelo Prophet identifica una tendencia creciente...",
    "recomendaciones": [
      "Considerar capacidad adicional para Q1",
      "Optimizar programación para días de alta demanda",
      "Revisar mantenimiento preventivo por mayor uso"
    ]
  }
}
```

### 6. Detección de Anomalías por Equipo
**GET** `/api/anomaly/{equipo}`

Obtiene análisis de anomalías para un equipo específico.

**Parámetros:**
- `equipo` (string): ID del equipo

**Respuesta:**
```json
{
  "equipo": "EQUIPO_001",
  "timestamp": "2024-12-19T10:30:00Z",
  "anomaly": {
    "anomaly_score": 0.23,
    "banda": "🟢 NORMAL",
    "anomalies_detected": 2,
    "anomaly_threshold": 0.5,
    "recent_anomalies": [
      {
        "fecha": "2024-01-15",
        "score": 0.67,
        "tipo": "uso_inusual",
        "descripcion": "Uso 40% superior al patrón normal"
      }
    ],
    "pattern_analysis": {
      "uso_promedio": 7.2,
      "desviacion_standard": 1.8,
      "coeficiente_variacion": 0.25,
      "patrones_detectados": ["picos_matutinos", "caidas_tarde"]
    },
    "feature_contributions": {
      "usage_pattern": 0.15,
      "maintenance_timing": 0.08,
      "failure_frequency": 0.12,
      "operational_context": 0.06
    },
    "explicacion": "El equipo muestra comportamiento normal con variabilidad...",
    "recomendaciones": [
      "Continuar monitoreo rutinario",
      "Establecer alertas para score > 0.5",
      "Revisar patrones de uso matutinos"
    ]
  }
}
```

### 7. Exportar Datos
**GET** `/api/export/{formato}`

Exporta datos en diferentes formatos para integración BI.

**Parámetros:**
- `formato` (string): Formato de exportación ("csv", "json", "excel", "powerbi")

**Para formato CSV:**
```
GET /api/export/csv
Content-Type: text/csv

equipo,mes,fr30_risk,fr30_banda,rul50_d,rul90_d,forecast_7d_h,forecast_30d_h,anomaly_score,anomaly_banda
EQUIPO_001,2024-01,0.35,"🟠 MEDIO",45,25,52.3,210.8,0.23,"🟢 NORMAL"
```

**Para formato JSON:**
```json
{
  "export_timestamp": "2024-12-19T10:30:00Z",
  "data_format": "powerbi",
  "data": [
    {
      "equipo": "EQUIPO_001",
      "mes": "2024-01",
      "fr30_risk": 0.35,
      "fr30_banda": "MEDIO",
      "fr30_color": "#FFA500",
      "rul50_d": 45,
      "rul90_d": 25,
      "forecast_7d_h": 52.3,
      "forecast_30d_h": 210.8,
      "anomaly_score": 0.23,
      "anomaly_banda": "NORMAL",
      "anomaly_color": "#28A745"
    }
  ],
  "metadata": {
    "total_records": 1,
    "columns": 11,
    "data_quality": "high",
    "last_update": "2024-12-19T10:00:00Z"
  }
}
```

### 8. Métricas Agregadas
**GET** `/api/metrics/summary`

Obtiene métricas agregadas para dashboards ejecutivos.

**Respuesta:**
```json
{
  "timestamp": "2024-12-19T10:30:00Z",
  "periodo": "2024-01",
  "summary": {
    "total_equipos": 15,
    "equipos_alto_riesgo": 3,
    "equipos_mantenimiento_urgente": 2,
    "equipos_con_anomalias": 1,
    "promedio_fr30": 0.28,
    "promedio_rul50": 52.3,
    "total_horas_pronosticadas_30d": 3162.5,
    "anomalias_detectadas_mes": 8
  },
  "distribucion_bandas": {
    "fr30": {
      "🔴 ALTO": 3,
      "🟠 MEDIO": 5,
      "🟢 BAJO": 7
    },
    "anomaly": {
      "🔴 CRÍTICO": 1,
      "🟠 ATENCIÓN": 2,
      "🟢 NORMAL": 12
    }
  },
  "tendencias": {
    "fr30_trend": "estable",
    "rul_trend": "mejorando",
    "usage_trend": "creciente",
    "anomaly_trend": "estable"
  }
}
```

## 🔧 Integración con Power BI

### Configuración de Conexión
1. **Obtener datos desde Web:**
   ```
   Datos > Obtener datos > Desde Web
   URL: https://tu-dominio-render.com/api/export/powerbi
   ```

2. **Configurar actualización automática:**
   - Programar actualización cada 4 horas
   - Configurar credenciales si es necesario

3. **Transformaciones recomendadas:**
   ```powerquery
   // Convertir columnas de banda a factores
   = Table.TransformColumnTypes(Source,{
       {"fr30_risk", type number},
       {"rul50_d", Int64.Type},
       {"anomaly_score", type number}
   })
   ```

### Medidas DAX Sugeridas
```dax
// Porcentaje de equipos en alto riesgo
EquiposAltoRiesgo% = 
DIVIDE(
    COUNTROWS(FILTER('KPIs', 'KPIs'[fr30_banda] = "ALTO")),
    COUNTROWS('KPIs'),
    0
) * 100

// Días promedio hasta mantenimiento
RULPromedio = AVERAGE('KPIs'[rul50_d])

// Alerta temprana
AlertaMantenimiento = 
IF('KPIs'[rul50_d] <= 14, "🔴 URGENTE",
   IF('KPIs'[rul50_d] <= 30, "🟠 PRÓXIMO", "🟢 OK"))
```

## 🔧 Integración con Tableau

### Configuración de Conexión
1. **Conectar a Servidor Web:**
   ```
   Datos > Nueva fuente de datos > Servidor Web
   URL: https://tu-dominio-render.com/api/export/json
   ```

2. **Configuración de Extracción:**
   - Extraer datos cada 4 horas
   - Usar incrementales si el volumen es alto

3. **Campos calculados sugeridos:**
   ```sql
   // Semáforo FR-30
   IF [fr30_risk] >= 0.5 THEN "Alto"
   ELSEIF [fr30_risk] >= 0.3 THEN "Medio"
   ELSE "Bajo"
   END

   // Días hasta mantenimiento crítico
   IF [rul50_d] <= 14 THEN [rul50_d]
   ELSE NULL
   END
   ```

## 🔧 Integración con Qlik Sense

### Script de Carga
```qlik
// Cargar datos desde API
KPIs:
LOAD
    equipo,
    mes,
    fr30_risk,
    fr30_banda,
    rul50_d,
    rul90_d,
    forecast_7d_h,
    forecast_30d_h,
    anomaly_score,
    anomaly_banda
FROM [https://tu-dominio-render.com/api/export/csv]
(txt, codepage is 28591, embedded labels, delimiter is ',', msq);

// Tabla de dimensiones
Bandas:
LOAD * INLINE [
    banda, color, prioridad
    "🔴 ALTO", "#DC3545", 3
    "🟠 MEDIO", "#FFC107", 2
    "🟢 BAJO", "#28A745", 1
];
```

## 📊 Ejemplos de Uso

### Dashboard Ejecutivo
```javascript
// Obtener métricas para dashboard ejecutivo
fetch('/api/metrics/summary')
  .then(response => response.json())
  .then(data => {
    updateExecutiveDashboard(data);
  });
```

### Monitoreo en Tiempo Real
```javascript
// Actualizar KPIs cada 15 minutos
setInterval(() => {
  const currentMonth = new Date().toISOString().slice(0, 7);
  fetch(`/kpis/${currentMonth}`)
    .then(response => response.json())
    .then(data => {
      updateRealTimeMonitoring(data);
    });
}, 15 * 60 * 1000);
```

### Alertas Automatizadas
```python
import requests
import json

# Verificar equipos en alto riesgo
response = requests.get('https://tu-dominio-render.com/api/metrics/summary')
data = response.json()

if data['summary']['equipos_alto_riesgo'] > 0:
    # Enviar alerta
    send_alert(f"⚠️ {data['summary']['equipos_alto_riesgo']} equipos requieren atención")
```

## 🔍 Códigos de Estado

| Código | Descripción |
|--------|-------------|
| 200 | Éxito |
| 400 | Solicitud inválida |
| 404 | Recurso no encontrado |
| 500 | Error interno del servidor |

## 📝 Notas de Implementación

1. **Rate Limiting:** Se recomienda no más de 60 requests por minuto
2. **Cacheo:** Los datos se cachean por 30 minutos
3. **Formatos de Fecha:** Usar ISO 8601 (YYYY-MM-DD)
4. **Encoding:** UTF-8 para todos los endpoints
5. **Compresión:** Respuestas comprimidas con gzip

## 🆘 Soporte

Para soporte técnico o consultas sobre la API:
- 📧 Email: soporte@cotema.com
- 📚 Documentación: /api/docs
- 🐛 Issues: GitHub repository
