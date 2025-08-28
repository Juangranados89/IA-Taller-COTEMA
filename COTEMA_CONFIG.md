# CONFIGURACIÓN ESPECÍFICA PARA COTEMA
# Basada en análisis real de datos

## MAPEO DE COLUMNAS CONFIRMADO:
```python
COLUMN_MAPPING = {
    'codigo_equipo': 'CODIGO',           # CG-TC06, AH-ED03, etc.
    'fecha_ingreso': 'FECHA IN',         # Fecha de ingreso al taller
    'fecha_salida': 'FECHA OUT',         # Fecha de salida del taller
    'sistema_afectado': 'SISTEMA AFECTADO', # HIDRÁULICO, MOTOR, etc.
    'tipo_atencion': 'TIPO ATENCION',    # CORRECTIVA, PREVENTIVA
    'mttr_horas': 'MTTR',                # Tiempo de reparación
    'dias_averia': 'Cont.Dias.Ave',     # Días desde última avería
    'horas_averia': 'Con.Hrs.Ave'       # Horas desde última avería
}
```

## EQUIPOS IDENTIFICADOS (Top 10):
- VD-CO30: Vibrocompactador con 45 ingresos
- VD-CO13: Vibrocompactador con 36 ingresos  
- VD-CO38: Vibrocompactador con 36 ingresos
- VD-CO22: Vibrocompactador con 35 ingresos
- VD-CO23: Vibrocompactador con 35 ingresos
- VD-CO37: Vibrocompactador con 33 ingresos
- VD-CO11: Vibrocompactador con 33 ingresos
- VD-CO36: Vibrocompactador con 33 ingresos
- VD-CO34: Vibrocompactador con 33 ingresos
- VD-TC48: Vibrocompactador con 33 ingresos

## SISTEMAS MÁS AFECTADOS:
1. CORRECTIVA: 89.5% de los casos
2. PREVENTIVA: 2.7% de los casos
3. Sistemas principales: HIDRÁULICO, MOTOR, VIBRATORIO

## CONFIGURACIÓN DE ALERTAS:
- FR-30 > 40% = 🔴 ALTO RIESGO
- RUL < 20 días = ⚠️ MANTENIMIENTO URGENTE  
- Anomaly Score > 0.5 = 🚨 COMPORTAMIENTO ANÓMALO

## PRÓXIMOS PASOS RECOMENDADOS:

### 1. DESPLIEGUE INMEDIATO:
```bash
# Subir a GitHub
git add .
git commit -m "COTEMA Analytics - Sistema optimizado para datos reales"
git push origin main

# Desplegar en Render
# - Conectar repositorio GitHub
# - Usar configuración automática (render.yaml)
# - Variables de entorno se configuran automáticamente
```

### 2. CONFIGURACIÓN OPERATIVA:
- Subir archivo Excel mensualmente
- Revisar KPIs cada semana
- Configurar alertas automáticas
- Integrar con Power BI/Tableau

### 3. OPTIMIZACIONES CONTINUAS:
- Calibrar thresholds según experiencia operativa
- Agregar nuevos equipos según crecimiento
- Expandir sistemas de falla catalogados
- Mejorar precisión con más datos históricos

## VALOR ECONÓMICO ESTIMADO:
- Reducción 20-30% en paradas no programadas
- Optimización 15-25% en inventario de repuestos  
- Mejora 10-20% en eficiencia operativa
- ROI estimado: 300-500% en primer año
