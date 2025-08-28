"""
Módulo de visualizaciones para el dashboard
Genera gráficos interactivos con Plotly
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def create_dashboard_plots(df):
    """
    Crea todos los gráficos para el dashboard principal
    """
    plots = {}
    
    try:
        # 1. Distribución temporal de ingresos
        plots['ingresos_temporales'] = create_temporal_distribution(df)
        
        # 2. Top equipos por frecuencia
        plots['top_equipos'] = create_top_equipment_chart(df)
        
        # 3. Distribución por sistemas afectados
        plots['sistemas_afectados'] = create_systems_distribution(df)
        
        # 4. Análisis de TBF
        plots['tbf_analysis'] = create_tbf_analysis(df)
        
        # 5. Heatmap de actividad
        plots['heatmap_actividad'] = create_activity_heatmap(df)
        
        # 6. Métricas MTTR
        plots['mttr_metrics'] = create_mttr_analysis(df)
        
    except Exception as e:
        print(f"Error creando gráficos: {str(e)}")
        plots['error'] = f"Error al generar visualizaciones: {str(e)}"
    
    return plots

def create_temporal_distribution(df):
    """
    Gráfico de distribución temporal de ingresos
    """
    try:
        # Agrupar por mes
        monthly_data = df.groupby(df['FECHA_IN'].dt.to_period('M')).size().reset_index()
        monthly_data['FECHA_IN'] = monthly_data['FECHA_IN'].astype(str)
        monthly_data.columns = ['Mes', 'Ingresos']
        
        # Crear gráfico
        fig = px.line(
            monthly_data, 
            x='Mes', 
            y='Ingresos',
            title='Evolución Temporal de Ingresos al Taller',
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Período",
            yaxis_title="Número de Ingresos",
            hovermode='x unified'
        )
        
        # Agregar línea de tendencia
        if len(monthly_data) > 3:
            fig.add_scatter(
                x=monthly_data['Mes'],
                y=monthly_data['Ingresos'].rolling(3).mean(),
                mode='lines',
                name='Tendencia (3 meses)',
                line=dict(dash='dash', color='red')
            )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en distribución temporal: {str(e)}")
        return create_error_plot("Error en distribución temporal")

def create_top_equipment_chart(df):
    """
    Gráfico de top equipos por frecuencia de ingresos
    """
    try:
        # Top 15 equipos
        top_equipos = df['CODIGO'].value_counts().head(15)
        
        fig = px.bar(
            x=top_equipos.values,
            y=top_equipos.index,
            orientation='h',
            title='Top 15 Equipos por Frecuencia de Ingresos',
            labels={'x': 'Número de Ingresos', 'y': 'Código de Equipo'}
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=500
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en top equipos: {str(e)}")
        return create_error_plot("Error en análisis de equipos")

def create_systems_distribution(df):
    """
    Distribución por sistemas afectados
    """
    try:
        if 'SISTEMA_AFECTADO' not in df.columns:
            return create_error_plot("No hay datos de sistemas afectados")
        
        sistemas = df['SISTEMA_AFECTADO'].value_counts().head(10)
        
        fig = px.pie(
            values=sistemas.values,
            names=sistemas.index,
            title='Distribución por Sistemas Afectados (Top 10)'
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en sistemas afectados: {str(e)}")
        return create_error_plot("Error en análisis de sistemas")

def create_tbf_analysis(df):
    """
    Análisis de Time Between Failures
    """
    try:
        # Filtrar valores válidos de TBF
        tbf_data = df[df['tbf_dias'].notna() & (df['tbf_dias'] > 0) & (df['tbf_dias'] < 365)]
        
        if len(tbf_data) == 0:
            return create_error_plot("No hay datos válidos de TBF")
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución TBF', 'TBF por Equipo (Top 10)', 
                          'Tendencia TBF Promedio', 'Box Plot TBF por Clase'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # 1. Histograma de TBF
        fig.add_trace(
            go.Histogram(x=tbf_data['tbf_dias'], nbinsx=30, name='TBF Distribution'),
            row=1, col=1
        )
        
        # 2. TBF promedio por equipo (top 10)
        tbf_por_equipo = tbf_data.groupby('CODIGO')['tbf_dias'].mean().sort_values(ascending=False).head(10)
        fig.add_trace(
            go.Bar(x=tbf_por_equipo.index, y=tbf_por_equipo.values, name='TBF Promedio'),
            row=1, col=2
        )
        
        # 3. Tendencia temporal de TBF
        tbf_monthly = tbf_data.groupby(tbf_data['FECHA_IN'].dt.to_period('M'))['tbf_dias'].mean()
        fig.add_trace(
            go.Scatter(
                x=[str(x) for x in tbf_monthly.index], 
                y=tbf_monthly.values, 
                mode='lines+markers',
                name='TBF Mensual'
            ),
            row=2, col=1
        )
        
        # 4. Box plot por clase
        if 'CLASE' in df.columns:
            for clase in tbf_data['CLASE'].unique()[:5]:  # Top 5 clases
                clase_data = tbf_data[tbf_data['CLASE'] == clase]['tbf_dias']
                fig.add_trace(
                    go.Box(y=clase_data, name=clase),
                    row=2, col=2
                )
        
        fig.update_layout(
            height=600,
            title_text="Análisis Completo de Time Between Failures (TBF)",
            showlegend=False
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en análisis TBF: {str(e)}")
        return create_error_plot("Error en análisis TBF")

def create_activity_heatmap(df):
    """
    Heatmap de actividad por día de la semana y hora
    """
    try:
        # Crear datos para el heatmap
        df['dia_semana'] = df['FECHA_IN'].dt.day_name()
        df['hora'] = df['FECHA_IN'].dt.hour
        
        # Crear matriz de actividad
        activity_matrix = df.groupby(['dia_semana', 'hora']).size().unstack(fill_value=0)
        
        # Reordenar días de la semana
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        activity_matrix = activity_matrix.reindex(day_order, fill_value=0)
        
        fig = px.imshow(
            activity_matrix,
            labels=dict(x="Hora del Día", y="Día de la Semana", color="Ingresos"),
            title="Heatmap de Actividad: Ingresos por Día y Hora",
            color_continuous_scale="Viridis"
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en heatmap: {str(e)}")
        return create_error_plot("Error en heatmap de actividad")

def create_mttr_analysis(df):
    """
    Análisis de Mean Time To Repair
    """
    try:
        # Filtrar datos válidos de MTTR
        mttr_data = df[df['ciclo_horas'].notna() & (df['ciclo_horas'] > 0) & (df['ciclo_horas'] < 200)]
        
        if len(mttr_data) == 0:
            return create_error_plot("No hay datos válidos de MTTR")
        
        # Crear subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Distribución MTTR (horas)', 'MTTR por Sistema', 
                          'Evolución MTTR Mensual', 'MTTR vs TBF'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. Distribución de MTTR
        fig.add_trace(
            go.Histogram(x=mttr_data['ciclo_horas'], nbinsx=25, name='MTTR Distribution'),
            row=1, col=1
        )
        
        # 2. MTTR promedio por sistema
        if 'SISTEMA_AFECTADO' in mttr_data.columns:
            mttr_por_sistema = mttr_data.groupby('SISTEMA_AFECTADO')['ciclo_horas'].mean().sort_values(ascending=False).head(10)
            fig.add_trace(
                go.Bar(x=mttr_por_sistema.index, y=mttr_por_sistema.values, name='MTTR por Sistema'),
                row=1, col=2
            )
        
        # 3. Evolución temporal de MTTR
        mttr_monthly = mttr_data.groupby(mttr_data['FECHA_IN'].dt.to_period('M'))['ciclo_horas'].mean()
        fig.add_trace(
            go.Scatter(
                x=[str(x) for x in mttr_monthly.index], 
                y=mttr_monthly.values, 
                mode='lines+markers',
                name='MTTR Mensual'
            ),
            row=2, col=1
        )
        
        # 4. MTTR vs TBF scatter
        scatter_data = mttr_data[mttr_data['tbf_dias'].notna()]
        if len(scatter_data) > 0:
            fig.add_trace(
                go.Scatter(
                    x=scatter_data['tbf_dias'], 
                    y=scatter_data['ciclo_horas'],
                    mode='markers',
                    name='MTTR vs TBF',
                    text=scatter_data['CODIGO'],
                    hovertemplate='<b>%{text}</b><br>TBF: %{x} días<br>MTTR: %{y} horas<extra></extra>'
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            height=600,
            title_text="Análisis de Mean Time To Repair (MTTR)",
            showlegend=False
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en análisis MTTR: {str(e)}")
        return create_error_plot("Error en análisis MTTR")

def create_kpi_dashboard(kpis_data, mes):
    """
    Crea dashboard específico de KPIs para un mes
    """
    try:
        plots = {}
        
        # 1. Gráfico de riesgo FR-30
        plots['fr30_risk'] = create_fr30_chart(kpis_data.get('fr30', {}))
        
        # 2. Gráfico de RUL
        plots['rul_chart'] = create_rul_chart(kpis_data.get('rul', {}))
        
        # 3. Pronóstico de uso
        plots['forecast_chart'] = create_forecast_chart(kpis_data.get('forecast', {}))
        
        # 4. Scatter de anomalías
        plots['anomaly_scatter'] = create_anomaly_scatter(kpis_data)
        
        # 5. Resumen por equipos
        plots['equipment_summary'] = create_equipment_summary_table(kpis_data)
        
        return plots
        
    except Exception as e:
        print(f"Error creando dashboard KPIs: {str(e)}")
        return {'error': f"Error al generar dashboard de KPIs: {str(e)}"}

def create_fr30_chart(fr30_data):
    """
    Gráfico de riesgo FR-30
    """
    try:
        if not fr30_data:
            return create_error_plot("No hay datos de FR-30")
        
        equipos = list(fr30_data.keys())
        risks = [fr30_data[eq].get('risk_30d', 0) for eq in equipos]
        bandas = [fr30_data[eq].get('banda', 'SIN DATO') for eq in equipos]
        
        # Colores por banda
        colors = []
        for banda in bandas:
            if '🔴' in banda:
                colors.append('red')
            elif '🟠' in banda:
                colors.append('orange')
            elif '🟢' in banda:
                colors.append('green')
            else:
                colors.append('gray')
        
        fig = go.Figure(data=go.Bar(
            x=equipos,
            y=risks,
            marker_color=colors,
            text=[f"{r:.1%}" for r in risks],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Riesgo: %{y:.1%}<br>Banda: %{text}<extra></extra>'
        ))
        
        fig.update_layout(
            title="FR-30: Riesgo de Falla en 30 Días por Equipo",
            xaxis_title="Equipos",
            yaxis_title="Probabilidad de Falla",
            yaxis=dict(tickformat='.0%'),
            height=400
        )
        
        # Líneas de referencia
        fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Umbral Alto (50%)")
        fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Umbral Medio (30%)")
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en gráfico FR-30: {str(e)}")
        return create_error_plot("Error en gráfico FR-30")

def create_rul_chart(rul_data):
    """
    Gráfico de Remaining Useful Life
    """
    try:
        if not rul_data:
            return create_error_plot("No hay datos de RUL")
        
        equipos = list(rul_data.keys())
        rul50 = [rul_data[eq].get('rul50_d', 0) for eq in equipos]
        rul90 = [rul_data[eq].get('rul90_d', 0) for eq in equipos]
        
        fig = go.Figure()
        
        # RUL-50 (mediana)
        fig.add_trace(go.Bar(
            name='RUL-50 (Mediana)',
            x=equipos,
            y=rul50,
            marker_color='lightblue',
            text=[f"{r:.0f}d" for r in rul50],
            textposition='auto'
        ))
        
        # RUL-90 (conservador)
        fig.add_trace(go.Bar(
            name='RUL-90 (Conservador)',
            x=equipos,
            y=rul90,
            marker_color='darkblue',
            text=[f"{r:.0f}d" for r in rul90],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="RUL: Vida Útil Restante por Equipo",
            xaxis_title="Equipos",
            yaxis_title="Días Restantes",
            barmode='group',
            height=400
        )
        
        # Línea de alerta (7 días)
        fig.add_hline(y=7, line_dash="dash", line_color="red", annotation_text="Alerta (7 días)")
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en gráfico RUL: {str(e)}")
        return create_error_plot("Error en gráfico RUL")

def create_forecast_chart(forecast_data):
    """
    Gráfico de pronóstico de uso
    """
    try:
        if not forecast_data:
            return create_error_plot("No hay datos de pronóstico")
        
        equipos = list(forecast_data.keys())
        forecast_7d = [forecast_data[eq].get('forecast_7d_h', 0) for eq in equipos]
        forecast_30d = [forecast_data[eq].get('forecast_30d_h', 0) for eq in equipos]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Pronóstico 7 días',
            x=equipos,
            y=forecast_7d,
            marker_color='lightgreen'
        ))
        
        fig.add_trace(go.Bar(
            name='Pronóstico 30 días',
            x=equipos,
            y=forecast_30d,
            marker_color='darkgreen'
        ))
        
        fig.update_layout(
            title="Pronóstico de Uso: Horas Proyectadas",
            xaxis_title="Equipos",
            yaxis_title="Horas de Operación",
            barmode='group',
            height=400
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en gráfico pronóstico: {str(e)}")
        return create_error_plot("Error en gráfico pronóstico")

def create_anomaly_scatter(kpis_data):
    """
    Scatter plot de anomalías vs riesgo
    """
    try:
        fr30_data = kpis_data.get('fr30', {})
        rul_data = kpis_data.get('rul', {})
        anomaly_data = kpis_data.get('anomaly', {})
        
        if not all([fr30_data, rul_data, anomaly_data]):
            return create_error_plot("Datos insuficientes para scatter de anomalías")
        
        equipos = list(fr30_data.keys())
        x_values = [rul_data.get(eq, {}).get('rul50_d', 0) for eq in equipos]
        y_values = [fr30_data.get(eq, {}).get('risk_30d', 0) for eq in equipos]
        colors = [anomaly_data.get(eq, {}).get('anomaly_score', 0) for eq in equipos]
        
        fig = go.Figure(data=go.Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            marker=dict(
                size=12,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="Score Anomalía"),
                showscale=True
            ),
            text=equipos,
            hovertemplate='<b>%{text}</b><br>RUL-50: %{x} días<br>FR-30: %{y:.1%}<br>Anomalía: %{marker.color:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Análisis Multidimensional: RUL vs Riesgo vs Anomalías",
            xaxis_title="RUL-50 (días)",
            yaxis_title="Riesgo FR-30",
            yaxis=dict(tickformat='.0%'),
            height=500
        )
        
        # Cuadrantes de referencia
        fig.add_vline(x=30, line_dash="dash", line_color="gray", annotation_text="RUL crítico")
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="Riesgo alto")
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en scatter anomalías: {str(e)}")
        return create_error_plot("Error en análisis multidimensional")

def create_equipment_summary_table(kpis_data):
    """
    Tabla resumen por equipos
    """
    try:
        fr30_data = kpis_data.get('fr30', {})
        rul_data = kpis_data.get('rul', {})
        forecast_data = kpis_data.get('forecast', {})
        anomaly_data = kpis_data.get('anomaly', {})
        
        equipos = list(fr30_data.keys())
        
        table_data = []
        for equipo in equipos:
            row = {
                'Equipo': equipo,
                'FR-30': f"{fr30_data.get(equipo, {}).get('risk_30d', 0):.1%}",
                'Banda FR-30': fr30_data.get(equipo, {}).get('banda', 'N/A'),
                'RUL-50': f"{rul_data.get(equipo, {}).get('rul50_d', 0):.0f}d",
                'RUL-90': f"{rul_data.get(equipo, {}).get('rul90_d', 0):.0f}d",
                'Pronóstico 7d': f"{forecast_data.get(equipo, {}).get('forecast_7d_h', 0):.0f}h",
                'Pronóstico 30d': f"{forecast_data.get(equipo, {}).get('forecast_30d_h', 0):.0f}h",
                'Anomalía': f"{anomaly_data.get(equipo, {}).get('anomaly_score', 0):.2f}",
                'Banda Anomalía': anomaly_data.get(equipo, {}).get('banda', 'N/A')
            }
            table_data.append(row)
        
        # Convertir a HTML table
        df_table = pd.DataFrame(table_data)
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df_table.columns),
                       fill_color='paleturquoise',
                       align='left'),
            cells=dict(values=[df_table[col] for col in df_table.columns],
                      fill_color='lavender',
                      align='left'))
        ])
        
        fig.update_layout(
            title="Resumen de KPIs por Equipo",
            height=600
        )
        
        return fig.to_json()
        
    except Exception as e:
        print(f"Error en tabla resumen: {str(e)}")
        return create_error_plot("Error en tabla resumen")

def create_error_plot(error_message):
    """
    Crea un gráfico de error
    """
    fig = go.Figure()
    fig.add_annotation(
        text=error_message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="red")
    )
    fig.update_layout(
        title="Error en Visualización",
        xaxis=dict(showticklabels=False),
        yaxis=dict(showticklabels=False),
        height=300
    )
    
    return fig.to_json()
