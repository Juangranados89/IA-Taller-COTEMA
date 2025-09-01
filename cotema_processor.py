"""
COTEMA Data Processor - Clean Version
Procesador de datos especializado sin dependencias problemáticas
"""

def process_cotema_data(df_raw):
    """
    Procesador COTEMA simplificado y robusto
    """
    import pandas as pd
    from datetime import datetime
    import unicodedata
    
    # Configurar timezone
    today = datetime.now().date()
    
    # Resultados
    dataset = []
    quality_report = {
        'total_registros': len(df_raw),
        'registros_abiertos': 0,
        'registros_cerrados': 0,
        'errores': {},
        'nulos_por_columna': {}
    }
    catalogos = {
        'flota': set(),
        'tipo_atencion': set(),
        'sistema_afectado': set(),
        'operador': set(),
        'ejecutor': set()
    }
    
    def safe_clean_text(text):
        """Limpiar texto de forma segura"""
        if text is None:
            return None
        if isinstance(text, float) and pd.isna(text):
            return None
        text = str(text).strip()
        if text.lower() in ['nan', 'none', '', 'null']:
            return None
        # Limpiar espacios especiales
        text = text.replace('\u00a0', ' ')
        text = ' '.join(text.split())
        return text if text else None
    
    def safe_parse_date(date_val):
        """Parsear fechas de forma segura"""
        if date_val is None:
            return None
        if isinstance(date_val, float) and pd.isna(date_val):
            return None
        if hasattr(date_val, '__str__') and str(date_val).lower() in ['nan', 'nat', 'none']:
            return None
            
        if hasattr(date_val, 'date'):  # datetime object
            return date_val.date()
        
        if isinstance(date_val, str):
            # Intentar múltiples formatos
            formats = ['%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d']
            for fmt in formats:
                try:
                    return datetime.strptime(date_val.strip(), fmt).date()
                except:
                    continue
        return None
    
    def safe_parse_numeric(value):
        """Convertir a numérico de forma segura"""
        if value is None:
            return None
        if isinstance(value, (int, float)) and not pd.isna(value):
            return max(0, int(value))
        
        try:
            if isinstance(value, str):
                value = value.replace(',', '.')
            num_val = float(value)
            if pd.isna(num_val):
                return None
            return max(0, int(num_val))
        except:
            return None
    
    # Mapeo de columnas
    column_mapping = {
        'codigo_limpio': ['CODIGO', 'Codigo', 'codigo'],
        'placa_limpia': ['PLACA', 'Placa', 'placa'],
        'descripcion_limpia': ['DESCRIPCION', 'Descripcion', 'descripcion'],
        'flota_normalizada': ['FLOTA', 'Flota', 'flota'],
        'fecha_in': ['FECHA IN', 'Fecha IN', 'fecha_in', 'FECHA_IN'],
        'fecha_out': ['FECHA OUT', 'Fecha OUT', 'fecha_out', 'FECHA_OUT'],
        'tipo_atencion_normalizado': ['TIPO ATENCION', 'Tipo Atencion', 'tipo_atencion'],
        'sistema_afectado_normalizado': ['SISTEMA AFECTADO', 'Sistema Afectado', 'sistema_afectado'],
        'dias_taller': ['Con.In.Taller', 'Con In Taller', 'dias_taller']
    }
    
    # Procesar cada fila
    for idx, row in df_raw.iterrows():
        try:
            record = {}
            errors = []
            
            # Procesar cada campo
            for new_col, possible_cols in column_mapping.items():
                raw_value = None
                for col in possible_cols:
                    if col in row.index and row[col] is not None:
                        raw_value = row[col]
                        break
                
                # Procesar según el tipo de campo
                if 'fecha' in new_col:
                    fecha = safe_parse_date(raw_value)
                    record[new_col] = fecha.isoformat() if fecha else None
                    
                    # Validación de fecha futura (solo para fecha_in)
                    if new_col == 'fecha_in' and fecha:
                        try:
                            if fecha > today:
                                errors.append('ERROR_FECHA_FUTURA')
                        except:
                            # Ignore comparison errors
                            pass
                            
                elif new_col == 'dias_taller':
                    record[new_col] = safe_parse_numeric(raw_value)
                else:
                    # Campos de texto
                    record[new_col] = safe_clean_text(raw_value)
            
            # Agregar metadatos
            record['errores'] = errors
            record['fila_original'] = idx + 1
            record['procesado_en'] = datetime.now().isoformat()
            
            # Calcular estado
            if record.get('fecha_out'):
                record['estado'] = 'CERRADO'
                quality_report['registros_cerrados'] += 1
            else:
                record['estado'] = 'ABIERTO'
                quality_report['registros_abiertos'] += 1
            
            # Actualizar catálogos
            if record.get('flota_normalizada'):
                catalogos['flota'].add(record['flota_normalizada'])
            if record.get('tipo_atencion_normalizado'):
                catalogos['tipo_atencion'].add(record['tipo_atencion_normalizado'])
            if record.get('sistema_afectado_normalizado'):
                catalogos['sistema_afectado'].add(record['sistema_afectado_normalizado'])
            
            dataset.append(record)
            
        except Exception as e:
            print(f"Error procesando fila {idx}: {e}")
            continue
    
    # Convertir sets a listas para JSON
    for key in catalogos:
        catalogos[key] = sorted(list(catalogos[key]))
    
    # Actualizar estadísticas de errores
    for record in dataset:
        for error in record.get('errores', []):
            quality_report['errores'][error] = quality_report['errores'].get(error, 0) + 1
    
    return dataset, quality_report, catalogos
