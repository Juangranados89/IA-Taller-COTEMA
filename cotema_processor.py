def process_cotema_data(df_raw):
    """
    Agente especializado de procesamiento de datos para COTEMA.
    Normaliza y valida datos de mantenimiento de equipos según especificaciones.
    """
    import pandas as pd
    import unicodedata
    import hashlib
    from datetime import datetime, timedelta
    import pytz
    
    # Configurar zona horaria
    bogota_tz = pytz.timezone('America/Bogota')
    today = datetime.now(bogota_tz).date()
    
    # A. Dataset normalizado
    dataset = []
    
    # B. Reporte de calidad
    quality_report = {
        'total_registros': len(df_raw),
        'registros_abiertos': 0,
        'registros_cerrados': 0,
        'errores': {},
        'nulos_por_columna': {}
    }
    
    # C. Catálogos
    catalogos = {
        'flota': set(),
        'tipo_atencion': set(),
        'sistema_afectado': set(),
        'operador': set(),
        'ejecutor': set()
    }
    
    def clean_text(text):
        """Limpiar texto: trimear, quitar espacios NO-BREAK, unificar espacios"""
        if text is None or (hasattr(text, '__len__') and len(str(text).strip()) == 0):
            return None
        if str(text).lower() in ['nan', 'none', '']:
            return None
        text = str(text).strip()
        # Quitar espacios NO-BREAK (U+00A0)
        text = text.replace('\u00a0', ' ')
        # Unificar espacios múltiples
        text = ' '.join(text.split())
        return text if text else None
    
    def normalize_system(system):
        """Normalizar sistema afectado: upper-case, quitar tildes"""
        if not system:
            return None
        # Quitar tildes
        normalized = unicodedata.normalize('NFD', system.upper())
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        # Equivalencias comunes
        equivalencias = {
            'SUSPENSIÓN': 'SUSPENSION',
            'ELÉCTRICO': 'ELECTRICO',
            'HIDRÁULICO': 'HIDRAULICO',
            'NEUMÁTICO': 'NEUMATICO'
        }
        return equivalencias.get(normalized, normalized)
    
    def parse_boolean(value):
        """Parsear booleanos flexibles"""
        if value is None or str(value).strip().lower() in ['nan', 'none', '']:
            return None
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        
        str_val = str(value).strip().upper()
        if str_val in ['SI', 'S', 'TRUE', '1', 'YES']:
            return True
        elif str_val in ['NO', 'N', 'FALSE', '0', 'NO']:
            return False
        return None
    
    def parse_date(date_val):
        """Parsear fechas con múltiples formatos"""
        if date_val is None or str(date_val).strip().lower() in ['nan', 'none', '']:
            return None
        
        if isinstance(date_val, datetime):
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
    
    def safe_numeric(value, min_val=0):
        """Convertir a numérico con validación"""
        if value is None or str(value).strip().lower() in ['nan', 'none', '']:
            return None
        
        try:
            # Manejar separador decimal ","
            if isinstance(value, str):
                value = value.replace(',', '.')
            
            num_val = float(value)
            return num_val if num_val >= min_val else None
        except:
            return None
    
    # Mapeo de columnas
    column_mapping = {
        'CODIGO': 'codigo_equipo',
        'PLACA': 'placa',
        'DESCRIPCION': 'descripcion_equipo',
        'FLOTA': 'flota',
        'Horas IN': 'horas_in',
        'Horometro IN': 'horometro_in',
        'Km IN': 'km_in',
        'FECHA IN': 'fecha_in',
        'OPERADOR': 'operador',
        'EJECUTOR': 'ejecutor',
        'FECHA OUT': 'fecha_out',
        'Horas OUT': 'horas_out',
        'TIPO ATENCION': 'tipo_atencion',
        'SISTEMA AFECTADO': 'sistema_afectado',
        'ORIGEN AVERIA': 'origen_averia',
        'DESCRIPCION INTERVENCION': 'desc_intervencion',
        'ATENCION LOCAL': 'atencion_local',
        'ATENCION EXTERNA': 'atencion_externa',
        'SCO/SSE': 'sco_sse',
        'ODC/ORS': 'odc_ors',
        'Cont.Dias.Ave': 'dias_averia',
        'Con.Hrs.Ave': 'horas_averia',
        'Con.In.Taller': 'en_taller',
        'MTTR': 'mttr'
    }
    
    # Procesar cada fila
    for idx, row in df_raw.iterrows():
        record = {}
        errors = []
        
        # Mapear campos básicos
        for orig_col, new_col in column_mapping.items():
            if orig_col in df_raw.columns:
                raw_value = row[orig_col]
                
                # Aplicar transformaciones según el tipo de campo
                if new_col in ['codigo_equipo', 'placa', 'descripcion_equipo', 'operador', 'ejecutor', 'desc_intervencion', 'sco_sse', 'odc_ors']:
                    record[new_col] = clean_text(raw_value)
                    
                elif new_col == 'flota':
                    cleaned = clean_text(raw_value)
                    record[new_col] = cleaned.upper() if cleaned else None
                    if cleaned:
                        catalogos['flota'].add(cleaned.upper())
                        
                elif new_col == 'origen_averia':
                    cleaned = clean_text(raw_value)
                    record[new_col] = cleaned if cleaned else "SIN ASIGNAR"
                    
                elif new_col in ['horas_in', 'horas_out', 'horometro_in', 'km_in', 'dias_averia', 'horas_averia', 'mttr']:
                    record[new_col] = safe_numeric(raw_value)
                    
                elif new_col == 'fecha_in':
                    fecha = parse_date(raw_value)
                    record[new_col] = fecha.isoformat() if fecha else None
                    if fecha and str(fecha) != 'NaT' and hasattr(fecha, 'year') and fecha > today:
                        errors.append('ERROR_FECHA_FUTURA')
                        
                elif new_col == 'fecha_out':
                    fecha = parse_date(raw_value)
                    record[new_col] = fecha.isoformat() if fecha else None
                    
                elif new_col in ['atencion_local', 'atencion_externa']:
                    record[new_col] = parse_boolean(raw_value)
                    
                elif new_col == 'en_taller':
                    en_taller = parse_boolean(raw_value)
                    record[new_col] = en_taller
                    
                elif new_col == 'tipo_atencion':
                    cleaned = clean_text(raw_value)
                    if cleaned:
                        # Normalizar tipos de atención
                        tipo_upper = cleaned.upper()
                        if 'PREVENT' in tipo_upper:
                            normalized_tipo = 'PREVENTIVA'
                        elif 'CORRECT' in tipo_upper:
                            normalized_tipo = 'CORRECTIVA'
                        elif 'ALIST' in tipo_upper or 'TC' in tipo_upper:
                            normalized_tipo = 'ALISTAMIENTO-TC'
                        else:
                            normalized_tipo = tipo_upper
                        
                        record[new_col] = normalized_tipo
                        catalogos['tipo_atencion'].add(normalized_tipo)
                    else:
                        record[new_col] = None
                        
                elif new_col == 'sistema_afectado':
                    cleaned = clean_text(raw_value)
                    if cleaned:
                        normalized = normalize_system(cleaned)
                        record[new_col] = normalized
                        catalogos['sistema_afectado'].add(normalized)
                    else:
                        record[new_col] = None
        
        # Agregar operador y ejecutor a catálogos
        if record.get('operador'):
            catalogos['operador'].add(record['operador'])
        if record.get('ejecutor'):
            catalogos['ejecutor'].add(record['ejecutor'])
        
        # Calcular campos derivados
        # ID Equipo
        record['id_equipo'] = (record.get('codigo_equipo') or 
                              record.get('placa') or 
                              record.get('descripcion_equipo') or 
                              f"EQUIPO_{idx}")
        
        # Estado
        fecha_out = record.get('fecha_out')
        en_taller = record.get('en_taller')
        
        if fecha_out is None or en_taller is True:
            record['estado'] = 'ABIERTO'
            quality_report['registros_abiertos'] += 1
        else:
            record['estado'] = 'CERRADO'
            quality_report['registros_cerrados'] += 1
        
        # Días en taller calculado
        fecha_in_str = record.get('fecha_in')
        if fecha_in_str:
            try:
                fecha_in = datetime.fromisoformat(fecha_in_str).date()
                if record['estado'] == 'ABIERTO':
                    dias_calc = (today - fecha_in).days
                else:
                    fecha_out_obj = datetime.fromisoformat(fecha_out).date()
                    dias_calc = (fecha_out_obj - fecha_in).days
                
                record['dias_en_taller_calc'] = max(0, dias_calc)
            except:
                record['dias_en_taller_calc'] = None
        else:
            record['dias_en_taller_calc'] = None
        
        # MTTR calculado (placeholder)
        record['mttr_calc'] = record.get('mttr')  # Por ahora usar el existente
        
        # Validaciones de calidad
        if fecha_in_str and fecha_out:
            try:
                fecha_in_obj = datetime.fromisoformat(fecha_in_str).date()
                fecha_out_obj = datetime.fromisoformat(fecha_out).date()
                if fecha_out_obj < fecha_in_obj:
                    errors.append('ERROR_FECHA')
            except:
                pass
        
        if record.get('horas_out') and record.get('horas_in'):
            if record['horas_out'] < record['horas_in']:
                errors.append('ERROR_HORAS')
        
        if record.get('atencion_local') and record.get('atencion_externa'):
            errors.append('CONFLICTO_CANAL')
        
        # Registrar errores
        for error in errors:
            quality_report['errores'][error] = quality_report['errores'].get(error, 0) + 1
        
        dataset.append(record)
    
    # Calcular porcentajes de nulos
    for col in column_mapping.values():
        null_count = sum(1 for record in dataset if record.get(col) is None)
        quality_report['nulos_por_columna'][col] = round((null_count / len(dataset)) * 100, 2)
    
    # Convertir sets a listas para JSON
    for key in catalogos:
        catalogos[key] = sorted(list(catalogos[key]))
    
    return dataset, quality_report, catalogos
