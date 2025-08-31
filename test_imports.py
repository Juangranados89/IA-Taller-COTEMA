#!/usr/bin/env python3
"""
Script de prueba para verificar que todas las dependencias se pueden importar correctamente
"""

import sys
print(f"Python version: {sys.version}")

try:
    import flask
    print(f"✅ Flask: {flask.__version__}")
except ImportError as e:
    print(f"❌ Flask: {e}")

try:
    import pandas as pd
    print(f"✅ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")

try:
    import openpyxl
    print(f"✅ Openpyxl: {openpyxl.__version__}")
except ImportError as e:
    print(f"❌ Openpyxl: {e}")

try:
    import sklearn
    print(f"✅ Scikit-learn: {sklearn.__version__}")
except ImportError as e:
    print(f"❌ Scikit-learn: {e}")

try:
    import numpy as np
    print(f"✅ Numpy: {np.__version__}")
except ImportError as e:
    print(f"❌ Numpy: {e}")

print("\n=== Probando importación de la aplicación ===")
try:
    import app
    print("✅ Aplicación Flask importada exitosamente")
    
    # Verificar configuración básica
    if hasattr(app, 'app'):
        print(f"✅ Flask app configurada: {app.app}")
        print(f"✅ Debug mode: {app.app.debug}")
        print(f"✅ Upload folder: {app.app.config.get('UPLOAD_FOLDER', 'No configurado')}")
    else:
        print("❌ No se encontró la instancia de Flask app")
        
except Exception as e:
    print(f"❌ Error importando aplicación: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test completado ===")
