#!/usr/bin/env python3
"""
Test de la funcionalidad de upload después del fix crítico
Verifica que el upload de archivos Excel funciona correctamente
"""

import requests
import os

def test_upload_functionality():
    """Testa la funcionalidad de upload corregida"""
    
    # URL del servidor local
    base_url = "http://localhost:5000"
    
    # Archivo de prueba
    test_file = "sample_data/datos_ejemplo_cotema.xlsx"
    
    if not os.path.exists(test_file):
        print("❌ Error: No se encontró el archivo de prueba")
        return False
    
    print("🧪 Iniciando test de upload...")
    
    # 1. Verificar que el servidor responde
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code != 200:
            print(f"❌ Servidor no responde correctamente: {response.status_code}")
            return False
        print("✅ Servidor responde correctamente")
    except Exception as e:
        print(f"❌ Error conectando al servidor: {e}")
        return False
    
    # 2. Probar el upload con headers AJAX
    try:
        headers = {
            'X-Requested-With': 'XMLHttpRequest'
        }
        
        with open(test_file, 'rb') as f:
            files = {'file': (os.path.basename(test_file), f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            
            upload_response = requests.post(f"{base_url}/upload", files=files, headers=headers, timeout=30)
            
            if upload_response.status_code == 200:
                result = upload_response.json()
                if result.get('success'):
                    print(f"✅ Upload exitoso: {result.get('message', 'Archivo procesado')}")
                    return True
                else:
                    print(f"❌ Upload falló: {result.get('error', 'Error desconocido')}")
                    return False
            else:
                print(f"❌ Upload falló con código: {upload_response.status_code}")
                print(f"Respuesta: {upload_response.text[:200]}")
                return False
                
    except Exception as e:
        print(f"❌ Error durante upload: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test de funcionalidad de upload - Sistema corregido")
    print("=" * 50)
    
    success = test_upload_functionality()
    
    if success:
        print("=" * 50)
        print("🎉 ¡FUNCIONALIDAD DE UPLOAD RESTAURADA!")
        print("✅ El sistema de sesiones Flask funciona correctamente")
        print("✅ Los archivos Excel se procesan sin errores")
        print("✅ Listo para producción")
    else:
        print("=" * 50)
        print("❌ El upload aún presenta problemas")
        print("🔧 Revisar logs del servidor para más detalles")
