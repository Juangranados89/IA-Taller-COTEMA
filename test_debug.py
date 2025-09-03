#!/usr/bin/env python3
"""
Script de debug para probar los componentes de la aplicación COTEMA
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import app, global_data, progress_state
import json

def test_progress_format():
    """Test the progress endpoint format"""
    print("=== TESTING PROGRESS FORMAT ===")
    
    # Test initial progress state
    print(f"Initial progress_state: {progress_state}")
    
    # Simulate progress updates
    from app import update_progress, reset_progress
    update_progress("Test task", 1, 3, "Testing...")
    print(f"After update_progress: {progress_state}")
    
    # Test response format
    response = progress_state.copy()
    response['percentage'] = response.get('progress', 0)
    response['details'] = response.get('message', '')
    print(f"Response format: {response}")
    
    reset_progress()
    print(f"After reset: {progress_state}")

def test_data_loading():
    """Test data loading simulation"""
    print("\n=== TESTING DATA LOADING ===")
    
    print(f"Global data loaded: {global_data.get('df') is not None}")
    print(f"Stats: {global_data.get('stats', {})}")
    
    # Check if sample data file exists
    sample_file = "sample_data/Registro_Entrada_Taller_COTEMA.xlsx"
    if os.path.exists(sample_file):
        print(f"Sample file exists: {sample_file}")
        try:
            import pandas as pd
            xl = pd.ExcelFile(sample_file)
            print(f"Available sheets: {xl.sheet_names}")
            
            # Test reading first sheet
            df = pd.read_excel(sample_file, sheet_name=xl.sheet_names[0], nrows=3)
            print(f"Sample columns: {list(df.columns)[:10]}")
            
        except Exception as e:
            print(f"Error reading sample file: {e}")
    else:
        print("No sample file found")

def test_fr30_analysis():
    """Test FR30 analysis with empty data"""
    print("\n=== TESTING FR30 ANALYSIS ===")
    
    from cotema_processor import get_fr30_analysis
    import pandas as pd
    
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    result = get_fr30_analysis(empty_df)
    print(f"Empty DF result: {result}")
    
    # Test with minimal DataFrame
    minimal_df = pd.DataFrame({
        'codigo': ['CG-TC01', 'AH-BC02', 'CG-TC01'],
        'tipo_atencion': ['CORRECTIVA', 'PREVENTIVA', 'CORRECTIVA'],
        'fecha_in': ['2025-08-01', '2025-08-15', '2025-09-01']
    })
    
    result = get_fr30_analysis(minimal_df, days=30)
    print(f"Minimal DF result: {result}")

if __name__ == "__main__":
    test_progress_format()
    test_data_loading()
    test_fr30_analysis()
    
    print("\n=== SUMMARY ===")
    print("Issues identified:")
    print("1. Progress endpoint now returns 'percentage' field as expected by frontend")
    print("2. Upload processing needs better feedback and error handling")
    print("3. FR30 analysis should handle edge cases better")
    print("4. Need to test with real uploaded file structure")
