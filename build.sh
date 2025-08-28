#!/bin/bash
# Build script para Render - COTEMA Analytics

echo "🔧 Iniciando build para COTEMA Analytics..."

# Actualizar pip y setuptools primero
echo "📦 Actualizando herramientas de build..."
python -m pip install --upgrade pip setuptools wheel

# Instalar dependencias
echo "📚 Instalando dependencias..."
pip install -r requirements.txt

echo "✅ Build completado exitosamente!"
