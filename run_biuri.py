
import sys
import os
from pathlib import Path

# Adicionar o diretório do projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from gui.biuri_app_complete import main
    
    if __name__ == "__main__":
        print("🚀 Iniciando BIURI - Explicable IA...")
        print("📱 Interface PyQt6 carregada")
        print("🧠 Sistema de interpretabilidade pronto")
        print("-" * 50)
        
        main()
        
except ImportError as e:
    print(f"❌ Erro ao importar dependências: {e}")
    print("\n💡 Instale as dependências com:")
    print("   pip install -r requirements-win.txt")
    print("\n📋 Dependências necessárias:")
    print("   - PyQt6")
    print("   - scikit-learn")
    print("   - pandas")
    print("   - numpy")
    print("   - matplotlib")
    print("   - scipy")
    print("   - liac-arff")
    
except Exception as e:
    print(f"❌ Erro inesperado: {e}")
    print("\n🔧 Verifique se todos os arquivos estão presentes:")
    print("   - gui/biuri_app_complete.py")
    print("   - core/ (módulos do core)")
    print("   - requirements-win.txt")
