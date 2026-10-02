import sys
import os
from pathlib import Path

def test_imports():
    
    print("🔍 Testando importações...")
    
    try:
        # Testar PyQt6
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 importado com sucesso")
        
        # Testar módulos do core
        sys.path.append(str(Path(__file__).parent))
        from core.trepan import TrepanReloaded
        print("✅ TrepanReloaded importado com sucesso")
        
        from core.metrics_comparator import MetricsComparator
        print("✅ MetricsComparator importado com sucesso")
        
        # Testar outras dependências
        import numpy as np
        print("✅ NumPy importado com sucesso")
        
        import pandas as pd
        print("✅ Pandas importado com sucesso")
        
        import matplotlib.pyplot as plt
        print("✅ Matplotlib importado com sucesso")
        
        from scipy.io import arff
        print("✅ SciPy importado com sucesso")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False

def test_gui_creation():
    
    print("\n🖥️ Testando criação da interface...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.biuri_app_complete import BiuriApp
        
        # Criar aplicação (sem mostrar)
        app = QApplication([])
        
        # Criar janela principal
        window = BiuriApp()
        
        print("✅ Interface criada com sucesso")
        print(f"✅ Título da janela: {window.windowTitle()}")
        print(f"✅ Tamanho da janela: {window.size().width()}x{window.size().height()}")
        
        # Limpar
        window.close()
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar interface: {e}")
        return False

def test_core_functionality():
    
    print("\n🧠 Testando funcionalidades do core...")
    
    try:
        from core.trepan import TrepanReloaded
        from core.metrics_comparator import MetricsComparator
        
        # Criar instâncias
        trepan = TrepanReloaded()
        metrics = MetricsComparator()
        
        print("✅ TrepanReloaded instanciado")
        print("✅ MetricsComparator instanciado")
        
        # Testar se tem os métodos necessários
        assert hasattr(trepan, 'train_mlp'), "Método train_mlp não encontrado"
        assert hasattr(trepan, 'extractor'), "Extractor não encontrado"
        assert hasattr(metrics, 'compare_all_models'), "Método compare_all_models não encontrado"
        
        print("✅ Métodos principais verificados")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro no core: {e}")
        return False

def test_data_files():
    
    print("\n📁 Testando arquivos de dados...")
    
    try:
        data_dir = Path(__file__).parent / "data"
        
        if not data_dir.exists():
            print("⚠️ Diretório 'data' não encontrado")
            return False
            
        # Verificar arquivos de exemplo
        sample_data = data_dir / "sample_data.arff"
        sample_ontology = data_dir / "sample_ontology.owl"
        
        if sample_data.exists():
            print("✅ sample_data.arff encontrado")
        else:
            print("⚠️ sample_data.arff não encontrado")
            
        if sample_ontology.exists():
            print("✅ sample_ontology.owl encontrado")
        else:
            print("⚠️ sample_ontology.owl não encontrado")
            
        return True
        
    except Exception as e:
        print(f"❌ Erro ao verificar arquivos: {e}")
        return False

def main():
    
    print("🧪 TESTE DA INTERFACE BIURI PYQT6")
    print("=" * 50)
    
    tests = [
        ("Importações", test_imports),
        ("Criação da Interface", test_gui_creation),
        ("Funcionalidades Core", test_core_functionality),
        ("Arquivos de Dados", test_data_files),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n🔬 Executando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro inesperado em {test_name}: {e}")
            results.append((test_name, False))
    
    # Resumo dos resultados
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n🎯 Resultado: {passed}/{total} testes passaram")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM!")
        print("✅ A interface PyQt6 está pronta para uso!")
        print("\n🚀 Para executar a aplicação:")
        print("   python run_biuri.py")
    else:
        print(f"\n⚠️ {total - passed} teste(s) falharam")
        print("🔧 Verifique as dependências e arquivos necessários")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
