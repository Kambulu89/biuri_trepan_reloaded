import sys
import os
from pathlib import Path

def test_imports():
    
    print("🔍 Testando importações dos novos widgets...")
    
    try:
        # Testar PyQt6
        from PyQt6.QtWidgets import QApplication
        print("✅ PyQt6 importado com sucesso")
        
        # Testar módulos do core
        sys.path.append(str(Path(__file__).parent))
        from core.trepan import TrepanReloaded
        print("✅ TrepanReloaded importado com sucesso")
        
        # Testar novos widgets PyQt6
        from gui.pyqt_tree_widget import InteractiveTreeWidget, TreeNode
        print("✅ InteractiveTreeWidget importado com sucesso")
        
        from gui.pyqt_tree_controls import TreeControlsWidget
        print("✅ TreeControlsWidget importado com sucesso")
        
        from gui.pyqt_explanation_widget import ExplanationWidget
        print("✅ ExplanationWidget importado com sucesso")
        
        from gui.pyqt_metrics_visualizer import MetricsVisualizer
        print("✅ MetricsVisualizer importado com sucesso")
        
        # Testar aplicação principal atualizada
        from gui.biuri_app_complete import BiuriApp
        print("✅ BiuriApp atualizada importada com sucesso")
        
        return True
        
    except ImportError as e:
        print(f"❌ Erro de importação: {e}")
        return False

def test_widget_creation():
    
    print("\n🖥️ Testando criação dos widgets...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.pyqt_tree_widget import InteractiveTreeWidget
        from gui.pyqt_tree_controls import TreeControlsWidget
        from gui.pyqt_explanation_widget import ExplanationWidget
        from gui.pyqt_metrics_visualizer import MetricsVisualizer
        
        # Criar aplicação (sem mostrar)
        app = QApplication([])
        
        # Testar criação dos widgets
        print("🔧 Testando InteractiveTreeWidget...")
        tree_widget = InteractiveTreeWidget(None, [], [])
        print("✅ InteractiveTreeWidget criado com sucesso")
        
        print("🔧 Testando TreeControlsWidget...")
        controls_widget = TreeControlsWidget(tree_widget)
        print("✅ TreeControlsWidget criado com sucesso")
        
        print("🔧 Testando ExplanationWidget...")
        explanation_widget = ExplanationWidget()
        print("✅ ExplanationWidget criado com sucesso")
        
        print("🔧 Testando MetricsVisualizer...")
        metrics_widget = MetricsVisualizer()
        print("✅ MetricsVisualizer criado com sucesso")
        
        # Limpar
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar widgets: {e}")
        return False

def test_tree_functionality():
    
    print("\n🌳 Testando funcionalidades da árvore...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.pyqt_tree_widget import InteractiveTreeWidget, TreeNode
        from sklearn.datasets import make_classification
        from sklearn.tree import DecisionTreeClassifier
        
        # Criar aplicação
        app = QApplication([])
        
        # Criar dados de teste
        X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
        tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree_model.fit(X, y)
        
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        class_names = [f"Class_{i}" for i in range(len(set(y)))]
        
        # Criar widget de árvore
        tree_widget = InteractiveTreeWidget(tree_model, feature_names, class_names)
        
        # Testar métodos principais
        print("🔧 Testando métodos de zoom...")
        tree_widget.zoom_in()
        tree_widget.zoom_out()
        tree_widget.reset_zoom()
        tree_widget.center_tree()
        print("✅ Métodos de zoom funcionando")
        
        print("🔧 Testando métodos de filtro...")
        tree_widget.toggle_uncertainty_display()
        tree_widget.set_complexity_filter(5, 3)
        print("✅ Métodos de filtro funcionando")
        
        # Testar TreeNode
        print("🔧 Testando TreeNode...")
        node = TreeNode(0, feature=0, threshold=2.5, samples=100, values=[50, 30, 20])
        print(f"✅ TreeNode criado: incerteza={node.uncertainty:.3f}")
        
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas funcionalidades da árvore: {e}")
        return False

def test_explanation_functionality():
    
    print("\n🧠 Testando funcionalidades de explicação...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.pyqt_explanation_widget import ExplanationWidget
        
        # Criar aplicação
        app = QApplication([])
        
        # Criar widget de explicação
        explanation_widget = ExplanationWidget()
        
        # Testar métodos principais
        print("🔧 Testando métodos de explicação...")
        explanation_widget._generate_summaries()
        explanation_widget._show_examples()
        print("✅ Métodos de explicação funcionando")
        
        # Testar análise de amostra
        print("🔧 Testando análise de amostra...")
        explanation_widget.sample_input.setText("2.5, 1.8, 0.7, 2.1")
        explanation_widget.feature_names = ["feature1", "feature2", "feature3", "feature4"]
        explanation_widget._analyze_sample()
        print("✅ Análise de amostra funcionando")
        
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas funcionalidades de explicação: {e}")
        return False

def test_metrics_functionality():
    
    print("\n📊 Testando funcionalidades de métricas...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.pyqt_metrics_visualizer import MetricsVisualizer
        
        # Criar aplicação
        app = QApplication([])
        
        # Criar widget de métricas
        metrics_widget = MetricsVisualizer()
        
        # Testar adição de dados
        print("🔧 Testando adição de dados...")
        metrics_widget.add_model_data("MLP", 85.2, 60.0)
        metrics_widget.add_model_data("Trepan-Original", 78.5, 80.0)
        metrics_widget.add_model_data("Trepan-Reloaded", 82.1, 90.0)
        print("✅ Adição de dados funcionando")
        
        # Testar métodos principais
        print("🔧 Testando métodos de métricas...")
        best_model = metrics_widget._find_best_model()
        print(f"✅ Melhor modelo encontrado: {best_model['name']}")
        
        key_points = metrics_widget._generate_key_points()
        print("✅ Pontos-chave gerados")
        
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro nas funcionalidades de métricas: {e}")
        return False

def test_integration():
    
    print("\n🔗 Testando integração com aplicação principal...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from gui.biuri_app_complete import BiuriApp
        
        # Criar aplicação
        app = QApplication([])
        
        # Criar aplicação principal
        biuri_app = BiuriApp()
        
        print("✅ BiuriApp criada com sucesso")
        print(f"✅ Título: {biuri_app.windowTitle()}")
        print(f"✅ Tamanho: {biuri_app.size().width()}x{biuri_app.size().height()}")
        
        # Testar se os novos widgets estão integrados
        print("🔧 Verificando integração dos widgets...")
        
        # Verificar se os métodos existem
        assert hasattr(biuri_app, 'show_natural_explanations'), "Método show_natural_explanations não encontrado"
        assert hasattr(biuri_app, 'visualize_tree'), "Método visualize_tree não encontrado"
        assert hasattr(biuri_app, 'compare_metrics'), "Método compare_metrics não encontrado"
        
        print("✅ Métodos integrados verificados")
        
        app.quit()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na integração: {e}")
        return False

def main():
    
    print("🧪 TESTE DAS FUNCIONALIDADES DA ÁRVORE DE VISUALIZAÇÃO")
    print("=" * 60)
    
    tests = [
        ("Importações", test_imports),
        ("Criação de Widgets", test_widget_creation),
        ("Funcionalidades da Árvore", test_tree_functionality),
        ("Funcionalidades de Explicação", test_explanation_functionality),
        ("Funcionalidades de Métricas", test_metrics_functionality),
        ("Integração Principal", test_integration),
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
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
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
        print("✅ As funcionalidades da árvore de visualização estão funcionando!")
        print("\n🚀 Funcionalidades implementadas:")
        print("   • 🌳 Visualização interativa de árvores")
        print("   • 🎛️ Controles de zoom, pan e filtros")
        print("   • 🧠 Explicações em linguagem natural")
        print("   • 📊 Visualização de métricas comparativas")
        print("   • 🔗 Integração completa com PyQt6")
        print("\n💡 Para executar a aplicação:")
        print("   python run_biuri.py")
    else:
        print(f"\n⚠️ {total - passed} teste(s) falharam")
        print("🔧 Verifique as dependências e implementações")
        
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
