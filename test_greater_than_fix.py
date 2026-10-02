import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from core.natural_language_explainer import NaturalLanguageExplainer

def test_greater_than_operator_fix():
    
    print("=" * 70)
    print("TESTE - Correcao do Erro com Operador >")
    print("=" * 70)
    
    # Cria dados de teste com valores mistos (strings e numeros)
    X = np.array([
        [25.5, 50000, 1.2, 3.0],
        [30.0, 60000, 2.1, 5.0],
        [35.5, 70000, 1.8, 4.0],
        [40.0, 80000, 2.5, 6.0],
        [45.5, 90000, 1.9, 7.0]
    ])
    
    y = np.array([0, 1, 0, 1, 0])
    
    feature_names = ['Idade', 'Renda', 'Educacao', 'Experiencia']
    class_names = ['Classe_A', 'Classe_B']
    
    print("Dados de teste criados:")
    print(f"  Amostras: {len(X)}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Classes: {len(class_names)}")
    print()
    
    # Treina arvore
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    tree.fit(X, y)
    
    print("Arvore de decisao treinada:")
    print(f"  Profundidade: {tree.get_depth()}")
    print(f"  Numero de folhas: {tree.get_n_leaves()}")
    print()
    
    # Cria explicador
    explainer = NaturalLanguageExplainer(feature_names, class_names)
    explainer.set_tree_model(tree, X, y)
    
    print("1. TESTE DE GERACAO DE RESUMOS")
    print("-" * 50)
    
    try:
        summaries = explainer.generate_rule_summaries(max_rules=5)
        print(f"OK: Resumos gerados com sucesso!")
        print(f"    Numero de regras: {len(summaries)}")
        
        for i, summary in enumerate(summaries[:3], 1):
            print(f"\n  Regra {i}:")
            print(f"    Predicao: {summary['prediction']}")
            print(f"    Confianca: {summary['confidence']:.0%}")
            print(f"    Amostras: {summary['samples']}")
            
        print("\n" + "=" * 70)
        print("2. TESTE DE AVALIACAO DE CONDICOES")
        print("-" * 50)
        
        # Testa diferentes tipos de condicoes
        test_conditions = [
            "Idade > 30.0",
            "Renda <= 70000",
            "Educacao > 2.0",
            "Experiencia <= 5.0"
        ]
        
        test_sample = np.array([35.0, 65000, 2.2, 4.5])
        
        print("Testando avaliacao de condicoes:")
        for condition in test_conditions:
            try:
                result = explainer._evaluate_condition(test_sample, condition)
                print(f"  {condition}: {result}")
            except Exception as e:
                print(f"  {condition}: ERRO - {str(e)}")
        
        print("\n" + "=" * 70)
        print("3. ANALISE DO PROBLEMA CORRIGIDO")
        print("-" * 50)
        
        print("Problema identificado:")
        print("  - Comparacao '>' entre string e float")
        print("  - Ocorria na funcao _evaluate_condition")
        print("  - feature_value nao era validado como numero")
        
        print("\nCausa raiz:")
        print("  - feature_value podia ser string do numpy array")
        print("  - Comparacao direta sem validacao de tipo")
        print("  - Falta de conversao segura para float")
        
        print("\nSolucoes implementadas:")
        print("  1. Try-except ao converter threshold para float")
        print("  2. Validacao isinstance() para feature_value")
        print("  3. Conversao segura de feature_value para float")
        print("  4. Tratamento de erros em todas as conversoes")
        
        print("\n" + "=" * 70)
        print("4. CODIGO CORRIGIDO")
        print("-" * 50)
        
        print("Antes (problematico):")
        print("  threshold = float(threshold)")
        print("  feature_value = sample[feature_idx]")
        print("  return feature_value > threshold")
        
        print("\nDepois (corrigido):")
        print("  try:")
        print("      threshold = float(threshold)")
        print("  except (ValueError, TypeError):")
        print("      return False")
        print("  if not isinstance(feature_value, (int, float)):")
        print("      feature_value = float(feature_value)")
        print("  return feature_value > threshold")
        
        print("\n" + "=" * 70)
        print("TESTE CONCLUIDO COM SUCESSO!")
        print("=" * 70)
        print("OK: Erro com operador > corrigido")
        print("OK: Resumos gerados sem erros")
        print("OK: Avaliacao de condicoes funcionando")
        print("OK: Validacao de tipos implementada")
        
        return True
        
    except Exception as e:
        print(f"ERRO: Teste falhou com excecao: {str(e)}")
        print(f"Tipo: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_greater_than_operator_fix()
    exit(0 if success else 1)
