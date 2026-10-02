import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Adicionar o diretório do projeto ao path
sys.path.append(str(Path(__file__).parent))

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel
from gui.pyqt_tree_widget import InteractiveTreeWidget
from gui.pyqt_tree_controls import TreeControlsWidget


class TreeTestWindow(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Teste - Melhorias da Árvore de Decisão")
        self.setGeometry(100, 100, 1200, 800)
        
        # Carregar dados de exemplo
        self.load_sample_data()
        
        # Configurar interface
        self.setup_ui()
        
    def load_sample_data(self):
        
        # Usar dataset Iris como exemplo
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Treinar árvore de decisão simples
        self.tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
        self.tree_model.fit(X, y)
        
        # Nomes das features e classes
        self.feature_names = iris.feature_names
        self.class_names = iris.target_names
        
        print(f"✅ Dados carregados: {len(X)} amostras, {len(self.feature_names)} features")
        print(f"✅ Árvore treinada com {self.tree_model.get_depth()} níveis de profundidade")
        
    def setup_ui(self):
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        
        # Título
        title = QLabel("🌳 Teste das Melhorias da Visualização da Árvore")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px;")
        layout.addWidget(title)
        
        # Instruções
        instructions = QLabel("""
        🎯 Funcionalidades testadas:
        • ✅ Zoom com scroll do mouse (scroll para cima/baixo)
        • ✅ Arrastar com mouse (clique e arraste em área vazia)
        • ✅ Árvore completa sempre visível (sem cortes)
        • ✅ Labels das arestas melhoradas
        • ✅ Centralização automática
        
        🖱️ Controles:
        • Scroll do mouse: Zoom in/out
        • Clique e arraste: Mover a árvore
        • Botões de controle: Zoom e centralização
        """)
        instructions.setStyleSheet("background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px;")
        layout.addWidget(instructions)
        
        # Widget principal com árvore e controles
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Criar widget da árvore
        self.tree_widget = InteractiveTreeWidget(
            self.tree_model,
            self.feature_names,
            self.class_names
        )
        
        # Criar controles
        self.tree_controls = TreeControlsWidget(self.tree_widget)
        
        # Layout horizontal para controles e árvore
        tree_layout = QVBoxLayout()
        
        # Controles no topo
        tree_layout.addWidget(self.tree_controls)
        
        # Árvore abaixo
        tree_layout.addWidget(self.tree_widget)
        
        main_widget.setLayout(tree_layout)
        layout.addWidget(main_widget)
        
        central_widget.setLayout(layout)
        
        print("✅ Interface de teste configurada")


def main():
    
    print("🚀 Iniciando teste das melhorias da árvore de decisão...")
    
    app = QApplication(sys.argv)
    
    # Configurar estilo
    app.setStyleSheet("""
        QMainWindow {
            background-color: #f5f5f5;
        }
        QApplication {
            font-family: 'Segoe UI', Arial, sans-serif;
        }
    """)
    
    # Criar e mostrar janela de teste
    window = TreeTestWindow()
    window.show()
    
    print("✅ Teste iniciado! Teste as funcionalidades:")
    print("   • Scroll do mouse para zoom")
    print("   • Clique e arraste para mover")
    print("   • Botões de controle")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
