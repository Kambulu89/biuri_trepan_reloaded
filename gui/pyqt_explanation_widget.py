from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QLineEdit, QScrollArea, QFrame, QTabWidget,
    QDialog, QDialogButtonBox, QListWidget, QListWidgetItem,
    QGroupBox, QSplitter, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QPalette, QColor
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Importar o explicador de linguagem natural
sys.path.append(str(Path(__file__).parent.parent))
from core.natural_language_explainer import NaturalLanguageExplainer


class ExplanationWidget(QWidget):
    
    def __init__(self, tree_model=None, feature_names=None, class_names=None, 
                 training_data=None, training_labels=None, parent=None):
        super().__init__(parent)
        
        # Inicializa o explicador
        self.feature_names = feature_names or []
        self.class_names = class_names or []
        self.tree_model = tree_model
        self.training_data = training_data
        self.training_labels = training_labels
        
        self.current_sample = None
        self.current_explanations = None
        self.question_history = []
        
        # Inicializa o NaturalLanguageExplainer se temos os dados necessários
        self.explainer = None
        if self.feature_names and self.class_names and self.tree_model:
            self.explainer = NaturalLanguageExplainer(self.feature_names, self.class_names)
            if self.tree_model and self.training_data is not None:
                self.explainer.set_tree_model(self.tree_model, self.training_data, self.training_labels)
        
        self.setup_ui()
        
    def setup_ui(self):
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Título
        title = QLabel("🧠 Explicaciones en Lenguaje Natural")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2c5aa0; margin: 10px;")
        layout.addWidget(title)
        
        # Seção de controles principais
        controls_layout = QHBoxLayout()
        
        # Botão para gerar resumos
        self.summaries_btn = QPushButton("📋 Generar Resúmenes")
        self.summaries_btn.clicked.connect(self._generate_summaries)
        
        # Botão para exemplos
        self.examples_btn = QPushButton("💡 Ver Ejemplos")
        self.examples_btn.clicked.connect(self._show_examples)
        
        # Botão para perguntas
        self.qa_btn = QPushButton("❓ Preguntas")
        self.qa_btn.clicked.connect(self._show_qa_interface)
        
        controls_layout.addWidget(self.summaries_btn)
        controls_layout.addWidget(self.examples_btn)
        controls_layout.addWidget(self.qa_btn)
        
        layout.addLayout(controls_layout)
        
        # Área de entrada de amostra
        sample_layout = QHBoxLayout()
        
        sample_label = QLabel("Muestra:")
        sample_label.setMinimumWidth(80)
        
        self.sample_input = QLineEdit()
        # Atualiza placeholder baseado nos nomes dos atributos
        if self.feature_names:
            placeholder = f"Ingrese valores separados por coma para: {', '.join(self.feature_names[:3])}{'...' if len(self.feature_names) > 3 else ''}"
        else:
            placeholder = "Ingrese valores separados por coma (ej: 25, 50000, 1)"
        self.sample_input.setPlaceholderText(placeholder)
        
        self.analyze_btn = QPushButton("🔍 Analizar")
        self.analyze_btn.clicked.connect(self._analyze_sample)
        
        sample_layout.addWidget(sample_label)
        sample_layout.addWidget(self.sample_input)
        sample_layout.addWidget(self.analyze_btn)
        
        layout.addLayout(sample_layout)
        
        # Área de conteúdo principal com tabs
        self.content_tabs = QTabWidget()
        
        # Tab de resumos
        self.summaries_tab = QWidget()
        self.setup_summaries_tab()
        self.content_tabs.addTab(self.summaries_tab, "📋 Resúmenes")
        
        # Tab de exemplos
        self.examples_tab = QWidget()
        self.setup_examples_tab()
        self.content_tabs.addTab(self.examples_tab, "💡 Ejemplos")
        
        # Tab de análise
        self.analysis_tab = QWidget()
        self.setup_analysis_tab()
        self.content_tabs.addTab(self.analysis_tab, "🔍 Análisis")
        
        layout.addWidget(self.content_tabs)
        
        self.setLayout(layout)
        
        # Mensagem inicial
        self._show_welcome_message()
        
    def setup_summaries_tab(self):
        
        layout = QVBoxLayout()
        
        self.summaries_text = QTextEdit()
        self.summaries_text.setReadOnly(True)
        self.summaries_text.setFont(QFont("Arial", 12))
        
        layout.addWidget(self.summaries_text)
        self.summaries_tab.setLayout(layout)
        
    def setup_examples_tab(self):
        
        layout = QVBoxLayout()
        
        self.examples_text = QTextEdit()
        self.examples_text.setReadOnly(True)
        self.examples_text.setFont(QFont("Arial", 12))
        
        layout.addWidget(self.examples_text)
        self.examples_tab.setLayout(layout)
        
    def setup_analysis_tab(self):
        
        layout = QVBoxLayout()
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFont(QFont("Arial", 12))
        
        layout.addWidget(self.analysis_text)
        self.analysis_tab.setLayout(layout)
        
    def _show_welcome_message(self):
        
        welcome_text = """
🧠 Sistema de Explicaciones en Lenguaje Natural

Este sistema genera explicaciones comprensibles sobre cómo funciona su árbol de decisión:

📋 Resúmenes Automáticos: Reglas más importantes en lenguaje natural
💡 Ejemplos Concretos: Casos reales que siguen cada regla
❓ Preguntas y Respuestas: Sistema interactivo para dudas

Para comenzar:
1. Haga clic en "Generar Resúmenes" para ver las reglas principales
2. Ingrese una muestra y haga clic en "Analizar" para explicaciones específicas
3. Use "Preguntas" para interactuar con el sistema
        """
        
        self.summaries_text.setPlainText(welcome_text)
        
    def _generate_summaries(self):
        
        if not self.tree_model:
            self._show_error("¡Ningún modelo de árbol disponible!")
            return
        
        if not self.explainer:
            if self.feature_names and self.class_names:
                self.explainer = NaturalLanguageExplainer(self.feature_names, self.class_names)
                if self.training_data is not None:
                    self.explainer.set_tree_model(self.tree_model, self.training_data, self.training_labels)
        
        if not self.explainer:
            self._show_error("¡No fue posible inicializar el sistema de explicaciones!")
            return
        
        try:
            summaries = self.explainer.generate_rule_summaries(max_rules=10)
            
            if not summaries:
                summaries_text = """
⚠️ Ninguna regla encontrada en el árbol de decisión.

Verifique que el modelo fue entrenado correctamente.
                """
                self.summaries_text.setPlainText(summaries_text)
                self.content_tabs.setCurrentIndex(0)
                return
            
            summaries_text = "📋 Resúmenes de las Reglas Más Importantes\n\n"
            
            for summary in summaries:
                rule_num = summary['rule_id']
                prediction = summary['prediction']
                text = summary['text']
                confidence = summary['confidence']
                samples = summary['samples']
                importance = summary['importance_score']
                
                summaries_text += f"**Regla {rule_num}: Clasificación {prediction}**\n"
                summaries_text += f"{text}\n"
                summaries_text += f"Confianza: {confidence:.0%} | Muestras: {samples} | Importancia: {importance:.2f}\n\n"
                
                if summary['examples']:
                    summaries_text += "💡 Ejemplos:\n"
                    for example in summary['examples'][:3]:
                        features_str = ", ".join([f"{name}={example['features'].get(name, 'N/A'):.2f}" 
                                                 for name in self.feature_names 
                                                 if name in example['features']])
                        status = "✅" if example['match'] else "❌"
                        summaries_text += f"• {features_str} → {example['actual_label']} {status}\n"
                    summaries_text += "\n"
            
            self.summaries_text.setPlainText(summaries_text)
            self.content_tabs.setCurrentIndex(0)
            
        except Exception as e:
            self._show_error(f"Error al generar resúmenes: {str(e)}")
        
    def _show_examples(self):
        
        if not self.tree_model:
            self._show_error("¡Ningún modelo de árbol disponible!")
            return
        
        if not self.explainer:
            if self.feature_names and self.class_names:
                self.explainer = NaturalLanguageExplainer(self.feature_names, self.class_names)
                if self.training_data is not None:
                    self.explainer.set_tree_model(self.tree_model, self.training_data, self.training_labels)
        
        if not self.explainer:
            self._show_error("¡No fue posible inicializar el sistema de explicaciones!")
            return
        
        try:
            examples = self.explainer.generate_concrete_examples(max_examples=10)
            
            if not examples:
                examples_text = """
⚠️ Ningún ejemplo encontrado.

Verifique que hay datos de entrenamiento disponibles.
                """
                self.examples_text.setPlainText(examples_text)
                self.content_tabs.setCurrentIndex(1)
                return
            
            examples_text = "💡 Ejemplos Concretos de las Reglas\n\n"
            
            for i, example in enumerate(examples, 1):
                example_id = example.get('sample_id', i)
                features = example.get('features', {})
                actual_label = example.get('actual_label', 'Desconocido')
                predicted_label = example.get('predicted_label', 'Desconocido')
                match = example.get('match', False)
                similarity = example.get('similarity', 0.0)
                
                features_str = ", ".join([f"{name}={value:.2f}" 
                                         for name, value in features.items() 
                                         if name in self.feature_names])
                
                status = "✅" if match else "❌"
                
                examples_text += f"Ejemplo {i} (ID: {example_id:03d})\n"
                examples_text += f"Características: {features_str}\n"
                examples_text += f"Clasificación Real: {actual_label} | Predicha: {predicted_label} {status}\n"
                if similarity > 0:
                    examples_text += f"Similitud: {similarity:.2f}\n"
                examples_text += "\n"
            
            self.examples_text.setPlainText(examples_text)
            self.content_tabs.setCurrentIndex(1)
            
        except Exception as e:
            self._show_error(f"Error al generar ejemplos: {str(e)}")
        
    def _show_qa_interface(self):
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Sistema de Preguntas y Respuestas")
        dialog.setModal(True)
        dialog.setMinimumSize(560, 420); dialog.resize(800, 600)
        
        layout = QVBoxLayout()
        
        # Título
        title = QLabel("❓ Sistema de Preguntas y Respuestas")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Preguntas sugeridas
        suggested_group = QGroupBox("Preguntas sugeridas:")
        suggested_layout = QVBoxLayout()
        
        questions = [
            "¿Cómo funciona el modelo?",
            "¿Cuándo clasifica el modelo como positivo?",
            "¿Qué analiza el modelo?",
            "¿Por qué se clasificó esta muestra así?",
            "¿Cuál es la regla más importante?",
            "¿Cuántas muestras analizó el modelo?",
            "¿Cuál es la confianza de esta decisión?",
            "¿Cuáles son las principales características analizadas?"
        ]
        
        for question in questions:
            btn = QPushButton(f"• {question}")
            btn.clicked.connect(lambda checked, q=question: self._ask_question(q, dialog))
            suggested_layout.addWidget(btn)
        
        suggested_group.setLayout(suggested_layout)
        layout.addWidget(suggested_group)
        
        # Área de entrada personalizada
        custom_layout = QHBoxLayout()
        
        self.custom_question_input = QLineEdit()
        self.custom_question_input.setPlaceholderText("Ingrese su pregunta aquí...")
        
        ask_btn = QPushButton("Preguntar")
        ask_btn.clicked.connect(lambda: self._ask_custom_question(dialog))
        
        history_btn = QPushButton("Historial")
        history_btn.clicked.connect(lambda: self._show_question_history(dialog))
        
        custom_layout.addWidget(self.custom_question_input)
        custom_layout.addWidget(ask_btn)
        custom_layout.addWidget(history_btn)
        
        layout.addLayout(custom_layout)
        
        # Área de resposta
        self.qa_answer_text = QTextEdit()
        self.qa_answer_text.setReadOnly(True)
        self.qa_answer_text.setFont(QFont("Arial", 12))
        self.qa_answer_text.setPlainText("Haga una pregunta para ver la respuesta aquí.")
        
        layout.addWidget(self.qa_answer_text)
        
        # Botão fechar
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def _ask_question(self, question, dialog):
        
        self.custom_question_input.setText(question)
        self._ask_custom_question(dialog)
        
    def _ask_custom_question(self, dialog):
        
        question = self.custom_question_input.text().strip()
        if not question:
            return
        
        try:
            self.question_history.append({
                'question': question,
                'timestamp': datetime.now().strftime("%H:%M:%S"),
                'sample': self.current_sample is not None
            })
            
            answer = self._simulate_answer(question)
            
            response_text = f"❓ Pregunta: {question}\n\n"
            response_text += f"💬 Respuesta:\n{answer}\n\n"
            response_text += f"🕒 Timestamp: {datetime.now().strftime('%H:%M:%S')}"
            
            self.qa_answer_text.setPlainText(response_text)
            
        except Exception as e:
            self.qa_answer_text.setPlainText(f"❌ Error al procesar pregunta: {str(e)}\n\n💡 Intente reformular su pregunta.")
    
    def _simulate_answer(self, question):
        
        if not self.explainer:
            if self.feature_names and self.class_names:
                self.explainer = NaturalLanguageExplainer(self.feature_names, self.class_names)
                if self.training_data is not None and self.tree_model:
                    self.explainer.set_tree_model(self.tree_model, self.training_data, self.training_labels)
        
        if not self.explainer:
            return "No fue posible inicializar el sistema de explicaciones. Verifique que el modelo fue entrenado."
        
        try:
            answer_dict = self.explainer.answer_question(question, self.current_sample)
            return answer_dict.get('answer', 'No fue posible generar una respuesta para esta pregunta.')
        except Exception as e:
            return f"Error al procesar pregunta: {str(e)}"
    
    def _show_question_history(self, dialog):
        
        if not self.question_history:
            self.qa_answer_text.setPlainText("Ninguna pregunta realizada todavía.\n\n¡Haga su primera pregunta para comenzar!")
            return
        
        history_text = "Historial de Preguntas:\n\n"
        for i, item in enumerate(self.question_history[-10:], 1):
            sample_indicator = "[CON MUESTRA]" if item['sample'] else "[GENERAL]"
            history_text += f"{i}. {sample_indicator} [{item['timestamp']}] {item['question']}\n"
        
        history_text += f"\nTotal de perguntas: {len(self.question_history)}"
        history_text += f"\nHaga clic en una pregunta sugerida para repetirla"
        
        self.qa_answer_text.setPlainText(history_text)
        
    def _analyze_sample(self):
        
        sample_text = self.sample_input.text().strip()
        if not sample_text:
            self._show_error("¡Ingrese una muestra para análisis!")
            return
        
        if not self.explainer:
            if self.feature_names and self.class_names:
                self.explainer = NaturalLanguageExplainer(self.feature_names, self.class_names)
                if self.training_data is not None and self.tree_model:
                    self.explainer.set_tree_model(self.tree_model, self.training_data, self.training_labels)
        
        if not self.explainer:
            self._show_error("¡No fue posible inicializar el sistema de explicaciones!")
            return
        
        try:
            values = [float(x.strip()) for x in sample_text.split(',')]
            
            if len(values) != len(self.feature_names):
                self._show_error(f"Se esperaban {len(self.feature_names)} valores, recibidos {len(values)}!\n\nAtributos esperados: {', '.join(self.feature_names)}")
                return
            
            sample = np.array(values)
            self.current_sample = sample
            
            explanation = self.explainer.generate_path_explanation(sample)
            
            analysis_text = f"🔍 Análisis de la Muestra\n\n"
            analysis_text += f"🎯 Predicción: {explanation['prediction']}\n"
            analysis_text += f"📊 Confianza: {explanation['confidence']:.0%}\n\n"
            
            if explanation.get('path_steps'):
                analysis_text += "🛤️ Camino de Decisión:\n"
                # Os path_steps já contêm os nomes dos atributos no formato gerado pelo NaturalLanguageExplainer
                # Extrai os nomes dos atributos dos passos (formato: "Passo N: nome_atributo (valor) condição")
                path_features = []
                for step_text in explanation['path_steps']:
                    # Procura por nomes de atributos no texto do passo
                    for feature_name in self.feature_names:
                        if feature_name in step_text:
                            if feature_name not in path_features:  # Evita duplicatas
                                path_features.append(feature_name)
                            break
                
                if path_features:
                    analysis_text += f"La muestra siguió el camino: {' → '.join(path_features)} → {explanation['prediction']}\n\n"
                else:
                    # Se não encontrou nomes, mostra o resumo do caminho se disponível
                    if explanation.get('summary'):
                        analysis_text += f"{explanation['summary']}\n\n"
            
            # Passos detalhados
            if explanation.get('path_steps'):
                analysis_text += "📋 Pasos Detallados:\n"
                for i, step_text in enumerate(explanation['path_steps'], 1):
                    analysis_text += f"{i}. {step_text}\n"
                analysis_text += "\n"
            
            # Resumo do caminho
            if explanation.get('summary'):
                analysis_text += f"📝 Resumen:\n{explanation['summary']}\n\n"
            
            # Advertencia de extrapolação se necessário
            if explanation.get('extrapolation', {}).get('is_extrapolation'):
                analysis_text += f"⚠️ {explanation['extrapolation']['message']}\n\n"
            
            # Exemplos similares
            if explanation.get('examples'):
                analysis_text += "💡 Ejemplos Similares:\n"
                for i, example in enumerate(explanation['examples'][:3], 1):
                    features_str = ", ".join([f"{name}={example['features'].get(name, 0):.2f}" 
                                            for name in self.feature_names 
                                            if name in example['features']])
                    similarity = example.get('similarity', 0.0)
                    analysis_text += f"{i}. {features_str} → {example.get('actual_label', 'Desconocido')} (similitud: {similarity:.2f}) ✅\n"
            
            self.analysis_text.setPlainText(analysis_text)
            self.content_tabs.setCurrentIndex(2)  # Muda para tab de análise
            
        except ValueError as e:
            self._show_error(f"Formato inválido de la muestra: {str(e)}")
        except Exception as e:
            self._show_error(f"Error al analizar muestra: {str(e)}")
    
    def _show_error(self, message):
        
        QMessageBox.warning(self, "Error", f"❌ {message}")
    
    def update_model(self, tree_model, feature_names, class_names, training_data=None, training_labels=None):
        
        self.tree_model = tree_model
        self.feature_names = feature_names
        self.class_names = class_names
        self.training_data = training_data
        self.training_labels = training_labels
        
        # Reinicializa o explicador com os novos dados
        if self.feature_names and self.class_names and self.tree_model:
            self.explainer = NaturalLanguageExplainer(self.feature_names, self.class_names)
            if self.training_data is not None:
                self.explainer.set_tree_model(self.tree_model, self.training_data, self.training_labels)
        else:
            self.explainer = None
        
        # Atualiza placeholder do campo de entrada
        if hasattr(self, 'sample_input') and self.feature_names:
            placeholder = f"Ingrese valores separados por coma para: {', '.join(self.feature_names[:3])}{'...' if len(self.feature_names) > 3 else ''}"
            self.sample_input.setPlaceholderText(placeholder)
        
        # Limpa conteúdo anterior
        self.summaries_text.clear()
        self.examples_text.clear()
        self.analysis_text.clear()
        self._show_welcome_message()
