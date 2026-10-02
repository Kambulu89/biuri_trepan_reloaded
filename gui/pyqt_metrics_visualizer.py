from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTableWidget, QTableWidgetItem, QScrollArea, QFrame,
    QDialog, QDialogButtonBox, QGroupBox, QSplitter, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen, QBrush
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from core.metrics_view_model import claim_banner


class MetricsVisualizer(QWidget):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.models_data = {}
        self.comparison_results = {}
        
        self.setup_ui()

    def setup_ui(self):
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("📊 Panel de Métricas Comparativas")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2c5aa0; margin: 10px;")
        layout.addWidget(title)
        
        action_layout = QHBoxLayout()
        
        compare_btn = QPushButton("📈 Comparar Modelos")
        compare_btn.setStyleSheet("background-color: #2ecc71; color: white; font-weight: bold; padding: 10px;")
        compare_btn.clicked.connect(self._show_comparison_chart)
        
        dashboard_btn = QPushButton("📋 Panel Resumido")
        dashboard_btn.setStyleSheet("background-color: #9b59b6; color: white; font-weight: bold; padding: 10px;")
        dashboard_btn.clicked.connect(self._show_dashboard_summary)
        
        data_viz_btn = QPushButton("📊 Visualizar Datos")
        data_viz_btn.setStyleSheet("background-color: #e67e22; color: white; font-weight: bold; padding: 10px;")
        data_viz_btn.clicked.connect(self._show_data_distribution)
        
        action_layout.addWidget(compare_btn)
        action_layout.addWidget(dashboard_btn)
        action_layout.addWidget(data_viz_btn)
        
        layout.addLayout(action_layout)
        
        self.content_area = QScrollArea()
        self.content_area.setWidgetResizable(True)
        self.content_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(15)
        
        self.content_widget.setLayout(self.content_layout)
        self.content_area.setWidget(self.content_widget)
        
        layout.addWidget(self.content_area)
        
        self.setLayout(layout)
        
    def clear_model_data(self):
        self.models_data = {}

    def add_model_data(self, model_name, precision, fidelity, accuracy=None, sample_size=None,
                       balanced_accuracy=None, macro_f1=None, oracle=None,
                       feature_space=None, canonical=True,
                       fidelity_to_active_oracle=None,
                       fidelity_to_mlp_original=None):
        self.models_data[model_name] = {
            'precision': precision,
            'fidelity': fidelity,
            'accuracy': accuracy,
            'sample_size': sample_size,
            'balanced_accuracy': balanced_accuracy,
            'macro_f1': macro_f1,
            'oracle': oracle,
            'feature_space': feature_space,
            'canonical': bool(canonical),
            'fidelity_to_active_oracle': fidelity_to_active_oracle,
            'fidelity_to_mlp_original': fidelity_to_mlp_original,
        }

    def set_comparison_results(self, results):
        self.comparison_results = results or {}
        
    def _fit_dialog_to_screen(self, dialog, preferred_width=900, preferred_height=680):
        """Ajusta e centra o diálogo na área útil do ecrã (exclui a barra de tarefas)."""
        margin = 24
        screen = QApplication.primaryScreen()
        if screen:
            available = screen.availableGeometry()
            width = min(preferred_width, available.width() - margin)
            height = min(preferred_height, available.height() - margin)
            dialog.setMinimumSize(min(width, 480), min(height, 360)); dialog.resize(width, height)
            x = available.x() + max(0, (available.width() - width) // 2)
            y = available.y() + max(0, (available.height() - height) // 2)
            dialog.move(x, y)
            return width, height
        dialog.setMinimumSize(480, 360); dialog.resize(preferred_width, preferred_height)
        return preferred_width, preferred_height

    def _c45_gate_summary_text(self):
        """Resumo do objetivo C4.5 sem ocultar/alterar qualquer métrica."""
        gate = (self.comparison_results or {}).get('c45_baseline_gate') or {}
        if gate.get('available'):
            parts = []
            for key, label in (
                ('trepan_original', 'TREPAN Original'),
                ('trepan_reloaded', 'TREPAN Reloaded'),
            ):
                item = gate.get(key) or {}
                if not item:
                    continue
                delta = (item.get('deltas') or {}).get('precision_macro')
                status = 'ATINGIU O BASELINE' if item.get('accepted') else 'ABAIXO DO BASELINE'
                delta_txt = '' if delta is None else f" ({float(delta) * 100:+.1f} pp em Precisão Macro)"
                parts.append(f"{label}: {status}{delta_txt}")
            if parts:
                return (
                    "Gate C4.5 (baseline supervisionado; não é oráculo): "
                    + "  |  ".join(parts)
                    + ". Os valores são reportados sem ajuste artificial."
                )

        # Fallback visual quando a estrutura de auditoria não estiver disponível.
        def find_value(fragment):
            norm = lambda value: str(value).lower().replace(' ', '').replace('-', '').replace('/', '')
            for name, data in self.models_data.items():
                if norm(fragment) in norm(name):
                    return float(data.get('precision', 0.0))
            return None

        c45 = find_value('C4.5-Nativo')
        if c45 is None:
            return 'Gate C4.5 indisponível nesta execução.'
        parts = []
        for label in ('Trepan-Original', 'Trepan-Reloaded'):
            value = find_value(label)
            if value is None:
                continue
            delta = value - c45
            status = 'ATINGIU O BASELINE' if delta >= -1e-9 else 'ABAIXO DO BASELINE'
            parts.append(f"{label}: {status} ({delta:+.1f} pp em Precisão Macro)")
        return (
            "Gate C4.5 (baseline supervisionado; não é oráculo): "
            + ('  |  '.join(parts) if parts else 'sem TREPAN comparável')
            + '. Os valores são reportados sem ajuste artificial.'
        )

    def _show_comparison_chart(self):
        if not self.models_data:
            self._show_no_data_message()
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Comparación de Modelos")
        dialog.setModal(True)
        dialog_width, dialog_height = self._fit_dialog_to_screen(dialog)
        
        layout = QVBoxLayout()
        
        title = QLabel("📈 Comparación de Modelos")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        gate_label = QLabel(self._c45_gate_summary_text())
        gate_label.setWordWrap(True)
        gate_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gate_label.setStyleSheet(
            "background-color: #f8f9fa; border: 1px solid #b0b0b0; "
            "padding: 7px; font-weight: bold;"
        )
        layout.addWidget(gate_label)
        
        chart_width = max(400, dialog_width - 30)
        chart_height = max(300, dialog_height - 120)
        figsize = (chart_width / 100, chart_height / 100)
        chart_widget = self._create_bar_chart(figsize=figsize)
        layout.addWidget(chart_widget)
        
        close_btn = QPushButton("Cerrar")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
        
    def _create_bar_chart(self, figsize=(9, 6)):
        print("[DIAG] Gráfico de barras — datos en self.models_data:")
        for name, data in self.models_data.items():
            print(
                f"  {name}: Precisão Macro={data['precision']:.1f}%, "
                f"Exatidão(auditoria)={float(data.get('accuracy') or 0):.1f}%, "
                f"Fidelidad={data['fidelity']:.1f}%"
            )

        precision_order = [
            'MLP Original',
            'MLP Ontológico',
            'Trepan-Original',
            'C4.5-Nativo',
            'Trepan-Reloaded',
        ]
        precision_colors = {
            'MLP Original': '#000000',
            'Trepan-Original': '#808080',
            'C4.5-Nativo': '#FFFFFF',
            'Trepan-Reloaded': '#FFFF00'
        }
        
        precision_models = []
        precision_values = []
        precision_colors_list = []
        
        for model_name in precision_order:
            matching_keys = [key for key in self.models_data.keys() 
                           if model_name.lower().replace(' ', '').replace('-', '').replace('/', '') in 
                              key.lower().replace(' ', '').replace('-', '').replace('/', '')]
            if matching_keys:
                key = matching_keys[0]
                precision_models.append(key)
                # Pedido do projeto: a comparação principal apresenta Precisão Macro,
                # nunca Accuracy disfarçada de Precisão.
                precision_values.append(self.models_data[key]['precision'])
                for ordered_name in precision_order:
                    if ordered_name.lower().replace(' ', '').replace('-', '').replace('/', '') in key.lower().replace(' ', '').replace('-', '').replace('/', ''):
                        precision_colors_list.append(precision_colors.get(ordered_name, '#808080'))
                        break
                else:
                    precision_colors_list.append('#808080')
        
        fidelity_models = []
        fidelity_values = []
        fidelity_colors = []
        
        fidelity_order = ['Trepan-Original', 'Trepan-Reloaded']
        fidelity_color_map = {
            'Trepan-Original': '#808080',
            'Trepan-Reloaded': '#FFFF00'
        }
        
        for model_name in fidelity_order:
            matching_keys = [key for key in self.models_data.keys() 
                           if model_name.lower().replace(' ', '').replace('-', '').replace('/', '') in 
                              key.lower().replace(' ', '').replace('-', '').replace('/', '')]
            if matching_keys:
                key = matching_keys[0]
                fidelity_models.append(key)
                # Comparação visual controlada: ambos os TREPAN usam a mesma
                # referência (MLP Original). A fidelidade ao oráculo ativo fica
                # visível no cartão do modelo, mas não é misturada neste gráfico.
                controlled = self.models_data[key].get('fidelity_to_mlp_original')
                fidelity_values.append(
                    self.models_data[key]['fidelity'] if controlled is None else controlled
                )
                for ordered_name in fidelity_order:
                    if ordered_name.lower().replace(' ', '').replace('-', '').replace('/', '') in key.lower().replace(' ', '').replace('-', '').replace('/', ''):
                        fidelity_colors.append(fidelity_color_map.get(ordered_name, '#808080'))
                        break
                else:
                    fidelity_colors.append('#808080')
        
        fig = Figure(figsize=figsize)
        fig.suptitle('Comparación de Métricas de los Modelos', fontsize=16, fontweight='bold')
        
        bar_width = 0.8
        
        ax1 = fig.add_subplot(2, 1, 1)
        if precision_models:
            x_pos1 = np.arange(len(precision_models))
            bars1 = ax1.bar(x_pos1, precision_values, width=bar_width, color=precision_colors_list, 
                           edgecolor='black', linewidth=1.5)
            ax1.set_title('Precisão Macro / Precision Macro', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Precisão Macro (%)')
            ax1.set_ylim(0, 110)
            ax1.set_facecolor('#F5F5F5')
            ax1.set_xticks(x_pos1)
            ax1.set_xticklabels(precision_models, rotation=0, ha='center')
            
            # C4.5 é mostrado como baseline real, nunca como oráculo. A linha
            # permite ver imediatamente se os TREPAN atingiram o objetivo sem
            # alterar as percentagens observadas.
            c45_value = None
            for model_name, value in zip(precision_models, precision_values):
                if 'c4.5' in model_name.lower():
                    c45_value = float(value)
                    break
            if c45_value is not None:
                ax1.axhline(
                    c45_value, linestyle='--', linewidth=1.2, color='black',
                    alpha=0.65, label=f'Baseline C4.5: {c45_value:.1f}%'
                )
                ax1.legend(loc='lower right', fontsize=9)

            for model_name, bar, value in zip(precision_models, bars1, precision_values):
                height = bar.get_height()
                label = f'{value:.1f}%'
                if c45_value is not None and 'trepan' in model_name.lower():
                    delta = float(value) - c45_value
                    status = 'OK C4.5' if delta >= -1e-9 else 'ABAIXO C4.5'
                    label += f'\n{status} ({delta:+.1f} pp)'
                ax1.text(
                    bar.get_x() + bar.get_width()/2., height + 1, label,
                    ha='center', va='bottom', fontweight='bold', color='black', fontsize=9
                )
        
        ax2 = fig.add_subplot(2, 1, 2)
        if fidelity_models:
            # Mantém os TREPAN alinhados com as respetivas posições no gráfico
            # superior, deixando explícito que MLP e C4.5 não têm barras de
            # fidelidade neste painel.
            if precision_models:
                normalize = lambda name: name.lower().replace(' ', '').replace('-', '').replace('/', '')
                precision_positions = {
                    normalize(name): index for index, name in enumerate(precision_models)
                }
                x_pos2 = [
                    precision_positions.get(normalize(name), index)
                    for index, name in enumerate(fidelity_models)
                ]
            else:
                x_pos2 = list(np.arange(len(fidelity_models)))
            fidelity_values_ordered = fidelity_values
            fidelity_colors_ordered = fidelity_colors
            fidelity_labels = fidelity_models
            
            bars2 = ax2.bar(x_pos2, fidelity_values_ordered, width=bar_width, color=fidelity_colors_ordered, 
                           edgecolor='black', linewidth=1.5)
            ax2.set_title('Fidelidade de controlo ao MLP Original', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Fidelidad (%)')
            ax2.set_ylim(0, 100)
            ax2.set_facecolor('#F5F5F5')
            
            if precision_models:
                ax2.set_xlim(ax1.get_xlim())
            
            ax2.set_xticks(x_pos2)
            ax2.set_xticklabels(fidelity_labels, rotation=0, ha='center')
            
            for bar, value in zip(bars2, fidelity_values_ordered):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', color='black')
        
        fig.tight_layout()
        
        canvas = FigureCanvas(fig)
        return canvas
        
    def _get_quality_color(self, value):
        if value >= 90:
            return '#2ecc71'
        elif value >= 75:
            return '#f39c12'
        else:
            return '#e74c3c'
        
    def _show_dashboard_summary(self):
        if not self.models_data:
            self._show_no_data_message()
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Panel Resumido")
        dialog.setModal(True)
        dialog.setMinimumSize(640, 480); dialog.resize(900, 700)
        
        main_layout = QVBoxLayout()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        content_widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)
        
        title = QLabel("📋 Panel Resumido - Resultados Principales")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2c5aa0; margin: 10px;")
        layout.addWidget(title)
        
        claim = claim_banner(self.comparison_results)
        summary_text = f"""
RESUMO DA EXECUÇÃO

Os modelos são apresentados sem ranking global automático.
A comparação principal apresenta Precisão Macro contra os rótulos reais. Accuracy/Exatidão permanece disponível apenas como métrica adicional de auditoria.
As fidelidades só são comparadas diretamente quando usam a mesma referência de oráculo.

{claim['text']}
        """
        
        summary_label = QLabel(summary_text)
        summary_label.setFont(QFont("Arial", 12))
        summary_label.setWordWrap(True)
        summary_label.setStyleSheet("background-color: #ecf0f1; padding: 15px; border-radius: 5px; margin: 10px;")
        layout.addWidget(summary_label)
        
        details_title = QLabel("📊 Detalles por Modelo")
        details_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        details_title.setStyleSheet("color: #8e44ad; margin: 10px;")
        layout.addWidget(details_title)
        
        for model_name, data in self.models_data.items():
            model_card = self._create_model_card(model_name, data)
            layout.addWidget(model_card)
        
        key_points = self._generate_key_points()
        points_title = QLabel("🔑 Puntos Clave para la Decisión")
        points_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        points_title.setStyleSheet("color: #e67e22; margin: 10px;")
        layout.addWidget(points_title)
        
        points_label = QLabel(key_points)
        points_label.setFont(QFont("Arial", 12))
        points_label.setWordWrap(True)
        points_label.setStyleSheet("background-color: #fef9e7; padding: 15px; border-radius: 5px; margin: 10px;")
        layout.addWidget(points_label)
        
        content_widget.setLayout(layout)
        scroll_area.setWidget(content_widget)
        
        main_layout.addWidget(scroll_area)
        
        close_btn = QPushButton("Cerrar Panel")
        close_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
        close_btn.clicked.connect(dialog.accept)
        main_layout.addWidget(close_btn)
        
        dialog.setLayout(main_layout)
        dialog.exec()
        
    def _create_model_card(self, model_name, data):
        
        card_frame = QFrame()
        card_frame.setFrameStyle(QFrame.Shape.Box)
        card_frame.setLineWidth(2)
        
        accuracy = data.get('accuracy') if data.get('accuracy') is not None else data['precision']
        balanced = data.get('balanced_accuracy') if data.get('balanced_accuracy') is not None else accuracy
        macro_f1 = data.get('macro_f1') if data.get('macro_f1') is not None else accuracy
        available = [accuracy, balanced, macro_f1]
        if data.get('fidelity') is not None and data.get('fidelity') > 0:
            available.append(data['fidelity'])
        avg_score = float(np.mean(available))
        bg_color = self._get_quality_color(avg_score)
        
        card_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: 10px;
                margin: 5px;
                padding: 10px;
            }}
        """)
        
        layout = QVBoxLayout()
        
        name_label = QLabel(f"🤖 {model_name}")
        name_label.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        name_label.setStyleSheet("color: white;")
        layout.addWidget(name_label)
        
        metrics_layout = QHBoxLayout()
        
        precision_label = QLabel(
            f"Precisão Macro: {data['precision']:.1f}%  |  Recall/BA: {balanced:.1f}%  |  Macro-F1: {macro_f1:.1f}%  |  Exatidão(auditoria): {accuracy:.1f}%"
        )
        precision_label.setFont(QFont("Arial", 12))
        precision_label.setStyleSheet("color: white;")
        
        if data.get('fidelity_to_active_oracle') is not None:
            fidelity_text = (
                f"Fidelidade ao oráculo ativo: {data['fidelity_to_active_oracle']:.1f}%"
            )
            if data.get('fidelity_to_mlp_original') is not None:
                fidelity_text += (
                    f"  |  Fidelidade de controlo ao MLP Original: "
                    f"{data['fidelity_to_mlp_original']:.1f}%"
                )
        else:
            fidelity_text = f"Fidelidade: {data['fidelity']:.1f}%"
        fidelity_label = QLabel(
            f"{fidelity_text}  |  Oráculo: {data.get('oracle') or 'n/a'}  |  "
            f"Espaço: {data.get('feature_space') or 'n/a'}"
        )
        fidelity_label.setFont(QFont("Arial", 12))
        fidelity_label.setStyleSheet("color: white;")
        
        metrics_layout.addWidget(precision_label)
        metrics_layout.addWidget(fidelity_label)
        
        layout.addLayout(metrics_layout)
        
        quality_indicator = self._create_quality_indicator(avg_score)
        layout.addWidget(quality_indicator)
        
        card_frame.setLayout(layout)
        return card_frame
        
    def _create_quality_indicator(self, score):
        indicator_layout = QHBoxLayout()
        
        if score >= 90:
            quality_text = "🌟 Excelente"
            color = "#27ae60"
        elif score >= 75:
            quality_text = "👍 Bueno"
            color = "#f39c12"
        else:
            quality_text = "⚠️ Necesita Mejorar"
            color = "#e74c3c"
        
        quality_label = QLabel(quality_text)
        quality_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        quality_label.setStyleSheet(f"color: {color};")
        indicator_layout.addWidget(quality_label)
        
        stars = "⭐" * min(5, int(score / 20))
        stars_label = QLabel(stars)
        stars_label.setFont(QFont("Arial", 14))
        indicator_layout.addWidget(stars_label)
        
        widget = QWidget()
        widget.setLayout(indicator_layout)
        return widget
        
    def _find_best_model(self):
        best_score = 0
        best_model = None
        
        for name, data in self.models_data.items():
            values = [
                data.get('accuracy'), data.get('balanced_accuracy'),
                data.get('macro_f1'),
            ]
            if data.get('fidelity', 0) > 0:
                values.append(data.get('fidelity'))
            values = [float(value) for value in values if value is not None]
            score = float(np.mean(values)) if values else 0.0
            
            if score > best_score:
                best_score = score
                best_model = {
                    'name': name,
                    'precision': data['precision'],
                    'fidelity': data['fidelity'],
                    'score': score,
                    'accuracy': data.get('accuracy') or 0,
                }
        
        return best_model
        
    def _generate_key_points(self):
        points = []
        
        precisions = [float(data['precision']) for data in self.models_data.values()]
        fidelities = [
            data.get('fidelity_to_mlp_original')
            for name, data in self.models_data.items()
            if 'trepan' in name.lower() and data.get('fidelity_to_mlp_original') is not None
        ]
        
        avg_precision = np.mean(precisions)
        avg_fidelity = np.mean(fidelities) if fidelities else float('nan')
        
        points.append(f"📈 Precisão Macro média geral: {avg_precision:.1f}%")
        points.append(f"🎯 Fidelidade média de controlo ao MLP Original (TREPAN): {avg_fidelity:.1f}%" if fidelities else "🎯 Fidelidade de controlo indisponível")
        points.append("📏 " + self._c45_gate_summary_text())
        
        if avg_precision < 80:
            points.append("⚠️ A Precisão Macro geral pode ser melhorada")
        
        if fidelities and avg_fidelity < 75:
            points.append("⚠️ La fidelidad general puede mejorarse")
        
        points.append("\n💡 Recomendaciones:")
        
        if 'Trepan-Reloaded' in self.models_data:
            trepan_data = self.models_data['Trepan-Reloaded']
            if trepan_data['fidelity'] > 85:
                points.append("✅ Trepan-Reloaded ofrece excelente fidelidad")
        
        if 'MLP' in self.models_data:
            mlp_data = self.models_data['MLP']
            if mlp_data['precision'] > 90:
                points.append("ℹ️ O MLP apresentou alta Precisão Macro nesta partição")
        
        points.append("🔄 Considere combinar modelos para obtener un mejor resultado")
        
        return "\n".join(points)
        
    def _show_data_distribution(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Distribución de Datos")
        dialog.setModal(True)
        dialog.setMinimumSize(560, 420); dialog.resize(800, 600)
        
        layout = QVBoxLayout()
        
        title = QLabel("📊 Distribución de Datos")
        title.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        if hasattr(self, 'data') and hasattr(self, 'meta') and self.data is not None and self.meta is not None:
            data_text = self._generate_real_data_distribution_text()
        else:
            data_text = """
⚠️ Datos no disponibles

Para visualizar la distribución de datos:
1. Cargue sus datos primero
2. Entrene un modelo o ejecute un análisis

Los datos se mostrarán automáticamente aquí después de la carga.
            """
        
        data_label = QLabel(data_text)
        data_label.setFont(QFont("Arial", 12))
        data_label.setWordWrap(True)
        data_label.setStyleSheet("background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin: 10px;")
        layout.addWidget(data_label)
        
        close_btn = QPushButton("Cerrar")
        close_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def _generate_real_data_distribution_text(self):
        
        import pandas as pd
        
        try:
            if isinstance(self.data, pd.DataFrame):
                df = self.data
            else:
                df = pd.DataFrame(self.data)
                if 'features' in self.meta and len(self.meta['features']) == df.shape[1]:
                    df.columns = self.meta['features']
            
            total_samples = len(df)
            num_features = len(df.columns)
            
            class_info = ""
            if 'target' in self.meta and self.meta['target'] in df.columns:
                target_col = self.meta['target']
                class_dist = df[target_col].value_counts()
                class_names = [str(c) for c in class_dist.index]
                class_info = f"• Clases: {len(class_names)} ({', '.join(class_names)})"
                
                dist_by_class = "\n📊 Distribución por Clase:\n"
                for cls, count in class_dist.items():
                    percentage = (count / total_samples) * 100
                    dist_by_class += f"• {cls}: {count} muestras ({percentage:.1f}%)\n"
            else:
                dist_by_class = ""
            
            feature_names = []
            if 'features' in self.meta:
                feature_names = self.meta['features']
            elif 'attributes' in self.meta:
                feature_names = [attr['name'] for attr in self.meta['attributes'] 
                               if attr.get('name') != self.meta.get('target')]
            else:
                feature_names = [col for col in df.columns 
                               if col != self.meta.get('target', '')]
            
            features_text = "\n📈 Distribución de Características:\n"
            numeric_features = []
            
            for feature_name in feature_names:
                if feature_name in df.columns:
                    feature_data = df[feature_name]
                    
                    if pd.api.types.is_numeric_dtype(feature_data):
                        numeric_features.append(feature_name)
                        mean_val = feature_data.mean()
                        std_val = feature_data.std()
                        min_val = feature_data.min()
                        max_val = feature_data.max()
                        features_text += f"• {feature_name}: Media={mean_val:.2f}, Desvío={std_val:.2f}, Rango=[{min_val:.2f}, {max_val:.2f}]\n"
                    else:
                        unique_count = feature_data.nunique()
                        features_text += f"• {feature_name}: {unique_count} valores únicos (categórico)\n"
            
            insights = []
            if len(df) > 0:
                insights.append(f"Total de {total_samples} muestras")
                if dist_by_class:
                    if 'target' in self.meta and self.meta['target'] in df.columns:
                        target_col = self.meta['target']
                        class_counts = df[target_col].value_counts()
                        max_count = class_counts.max()
                        min_count = class_counts.min()
                        if min_count > 0:
                            imbalance_ratio = max_count / min_count
                            if imbalance_ratio < 2:
                                insights.append("Los datos están bien balanceados entre las clases")
                            elif imbalance_ratio < 5:
                                insights.append("Los datos tienen desbalance moderado")
                            else:
                                insights.append("Los datos tienen desbalance significativo")
                
                missing_count = df.isnull().sum().sum()
                if missing_count == 0:
                    insights.append("No hay valores ausentes significativos")
                else:
                    insights.append(f"Existen {missing_count} valores ausentes en el conjunto de datos")
                
                if numeric_features:
                    stds = [df[feat].std() for feat in numeric_features if feat in df.columns]
                    if stds:
                        max_var_feature_idx = np.argmax(stds)
                        if max_var_feature_idx < len(numeric_features):
                            insights.append(f"{numeric_features[max_var_feature_idx]} tiene mayor variabilidad")
                        insights.append("Las características son numéricamente estables")
            
            insights_text = "\n💡 Observaciones:\n"
            for insight in insights:
                insights_text += f"• {insight}\n"
            
            data_text = f"""
📊 Análisis de la Distribución de Datos

📈 Características Principales:
• Total de muestras: {total_samples}
• Número de características: {num_features}
{class_info}

{dist_by_class}{features_text}
{insights_text}
            """
            
            return data_text.strip()
            
        except Exception as e:
            return f"""
⚠️ Error al generar la distribución de datos

Detalles del error: {str(e)}

Verifique que los datos estén en el formato correcto.
            """
        
    def set_data_info(self, data, meta):
        self.data = data
        self.meta = meta
        
    def _show_no_data_message(self):    
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "⚠️ Sin Datos", 
                           "Ningún modelo ha sido analizado todavía.\n\n"
                           "Ejecute el análisis primero para visualizar las métricas.")
