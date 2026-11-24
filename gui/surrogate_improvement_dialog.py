"""Configuração da melhoria contrafactual de árvores substitutas."""
from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
)


class SurrogateImprovementDialog(QDialog):
    """Recolhe apenas escolhas que não alteram a partição de teste."""

    def __init__(self, *, ontology_active: bool, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Melhorar árvore substituta com contrafactuais")
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        intro = QLabel(
            "Cria uma árvore candidata e compara: árvore atual × reextração sem CF "
            "× reextração com CF. A configuração é escolhida apenas na validação "
            "interna; o teste final permanece bloqueado."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.addItem("Protocolo BIURI (recomendado)", "biuri")
        self.profile_combo.addItem("Método da tese (González Pérez, 2026)", "tese")
        self.profile_combo.setToolTip(
            "Tese: validação cruzada de 2 partições a maximizar a fidelidade, grelha max_cfs {5, 10} × tipos "
            "× fator de fronteira, reforço adaptativo dos divergentes, ≥ 50 âncoras. Os CFs de cada partição "
            "vêm só das linhas de ajuste dessa partição, e o gate de aceitação do BIURI mantém-se."
        )
        form.addRow("Perfil do método", self.profile_combo)

        self.target_combo = QComboBox()
        self.target_combo.addItem("TREPAN Original e Reloaded", "both")
        self.target_combo.addItem("Somente TREPAN Original", "trepan_original")
        self.target_combo.addItem("Somente TREPAN Reloaded", "trepan_reloaded")
        form.addRow("Árvore substituta", self.target_combo)

        self.method_combo = QComboBox()
        self.method_combo.addItem("COGS", "COGS")
        self.method_combo.addItem("CLEAR", "CLEAR")
        self.method_combo.addItem("CLEAR + COGS", "COMBINED")
        form.addRow("Gerador contrafactual", self.method_combo)

        self.max_origins_spin = QSpinBox()
        self.max_origins_spin.setRange(4, 100)
        self.max_origins_spin.setValue(24)
        form.addRow("Máximo de instâncias factuais", self.max_origins_spin)

        self.cfs_per_origin_spin = QSpinBox()
        self.cfs_per_origin_spin.setRange(1, 10)
        self.cfs_per_origin_spin.setValue(3)
        form.addRow("CFs por instância e método", self.cfs_per_origin_spin)

        self.max_cfs_spin = QSpinBox()
        self.max_cfs_spin.setRange(4, 100)
        self.max_cfs_spin.setValue(30)
        form.addRow("Máximo de CFs no treino", self.max_cfs_spin)

        self.sample_size_spin = QSpinBox()
        self.sample_size_spin.setRange(400, 5000)
        self.sample_size_spin.setSingleStep(200)
        self.sample_size_spin.setValue(1200)
        form.addRow("Orçamento sintético por candidata", self.sample_size_spin)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.50, 0.99)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(0.60)
        form.addRow("Confiança mínima do professor", self.confidence_spin)

        self.density_spin = QDoubleSpinBox()
        self.density_spin.setRange(80.0, 99.9)
        self.density_spin.setSingleStep(1.0)
        self.density_spin.setDecimals(1)
        self.density_spin.setValue(95.0)
        form.addRow("Percentil de densidade", self.density_spin)

        self.strict_density = QCheckBox("Rejeitar CFs fora do manifold empírico")
        self.strict_density.setChecked(True)
        form.addRow("Gate de plausibilidade", self.strict_density)
        root.addLayout(form)
        self.profile_combo.currentIndexChanged.connect(self._apply_profile)

        note = QLabel(
            "O Reloaded usa o professor e o espaço ontológico apenas quando a OWL "
            "passou nos quality gates. Sem OWL aceite, ele espelha o TREPAN Original. "
            "C4.5 não aparece porque não é um substituto do MLP."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #5f6368; padding: 8px 0;")
        if not ontology_active:
            note.setText(
                note.text() + " Nesta sessão não há enriquecimento OWL aceite."
            )
        root.addWidget(note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Executar experimento")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

    def _apply_profile(self):
        thesis = self.profile_combo.currentData() == "tese"
        # A grelha de max_cfs da tese é fixa ({5, 10}); a amostra sintética do extrator na tese é 2000.
        self.max_cfs_spin.setEnabled(not thesis)
        if thesis:
            self.sample_size_spin.setValue(2000)
            self.confidence_spin.setValue(0.60)
            self.density_spin.setValue(95.0)
            self.strict_density.setChecked(True)

    def options(self):
        if self.profile_combo.currentData() == "tese":
            return {
                "profile": "tese",
                "target_model": self.target_combo.currentData(),
                "counterfactual_method": self.method_combo.currentData(),
                "max_origins": int(self.max_origins_spin.value()),
                "cfs_per_origin": int(self.cfs_per_origin_spin.value()),
                "sample_size": int(self.sample_size_spin.value()),
                "confidence_threshold": float(self.confidence_spin.value()),
                "density_percentile": float(self.density_spin.value()),
                "strict_density_gate": bool(self.strict_density.isChecked()),
            }
        maximum = int(self.max_cfs_spin.value())
        max_grid = tuple(sorted({min(8, maximum), min(15, maximum), maximum}))
        return {
            "target_model": self.target_combo.currentData(),
            "counterfactual_method": self.method_combo.currentData(),
            "max_origins": int(self.max_origins_spin.value()),
            "cfs_per_origin": int(self.cfs_per_origin_spin.value()),
            "max_cfs_grid": max_grid,
            "sample_size": int(self.sample_size_spin.value()),
            "confidence_threshold": float(self.confidence_spin.value()),
            "density_percentile": float(self.density_spin.value()),
            "strict_density_gate": bool(self.strict_density.isChecked()),
        }


__all__ = ["SurrogateImprovementDialog"]
