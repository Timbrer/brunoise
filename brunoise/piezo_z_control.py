from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDoubleSpinBox, QLabel, QSlider, QVBoxLayout, QWidget

PIEZO_MAX_UM = 450.0


class PiezoZControl(QWidget):
    slider_scale = 10

    def __init__(self, state):
        super().__init__()
        self.state = state
        self.setLayout(QVBoxLayout())

        self.label = QLabel("Piezo z position (um)")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, int(PIEZO_MAX_UM * self.slider_scale))
        self.slider.setSingleStep(1)
        self.slider.setPageStep(10)

        self.spin_box = QDoubleSpinBox()
        self.spin_box.setRange(0.0, PIEZO_MAX_UM)
        self.spin_box.setDecimals(1)
        self.spin_box.setSingleStep(0.1)
        self.spin_box.setSuffix(" um")

        self.layout().addWidget(self.label)
        self.layout().addWidget(self.slider)
        self.layout().addWidget(self.spin_box)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.spin_box.valueChanged.connect(self._on_spin_box_changed)
        self.sync_state()

    def _set_state_value(self, value_um):
        self.state.set_piezo_z_um(value_um)
        self.sync_state()

    def _on_slider_changed(self, slider_value):
        self._set_state_value(slider_value / self.slider_scale)

    def _on_spin_box_changed(self, value_um):
        self._set_state_value(value_um)

    def sync_state(self):
        value_um = self.state.piezo_z_um
        slider_value = int(round(value_um * self.slider_scale))
        if self.slider.value() != slider_value:
            was_blocked = self.slider.blockSignals(True)
            self.slider.setValue(slider_value)
            self.slider.blockSignals(was_blocked)
        if self.spin_box.value() != value_um:
            was_blocked = self.spin_box.blockSignals(True)
            self.spin_box.setValue(value_um)
            self.spin_box.blockSignals(was_blocked)

        controls_enabled = not (self.state.saving or self.state.paused)
        self.slider.setEnabled(controls_enabled)
        self.spin_box.setEnabled(controls_enabled)
