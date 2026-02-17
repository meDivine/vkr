import sys
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from skimage.color import rgb2lab, lab2rgb, deltaE_cie76
from skimage.io import imread
from skimage import img_as_float
from scipy.interpolate import RegularGridInterpolator


# ----------------- Highlighter -----------------
class PythonHighlighter(QSyntaxHighlighter):
    def __init__(self, doc):
        super().__init__(doc)
        self.rules = []
        fmt = QTextCharFormat()
        fmt.setForeground(Qt.GlobalColor.blue)
        for kw in ["np", "lab", "sin", "cos"]:
            self.rules.append((QRegularExpression(rf"\b{kw}\b"), fmt))

    def highlightBlock(self, text):
        for pattern, fmt in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), fmt)


# ----------------- Worker Thread -----------------
class LUTWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object, dict)
    error = pyqtSignal(str)

    def __init__(self, N, code, clip, use_gpu=False):
        super().__init__()
        self.N = N
        self.code = code
        self.clip = clip
        self.use_gpu = use_gpu

    def run(self):
        try:
            # Здесь можно добавить поддержку GPU в будущем
            if self.use_gpu:
                print("Поддержка GPU пока не реализована! Используйте CPU")

            # Генерация сетки значений
            grid = np.linspace(0, 1, self.N)
            r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
            rgb_nodes = np.stack([r, g, b], -1).reshape(-1, 3)

            # Конвертация в Lab
            lab_nodes = rgb2lab(rgb_nodes.reshape(-1, 1, 3)).reshape(-1, 3)

            # Применение пользовательской формулы
            env = {"np": np, "lab": lab_nodes}
            exec(self.code, {"__builtins__": {}}, env)
            lab_nodes = env["lab"]

            if self.clip:
                lab_nodes[:, 0] = np.clip(lab_nodes[:, 0], 0, 100)
                lab_nodes[:, 1:] = np.clip(lab_nodes[:, 1:], -128, 127)

            # Обратная конвертация в RGB он же LUT
            rgb_out_nodes = lab2rgb(lab_nodes.reshape(-1, 1, 3)).reshape(-1, 3)
            rgb_out_nodes = np.clip(rgb_out_nodes, 0, 1)
            lut = rgb_out_nodes.reshape(self.N, self.N, self.N, 3)

            metrics = {}

            # Создаем интерполятор по сгенерированному LUT
            interp = RegularGridInterpolator((grid, grid, grid), lut)

            # Генерируем случайные тестовые точки
            np.random.seed(42)
            n_samples = 10000
            test_rgb = np.random.rand(n_samples, 3)

            # Прямое преобразование
            test_lab = rgb2lab(test_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
            env_test = {"np": np, "lab": test_lab}
            exec(self.code, {"__builtins__": {}}, env_test)
            test_lab_transformed = env_test["lab"]
            if self.clip:
                test_lab_transformed[:, 0] = np.clip(test_lab_transformed[:, 0], 0, 100)
                test_lab_transformed[:, 1:] = np.clip(test_lab_transformed[:, 1:], -128, 127)

            # Преобразование через LUT
            test_rgb_lut = interp(test_rgb)

            # Сравнение в пространстве Lab
            test_lab_lut = rgb2lab(test_rgb_lut.reshape(-1, 1, 3)).reshape(-1, 3)

            # Рассчитываем ΔE
            delta_e = deltaE_cie76(test_lab_transformed, test_lab_lut)

            metrics['Mean Delta E'] = np.mean(delta_e)
            metrics['Max Delta E'] = np.max(delta_e)

            # Сравнение в пространстве RGB
            test_rgb_truth = lab2rgb(test_lab_transformed.reshape(-1, 1, 3)).reshape(-1, 3)

            mse = np.mean((test_rgb_truth - test_rgb_lut) ** 2)
            if mse == 0:
                metrics['PSNR (dB)'] = float('inf')
            else:
                metrics['PSNR (dB)'] = 20 * np.log10(1.0 / np.sqrt(mse))

            self.finished.emit(lut, metrics)

        except Exception as e:
            import traceback
            self.error.emit(str(traceback.format_exc()))


# ----------------- Main Window -----------------
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RGB <-> Lab LUT Generator")
        self.resize(1000, 750)

        self.lut = None
        self.image = None

        self.status = self.statusBar()
        self.progress = QProgressBar()
        self.status.addPermanentWidget(self.progress)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.init_function_tab()
        self.init_preview_tab()

    # -------- TAB 1 --------
    def init_function_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(5)

        # Верхняя панель с настройками
        top = QHBoxLayout()
        top.setSpacing(5)

        top.addWidget(QLabel("Размер LUT:"))

        # Инпут ввода размера лута
        self.lut_size_spin = QSpinBox()
        self.lut_size_spin.setRange(2, 256)
        self.lut_size_spin.setValue(33)
        self.lut_size_spin.setFixedWidth(100)  # Увеличил ширину
        self.lut_size_spin.setToolTip(
            "Размер сетки LUT. Можно ввести число от 2 до 256.\n"
            "Рекомендации:\n"
            "• 17  - быстро, низкое качество\n"
            "• 33  - стандарт, хороший баланс\n"
            "• 65  - высокое качество\n"
            "• 129 - максимальное качество (может быть медленно)"
        )
        # Активируем ручной ввод
        self.lut_size_spin.setKeyboardTracking(True)
        self.lut_size_spin.setAccelerated(True)

        top.addWidget(self.lut_size_spin)

        self.illuminant = QLineEdit("D65")
        self.illuminant.textChanged.connect(self.validate_illuminant)
        self.illuminant.setFixedWidth(50)
        top.addWidget(QLabel("Ill:"))
        top.addWidget(self.illuminant)

        self.clip_cb = QCheckBox("Клиппинг")
        self.clip_cb.setChecked(True)
        top.addWidget(self.clip_cb)

        self.gpu_cb = QCheckBox("GPU")
        self.gpu_cb.setEnabled(True)
        self.gpu_cb.setToolTip("Использовать GPU для ускорения (в разработке)")
        top.addWidget(self.gpu_cb)

        top.addStretch()

        layout.addLayout(top)

        # Добавляем подсказку для размера
        size_hint = QLabel("Рекомендуемые значения: 17 (быстро), 33 (стандарт), 65 (качество), 129 (макс. качество)")
        size_hint.setStyleSheet("color: gray; font-size: 9pt;")
        size_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(size_hint)

        # Компактное окно ввода формулы
        formula_group = QGroupBox("Формула преобразования (работает с переменной 'lab')")
        formula_layout = QVBoxLayout()
        formula_layout.setSpacing(3)

        # поле ввода
        self.editor = QTextEdit()
        self.editor.setPlainText("lab[:,0] *= 1.2  # Увеличить яркость")
        self.editor.setMaximumHeight(80)
        self.editor.setFont(QFont("Courier New", 10))
        PythonHighlighter(self.editor.document())
        formula_layout.addWidget(self.editor)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(3)

        check_btn = QPushButton("Проверить")
        check_btn.setFixedHeight(25)
        check_btn.clicked.connect(self.check_syntax)
        btn_layout.addWidget(check_btn)

        gen_btn = QPushButton("Сгенерировать LUT")
        gen_btn.setFixedHeight(25)
        gen_btn.clicked.connect(self.generate_lut)
        btn_layout.addWidget(gen_btn)

        btn_layout.addStretch()
        formula_layout.addLayout(btn_layout)

        formula_group.setLayout(formula_layout)
        layout.addWidget(formula_group)

        # Поле для метрик
        metrics_group = QGroupBox("Метрики качества LUT")
        metrics_layout = QVBoxLayout()
        metrics_layout.setSpacing(3)

        self.metrics_label = QLabel("Метрики появятся после генерации...")
        self.metrics_label.setStyleSheet("padding: 5px; border: 1px solid #ccc;")
        self.metrics_label.setWordWrap(True)
        self.metrics_label.setMinimumHeight(60)
        self.metrics_label.setMaximumHeight(80)
        metrics_layout.addWidget(self.metrics_label)

        metrics_group.setLayout(metrics_layout)
        layout.addWidget(metrics_group)

        layout.addStretch()

        self.tabs.addTab(tab, "Функция")

    def init_preview_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        img_row = QHBoxLayout()
        self.orig_label = QLabel("Оригинал")
        self.result_label = QLabel("После LUT")
        for lbl in [self.orig_label, self.result_label]:
            lbl.setFixedSize(400, 300)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: black;")
        img_row.addWidget(self.orig_label)
        img_row.addWidget(self.result_label)
        layout.addLayout(img_row)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        btn_row.addWidget(load_btn)
        apply_btn = QPushButton("Применить LUT")
        apply_btn.clicked.connect(self.apply_lut)
        btn_row.addWidget(apply_btn)
        layout.addLayout(btn_row)

        self.tabs.addTab(tab, "Предпросмотр")

    def generate_lut(self):
        self.progress.setValue(0)
        self.metrics_label.setText("Расчет...")

        N = self.lut_size_spin.value()
        use_gpu = self.gpu_cb.isChecked()

        self.worker = LUTWorker(
            N,
            self.editor.toPlainText(),
            self.clip_cb.isChecked(),
            use_gpu
        )
        self.worker.finished.connect(self.lut_ready)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def lut_ready(self, lut, metrics):
        self.lut = lut
        self.progress.setValue(100)

        # Формируем текст метрик
        text = f"<b>Размер LUT: {self.lut_size_spin.value()}³</b><br>"
        text += f"Средняя ΔE: <b>{metrics['Mean Delta E']:.4f}</b><br>"
        text += f"Макс. ΔE: <b>{metrics['Max Delta E']:.4f}</b><br>"
        text += f"PSNR: <b>{metrics['PSNR (dB)']:.2f} dB</b>"

        self.metrics_label.setText(text)
        self.status.showMessage("LUT сгенерирован", 3000)

    def on_error(self, e):
        QMessageBox.critical(self, "Ошибка", str(e))
        self.metrics_label.setText("Ошибка при расчете.")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Images (*.png *.jpg *.jpeg *.tiff)")
        if path:
            self.image = img_as_float(imread(path))
            if self.image.shape[-1] == 4:
                self.image = self.image[..., :3]
            self.show_img(self.image, self.orig_label)

    def apply_lut(self):
        if self.lut is None or self.image is None:
            QMessageBox.warning(self, "Ошибка", "Сначала сгенерируйте LUT и загрузите изображение")
            return

        grid = np.linspace(0, 1, self.lut.shape[0])
        interp = RegularGridInterpolator((grid, grid, grid), self.lut)
        h, w, _ = self.image.shape

        # Применяем LUT
        result = interp(self.image.reshape(-1, 3))
        result = np.clip(result.reshape(h, w, 3), 0, 1)
        self.show_img(result, self.result_label)

    def show_img(self, img, label):
        img8 = (img * 255).astype(np.uint8)
        h, w, _ = img8.shape
        qimg = QImage(img8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))

    def validate_illuminant(self, text):
        if text not in ["A", "B", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11"]:
            self.illuminant.setStyleSheet("border: 1px solid red;")
        else:
            self.illuminant.setStyleSheet("")

    def check_syntax(self):
        try:
            env = {"np": np, "lab": np.zeros((10, 3))}
            compile(self.editor.toPlainText(), '<string>', 'exec')
            QMessageBox.information(self, "Проверка синтаксиса", "Синтаксис корректен!")
        except SyntaxError as e:
            QMessageBox.critical(self, "Ошибка синтаксиса", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())