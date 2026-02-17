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

            # ------------------ Расчет метрик ------------------
            metrics = {}
            interp = RegularGridInterpolator((grid, grid, grid), lut)

            np.random.seed(42)
            n_samples = 10000
            test_rgb = np.random.rand(n_samples, 3)

            # Эталонное преобразование
            test_lab = rgb2lab(test_rgb.reshape(-1, 1, 3)).reshape(-1, 3)
            env_test = {"np": np, "lab": test_lab}
            exec(self.code, {"__builtins__": {}}, env_test)
            test_lab_transformed = env_test["lab"]
            if self.clip:
                test_lab_transformed[:, 0] = np.clip(test_lab_transformed[:, 0], 0, 100)
                test_lab_transformed[:, 1:] = np.clip(test_lab_transformed[:, 1:], -128, 127)

            # Преобразование через LUT
            test_rgb_lut = interp(test_rgb)

            # Сравнение
            test_lab_lut = rgb2lab(test_rgb_lut.reshape(-1, 1, 3)).reshape(-1, 3)
            delta_e = deltaE_cie76(test_lab_transformed, test_lab_lut)

            metrics['Mean Delta E'] = np.mean(delta_e)
            metrics['Max Delta E'] = np.max(delta_e)

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
        self.resize(1000, 900)  # Немного увеличил высоту для одной страницы

        self.lut = None
        self.image = None

        self.status = self.statusBar()
        self.progress = QProgressBar()
        self.status.addPermanentWidget(self.progress)

        # Центральный виджет и главный Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)

        # ================= БЛОК 1: НАСТРОЙКИ =================
        settings_group = QGroupBox("Настройки генерации")
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(10)

        # Размер LUT
        settings_layout.addWidget(QLabel("Размер LUT:"))
        self.lut_size_spin = QSpinBox()
        self.lut_size_spin.setRange(2, 256)
        self.lut_size_spin.setValue(33)
        self.lut_size_spin.setFixedWidth(120)
        self.lut_size_spin.setToolTip("Рекомендуется: 17, 33, 65")
        settings_layout.addWidget(self.lut_size_spin)

        # Иллюминант
        settings_layout.addWidget(QLabel("Ill:"))
        self.illuminant = QLineEdit("D65")
        self.illuminant.setFixedWidth(50)
        self.illuminant.textChanged.connect(self.validate_illuminant)
        settings_layout.addWidget(self.illuminant)

        # Клиппинг
        self.clip_cb = QCheckBox("Клиппинг гамута")
        self.clip_cb.setChecked(True)
        settings_layout.addWidget(self.clip_cb)

        # GPU
        self.gpu_cb = QCheckBox("GPU")
        self.gpu_cb.setToolTip("В разработке")
        settings_layout.addWidget(self.gpu_cb)

        settings_layout.addStretch()
        settings_group.setLayout(settings_layout)
        main_layout.addWidget(settings_group)

        # ================= БЛОК 2: ФОРМУЛА =================
        formula_group = QGroupBox("Формула преобразования (Python код)")
        formula_layout = QVBoxLayout()

        self.editor = QTextEdit()
        self.editor.setPlainText("lab[:,0] *= 1.2  # Увеличить яркость")
        self.editor.setMaximumHeight(80)
        self.editor.setFont(QFont("Courier New", 10))
        PythonHighlighter(self.editor.document())
        formula_layout.addWidget(self.editor)

        # Кнопки управления формулой
        btn_formula_layout = QHBoxLayout()
        check_btn = QPushButton("Проверить синтаксис")
        check_btn.clicked.connect(self.check_syntax)
        btn_formula_layout.addWidget(check_btn)

        gen_btn = QPushButton("Сгенерировать LUT")
        gen_btn.setStyleSheet(" font-weight: bold;")
        gen_btn.clicked.connect(self.generate_lut)
        btn_formula_layout.addWidget(gen_btn)

        btn_formula_layout.addStretch()
        formula_layout.addLayout(btn_formula_layout)

        formula_group.setLayout(formula_layout)
        main_layout.addWidget(formula_group)

        # ================= БЛОК 3: МЕТРИКИ =================
        self.metrics_label = QLabel("Метрики качества: ожидание генерации...")
        self.metrics_label.setStyleSheet(
            "padding: 8px; border: 1px solid #dee2e6; border-radius: 4px;")
        self.metrics_label.setWordWrap(True)
        main_layout.addWidget(self.metrics_label)

        # ================= БЛОК 4: ПРЕДПРОСМОТР =================
        preview_group = QGroupBox("Предпросмотр")
        preview_layout = QVBoxLayout()

        # Ряд с картинками
        images_row = QHBoxLayout()
        self.orig_label = QLabel("Оригинал")
        self.result_label = QLabel("Результат LUT")
        for lbl in [self.orig_label, self.result_label]:
            lbl.setFixedSize(450, 300)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #333333; color: white; border: 1px solid #555;")
            lbl.setScaledContents(False)
        images_row.addWidget(self.orig_label)
        images_row.addWidget(self.result_label)
        preview_layout.addLayout(images_row)

        # Ряд кнопок предпросмотра
        btn_preview_row = QHBoxLayout()
        load_btn = QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        btn_preview_row.addWidget(load_btn)

        apply_btn = QPushButton("Применить LUT к изображению")
        apply_btn.clicked.connect(self.apply_lut)
        btn_preview_row.addWidget(apply_btn)

        export_btn = QPushButton("Экспорт .cube")
        export_btn.clicked.connect(self.export_lut)
        btn_preview_row.addWidget(export_btn)

        btn_preview_row.addStretch()
        preview_layout.addLayout(btn_preview_row)

        preview_group.setLayout(preview_layout)
        main_layout.addWidget(preview_group, stretch=1)  # stretch=1 заставляет этот блок занимать все оставшееся место

    # ----------------- Логика -----------------
    def validate_illuminant(self, text):
        valid = text in ["A", "B", "C", "D50", "D55", "D65", "D75", "E", "F2", "F7", "F11"]
        self.illuminant.setStyleSheet("border: 1px solid red;" if not valid else "")

    def check_syntax(self):
        try:
            code = compile(self.editor.toPlainText(), '<string>', 'exec')
            # Простая проверка переменных
            env = {"np": np, "lab": np.zeros((10, 3))}
            exec(code, {"__builtins__": {}}, env)
            QMessageBox.information(self, "Успех", "Синтаксис корректен.")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка синтаксиса", str(e))

    def generate_lut(self):
        self.progress.setValue(0)
        self.metrics_label.setText("Генерация и расчет метрик...")

        N = self.lut_size_spin.value()
        use_gpu = self.gpu_cb.isChecked()

        self.worker = LUTWorker(N, self.editor.toPlainText(), self.clip_cb.isChecked(), use_gpu)
        self.worker.finished.connect(self.lut_ready)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def lut_ready(self, lut, metrics):
        self.lut = lut
        self.progress.setValue(100)

        text = (f"<b>Метрики (LUT {self.lut_size_spin.value()}³):</b> &nbsp;&nbsp;&nbsp;"
                f"Средняя ΔE: <b>{metrics['Mean Delta E']:.4f}</b> &nbsp;&nbsp;&nbsp;"
                f"Макс. ΔE: <b>{metrics['Max Delta E']:.4f}</b> &nbsp;&nbsp;&nbsp;"
                f"PSNR: <b>{metrics['PSNR (dB)']:.2f} dB</b>")

        self.metrics_label.setText(text)
        self.status.showMessage("LUT успешно сгенерирован", 3000)

    def on_error(self, e):
        QMessageBox.critical(self, "Ошибка генерации", str(e))
        self.metrics_label.setText(f"<b style='color:red'>Ошибка:</b> {e}")
        self.progress.setValue(0)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Открыть изображение", "", "Images (*.png *.jpg *.jpeg *.tiff)")
        if path:
            self.image = img_as_float(imread(path))
            if self.image.shape[-1] == 4:
                self.image = self.image[..., :3]
            self.show_img(self.image, self.orig_label)
            self.result_label.clear()
            self.result_label.setText("Нажмите 'Применить LUT'")

    def apply_lut(self):
        if self.lut is None:
            QMessageBox.warning(self, "Внимание", "Сначала сгенерируйте LUT.")
            return
        if self.image is None:
            QMessageBox.warning(self, "Внимание", "Сначала загрузите изображение.")
            return

        # Применение
        grid = np.linspace(0, 1, self.lut.shape[0])
        interp = RegularGridInterpolator((grid, grid, grid), self.lut)
        h, w, _ = self.image.shape

        # Обработка изображений с 4 каналами RGBA или серых
        if len(self.image.shape) == 2:
            self.image = np.stack([self.image] * 3, axis=-1)

        result = interp(self.image.reshape(-1, 3))
        result = np.clip(result.reshape(h, w, 3), 0, 1)
        self.show_img(result, self.result_label)

    def export_lut(self):
        if self.lut is None:
            QMessageBox.warning(self, "Внимание", "Нечего экспортировать. Сгенерируйте LUT.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Сохранить LUT", "lut.cube", "Cube LUT (*.cube)")
        if path:
            with open(path, "w") as f:
                N = self.lut.shape[0]
                f.write(f"LUT_3D_SIZE {N}\n")
                for r in range(N):
                    for g in range(N):
                        for b in range(N):
                            f.write("{:.6f} {:.6f} {:.6f}\n".format(*self.lut[r, g, b]))
            self.status.showMessage(f"LUT сохранен в {path}", 5000)

    def show_img(self, img, label):
        if img.dtype.kind == 'f':
            data = (img * 255).astype(np.uint8)
        else:
            data = img

        h, w, _ = data.shape

        qimg = QImage(data.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())