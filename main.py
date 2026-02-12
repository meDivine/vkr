import sys
import numpy as np
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
from PyQt6.QtCore import *
from skimage.color import rgb2lab, lab2rgb
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
                self.setFormat(match.capturedStart(),
                               match.capturedLength(), fmt)


# ----------------- Worker Thread -----------------
class LUTWorker(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, N, code, clip):
        super().__init__()
        self.N = N
        self.code = code
        self.clip = clip

    def run(self):
        try:
            grid = np.linspace(0, 1, self.N)
            r, g, b = np.meshgrid(grid, grid, grid, indexing="ij")
            rgb = np.stack([r, g, b], -1).reshape(-1, 3)

            lab = rgb2lab(rgb.reshape(-1, 1, 3)).reshape(-1, 3)

            env = {"np": np, "lab": lab}
            exec(self.code, {"__builtins__": {}}, env)
            lab = env["lab"]

            if self.clip:
                lab[:, 0] = np.clip(lab[:, 0], 0, 100)
                lab[:, 1:] = np.clip(lab[:, 1:], -128, 127)

            rgb_out = lab2rgb(lab.reshape(-1, 1, 3)).reshape(-1, 3)
            rgb_out = np.clip(rgb_out, 0, 1)

            lut = rgb_out.reshape(self.N, self.N, self.N, 3)
            self.finished.emit(lut)

        except Exception as e:
            self.error.emit(str(e))


# ----------------- Main Window -----------------
class App(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RGB↔Lab LUT Generator")
        self.resize(1000, 700)

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

        top = QHBoxLayout()
        self.illuminant = QLineEdit("D65")
        self.illuminant.textChanged.connect(self.validate_illuminant)
        top.addWidget(QLabel("Illuminant:"))
        top.addWidget(self.illuminant)

        self.clip_cb = QCheckBox("Включить клиппинг гамута")
        self.clip_cb.setChecked(True)
        top.addWidget(self.clip_cb)

        self.gpu_cb = QCheckBox("Использовать GPU")
        top.addWidget(self.gpu_cb)

        layout.addLayout(top)

        self.editor = QTextEdit("lab[:,0] *= 1.2")
        PythonHighlighter(self.editor.document())
        layout.addWidget(self.editor)

        btn_row = QHBoxLayout()

        check_btn = QPushButton("Проверить синтаксис")
        check_btn.clicked.connect(self.check_syntax)
        btn_row.addWidget(check_btn)

        gen_btn = QPushButton("Сгенерировать LUT")
        gen_btn.clicked.connect(self.generate_lut)
        btn_row.addWidget(gen_btn)

        preset_menu = QMenu()
        preset_menu.addAction("Увеличить контраст",
                              lambda: self.editor.setText("lab[:,0]=(lab[:,0]-50)*1.2+50"))
        preset_menu.addAction("Теплый тон",
                              lambda: self.editor.setText("lab[:,1]+=10; lab[:,2]-=5"))
        preset_menu.addAction("Насыщенность",
                              lambda: self.editor.setText(
                                  "chroma=np.sqrt(lab[:,1]**2+lab[:,2]**2); lab[:,1:] *= (1+0.2*np.sin(lab[:,0]/100))"))

        preset_btn = QPushButton("Пресеты")
        preset_btn.setMenu(preset_menu)
        btn_row.addWidget(preset_btn)

        layout.addLayout(btn_row)
        self.tabs.addTab(tab, "Функция")

    # -------- TAB 2 --------
    def init_preview_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        img_row = QHBoxLayout()

        self.orig_label = QLabel("Оригинал")
        self.result_label = QLabel("После LUT")

        for lbl in [self.orig_label, self.result_label]:
            lbl.setFixedSize(400, 300)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        img_row.addWidget(self.orig_label)
        img_row.addWidget(self.result_label)

        layout.addLayout(img_row)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Загрузить изображение")
        load_btn.clicked.connect(self.load_image)
        btn_row.addWidget(load_btn)

        apply_btn = QPushButton("Применить")
        apply_btn.clicked.connect(self.apply_lut)
        btn_row.addWidget(apply_btn)

        export_btn = QPushButton("Экспорт")
        export_btn.clicked.connect(self.export_lut)
        btn_row.addWidget(export_btn)

        layout.addLayout(btn_row)
        self.tabs.addTab(tab, "Предпросмотр")

    # -------- Логика --------
    def validate_illuminant(self):
        if self.illuminant.text() not in ["D65", "D50"]:
            self.status.showMessage("Допустимо: D65 или D50", 3000)

    def check_syntax(self):
        try:
            lab = np.zeros((10, 3))
            exec(self.editor.toPlainText(),
                 {"__builtins__": {}, "np": np},
                 {"lab": lab})
            self.status.showMessage("Синтаксис корректен", 3000)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    def generate_lut(self):
        self.progress.setValue(0)
        self.worker = LUTWorker(
            33,
            self.editor.toPlainText(),
            self.clip_cb.isChecked()
        )
        self.worker.finished.connect(self.lut_ready)
        self.worker.error.connect(lambda e: QMessageBox.critical(self, "Ошибка", e))
        self.worker.start()

    def lut_ready(self, lut):
        self.lut = lut
        self.progress.setValue(100)
        self.status.showMessage("LUT сгенерирован", 3000)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open", "", "Images (*.png *.jpg *.jpeg *.tiff)")
        if path:
            self.image = img_as_float(imread(path))
            self.show_img(self.image, self.orig_label)

    def apply_lut(self):
        if self.lut is None or self.image is None:
            QMessageBox.warning(self, "Ошибка", "Нет LUT или изображения")
            return

        grid = np.linspace(0, 1, self.lut.shape[0])
        interp = RegularGridInterpolator((grid, grid, grid), self.lut)
        h, w, _ = self.image.shape
        result = interp(self.image.reshape(-1, 3))
        result = np.clip(result.reshape(h, w, 3), 0, 1)
        self.show_img(result, self.result_label)

    def export_lut(self):
        if self.lut is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save LUT", "", "Cube (*.cube);;NumPy (*.npy)")
        if path.endswith(".npy"):
            np.save(path, self.lut)
        else:
            with open(path, "w") as f:
                N = self.lut.shape[0]
                f.write(f"LUT_3D_SIZE {N}\n")
                for r in range(N):
                    for g in range(N):
                        for b in range(N):
                            f.write("{:.6f} {:.6f} {:.6f}\n".format(
                                *self.lut[r, g, b]))

    def show_img(self, img, label):
        img8 = (img * 255).astype(np.uint8)
        h, w, _ = img8.shape
        qimg = QImage(img8.data, w, h, 3 * w, QImage.Format.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.AspectRatioMode.KeepAspectRatio))


# ----------------- Run -----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = App()
    win.show()
    sys.exit(app.exec())
