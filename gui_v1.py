import sys
import cv2
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QLabel, QPushButton, QFileDialog,
    QSpinBox, QFormLayout, QHBoxLayout,
    QVBoxLayout, QTableWidget, QTableWidgetItem,
    QMessageBox, QSizePolicy, QSplitter
)
from PySide6.QtCore import Qt, QByteArray
from PySide6.QtGui import QPixmap, QImage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Morphology Analyzer")
        self.image_path = None

        # --- Виджеты управления параметрами
        self.dpi_spin = QSpinBox(value=300, minimum=50, maximum=1200, suffix=" DPI")
        self.min_area_spin = QSpinBox(value=20, minimum=1, maximum=10000, suffix=" px")

        self.open_btn = QPushButton("Открыть изображение")
        self.open_btn.clicked.connect(self.open_image)

        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run_analysis)
        self.run_btn.setEnabled(False)

        form = QFormLayout()
        form.addRow("DPI:", self.dpi_spin)
        form.addRow("min_area_px:", self.min_area_spin)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.addWidget(self.open_btn)
        ctrl_layout.addLayout(form)
        ctrl_layout.addWidget(self.run_btn)
        ctrl_layout.addStretch()

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl_layout)

        # --- Виджет для показа изображения
        self.img_label = QLabel("Здесь будет изображение")
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Таблица результатов
        self.tbl = QTableWidget()
        self.tbl.setColumnCount(3)
        self.tbl.setHorizontalHeaderLabels(["№ фигуры", "Площадь, px", "Площадь, кв.см"])

        # --- Собираем всё в сплиттер
        right_layout = QVBoxLayout()
        right_layout.addWidget(self.img_label, 3)
        right_layout.addWidget(self.tbl, 2)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(ctrl_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 3)

        self.setCentralWidget(splitter)
        self.resize(900, 600)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите изображение", filter="Images (*.png *.jpg *.bmp)")
        if not path:
            return
        self.image_path = path
        pix = QPixmap(path).scaled(self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(pix)
        self.run_btn.setEnabled(True)

    def run_analysis(self):
        if not self.image_path:
            return
        # 1) Параметры
        _DPI = self.dpi_spin.value()
        mm_per_pixel = 25.4 / _DPI
        cm2_per_pixel = (mm_per_pixel / 10) ** 2
        min_area_px = self.min_area_spin.value()

        # 2) Загрузка и бинаризация
        img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            QMessageBox.critical(self, "Ошибка", "Невозможно загрузить изображение.")
            return

        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(th == 255) > np.count_nonzero(th == 0):
            th = cv2.bitwise_not(th)
        # морфология
        kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kern)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kern)

        # 3) connectedComponentsWithStats + центроиды
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(th, connectivity=8)
        h, w = th.shape
        records = []

        for label in range(1, nlabels):
            x, y, w_comp, h_comp, area_px = stats[label]
            if area_px < min_area_px:
                continue
            # фильтр «касающихся края»
            if x == 0 or y == 0 or x + w_comp >= w or y + h_comp >= h:
                continue
            cx, cy = centroids[label]
            records.append({
                "№ фигуры": len(records) + 1,
                "Площадь, px": int(area_px),
                "Площадь, кв.см": round(area_px * cm2_per_pixel, 3),
                "centroid": (int(cx), int(cy))
            })

        # 4) Отметить фигуры на изображении
        # Переводим бинарное в цветное
        vis = cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)
        for rec in records:
            num = rec["№ фигуры"]
            area_cm = rec["Площадь, кв.см"]
            cx, cy = rec["centroid"]
            text = f"{num}: {area_cm}"
            cv2.putText(vis, text, (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # 5) Конвертируем в QPixmap и показываем
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h2, w2, ch = rgb.shape
        bytes_per_line = ch * w2
        qt_img = QImage(rgb.data, w2, h2, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qt_img).scaled(
            self.img_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.img_label.setPixmap(pix)

        # 6) Заполняем таблицу
        df = pd.DataFrame(records)
        summary = {
            "№ фигуры": "Итого",
            "Площадь, px": df["Площадь, px"].sum(),
            "Площадь, кв.см": round(df["Площадь, кв.см"].sum(), 3)
        }
        df = pd.concat([df.drop(columns="centroid"), pd.DataFrame([summary])], ignore_index=True)

        self.tbl.setRowCount(len(df))
        for i, row in df.iterrows():
            for j, col in enumerate(df.columns):
                self.tbl.setItem(i, j, QTableWidgetItem(str(row[col])))
        self.tbl.resizeColumnsToContents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
