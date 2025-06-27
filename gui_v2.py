import os
import sys
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QSpinBox, QFormLayout, QVBoxLayout, QHBoxLayout, QTableWidget,
    QTableWidgetItem, QSplitter, QHeaderView, QScrollArea,
    QComboBox, QLineEdit
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("--- Расчет площади фигур на изображении методом подсчета пикселей--- GUI")

        # --- Константы и переменные ---
        self.COLUMN_HEADERS = ["Имя файла", "№ фигуры", "Кол-во пикселей", "Площадь"]
        self.PROCESSED_PATH_COLUMN = "Путь к обработанному файлу"
        self.MSG_SELECT_IMAGE = "Выберите папку или файл и нажмите 'Обработать'"

        self.input_folder = os.getcwd()
        self.output_folder = os.path.join(os.getcwd(), "results")
        os.makedirs(self.output_folder, exist_ok=True)

        self.df = pd.DataFrame()
        self.scale = 1.0
        self.original_pixmap = None

        # --- Настройки ---
        # self.dpi_spin = QSpinBox(value=300, minimum=150, maximum=1200, suffix=" DPI")
        # self.min_area_spin = QSpinBox(value=30, minimum=1, maximum=100000, suffix=" px")

        self.dpi_spin = QSpinBox()
        self.dpi_spin.setMinimum(50)
        self.dpi_spin.setMaximum(1200)
        self.dpi_spin.setValue(300)  # Значение по-умолчанию
        self.dpi_spin.setSuffix(" DPI")

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setMinimum(1)
        self.min_area_spin.setMaximum(100000)
        self.min_area_spin.setValue(30)  # Значение по-умолчанию
        self.min_area_spin.setSuffix(" px")

        self.conn_combo = QComboBox()
        # Используем стандартные и понятные термины
        self.conn_combo.addItems(["4-связность", "8-связность"])

        # --- Виджеты для выбора путей ---
        self.input_folder_edit = QLineEdit(self.input_folder)
        input_folder_btn = QPushButton("…")
        input_folder_btn.clicked.connect(self.browse_input_folder)

        self.input_file_edit = QLineEdit()
        input_file_btn = QPushButton("…")
        input_file_btn.clicked.connect(self.browse_input_file)

        self.output_folder_edit = QLineEdit(self.output_folder)
        output_folder_btn = QPushButton("…")
        output_folder_btn.clicked.connect(self.browse_output_folder)

        open_output_btn = QPushButton("Открыть папку сохранения")
        open_output_btn.clicked.connect(self.open_output_folder)

        # --- Кнопки управления ---
        self.run_btn = QPushButton("Обработать всю папку")
        self.run_btn.clicked.connect(self.process_input_folder)

        self.run_file_btn = QPushButton("Обработать один файл")
        self.run_file_btn.clicked.connect(self.process_single_file)

        self.clear_btn = QPushButton("Очистить результаты")
        self.clear_btn.clicked.connect(self.clear_results)
        self.clear_btn.setEnabled(False)  # Изначально неактивна

        self.excel_btn = QPushButton("Экспорт в Excel")
        self.excel_btn.clicked.connect(self.export_excel)
        self.excel_btn.setEnabled(False)

        # --- Кнопки масштабирования ---
        zoom_in_btn = QPushButton("Увеличить +")
        zoom_in_btn.clicked.connect(lambda: self.change_scale(1.25))
        zoom_out_btn = QPushButton("Уменьшить -")
        zoom_out_btn.clicked.connect(lambda: self.change_scale(0.8))
        fit_btn = QPushButton("По размеру окна")
        fit_btn.clicked.connect(self.fit_to_window)

        # --- Левая панель ---
        form = QFormLayout()
        h_input = QHBoxLayout()
        h_input.addWidget(self.input_folder_edit)
        h_input.addWidget(input_folder_btn)
        form.addRow("Папка с файлами:", h_input)

        h_file = QHBoxLayout()
        h_file.addWidget(self.input_file_edit)
        h_file.addWidget(input_file_btn)
        form.addRow("Отдельный файл:", h_file)

        h_output = QHBoxLayout()
        h_output.addWidget(self.output_folder_edit)
        h_output.addWidget(output_folder_btn)
        form.addRow("Папка для сохранения:", h_output)

        form.addRow("Разрешение (DPI):", self.dpi_spin)
        form.addRow("Минимальная площадь (px):", self.min_area_spin)
        form.addRow("Тип связности (поиск шума):", self.conn_combo)

        left_layout = QVBoxLayout()
        left_layout.addLayout(form)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.run_file_btn)
        left_layout.addStretch(1)

        zoom_layout = QHBoxLayout()
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        left_layout.addLayout(zoom_layout)
        left_layout.addWidget(fit_btn)
        left_layout.addStretch(1)

        left_layout.addWidget(open_output_btn)
        # --- Группируем кнопки управления результатами ---
        results_actions_layout = QHBoxLayout()
        results_actions_layout.addWidget(self.excel_btn)
        results_actions_layout.addWidget(self.clear_btn)
        left_layout.addLayout(results_actions_layout)

        # # ===============================================================
        # # ДОНАТ
        # # ===============================================================
        #
        # left_layout.addStretch(2)  # Добавим побольше отступ перед этим блоком
        # tg_link = QLabel("<a href='https://t.me/PyOpsMaster'>Telegram чат с разработчиком</a>")
        # tg_link.setTextFormat(Qt.RichText)
        # tg_link.setOpenExternalLinks(True)
        # tg_link.setAlignment(Qt.AlignCenter)
        # donation_link = QLabel("<a href='https://pay.cloudtips.ru/p/85cd51e7'>"
        #                        "# Поблагодарить за классный код #</a>")
        # donation_link.setTextFormat(Qt.RichText)
        # donation_link.setOpenExternalLinks(True)
        # donation_link.setAlignment(Qt.AlignCenter)
        # left_layout.addWidget(tg_link)
        # left_layout.addWidget(donation_link)
        #
        # # ===============================================================
        # # ДОНАТ КОНЕЦ
        # # ===============================================================

        left_w = QWidget()
        left_w.setLayout(left_layout)

        # --- Правая панель ---
        self.img_label = QLabel(self.MSG_SELECT_IMAGE)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.img_label)

        self.tbl = QTableWidget(0, 4)
        self.tbl.setHorizontalHeaderLabels(self.COLUMN_HEADERS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.tbl.setStyleSheet("QTableWidget::item { padding: 1px; }")
        header_style = """
            QHeaderView::section {
                background-color: #f0f0f0; /* Светло-серый фон */
                padding: 2px;
                border-top: 1px solid #d0d0d0;
                border-bottom: 2px solid #b0b0b0; /* Более жирная линия снизу */
                border-right: 1px solid #d0d0d0;
            }
        """
        self.tbl.horizontalHeader().setStyleSheet(header_style)

        # self.tbl.horizontalHeader().setStyleSheet(header_style)

        self.tbl.verticalHeader().setDefaultSectionSize(20)  # Высота для всех строк таблицы
        # --- Клик по кнопке очистить таблицу ---
        self.tbl.cellClicked.connect(self.on_table_cell_clicked)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.tbl, 2)  # Таблица, коэф. растяжения 2
        right_layout.addWidget(self.scroll, 3)  # Картинка, коэф. растяжения 3
        right_w = QWidget()
        right_w.setLayout(right_layout)

        # --- Сборка интерфейса ---
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)  # Левая панель
        splitter.setStretchFactor(1, 3)  # Правая панель
        self.setCentralWidget(splitter)

        screen = self.screen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

        # # --- Правая панель --- виджеты расположены горизонтально один над другим
        # self.img_label = QLabel("Выберите папку с изображениями или файл и нажмите 'Process'")
        # self.img_label.setAlignment(Qt.AlignCenter)
        # self.scroll = QScrollArea()
        # self.scroll.setWidgetResizable(True)
        # self.scroll.setWidget(self.img_label)
        #
        # self.tbl = QTableWidget(0, 4)
        # self.tbl.setHorizontalHeaderLabels(["Filename", "№ фигуры", "Area px", "Area cm²"])
        # self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # self.tbl.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        #
        # # Уменьшаем отступы в ячейках
        # self.tbl.setStyleSheet("QTableWidget::item { padding: 1px; }")
        #
        # right_layout = QVBoxLayout()
        # right_layout.addWidget(self.scroll, 1)  # Равные пропорции
        # right_layout.addWidget(self.tbl, 1)  # Равные пропорции
        # right_w = QWidget()
        # right_w.setLayout(right_layout)
        #
        # # --- Сборка интерфейса ---
        # splitter = QSplitter(Qt.Horizontal)
        # splitter.addWidget(left_w)
        # splitter.addWidget(right_w)
        # splitter.setStretchFactor(0, 1)
        # splitter.setStretchFactor(1, 3)
        # self.setCentralWidget(splitter)
        # self.resize(1200, 700)

    def on_table_cell_clicked(self, row, column):
        """Отображает изображение, соответствующее выбранной строке в таблице."""
        if self.df is None or self.df.empty:
            return

        # Получаем путь из 'скрытой' колонки DataFrame по индексу строки
        image_path = self.df.iloc[row][self.PROCESSED_PATH_COLUMN]

        if image_path and os.path.exists(image_path):
            self.display_image(image_path)

    def clear_results(self):
        """Очищает таблицу, DataFrame, изображение и деактивирует кнопки."""
        self.tbl.setRowCount(0)
        self.df = pd.DataFrame()
        self.img_label.clear()
        self.img_label.setText(self.MSG_SELECT_IMAGE)
        self.original_pixmap = None

        self.excel_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

    # --- Методы для работы с файлами и папками ---
    def browse_input_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Выберите папку где лежат изображения", self.input_folder)
        if d:
            self.input_folder = d
            self.input_folder_edit.setText(d)

    def browse_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Выберите папку куда сохранять изображения", self.output_folder)
        if d:
            self.output_folder = d
            self.output_folder_edit.setText(d)

    def open_output_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_folder))

    # --- Методы для работы с изображением ---
    def display_image(self, path):
        self.original_pixmap = QPixmap(path)
        if self.original_pixmap.isNull():
            self.img_label.setText(f"Не удалось загрузить:\n{path}")
            self.original_pixmap = None
            return
        self.fit_to_window()

    def fit_to_window(self):
        if not self.original_pixmap:
            return
        self.scale = 1.0
        # Вписываем изображение в размер ScrollArea
        scaled_pixmap = self.original_pixmap.scaled(self.scroll.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled_pixmap)

    def change_scale(self, factor):
        if not self.img_label.pixmap():  # Проверяем, есть ли что масштабировать
            return

        # Берем текущий размер изображения в QLabel
        # Вычисляем новый размер
        current_size = self.img_label.pixmap().size()
        new_size = current_size * factor

        # Масштабируем оригинальный pixmap до нового размера
        # Это дает лучшее качество, чем масштабирование уже масштабированного изображения
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(new_size,
                                                        Qt.KeepAspectRatio,
                                                        Qt.SmoothTransformation)
            self.img_label.setPixmap(scaled_pixmap)
            self.img_label.adjustSize()  # Важно для корректной работы прокрутки

    # --- Основная логика ---
    def process_input_folder(self):
        self.input_folder = self.input_folder_edit.text()
        self.output_folder = self.output_folder_edit.text()
        os.makedirs(self.output_folder, exist_ok=True)

        image_files = [f for f in os.listdir(self.input_folder) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        if not image_files:
            self.img_label.setText("В указанной папке нет изображений.")
            return

        all_records = []
        # last_image_path = None

        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            records, result_path = self.run_single_analysis(image_path)

            # Проверяем, что обработка прошла успешно
            if not result_path:
                continue

            # Добавляем имя файла и путь к обработанному файлу к каждой записи
            for rec in records:
                rec['Имя файла'] = image_file
                rec[self.PROCESSED_PATH_COLUMN] = result_path  # <-- НОВАЯ СТРОКА
            all_records.extend(records)

        if all_records:
            self.df = pd.DataFrame(all_records)
            # Получаем полный список колонок, включая скрытую
            all_columns = self.COLUMN_HEADERS + [self.PROCESSED_PATH_COLUMN]
            # Переупорядочиваем колонки для лучшего вида
            self.df = self.df[all_columns]
            self.populate_table()
            self.excel_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)  # <-- Активируем кнопку очистки

            # Отображаем первое из обработанных изображений, если они есть
        if not self.df.empty:
            first_image_path = self.df.iloc[0][self.PROCESSED_PATH_COLUMN]
            if first_image_path and os.path.exists(first_image_path):
                self.display_image(first_image_path)

    def run_single_analysis(self, image_path):
        _DPI = self.dpi_spin.value()
        min_area_px = self.min_area_spin.value()
        conn = 4 if self.conn_combo.currentIndex() == 0 else 8
        cm2_per_px = (2.54 / _DPI) ** 2

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return [], None

        _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.count_nonzero(th == 255) > np.count_nonzero(th == 0):
            th = cv2.bitwise_not(th)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=2)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)

        n, lbl, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=conn)
        h, w = th.shape
        records = []
        color_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(1, n):
            x, y, wx, hy, area = stats[i]
            if area < min_area_px or x == 0 or y == 0 or x + wx >= w or y + hy >= h:
                continue

            cm2 = round(area * cm2_per_px, 3)
            num = len(records) + 1
            records.append({"№ фигуры": num, "Кол-во пикселей": int(area), "Площадь": cm2})

            cx, cy = x + wx // 2, y + hy // 2
            cv2.putText(color_img, str(num), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        2.1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.rectangle(color_img, (x, y), (x + wx, y + hy), (0, 255, 0), 3)

        filename = os.path.basename(image_path)
        name_base = os.path.splitext(filename)[0]
        # Добавляем дату и время к имени файла
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_out = f"{name_base}_processed_{timestamp}.png"
        save_path = os.path.join(self.output_folder, name_out)
        cv2.imwrite(save_path, color_img)

        return records, save_path

    # --- метод для обработки одного файла
    def process_single_file(self):
        file_path = self.input_file_edit.text()
        if not file_path or not os.path.exists(file_path):
            self.img_label.setText("Выберите корректный файл изображения.")
            return

        self.output_folder = self.output_folder_edit.text()
        os.makedirs(self.output_folder, exist_ok=True)

        records, result_path = self.run_single_analysis(file_path)
        if records:
            filename = os.path.basename(file_path)
            for rec in records:
                rec['Имя файла'] = filename
                rec[self.PROCESSED_PATH_COLUMN] = result_path  # <-- НОВАЯ СТРОКА

            all_columns = self.COLUMN_HEADERS + [self.PROCESSED_PATH_COLUMN]
            self.df = pd.DataFrame(records)
            self.df = self.df[all_columns]
            self.populate_table()
            self.excel_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)  # <-- НОВАЯ СТРОКА

        if result_path:
            self.display_image(result_path)

    # ----- метод для выбора одного файла
    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл изображения для обработки",
            self.input_folder,
            "Допустимые форматы (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.input_file_edit.setText(file_path)

    # --- Методы для вывода результатов ---
    def populate_table(self):
        self.tbl.setRowCount(0)
        if self.df is None or self.df.empty:
            return

        self.tbl.setRowCount(len(self.df))
        for row, rec in self.df.iterrows():
            for col, key in enumerate(self.df.columns):
                item = QTableWidgetItem(str(rec[key]))
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(row, col, item)

    def export_excel(self):
        # Важно: для работы требуется `pip install openpyxl`
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn, _ = QFileDialog.getSaveFileName(self, "Save Excel", os.path.join(self.output_folder,
                                                                             f"report_{timestamp}.xlsx"),
                                            "Excel (*.xlsx)")
        if fn:
            if not fn.lower().endswith('.xlsx'):
                fn += '.xlsx'
            try:
                self.df.to_excel(fn, index=False)
            except Exception as e:
                print(f"Ошибка операции экспорта в файл Excel: {e}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec())
