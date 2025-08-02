"""
Класс реализации компонента GUI с минимальной логикой
"""

import os
from datetime import datetime

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QWidget, QLabel, QPushButton, QFileDialog,
    QSpinBox, QFormLayout, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QSplitter, QHeaderView,
    QScrollArea, QComboBox, QLineEdit, QMessageBox
)

from src.config import COLUMN_HEADERS, PROCESSED_PATH_COLUMN, MSG_SELECT_IMAGE
from src.file_manager import normalize_path, check_path_encoding
from src.image_processor import analyze_image


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("--- Расчет площади фигур на изображении методом подсчета пикселей--- GUI")

        self.COLUMN_HEADERS = COLUMN_HEADERS
        self.PROCESSED_PATH_COLUMN = PROCESSED_PATH_COLUMN
        self.MSG_SELECT_IMAGE = MSG_SELECT_IMAGE

        self.input_folder = os.getcwd()
        self.output_folder = os.path.join(os.getcwd(), "results")
        os.makedirs(self.output_folder, exist_ok=True)

        self.df = None
        self.scale = 1.0
        self.original_pixmap = None

        self.build_ui()

    def build_ui(self):
        # Настройки (DPI, площади, связность)
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setMinimum(50)
        self.dpi_spin.setMaximum(1200)
        self.dpi_spin.setValue(300)
        self.dpi_spin.setSuffix(" DPI")

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setMinimum(1)
        self.min_area_spin.setMaximum(100000)
        self.min_area_spin.setValue(3000)
        self.min_area_spin.setSuffix(" px")

        self.conn_combo = QComboBox()
        self.conn_combo.addItems(["4-связность", "8-связность"])

        # Путь к папкам/файлам
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

        # Кнопки управления
        self.run_btn = QPushButton("Обработать всю папку")
        self.run_btn.clicked.connect(self.process_input_folder)

        self.run_file_btn = QPushButton("Обработать один файл")
        self.run_file_btn.clicked.connect(self.process_single_file)

        self.clear_btn = QPushButton("Очистить результаты")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self.clear_results)

        self.excel_btn = QPushButton("Экспорт в Excel")
        self.excel_btn.setEnabled(False)
        self.excel_btn.clicked.connect(self.export_excel)

        # Масштабирование
        zoom_in_btn = QPushButton("Увеличить +")
        zoom_in_btn.clicked.connect(lambda: self.change_scale(1.25))
        zoom_out_btn = QPushButton("Уменьшить -")
        zoom_out_btn.clicked.connect(lambda: self.change_scale(0.8))
        fit_btn = QPushButton("По размеру окна")
        fit_btn.clicked.connect(self.fit_to_window)

        # Левая панель
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

        results_actions_layout = QHBoxLayout()
        results_actions_layout.addWidget(self.excel_btn)
        results_actions_layout.addWidget(self.clear_btn)
        left_layout.addLayout(results_actions_layout)

        # Донат ссылки
        left_layout.addStretch(2)
        from PySide6.QtWidgets import QLabel
        tg_link = QLabel("<a href='https://t.me/PyOpsMaster'>Telegram чат с разработчиком</a>")
        tg_link.setTextFormat(Qt.RichText)
        tg_link.setOpenExternalLinks(True)
        tg_link.setAlignment(Qt.AlignCenter)
        donation_link = QLabel("<a href='https://pay.cloudtips.ru/p/85cd51e7'># Поблагодарить за классный код #</a>")
        donation_link.setTextFormat(Qt.RichText)
        donation_link.setOpenExternalLinks(True)
        donation_link.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(tg_link)
        left_layout.addWidget(donation_link)

        left_w = QWidget()
        left_w.setLayout(left_layout)

        # Правая панель
        self.img_label = QLabel(self.MSG_SELECT_IMAGE)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.img_label)

        self.tbl = QTableWidget(0, len(self.COLUMN_HEADERS))
        self.tbl.setHorizontalHeaderLabels(self.COLUMN_HEADERS)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
        self.tbl.setStyleSheet("QTableWidget::item { padding: 1px; }")
        header_style = """
            QHeaderView::section {
                background-color: #f0f0f0;
                padding: 2px;
                border-top: 1px solid #d0d0d0;
                border-bottom: 2px solid #b0b0b0;
                border-right: 1px solid #d0d0d0;
            }
        """
        self.tbl.horizontalHeader().setStyleSheet(header_style)
        self.tbl.verticalHeader().setDefaultSectionSize(20)
        self.tbl.cellClicked.connect(self.on_table_cell_clicked)

        right_layout = QHBoxLayout()
        right_layout.addWidget(self.tbl, 2)
        right_layout.addWidget(self.scroll, 3)
        right_w = QWidget()
        right_w.setLayout(right_layout)

        from PySide6.QtWidgets import QSplitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def on_table_cell_clicked(self, row, column):
        if self.df is None or self.df.empty:
            return
        image_path = self.df.iloc[row].get(self.PROCESSED_PATH_COLUMN, "")
        if image_path and os.path.exists(image_path):
            self.display_image(image_path)

    def clear_results(self):
        self.tbl.setRowCount(0)
        self.df = None
        self.img_label.clear()
        self.img_label.setText(self.MSG_SELECT_IMAGE)
        self.original_pixmap = None
        self.excel_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

    def browse_input_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Выберите папку где лежат изображения", self.input_folder_edit.text())
        if d:
            self.input_folder = d
            self.input_folder_edit.setText(d)

    def browse_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Выберите папку куда сохранять изображения", self.output_folder_edit.text())
        if d:
            self.output_folder = d
            self.output_folder_edit.setText(d)

    def open_output_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_folder_edit.text()))

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл изображения для обработки",
            self.input_folder_edit.text(),
            "Допустимые форматы (*.png *.jpg *.jpeg *.tif *.tiff)"
        )
        if file_path:
            self.input_file_edit.setText(file_path)

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
        scaled_pixmap = self.original_pixmap.scaled(self.scroll.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled_pixmap)

    def change_scale(self, factor):
        if not self.original_pixmap:
            return
        self.scale *= factor
        new_size = self.original_pixmap.size() * self.scale
        new_size = new_size.toSize()
        scaled_pixmap = self.original_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled_pixmap)
        self.img_label.adjustSize()

    def process_input_folder(self):
        self.input_folder = normalize_path(self.input_folder_edit.text())
        self.output_folder = normalize_path(self.output_folder_edit.text())

        if not check_path_encoding(self.input_folder):
            self.img_label.setText("Ошибка: проблема с кодировкой пути к входной папке")
            return

        os.makedirs(self.output_folder, exist_ok=True)

        image_files = [f for f in os.listdir(self.input_folder) if
                       f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        if not image_files:
            self.img_label.setText("В указанной папке нет изображений.")
            return

        all_records = []

        for image_file in image_files:
            image_path = os.path.join(self.input_folder, image_file)
            records, result_path = analyze_image(
                image_path,
                self.output_folder,
                self.dpi_spin.value(),
                self.min_area_spin.value(),
                4 if self.conn_combo.currentIndex() == 0 else 8
            )

            if not result_path:
                continue

            for rec in records:
                rec['Имя файла'] = image_file
                rec[self.PROCESSED_PATH_COLUMN] = result_path
            all_records.extend(records)

        if all_records:
            import pandas as pd
            self.df = pd.DataFrame(all_records)
            all_cols = self.COLUMN_HEADERS + [self.PROCESSED_PATH_COLUMN]
            self.df = self.df[all_cols]
            self.populate_table()
            self.excel_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)

            if not self.df.empty:
                first_image_path = self.df.iloc[0][self.PROCESSED_PATH_COLUMN]
                if first_image_path and os.path.exists(first_image_path):
                    self.display_image(first_image_path)

    def process_single_file(self):
        file_path = self.input_file_edit.text()
        if not file_path or not os.path.exists(file_path):
            self.img_label.setText("Выберите корректный файл изображения.")
            return

        self.output_folder = self.output_folder_edit.text()
        os.makedirs(self.output_folder, exist_ok=True)

        records, result_path = analyze_image(
            file_path,
            self.output_folder,
            self.dpi_spin.value(),
            self.min_area_spin.value(),
            4 if self.conn_combo.currentIndex() == 0 else 8
        )
        if records:
            filename = os.path.basename(file_path)
            for rec in records:
                rec['Имя файла'] = filename
                rec[self.PROCESSED_PATH_COLUMN] = result_path
            import pandas as pd
            self.df = pd.DataFrame(records)
            all_cols = self.COLUMN_HEADERS + [self.PROCESSED_PATH_COLUMN]
            self.df = self.df[all_cols]
            self.populate_table()
            self.excel_btn.setEnabled(True)
            self.clear_btn.setEnabled(True)

        if result_path:
            self.display_image(result_path)

    def populate_table(self):
        self.tbl.setRowCount(0)
        if self.df is None or self.df.empty:
            return

        self.tbl.setRowCount(len(self.df))
        for row, rec in self.df.iterrows():
            for col, key in enumerate(self.COLUMN_HEADERS):
                item = QTableWidgetItem(str(rec.get(key, "")))
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(row, col, item)

    def export_excel(self):
        if self.df is None or self.df.empty:
            QMessageBox.warning(self, "Внимание", "Нет данных для экспорта.")
            return

        from PySide6.QtWidgets import QFileDialog

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn, _ = QFileDialog.getSaveFileName(self, "Сохранить Excel", os.path.join(self.output_folder_edit.text(),
                                                                                 f"report_{timestamp}.xlsx"),
                                            "Excel (*.xlsx)")
        if fn:
            if not fn.lower().endswith('.xlsx'):
                fn += '.xlsx'
            try:
                self.df.to_excel(fn, index=False)
                QMessageBox.information(self, "Успех", f"Данные экспортированы в файл:\n{fn}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Ошибка операции экспорта в файл Excel:\n{e}")
