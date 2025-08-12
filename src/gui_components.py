"""
Класс реализации компонента GUI:
- Удаление рамки (auto/fixed) + толщина (px)
- Эталон (A4/A3/Letter/Legal/Нет) и Произвольный (см)
- Кнопка "Очистить debug"
- Обработка PDF: извлечение изображений и запуск процессора для каждого
- И другие элементы которые присутствовали в прошлом релизе
"""

import os
import shutil
from datetime import datetime
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices, QPixmap
from PySide6.QtWidgets import (
    QWidget, QPushButton, QFileDialog, QDoubleSpinBox,
    QSpinBox, QFormLayout, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox,
    QScrollArea, QComboBox, QLineEdit, QMessageBox, QLabel, QSplitter
)
from src.config import COLUMN_HEADERS, PROCESSED_PATH_COLUMN, MSG_SELECT_IMAGE
from src.file_manager import normalize_path, check_path_encoding
from src.image_processor import analyze_image


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        # self.run_only_process_frame_btn = None
        self.tbl = None
        self.scroll = None
        self.img_label = None
        self.excel_btn = None
        self.clear_btn = None
        self.clear_debug_btn = None
        self.run_file_btn = None
        self.run_btn = None
        self.output_folder_edit = None
        self.input_file_edit = None
        self.input_folder_edit = None
        self.conn_combo = None
        self.min_area_spin = None
        self.dpi_spin = None

        # Новые элементы
        self.frame_check = None
        self.frame_mode_combo = None
        self.frame_width_spin = None
        self.ref_mode_combo = None
        self.ref_preset_combo = None
        self.ref_width_cm = None
        self.ref_height_cm = None

        self.setWindowTitle("--- Расчет площади фигур на изображении методом подсчета пикселей --- GUI")

        self.COLUMN_HEADERS = COLUMN_HEADERS
        self.PROCESSED_PATH_COLUMN = PROCESSED_PATH_COLUMN
        self.MSG_SELECT_IMAGE = MSG_SELECT_IMAGE

        self.input_folder = os.getcwd()
        self.output_folder = os.path.join(os.getcwd(), "results")
        os.makedirs(self.output_folder, exist_ok=True)
        self.debug_folder = os.path.join(self.output_folder, "debug")

        # Модель данных
        self.df_records = []  # список dict с результатами

        self.scale = 1.0
        self.original_pixmap = None

        self.build_ui()

    # --------------------------- UI ---------------------------

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
        self.min_area_spin.setValue(12000)
        self.min_area_spin.setSuffix(" px")

        self.conn_combo = QComboBox()
        self.conn_combo.addItems(["4-связность", "8-связность"])

        # Новые настройки: Удалять рамку
        self.frame_check = QCheckBox("Удалять рамку")
        self.frame_check.setChecked(True)
        self.frame_mode_combo = QComboBox()
        self.frame_mode_combo.addItems(["auto", "fixed"])
        self.frame_width_spin = QSpinBox()
        self.frame_width_spin.setRange(0, 400)
        self.frame_width_spin.setValue(30)
        self.frame_width_spin.setSuffix(" px")
        # Управление доступностью поля толщины
        self.frame_mode_combo.currentTextChanged.connect(
            lambda m: self.frame_width_spin.setEnabled(m == "fixed")
        )
        # Инициализация
        self.frame_width_spin.setEnabled(self.frame_mode_combo.currentText() == "fixed")

        # Новые настройки: Эталон/Произвольный/Нет
        self.ref_mode_combo = QComboBox()
        self.ref_mode_combo.addItems(["preset", "custom", "none"])
        self.ref_preset_combo = QComboBox()
        self.ref_preset_combo.addItems(["A4", "A3", "Letter", "Legal"])
        self.ref_width_cm = QDoubleSpinBox()
        self.ref_width_cm.setRange(0.1, 500.0)
        self.ref_width_cm.setValue(21.0)
        self.ref_width_cm.setSuffix(" см")
        self.ref_width_cm.setDecimals(2)
        self.ref_height_cm = QDoubleSpinBox()
        self.ref_height_cm.setRange(0.1, 500.0)
        self.ref_height_cm.setValue(29.7)
        self.ref_height_cm.setSuffix(" см")
        self.ref_height_cm.setDecimals(2)

        # Включение/выключение контролов по режиму эталона
        def apply_ref_mode(mode: str):
            is_preset = mode == "preset"
            is_custom = mode == "custom"
            self.ref_preset_combo.setEnabled(is_preset)
            self.ref_width_cm.setEnabled(is_custom)
            self.ref_height_cm.setEnabled(is_custom)

        self.ref_mode_combo.currentTextChanged.connect(apply_ref_mode)
        apply_ref_mode(self.ref_mode_combo.currentText())

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

        # self.run_only_process_frame_btn = QPushButton("Обработать только рамку изображения")
        # self.run_only_process_frame_btn.clicked.connect(self.run_only_process_frame)

        self.clear_btn = QPushButton("Очистить результаты")
        self.clear_btn.setEnabled(False)
        self.clear_btn.clicked.connect(self.clear_results)

        self.excel_btn = QPushButton("Экспорт в Excel")
        self.excel_btn.setEnabled(False)
        self.excel_btn.clicked.connect(self.export_excel)

        # Новая кнопка: Очистить debug
        self.clear_debug_btn = QPushButton("Очистить debug")
        self.clear_debug_btn.clicked.connect(self.clear_debug_folder)

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

        # Новые строки формы — удаление рамки
        frame_line = QHBoxLayout()
        frame_line.addWidget(self.frame_check)
        frame_line.addStretch(1)
        form.addRow("Рамка:", frame_line)

        frame_mode_line = QHBoxLayout()
        frame_mode_line.addWidget(self.frame_mode_combo)
        frame_mode_line.addWidget(self.frame_width_spin)
        frame_mode_line.addStretch(1)
        form.addRow("Режим рамки:", frame_mode_line)

        # Новые строки формы — эталон
        ref_mode_line = QHBoxLayout()
        ref_mode_line.addWidget(self.ref_mode_combo)
        ref_mode_line.addStretch(1)
        form.addRow("Эталон режим:", ref_mode_line)

        ref_preset_line = QHBoxLayout()
        ref_preset_line.addWidget(self.ref_preset_combo)
        ref_preset_line.addStretch(1)
        form.addRow("Эталон (preset):", ref_preset_line)

        ref_custom_line = QHBoxLayout()
        ref_custom_line.addWidget(self.ref_width_cm)
        ref_custom_line.addWidget(self.ref_height_cm)
        ref_custom_line.addStretch(1)
        form.addRow("Эталон (см):", ref_custom_line)

        left_layout = QVBoxLayout()
        left_layout.addLayout(form)
        left_layout.addWidget(self.run_btn)
        left_layout.addWidget(self.run_file_btn)
        # left_layout.addWidget(self.run_only_process_frame_btn)

        left_layout.addStretch(1)
        left_layout.addWidget(open_output_btn)

        results_actions_layout = QHBoxLayout()
        results_actions_layout.addWidget(self.excel_btn)
        results_actions_layout.addWidget(self.clear_btn)
        results_actions_layout.addWidget(self.clear_debug_btn)
        left_layout.addLayout(results_actions_layout)

        # Донат ссылки
        left_layout.addStretch(2)
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

        # Масштабирование
        zoom_in_btn = QPushButton("Увеличить +")
        zoom_in_btn.clicked.connect(lambda: self.change_scale(1.25))
        zoom_out_btn = QPushButton("Уменьшить -")
        zoom_out_btn.clicked.connect(lambda: self.change_scale(0.8))
        fit_btn = QPushButton("По размеру окна")
        fit_btn.clicked.connect(self.fit_to_window)

        # Виджет для изображения
        self.img_label = QLabel(self.MSG_SELECT_IMAGE)
        self.img_label.setAlignment(Qt.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.img_label)

        # Горизонтальная линия для кнопок
        zoom_layout = QHBoxLayout()
        zoom_layout.addStretch(1)
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(fit_btn)
        zoom_layout.addStretch(1)

        # Вертикальный контейнер: картинка + кнопки
        image_panel = QVBoxLayout()
        image_panel.addWidget(self.scroll, stretch=1)
        image_panel.addLayout(zoom_layout)

        image_widget = QWidget()
        image_widget.setLayout(image_panel)

        # Таблица
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

        # основной правый layout (таблица и панель с изображением)
        right_layout = QHBoxLayout()
        right_layout.addWidget(self.tbl, 2)
        right_layout.addWidget(image_widget, 3)

        right_w = QWidget()
        right_w.setLayout(right_layout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_w)
        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

        screen = self.screen().availableGeometry()
        self.resize(int(screen.width() * 0.8), int(screen.height() * 0.8))

    # --------------------------- Хэндлеры UI ---------------------------

    def on_table_cell_clicked(self, row: int, column: int):
        if not self.df_records or row < 0 or row >= len(self.df_records):
            return
        image_path = self.df_records[row].get(self.PROCESSED_PATH_COLUMN, "")
        if image_path and os.path.exists(image_path):
            self.display_image(image_path)

    def clear_results(self):
        self.tbl.setRowCount(0)
        self.df_records = []
        self.img_label.clear()
        self.img_label.setText(self.MSG_SELECT_IMAGE)
        self.original_pixmap = None
        self.excel_btn.setEnabled(False)
        self.clear_btn.setEnabled(False)

    def clear_debug_folder(self):
        try:
            if os.path.isdir(self.debug_folder):
                shutil.rmtree(self.debug_folder, ignore_errors=True)
            os.makedirs(self.debug_folder, exist_ok=True)
            QMessageBox.information(self, "Готово", "Папка debug очищена.")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось очистить debug: {e}")

    def browse_input_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Выберите папку где лежат изображения",
                                             self.input_folder_edit.text())
        if d:
            self.input_folder = d
            self.input_folder_edit.setText(d)

    def browse_output_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Выберите папку куда сохранять изображения",
                                             self.output_folder_edit.text())
        if d:
            self.output_folder = d
            self.output_folder_edit.setText(d)
            # Обновим и путь debug при смене папки результатов
            self.debug_folder = os.path.join(self.output_folder, "debug")

    def open_output_folder(self):
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_folder_edit.text()))

    def browse_input_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл изображения для обработки",
            self.input_folder_edit.text(),
            "Допустимые форматы (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.pdf)"
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
        if not self.original_pixmap or self.original_pixmap.isNull():
            return

        scaled_pixmap = self.original_pixmap.scaled(self.scroll.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        if self.original_pixmap.width() > 0:
            self.scale = scaled_pixmap.width() / self.original_pixmap.width()
        else:
            self.scale = 1.0

        self.img_label.setPixmap(scaled_pixmap)

    def change_scale(self, factor):
        if not self.original_pixmap:
            return
        self.scale *= factor
        new_size = self.original_pixmap.size() * self.scale
        scaled_pixmap = self.original_pixmap.scaled(new_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.img_label.setPixmap(scaled_pixmap)
        self.img_label.adjustSize()

    # --------------------------- Логика обработки ---------------------------

    def process_input_folder(self):
        self.input_folder = normalize_path(self.input_folder_edit.text())
        self.output_folder = normalize_path(self.output_folder_edit.text())
        self.debug_folder = os.path.join(self.output_folder, "debug")

        if not check_path_encoding(self.input_folder):
            self.img_label.setText("Ошибка: проблема с кодировкой пути к входной папке")
            return

        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.debug_folder, exist_ok=True)

        image_files = [f for f in os.listdir(self.input_folder)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        if not image_files:
            self.img_label.setText("В указанной папке нет изображений.")
            return

        image_paths = [os.path.join(self.input_folder, f) for f in image_files]
        self._process_image_paths(image_paths)

    def process_single_file(self):
        in_path = normalize_path(self.input_file_edit.text().strip())
        if not in_path:
            QMessageBox.information(self, "Файл не выбран", "Укажите файл для обработки.")
            return

        self.output_folder = normalize_path(self.output_folder_edit.text())
        self.debug_folder = os.path.join(self.output_folder, "debug")
        os.makedirs(self.output_folder, exist_ok=True)
        os.makedirs(self.debug_folder, exist_ok=True)

        ext = os.path.splitext(in_path)[1].lower()
        if ext == ".pdf":
            # Обработка PDF: извлечение изображений во временную папку и обработка
            try:
                tmp_dir = self._extract_images_from_pdf(in_path)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка PDF", f"Не удалось извлечь изображения из PDF:\n{e}")
                return
            image_paths = [os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
            if not image_paths:
                QMessageBox.information(self, "Пусто", "В PDF не обнаружены изображения.")
                return
            self._process_image_paths(image_paths)
        else:
            if not os.path.isfile(in_path):
                QMessageBox.warning(self, "Ошибка", "Указанный файл не найден.")
                return
            self._process_image_paths([in_path])

    def run_only_process_frame(self):
        pass
        # in_path = normalize_path(self.input_file_edit.text().strip())
        # if not in_path:
        #     QMessageBox.information(self, "Файл не выбран", "Укажите файл для обработки.")
        #     return
        # if not os.path.isfile(in_path):
        #     QMessageBox.warning(self, "Ошибка", "Указанный файл не найден.")
        #     return
        #
        # self.output_folder = normalize_path(self.output_folder_edit.text())
        # self.debug_folder = os.path.join(self.output_folder, "debug")
        # os.makedirs(self.output_folder, exist_ok=True)
        # os.makedirs(self.debug_folder, exist_ok=True)
        #
        # try:
        #     img = cv2.imread(in_path)
        #     if img is None:
        #         raise IOError("Не удалось прочитать файл изображения. Проверьте путь и целостность файла.")
        #
        #     params = {
        #         '_remove_frame': self.frame_check.isChecked(),
        #         'frame_mode': self.frame_mode_combo.currentText(),
        #         'frame_width': self.frame_width_spin.value(),
        #         'debug_folder': self.debug_folder
        #     }
        #
        #     processed_img = remove_frame(img, **params)
        #
        #     filename = os.path.basename(in_path)
        #     name, ext = os.path.splitext(filename)
        #     result_path = os.path.join(self.output_folder, f"{name}_no_frame{ext}")
        #
        #     if not cv2.imwrite(result_path, processed_img):
        #         raise IOError("Не удалось сохранить обработанное изображение.")
        #
        #     self.display_image(result_path)
        #     QMessageBox.information(self, "Готово", f"Обработка рамки завершена.\nФайл сохранен: {result_path}")
        #
        # except Exception as e:
        #     QMessageBox.critical(self, "Критическая ошибка", f"Произошла ошибка при обработке рамки:\n{e}")

    def _process_image_paths(self, image_paths):
        """
        Унифицированная обработка набора изображений с заполнением таблицы.
        """
        all_records = []

        output_folder = self.output_folder_edit.text()
        dpi = self.dpi_spin.value()
        min_area_px = self.min_area_spin.value()
        connectivity = 4 if self.conn_combo.currentText() == "4-связность" else 8
        remove_frame_flag = self.frame_check.isChecked()
        frame_mode = self.frame_mode_combo.currentText()
        frame_width = self.frame_width_spin.value()
        file_name_column = self.COLUMN_HEADERS[0]

        for image_path in image_paths:
            filename = os.path.basename(image_path)
            try:
                records, result_path = analyze_image(
                    image_path=image_path,
                    output_folder=output_folder,
                    dpi=dpi,
                    min_area_px=min_area_px,
                    connectivity=connectivity,
                    remove_frame_flag=remove_frame_flag,
                    frame_mode=frame_mode,
                    frame_width=frame_width
                )
            except TypeError:
                # На случай если сигнатура analyze_image старая (без дополнительных аргументов)
                records, result_path = analyze_image(
                    image_path,
                    self.output_folder,
                    self.dpi_spin.value()
                )
            except Exception as e:
                QMessageBox.warning(self, "Ошибка обработки", f"{os.path.basename(image_path)}:\n{e}")
                continue

            # Ожидается, что records — список словарей по колонкам таблицы
            for rec in records:
                # Добавим путь обработанного файла, если его нет в записи
                rec[file_name_column] = filename
                if self.PROCESSED_PATH_COLUMN and self.PROCESSED_PATH_COLUMN not in rec and result_path:
                    rec[self.PROCESSED_PATH_COLUMN] = result_path
                all_records.append(rec)

        if not all_records:
            self.img_label.setText("Нет результатов для отображения.")
            return

        # Обновление таблицы
        start_row = self.tbl.rowCount()
        self.tbl.setRowCount(start_row + len(all_records))
        for i, rec in enumerate(all_records, start=start_row):
            for col_idx, col_name in enumerate(self.COLUMN_HEADERS):
                val = rec.get(col_name, "")
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.tbl.setItem(i, col_idx, item)

        # Запоминаем результаты для кликов по таблице
        self.df_records.extend(all_records)

        # UI state
        self.excel_btn.setEnabled(True)
        self.clear_btn.setEnabled(True)

        # Показать первый результат, если есть путь изображения
        processed_col_index = (self.COLUMN_HEADERS.index(self.PROCESSED_PATH_COLUMN)
                               if self.PROCESSED_PATH_COLUMN in self.COLUMN_HEADERS else None)
        if processed_col_index is not None and self.tbl.rowCount() > 0:
            first_path = self.tbl.item(start_row, processed_col_index).text()
            if first_path and os.path.exists(first_path):
                self.display_image(first_path)

    def export_excel(self):
        try:
            import pandas as pd
        except ImportError:
            QMessageBox.warning(self, "Требуется pandas", "Для экспорта в Excel установите пакет pandas.")
            return

        if not self.df_records:
            QMessageBox.information(self, "Пусто", "Нет данных для экспорта.")
            return

        df = pd.DataFrame(self.df_records, columns=self.COLUMN_HEADERS)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        xlsx_path = os.path.join(self.output_folder, f"results_{ts}.xlsx")
        try:
            df.to_excel(xlsx_path, index=False)
            QMessageBox.information(self, "Готово", f"Экспортировано в:\n{xlsx_path}")
        except Exception as e:
            QMessageBox.warning(self, "Ошибка экспорта", f"Не удалось сохранить Excel:\n{e}")

    def _extract_images_from_pdf(self, pdf_path: str) -> str:
        """
        Извлекает изображения из PDF в временную папку внутри output.
        Использует PyMuPDF (fitz). Требует установки: pip install pymupdf
        Возвращает путь к папке с изображениями.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise RuntimeError("Для извлечения изображений из PDF установите пакет 'pymupdf' (PyMuPDF).") from e

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = os.path.join(self.output_folder, f"tmp_pdf_images_{ts}")
        os.makedirs(tmp_dir, exist_ok=True)

        doc = fitz.open(pdf_path)
        img_idx = 0
        for page_index in range(len(doc)):
            page = doc[page_index]
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    # CMYK -> RGB(A)
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                out_path = os.path.join(tmp_dir, f"page{page_index + 1:03d}_img{img_idx + 1:03d}.png")
                pix.save(out_path)
                pix = None
                img_idx += 1
        doc.close()

        if img_idx == 0:
            # Альтернатива: рендерить страницы, если встроенных картинок нет
            for page_index in range(len(doc)):
                page = doc.load_page(page_index)
                pix = page.get_pixmap(dpi=self.dpi_spin.value())
                out_path = os.path.join(tmp_dir, f"page{page_index + 1:03d}.png")
                pix.save(out_path)

        return tmp_dir
