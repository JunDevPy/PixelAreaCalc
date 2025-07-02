"""
Простой и удобный GUI-инструмент для автоматического расчета площади объектов (фигур) на изображениях.
Приложение анализирует выбранные файлы, находит на них отдельные фигуры,
вычисляет их площадь в пикселях и квадратных сантиметрах,
а затем представляет результаты в виде таблицы и размеченного изображения.
"""


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
        self.dpi_spin = QSpinBox()
        self.dpi_spin.setMinimum(50)
        self.dpi_spin.setMaximum(1200)
        self.dpi_spin.setValue(300)  # Значение по-умолчанию
        self.dpi_spin.setSuffix(" DPI")

        self.min_area_spin = QSpinBox()
        self.min_area_spin.setMinimum(1)
        self.min_area_spin.setMaximum(100000)
        self.min_area_spin.setValue(1000)  # Значение по-умолчанию
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

        # ===============================================================
        # ДОНАТ
        # ===============================================================

        left_layout.addStretch(2)  # Добавим побольше отступ перед этим блоком
        tg_link = QLabel("<a href='https://t.me/PyOpsMaster'>Telegram чат с разработчиком</a>")
        tg_link.setTextFormat(Qt.RichText)
        tg_link.setOpenExternalLinks(True)
        tg_link.setAlignment(Qt.AlignCenter)
        donation_link = QLabel("<a href='https://pay.cloudtips.ru/p/85cd51e7'>"
                               "# Поблагодарить за классный код #</a>")
        donation_link.setTextFormat(Qt.RichText)
        donation_link.setOpenExternalLinks(True)
        donation_link.setAlignment(Qt.AlignCenter)
        left_layout.addWidget(tg_link)
        left_layout.addWidget(donation_link)

        # ===============================================================
        # ДОНАТ КОНЕЦ
        # ===============================================================

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
        # self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
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
        DPI = self.dpi_spin.value()
        min_area_px = self.min_area_spin.value()
        conn = 4 if self.conn_combo.currentIndex() == 0 else 8
        cm_per_px = 2.54 / DPI
        cm2_per_px = cm_per_px ** 2

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return [], None

        h, w = img.shape

        debug_root = os.path.join(self.output_folder, "debug")
        os.makedirs(debug_root, exist_ok=True)
        filename = os.path.basename(image_path)
        name_base = os.path.splitext(filename)[0]
        debug_folder = os.path.join(debug_root, name_base)
        os.makedirs(debug_folder, exist_ok=True)

        log_path = os.path.join(debug_folder, f"{name_base}.txt")
        log = open(log_path, "w", encoding="utf-8")

        def write(line):
            log.write(line + "\n")

        try:
            write(f"=== ОТЛАДКА для {filename} ===")
            write(f"Размер изображения: {w}x{h} пикселей")
            write(f"DPI: {DPI}")
            write(f"Сантиметров на пиксель: {cm_per_px:.6f}")
            write(f"Площадь одного пикселя: {cm2_per_px:.8f} см²")
            write(f"Минимальная площадь фигуры: {min_area_px} пикселей")
            write(f"Связность: {conn}")

            # Бинаризация
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            white_count = np.sum(binary == 255)
            black_count = np.sum(binary == 0)
            write(f"После бинаризации (OTSU): белых={white_count}, черных={black_count}")

            # Анализ краев для определения фона
            edges = np.concatenate([
                binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]
            ])
            edge_white = np.sum(edges == 255)
            edge_black = np.sum(edges == 0)
            write(f"Анализ краев: белых={edge_white}, черных={edge_black}")

            # Определение фона
            background_is_white = edge_white > edge_black
            write(f"Фон определен как: {'белый' if background_is_white else 'черный'}")

            # Инверсия если нужно (объекты должны быть белыми на черном фоне)
            if background_is_white:
                binary = cv2.bitwise_not(binary)
                write("Применена инверсия изображения")

            # Сохранение промежуточных результатов
            cv2.imwrite(os.path.join(debug_folder, "01_original.png"), img)
            cv2.imwrite(os.path.join(debug_folder, "02_binary.png"), binary)

            # Морфологические операции для улучшения качества
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            write("Применены улучшенные морфологические операции")
            cv2.imwrite(os.path.join(debug_folder, "03_morphology.png"), binary)

            # FloodFill из углов для удаления краевых артефактов
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
            flood_count = 0

            for corner in corners:
                if binary[corner] == 255:  # Если угол белый (объект)
                    # cv2.floodFill(binary, flood_mask, corner, 0)
                    cv2.floodFill(binary, flood_mask, corner, (0,))
                    flood_count += 1

            write(f"FloodFill применен из {flood_count} углов")
            cv2.imwrite(os.path.join(debug_folder, "04_floodfill.png"), binary)

            # Заполнение дыр
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # cv2.fillPoly(binary, contours, 255)
            cv2.fillPoly(binary, contours, (255,))
            write("Применено заполнение дыр")
            cv2.imwrite(os.path.join(debug_folder, "05_filled.png"), binary)

            # Поиск связанных компонентов
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary, connectivity=conn
            )

            # Исключаем фон (метка 0)
            actual_components = num_labels - 1
            write(f"Найдено связанных компонентов: {actual_components} (исключая фон)")

            # Калибровка
            total_area_cm2 = (w * h) * cm2_per_px
            a4_area_cm2 = 21.0 * 29.7
            calibration_factor = a4_area_cm2 / total_area_cm2 if total_area_cm2 > 0 else 1.0

            write(f"\nКАЛИБРОВКА:")
            write(f"Теоретическая площадь изображения: {total_area_cm2:.2f} см²")
            write(f"Площадь листа A4: {a4_area_cm2:.2f} см²")
            write(f"Коэффициент калибровки: {calibration_factor:.4f}")

            # Анализ компонентов
            results = []
            accepted_count = 0
            filtered_count = 0
            total_accepted_area = 0.0

            write(f"\nАНАЛИЗ КОМПОНЕНТОВ:")

            # Создаем цветное изображение для визуализации
            result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            for i in range(1, num_labels):  # Пропускаем фон (i=0)
                area_px = stats[i, cv2.CC_STAT_AREA]

                # Исправленная логика фильтрации
                max_reasonable_area = (w * h) * 0.8  # Максимум 80% от общей площади

                if area_px < min_area_px:
                    write(f"  Компонент {i}: ОТФИЛЬТРОВАН - слишком маленький ({area_px} < {min_area_px} пикселей)")
                    filtered_count += 1
                    continue

                if area_px > max_reasonable_area:
                    write(
                        f"  Компонент {i}: ОТФИЛЬТРОВАН - "
                        f"слишком большой ({area_px} > {max_reasonable_area:.0f} пикселей)")
                    filtered_count += 1
                    continue

                # Компонент принят
                area_cm2 = area_px * cm2_per_px * calibration_factor
                x, y, w_comp, h_comp = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[
                    i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                cx, cy = centroids[i]

                num = len(results) + 1
                results.append({
                    "№ фигуры": num,
                    "Кол-во пикселей": int(area_px),
                    "Площадь": round(area_cm2, 3),  # Округляем до 3 знаков после запятой
                    "Центроид": (int(cx), int(cy)),
                    "Рамка": (x, y, w_comp, h_comp)
                })

                accepted_count += 1
                total_accepted_area += area_cm2

                write(f"  Компонент {i}: ПРИНЯТ - площадь {area_px} пикс ({area_cm2:.3f} см²)")

                # Рисуем рамку вокруг принятого компонента
                cv2.rectangle(result_img, (int(x), int(y)), (int(x + w_comp), int(y + h_comp)), (0, 255, 0), 5)
                label = f"{num}"
                cv2.putText(result_img, label, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                            3, (0, 0, 255), 5, cv2.LINE_AA)

            write(f"\nРЕЗУЛЬТАТ:")
            write(f"Принято фигур: {accepted_count}")
            write(f"Отфильтровано: {filtered_count}")
            write(f"Общая площадь принятых фигур: {total_accepted_area:.3f} см²")

            # Сохраняем результат
            output_filename = f"{name_base}_processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            output_path = os.path.join(self.output_folder, output_filename)
            cv2.imwrite(output_path, result_img)
            write(f"Результат сохранен: {output_filename}")

            # Сохраняем дополнительные отладочные изображения
            cv2.imwrite(os.path.join(debug_folder, "06_components.png"), result_img)

            # Создаем изображение с подписанными компонентами
            labeled_img = cv2.applyColorMap((labels * 255 // num_labels).astype(np.uint8), cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(debug_folder, "07_labeled.png"), labeled_img)

            debug_files_count = len([f for f in os.listdir(debug_folder) if f.endswith('.png')])
            write(
                f"Отладочные изображения ({debug_files_count} шт.) "
                f"сохранены в папку: {os.path.relpath(debug_folder, self.output_folder)}/")
            write("=" * 50)

            result_filename = f"processed_{os.path.basename(image_path)}"
            result_path = os.path.join(self.output_folder, result_filename)
            cv2.imwrite(result_path, result_img)

            return results, result_path

        except Exception as e:
            write(f"ОШИБКА: {str(e)}")
            import traceback
            write(traceback.format_exc())
            return [], None
        finally:
            log.close()

    def process_single_file(self):
        """Метод для обработки одного файла"""
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
