"""
Класс который непосредственно проводит обработку и анализ изображений
"""

import os
from datetime import datetime

import cv2
import numpy as np

from src.file_manager import safe_imread, safe_img_write


def analyze_image(image_path: str, output_folder: str, dpi: int, min_area_px: int, connectivity: int):
    """
    Анализ изображения - поиск фигур, расчет площади,
    сохранение результатов (отладка, визуализация).
    Возвращает: список результатов и путь к итоговому изображению.
    """
    img = safe_imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return [], None

    h, w = img.shape

    debug_root = os.path.join(output_folder, "debug")
    os.makedirs(debug_root, exist_ok=True)
    filename = os.path.basename(image_path)
    name_base = os.path.splitext(filename)[0]
    debug_folder = os.path.join(debug_root, name_base)
    os.makedirs(debug_folder, exist_ok=True)

    log_path = os.path.join(debug_folder, f"{name_base}.txt")
    log = open(log_path, "w", encoding="utf-8")

    def write(line: str):
        log.write(line + "\n")

    try:
        cm_per_px = 2.54 / dpi
        cm2_per_px = cm_per_px ** 2

        write(f"=== ОТЛАДКА для {filename} ===")
        write(f"Размер изображения: {w}x{h} пикселей")
        write(f"DPI: {dpi}")
        write(f"Сантиметров на пиксель: {cm_per_px:.6f}")
        write(f"Площадь одного пикселя: {cm2_per_px:.8f} см²")
        write(f"Минимальная площадь фигуры: {min_area_px} пикселей")
        write(f"Связность: {connectivity}")

        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        white_count = np.sum(binary == 255)
        black_count = np.sum(binary == 0)
        write(f"После бинаризации (OTSU): белых={white_count}, черных={black_count}")

        edges = np.concatenate([binary[0, :], binary[-1, :], binary[:, 0], binary[:, -1]])
        edge_white = np.sum(edges == 255)
        edge_black = np.sum(edges == 0)
        write(f"Анализ краев: белых={edge_white}, черных={edge_black}")

        background_is_white = edge_white > edge_black
        write(f"Фон определен как: {'белый' if background_is_white else 'черный'}")

        if background_is_white:
            binary = cv2.bitwise_not(binary)
            write("Применена инверсия изображения")

        cv2.imwrite(os.path.join(debug_folder, "01_original.png"), img)
        cv2.imwrite(os.path.join(debug_folder, "02_binary.png"), binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
        write("Применены улучшенные морфологические операции")
        cv2.imwrite(os.path.join(debug_folder, "03_morphology.png"), binary)

        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        flood_count = 0
        for corner in corners:
            if binary[corner] == 255:
                cv2.floodFill(binary, flood_mask, corner, (0,))
                flood_count += 1
        write(f"FloodFill применен из {flood_count} углов")
        cv2.imwrite(os.path.join(debug_folder, "04_floodfill.png"), binary)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(binary, contours, (255,))
        write("Применено заполнение дыр")
        cv2.imwrite(os.path.join(debug_folder, "05_filled.png"), binary)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=connectivity)

        actual_components = num_labels - 1
        write(f"Найдено связанных компонентов: {actual_components} (исключая фон)")

        total_area_cm2 = (w * h) * cm2_per_px
        a4_area_cm2 = 21.0 * 29.7
        calibration_factor = a4_area_cm2 / total_area_cm2 if total_area_cm2 > 0 else 1.0

        write(f"\nКАЛИБРОВКА:")
        write(f"Теоретическая площадь изображения: {total_area_cm2:.2f} см²")
        write(f"Площадь листа A4: {a4_area_cm2:.2f} см²")
        write(f"Коэффициент калибровки: {calibration_factor:.4f}")

        results = []
        accepted_count = 0
        filtered_count = 0
        total_accepted_area = 0.0

        write(f"\nАНАЛИЗ КОМПОНЕНТОВ:")

        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in range(1, num_labels):
            area_px = stats[i, cv2.CC_STAT_AREA]
            max_reasonable_area = (w * h) * 0.8

            if area_px < min_area_px:
                write(f"  Компонент {i}: ОТФИЛЬТРОВАН - слишком маленький ({area_px} < {min_area_px} пикселей)")
                filtered_count += 1
                continue

            if area_px > max_reasonable_area:
                write(f"  Компонент {i}: ОТФИЛЬТРОВАН - слишком большой ({area_px} > {max_reasonable_area:.0f} пикселей)")
                filtered_count += 1
                continue

            area_cm2 = area_px * cm2_per_px * calibration_factor
            x, y, w_comp, h_comp = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                                   stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            cx, cy = centroids[i]

            num = len(results) + 1
            results.append({
                "№ фигуры": num,
                "Кол-во пикселей": int(area_px),
                "Площадь": round(area_cm2, 3),
                "Центроид": (int(cx), int(cy)),
                "Рамка": (x, y, w_comp, h_comp)
            })

            accepted_count += 1
            total_accepted_area += area_cm2

            write(f"  Компонент {i}: ПРИНЯТ - площадь {area_px} пикс ({area_cm2:.3f} см²)")

            cv2.rectangle(result_img, (x, y), (x + w_comp, y + h_comp), (0, 255, 0), 5)
            cv2.putText(result_img, str(num), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (0, 0, 255), 5, cv2.LINE_AA)

        write(f"\nРЕЗУЛЬТАТ:")
        write(f"Принято фигур: {accepted_count}")
        write(f"Отфильтровано: {filtered_count}")
        write(f"Общая площадь принятых фигур: {total_accepted_area:.3f} см²")

        result_filename = f"processed_{filename}"
        result_path = os.path.join(output_folder, result_filename)

        if safe_img_write(result_path, result_img):
            write(f"Результат сохранен: {result_filename}")
        else:
            write(f"ОШИБКА: Не удалось сохранить результат: {result_filename}")
            result_path = None

        cv2.imwrite(os.path.join(debug_folder, "06_components.png"), result_img)

        labeled_img = cv2.applyColorMap((labels * 255 // max(num_labels - 1, 1)).astype(np.uint8), cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(debug_folder, "07_labeled.png"), labeled_img)

        debug_files_count = len([f for f in os.listdir(debug_folder) if f.endswith('.png')])
        write(f"Отладочные изображения ({debug_files_count} шт.) сохранены в папку: {os.path.relpath(debug_folder, output_folder)}/")
        write("=" * 50)

        return results, result_path

    except Exception as e:
        write(f"ОШИБКА: {str(e)}")
        import traceback
        write(traceback.format_exc())
        return [], None
    finally:
        log.close()
