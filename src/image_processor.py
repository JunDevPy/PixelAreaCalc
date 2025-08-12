"""
Класс который непосредственно проводит обработку и анализ изображений
"""

import cv2
import numpy as np
import os
from typing import Tuple
from src.file_manager import safe_imread, safe_img_write


def _run_length_from_start(arr: np.ndarray, tolerance: int, start_val: int) -> int:
    """Считает длину начального пробега значений, близких к start_val (по оси arr)."""
    count = 0
    for px in arr:
        if abs(int(px) - int(start_val)) <= tolerance:
            count += 1
        else:
            break
    return count


def _estimate_edge_thicknesses(binary: np.ndarray, tolerance: int = 10, samples: int = 50,
                               max_frac: float = 0.25, ) -> Tuple[int, int, int, int]:
    """
    Оценка толщины рамки по каждому краю: (top, right, bottom, left).
    Используется несколько срезов и берётся низкий перцентиль для устойчивости.
    """
    h, w = binary.shape

    # какие строки/столбцы пробовать
    rows = np.linspace(0, h - 1, num=min(samples, h), dtype=int)
    cols = np.linspace(0, w - 1, num=min(samples, w), dtype=int)

    left_runs = []
    right_runs = []
    top_runs = []
    bottom_runs = []

    # Левый/правый: считаем по строкам, вдоль X
    for r in rows:
        row = binary[r, :]

        # слева
        left_start = int(row[0])
        left_runs.append(_run_length_from_start(row, tolerance, left_start))

        # справа
        right_start = int(row[-1])
        right_runs.append(_run_length_from_start(row[::-1], tolerance, right_start))

    # Верх/низ: считаем по столбцам, вдоль Y
    for c in cols:
        col = binary[:, c]

        # сверху
        top_start = int(col[0])
        top_runs.append(_run_length_from_start(col, tolerance, top_start))

        # снизу
        bottom_start = int(col[-1])
        bottom_runs.append(_run_length_from_start(col[::-1], tolerance, bottom_start))

    # Берём «консервативную» оценку: 10-й перцентиль (а не максимум),
    # чтобы единичные большие участки сплошного фона не сносили всё.
    def robust_width(runs, limit):
        if not runs:
            return 0
        val = int(np.percentile(runs, 10))
        # если пробег практически равен всему размеру — считаем, что рамки нет
        if val >= limit * 0.95:
            return 0
        # ограничим разумной долей от размера
        return int(min(val, limit * max_frac))

    top = robust_width(top_runs, h)
    bottom = robust_width(bottom_runs, h)
    left = robust_width(left_runs, w)
    right = robust_width(right_runs, w)

    return top, right, bottom, left


def estimate_frame_width(binary: np.ndarray, tolerance: int = 10) -> int:
    """
    Оставлено для обратной совместимости: возвращает максимальную толщину
    по краям, используя улучшённую оценку.
    """
    t, r, b, l = _estimate_edge_thicknesses(binary, tolerance=tolerance)
    return max(t, r, b, l)


def remove_frame(
        binary: np.ndarray,
        remove_frame_flag: bool,
        frame_mode: str = "auto",
        frame_width_px: int = 0,
        tolerance: int = 10,
        debug_folder: str = None,
        per_edge: bool = True,
) -> np.ndarray:
    """
    Убирает рамку в бинарном изображении, устанавливая её в 0 (чёрный).
    Внимание: работает с копией входного массива.
    """
    if not remove_frame_flag:
        return binary

    assert binary.ndim == 2, "Ожидается бинарное изображение (H, W)"
    h, w = binary.shape
    out = binary.copy()

    if frame_mode.lower() == "fixed":
        fw = max(0, int(frame_width_px))
        top = bottom = left = right = fw
    else:
        # автооценка по каждому краю
        top, right, bottom, left = _estimate_edge_thicknesses(out, tolerance=tolerance)
        if not per_edge:
            # опционально можно использовать единое значение
            fw = max(top, right, bottom, left)
            top = bottom = left = right = fw

    # Дополнительные предохранители: не позволяем «съесть» более трети изображения
    max_top = min(top, h // 3)
    max_bottom = min(bottom, h // 3)
    max_left = min(left, w // 3)
    max_right = min(right, w // 3)

    if max_left > 0:
        out[:, :max_left] = 0
    if max_right > 0:
        out[:, w - max_right:] = 0
    if max_top > 0:
        out[:max_top, :] = 0
    if max_bottom > 0:
        out[h - max_bottom:, :] = 0

    if debug_folder:
        os.makedirs(debug_folder, exist_ok=True)
        cv2.imwrite(os.path.join(debug_folder, "03b_frame_removed.png"), out)

    return out


def analyze_image(image_path: str, output_folder: str, dpi: int, min_area_px: int, connectivity: int,
                  remove_frame_flag: bool = False, frame_mode: str = 'auto', frame_width: int = 30):
    """
    Анализ изображения - поиск фигур, расчет площади,
    сохранение результатов (отладка, визуализация).
    Возвращает: список результатов и путь к итоговому изображению.
    Добавлена опция удаления рамки.
    """
    img = safe_imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # raise FileNotFoundError(f"Не удалось открыть {image_path}")
        return [], None

    # Удаление рамки, если включено
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # img = remove_frame(img, remove_frame_flag, frame_mode, frame_width)
    img = remove_frame(binary=binary_img, remove_frame_flag=remove_frame_flag, frame_mode=frame_mode,
                       frame_width_px=frame_width, tolerance=12, debug_folder="debug", per_edge=True, )

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
                write(f"  Компонент {i}: ОТФИЛЬТРОВАН - слишком большой "
                      f"({area_px} > {max_reasonable_area:.0f} пикселей)")
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
        write(
            f"Отладочные изображения ({debug_files_count} шт.) сохранены в папку: "
            f"{os.path.relpath(debug_folder, output_folder)}/")
        write("=" * 50)

        return results, result_path

    except Exception as e:
        write(f"ОШИБКА: {str(e)}")
        import traceback
        write(traceback.format_exc())
        return [], None
    finally:
        log.close()
