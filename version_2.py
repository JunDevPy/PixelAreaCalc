import os
from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1) Параметры
name_image = datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
folder_image_result = 'result_img'
DPI = 300
mm_per_pixel = 25.4 / DPI
cm2_per_pixel = (mm_per_pixel / 10) ** 2
min_area_px = 20  # фильтр мелкого мусора
image = 'img/1.png'
save_path = f'{folder_image_result}/{name_image}'

if not os.path.exists(folder_image_result):
    os.mkdir(folder_image_result)

# 2) Загрузка исходного изображения и его вывод в консоль

img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title('Исходный исходный загруженный скан')
plt.axis('off')
plt.show()

# 3) Определение порога и автоинверсия если требуется
_, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
if np.count_nonzero(th == 255) > np.count_nonzero(th == 0):
    th = cv2.bitwise_not(th)

# морфология
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

# 4) connectedComponents
nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)
h_img, w_img = th.shape
records = []
shape_coords = []  # для хранения координат фигур

for label in range(1, nlabels):
    x, y, w_comp, h_comp, area_px = stats[label]
    if area_px < min_area_px:
        continue
    # фильтр «касающихся края»
    if x == 0 or y == 0 or x + w_comp >= w_img or y + h_comp >= h_img:
        continue

    records.append({
        '№ фигуры': len(records) + 1,
        'Площадь, px': int(area_px),
        'Площадь, кв.см': round(area_px * cm2_per_pixel, 3)
    })

    # Сохраняем координаты для маркировки
    shape_coords.append({
        'x': x,
        'y': y,
        'width': w_comp,
        'height': h_comp,
        'area_cm2': round(area_px * cm2_per_pixel, 3),
        'shape_num': len(records)
    })

df = pd.DataFrame(records)

# 5) Визуализация получившегося изображения после обработки
plt.figure(figsize=(4, 4))
plt.title('Проверка изображения\nфон должен быть чёрный, объекты белые')
plt.imshow(th, cmap='gray')
plt.axis('off')
plt.show()


# print(shape_coords)


# 6) НОВАЯ ФУНКЦИЯ: Маркировка фигур на нормализованном изображении

def create_marked_image(original_img, shape_coordinates):
    """
    Создает изображение с маркированными фигурами
    """
    # Нормализация исходного изображения
    normalized_img = cv2.normalize(original_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Создаем цветное изображение для маркировки
    marked_img = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)

    # Настройки для текста и прямоугольников
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.6
    color = (255, 0, 0)  # красный цвет
    thickness = 7

    for shape in shape_coordinates:
        _x, _y, width, height = shape['x'], shape['y'], shape['width'], shape['height']
        area_cm2 = shape['area_cm2']
        shape_num = shape['shape_num']

        # НОВОЕ: Заливка внутренней области фигуры белым
        # cv2.rectangle(marked_img, (x, y), (x + width, y + height), (200 , 255, 255), -1)

        # Рисуем рамку вокруг фигуры
        cv2.rectangle(marked_img, (_x, _y), (_x + width, _y + height), color, thickness)

        # Подготавливаем текст
        text_lines = [
            f"#{shape_num}",
            # f"{area_cm2:.3f} cm2"
        ]

        # Позиция для текста (слева сверху от прямоугольника)
        # text_x = x
        text_x = max(_x + 10, 30)
        text_y = max(_y + 80, 200)  # минимум 30 пикселей от верха

        # Рисуем текст
        for i, line in enumerate(text_lines):
            line_y = text_y + i * 55  # increased line spacing
            cv2.putText(marked_img, line, (text_x, line_y),
                        font, font_scale, color, thickness)

    if save_path:
        cv2.imwrite(save_path, marked_img)
        print(f"Изображение сохранено: {save_path}")

    return marked_img


# 7) Создание и отображение маркированного изображения (только если DataFrame не пустой)
if not df.empty:
    marked_image = create_marked_image(img, shape_coords)

    plt.figure(figsize=(10, 8))
    plt.imshow(marked_image)
    plt.title('Изображение с маркированными фигурами')
    plt.axis('off')
    plt.show()

# 8) Итоговая строка в таблице DataFrame
summary = {
    '№ фигуры': 'Итого',
    'Площадь, px': df['Площадь, px'].sum() if not df.empty else 0,
    'Площадь, кв.см': round(df['Площадь, кв.см'].sum(), 3) if not df.empty else 0
}
df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)

# 9) Выводим в лог итоговый DataFrame (таблицу)
print("\nТаблица с результатами анализа:")
print(df.to_string(index=False))
