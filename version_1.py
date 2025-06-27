import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1) Параметры
DPI = 300
mm_per_pixel = 25.4 / DPI
cm2_per_pixel = (mm_per_pixel / 10) ** 2
min_area_px = 20  # фильтр мелкого мусора

# 2) Загрузка
img = cv2.imread('img/1.png', cv2.IMREAD_GRAYSCALE)

# 3) Оцу-порог
_, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 4) connectedComponentsWithStats
nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=8)

records = []
h_img, w_img = th.shape
# num_label = 1
for label in range(1, nlabels):
    x, y, w_comp, h_comp, area_px = stats[label]

    # 1) фильтруем «мусор»
    if area_px < min_area_px:
        continue

    # 2) фильтруем всё, что касается границ:
    #    если любая из четырёх сторон боксика = 0 или = размеру картинки
    if x == 0 or y == 0 or x + w_comp >= w_img or y + h_comp >= h_img:
        # num_label +=1
        continue

    area_cm2 = area_px * cm2_per_pixel
    records.append({
        '№ фигуры': len(records) + 1,
        'Площадь, px': int(area_px),
        'Площадь, cm²': round(area_cm2, 3)
    })

# 5) сбор в DataFrame
df = pd.DataFrame(records)
total_px = df['Площадь, px'].sum()
total_cm2 = df['Площадь, cm²'].sum()
df.loc[len(df)] = ['Итого', total_px, round(total_cm2, 3)]
print(df)

# 7) Визуальная проверка - вывод избражения на экран уже бинарника
plt.figure(figsize=(4, 4))
plt.title('bw — фон должен быть чёрный, объекты белые. Тогда все отработало правильно!')
plt.imshow(th, cmap='gray')
plt.axis('off')
plt.show()