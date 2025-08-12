"""
Класс для работы с PDF и изображениями которые в PDF
"""

import os
from typing import List

import fitz


def images_from_pdf(pdf_path: str, out_folder: str) -> List[str]:
    """
    Извлекает изображения как файловые пути (PNG).
    Для сканов-страниц делаем рендер страницы.
    """
    os.makedirs(out_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    saved = []
    for pno in range(len(doc)):
        page = doc[pno]
        # Пробуем "как есть" изображения
        for i, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:
                pix = fitz.Pixmap(fitz.csRGB, pix)
            out = os.path.join(out_folder, f"pdfimg_p{pno}_{i}.png")
            pix.save(out)
            saved.append(out)
        # Если на странице нет встроенных изображений — рендер всей страницы
        if not any(page.get_images(full=True)):
            mat = fitz.Matrix(2, 2)  # x2 масштаб для качества
            pix = page.get_pixmap(matrix=mat, alpha=False)
            out = os.path.join(out_folder, f"pdfrender_p{pno}.png")
            pix.save(out)
            saved.append(out)
    return saved
