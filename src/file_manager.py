"""
Класс в котором содержатся методы для работы с файлами и путями
"""

import os
import numpy as np
import cv2


def normalize_path(path: str) -> str:
    """Нормализация пути, поддержка русских символов"""
    return os.path.normpath(os.path.abspath(path))


def check_path_encoding(path: str) -> bool:
    try:
        if os.path.isdir(path):
            os.listdir(path)
        elif os.path.isfile(path):
            with open(path, 'rb'):
                pass
        return True
    except (UnicodeDecodeError, UnicodeEncodeError, OSError) as e:
        print(f"Проблема с кодировкой пути {path}: {e}")
        return False


def safe_imread(image_path: str, flags=cv2.IMREAD_GRAYSCALE):
    """Чтение изображения с поддержкой русских путей"""
    try:
        with open(image_path, 'rb') as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, flags)
        return img
    except Exception as e:
        print(f"Ошибка чтения файла {image_path}: {e}")
        return None


def safe_img_write(filename: str, img) -> bool:
    """Запись изображения с поддержкой русских путей"""
    try:
        ext = os.path.splitext(filename)[1]
        if not ext:
            ext = '.png'
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            with open(filename, 'wb') as f:
                f.write(encoded_img)
            return True
        return False
    except Exception as e:
        print(f"Ошибка записи файла {filename}: {e}")
        return False
