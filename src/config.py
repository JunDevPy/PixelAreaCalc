"""
Файл конфигурации, тут расположены константы которые используется в разных файлах и модулях проекта
"""

COLUMN_HEADERS = ["Имя файла", "№ фигуры", "Кол-во пикселей", "Площадь"]
PROCESSED_PATH_COLUMN = "Путь к обработанному файлу"
MSG_SELECT_IMAGE = "Выберите папку или файл и нажмите 'Обработать'"

# Поддерживаемые расширения входных изображений
IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"
}

# Имена файлов отладочных артефактов в analyze_image
DEBUG_DIR_NAME = "debug"
VISUAL_SUFFIX = "_visual"
LOG_EXT = ".txt"

# Тестовые метки для интерфейса
LABEL_TEXT_ZOOM_IN_BTN = "Увеличить +"
LABEL_TEXT_ZOOM_OUT_BTN = "Уменьшить -"
LABEL_TEXT_FIT_BTN = "По размеру окна"
LABEL_TEXT_TG_LINK = "<a href='https://t.me/PyOpsMaster'>Telegram чат с разработчиком</a>"
LABEL_TEXT_DONATION_LINK = "<a href='https://pay.cloudtips.ru/p/85cd51e7'># Поблагодарить за классный код #</a>"

