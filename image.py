import os

import cv2
import numpy as np


def detect_contours_in_water(image_path):
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось загрузить изображение.")
        return None

    # Преобразование в оттенки серого
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # img_grey = cv2.equalizeHist(img_grey)

    # Применение размытия для снижения шума
    blurred = cv2.GaussianBlur(img_grey, (5, 5), 0)

    # Применение пороговой обработки для выделения объектов
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Морфологические операции для удаления мелких шумов
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Заполнение "дыр" внутри объектов
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    img_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))

    # Отрисовка контуров на исходном изображении
    cv2.drawContours(image, contours, -1, (255,0,0), 3)

    output_dir = 'photos_out'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = os.path.join(output_dir, f"{os.path.basename(image_path)}_contours.png")

    cv2.imwrite(output_path, image)

    # Отображение результата
    # cv2.imshow("Contours", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return contours


if __name__ == "__main__":
    for filename in os.listdir("photos_in"):
        image_path = os.path.join("photos_in", filename)
        detect_contours_in_water(image_path)
