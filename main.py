import cv2


def main():
    # Инициализация камеры
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Не удалось открыть камеру")
        return

    while True:
        # Захват кадра
        ret, frame = cap.read()

        # Проверка успешности захвата кадра
        if not ret:
            print("Не удалось получить кадр. Выход...")
            break

        # Обработка кадра
        processed_frame, img_grey, contours = process_frame(frame)

        # Отображение обработанного кадра
        cv2.imshow('Detected counters', processed_frame)
        cv2.imshow('Gray', img_grey)
        cv2.imshow('Counters', contours)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    # Преобразование в оттенки серого
    img_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Повышение контрастности
    img_grey = cv2.equalizeHist(img_grey)

    # Применение размытия для снижения шума
    blurred = cv2.GaussianBlur(img_grey, (5, 5), 0)

    # Применение пороговой обработки для выделения объектов
    ret, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Морфологические операции для удаления мелких шумов
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Заполнение "дыр" внутри объектов
    thresh = cv2.dilate(thresh, kernel, iterations=3)

    edged = cv2.Canny(img_grey, 50, 150)

    # Поиск контуров
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)

    output = frame.copy()
    # Отрисовка контуров на исходном изображении
    cv2.drawContours(output, contours, -1, (255,0,0), 3)

    return output, img_grey, edged

if __name__ == "__main__":
    main()
