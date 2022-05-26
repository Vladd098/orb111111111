import cv2

# Включаем первую камеру
cap = cv2.VideoCapture(0)

# "Прогреваем" камеру, чтобы снимок не был тёмным
for i in range(30):
	cap.read()

	# Делаем снимок
	ret, frame = cap.read()

# Записываем в файл
cv2.imwrite('cam.png', frame)

# Отключаем камеру
cap.release()