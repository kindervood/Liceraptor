								Проект Лицераптор
Лицераптор позволяет считать лицо человека и проанализировать его: узнать о наличии маски на лице, возраст и настроение.
Позволяет быстро и эффективно обслуживать клиентов, предлагать клинету более персонализированные продукты,
в режиме онлайн подстраиваться под клиента, анализруяя его реакцию. Таким образом проект улучшает качество предоставляемых услуг.

Чтобы запустить код, требуется:
-установить python3
загрузить библиотеки:
DeepFace, face_recognition, imutils, OpenCV

существует два варианты работы:

1) определяет маску на лице
-запустить detect_mask_webcam.py в папке andrey
больше ничего не требуется, улыбаемся и машем)
	
2) определяет атрибуты лица
-запустить facial_req.py в папке lol
при нажатии на пробел, производится анализ вашего возраста и настроения. Информация выводится в консоль и сохраняется в base.txt


Внедрение в код:

Как научить нейронку определять твое лицо? 

Чтобы загрузить себя в базу данных, следует: 

1)зайти в папку dataset/names и создать тут пустую папку с вашим именем на английском

2)открыть файл headshots.py в 6 строке заменить имя Andrey на ваше

3)далее запустить программу и начать двигать головой, переодически нажимая кнопку "space" для фиксирования фотографии в базу данных. Сойдет порядка 30 фотографий.

Далее открыть файл train_model.py и запустить его. На ваших фотографиях он начнет учиться.

