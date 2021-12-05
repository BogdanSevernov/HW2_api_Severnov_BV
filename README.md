Документация к API 

С помощью API можно решать две задачи:
1) классификация
2) регрессия

Для получения информации о доступных моделях необходимо написать get запрос c url 'http://localhost:8080/api/ml_models_info'(В ответ будет выведен json)
Для обучения модели необходимо подготовить dataframe. Dataframe обрабатывать не нужно, API делает это сама. Чтобы передать в API данные для обучения нужно передать в post запрос словарь с необходимыми данными. Структура словаря { 'data': 'Данные которые хотим передать', # Прежде чем передвать Dataframe его нужно преобразовать в json. Сделать это можно с помощью метода to_json() 'args': гипер параметры моделей, # гипер параметры нужно передавать в виде словаря, например {'max_d':5,'lr':0.001} 'target': название целевой переменной, # это переменная, которую мы хоти прогнозировать если решаем задачу регресии или имина классов, в случае задачи классификации 'model_name': название модели # назавание сохраненной модели } url для обучения модели 'http://127.0.0.1:5000/api/ml_models/train/id' # id - это id модели, его можно узнать в ответе get запроса из пункта 1, и указать в url.
Для тестирования модели необходимо подготовить Dataframe с тестовыми данными (без целевой переменной) и передать в api. Чтобы передать в API данные для обучения нужно передать в post запрос словарь с необходимыми данными. Структура словаря: { 'data': 'Данные которые хотим передать', # Прежде чем передвать Dataframe его нужно преобразовать в json. Сделать это можно с помощью метода to_json() 'model_name': название модели # назавание сохраненной модели } url для обучения модели 'http://localhost:8080/api/ml_models/test/id' # id - это id модели, его можно узнать в ответе get запроса из пункта 1, и указать в url.
Для удаления моделей нужно написать delete запрос в API. В качесве параметра запроса необходимо передать название файла с моделью.
Чтобы прочитать информацию о модели из БД нужно написать get запрос с url 'http://localhost:8080/api/read/1' и указать id модели.
Для удаления информации о модели из БД нужно написать delete запрос с url 'http://localhost:8080/api/delete/1' и указать id модели.
Код самой API лежит в main.py Код для тестирования APi лежит в test.py.
Ссылка на docker hub: https://hub.docker.com/r/24534543/bogdan_api