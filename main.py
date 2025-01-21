import requests
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
from io import BytesIO
import os
import shutil
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.models import Sequential
from PIL import Image
from sklearn.utils import shuffle


base_cats_url = "https://catsoftheweb.com/cats/?cst"
base_dogs_url = "https://lzsfoto.ru/fotootchet/vistavki/2010_omsk/2010_omsk_kob/"
#base_cats_url = "https://www.kaggle.com/datasets/crawford/cat-dataset"
#base_dogs_url = "https://sobakovod.club/"

all_cat_links = []
all_dog_links = []


for page_num in range(1, 10):
    url_c = f"https://catsoftheweb.com/cats/?query-0-page={page_num}&cst"
    #print(url_c)
    response_1 = requests.get(url_c, headers={'User-Agent': UserAgent().chrome})
    html_1 = response_1.content
    soup_1 = BeautifulSoup(html_1, 'html.parser')
    obj_1 = soup_1.find(lambda tag: tag.name == 'img')
    src_1 = obj_1.attrs['src']

    cat_links = soup_1.findAll(lambda tag: tag.name == 'img')
    cat_links = [link.attrs['src'] for link in cat_links]
    all_cat_links.extend(cat_links)
    
    url_d = f"{base_dogs_url}/page/{page_num}/"
    response_2 = requests.get(url_d, headers={'User-Agent': UserAgent().chrome})
    html_2 = response_2.content
    soup_2 = BeautifulSoup(html_2, 'html.parser')
    obj_2 = soup_2.find(lambda tag: tag.name == 'img')
    src_2 = obj_2.attrs['src']
    dog_links = soup_2.findAll(lambda tag: tag.name == 'img')
    dog_links = [link.attrs['src'] for link in dog_links]
    cleaned_dog_links = []
    for link in dog_links:
        start = link.find("src=") + len("src=")
        end = link.find("&", start)
        cleaned_link = link[start:end] if "src=" in link else link
        cleaned_dog_links.append(cleaned_link)
    all_dog_links.extend(cleaned_dog_links)
    
response_2 = requests.get(base_dogs_url, headers={'User-Agent': UserAgent().chrome})
html_2 = response_2.content
soup_2 = BeautifulSoup(html_2, 'html.parser')
obj_2 = soup_2.find(lambda tag: tag.name == 'img')
src_2 = obj_2.attrs['src']
dog_links = soup_2.findAll(lambda tag: tag.name == 'img')
dog_links = [link.attrs['src'] for link in dog_links]

#print(dog_links[:3])
#print(len(dog_links))
#print(len(all_cat_links))

def download_and_save_image(url, save_dir, image_size=(128, 128)):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize(image_size)
        filename = os.path.join(save_dir, url.split('/')[-1])
        img.save(filename)
    except Exception as e:
        print(f"Ошибка при загрузке {url}: {e}")

os.makedirs('data/cats', exist_ok=True)
os.makedirs('data/dogs', exist_ok=True)

image_size = (128, 128)

print("Сохранение изображений кошек...")
for url in all_cat_links:
    download_and_save_image(url, 'data/cats', image_size)

print("Сохранение изображений собак...")
for url in dog_links:
    download_and_save_image(url, 'data/dogs', image_size)

cat_dir = "data/cats"
dog_dir = "data/dogs"

def load_images_from_directory(directory, label):
    images = []
    labels = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            img = load_img(file_path, target_size=image_size)
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)
        except Exception as e:
            print(f"Ошибка загрузки {file_path}: {e}")
    return images, labels

###
image_size = (128, 128)

# Подготовка данных
cat_images, cat_labels = load_images_from_directory(cat_dir, label=0)
dog_images, dog_labels = load_images_from_directory(dog_dir, label=1)

all_images = np.array(cat_images + dog_images)
all_labels = np.array(cat_labels + dog_labels)

# Перемешивание, сохраняя соответствие изображения и подписи
all_images, all_labels = shuffle(all_images, all_labels, random_state=42)

# Масштабирование и преобразование в градации серого
#all_images = all_images / 255.0
all_images = tf.image.rgb_to_grayscale(all_images).numpy()

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(all_images, all_labels, test_size=0.2, random_state=42)

print(f"Обучающая выборка: {X_train.shape[0]} изображений")
print(f"Тестовая выборка: {X_test.shape[0]} изображений")

# Определение модели
model = Sequential([
    Conv2D(8, 3, padding='same', activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2), strides=2, padding='valid'),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Обучение модели
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Предсказания
predictions = model.predict(X_test[:20])
print("Предсказания:", np.argmax(predictions, axis=1))
print("Истинные метки:", y_test[:20])

class_names = [
    'Cat', 'Dog'
]
# Визуализация предсказаний
for n in range(20):
    plt.imshow(X_test[n].reshape(128, 128), cmap=plt.cm.binary)
    plt.show()
    print("Ожидание: ", class_names[y_test[n]],
          "Результат: ", class_names[np.argmax(predictions[n])])


'''
page_link_1 = "https://catsoftheweb.com/cats/"
response_1 = requests.get(page_link_1, headers={'User-Agent':UserAgent().chrome})
html_1 = response_1.content
soup_1 = BeautifulSoup(html_1, 'html.parser')
obj_1 = soup_1.find(lambda tag: tag.name=='img')
src_1 = obj_1.attrs['src']
print(type(obj_1.attrs['src']))

page_link_2 = "https://sobakovod.club/"
response_2 = requests.get(page_link_2, headers={'User-Agent':UserAgent().chrome})
html_2 = response_2.content
soup_2 = BeautifulSoup(html_2, 'html.parser')
obj_2 = soup_2.find(lambda tag: tag.name=='img')
src_2 = obj_2.attrs['src']
print(type(obj_2.attrs['src']))


cat_links = soup_1.findAll(lambda tag: tag.name=='img')
cat_links = [link.attrs['src'] for link in cat_links]
print(len(cat_links))
dog_links = soup_2.findAll(lambda tag: tag.name == 'img')
dog_links = [link.attrs['src'] for link in dog_links]
cleaned_dog_links = []
for link in dog_links:
    start = link.find("src=") + len("src=")
    end = link.find("&", start)
    cleaned_link = link[start:end] if "src=" in link else link
    cleaned_dog_links.append(cleaned_link)
print(len(cleaned_dog_links))
'''


'''
print(f"Обучающая выборка: {X_train.shape[0]} изображений")
print(f"Тестовая выборка: {X_test.shape[0]} изображений")

class_names = [
    'Cat', 'Dog'
]

# Определение модели
num_filters = 8
filter_size = 3
pool_size = 2

model = Sequential([
    Conv2D(num_filters, filter_size, padding='same', activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), strides=2, padding='valid'),
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    MaxPooling2D((2, 2), strides=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

print(y_train[:5])
#метки- целые числа значит можно использовать sparse_categorical_crossentropy
#тогда в fit не нужен to_categorical

# Обучение модели
model.fit(
    X_train,
    y_train,
    epochs=3,
    validation_data=(X_test, y_test)
)

# Предсказания
predictions = model.predict(X_test[:10])
print(np.argmax(predictions, axis=1))
print(y_test[:10])

# Визуализация предсказаний
for n in range(10):
    plt.imshow(X_test[n].reshape(28, 28), cmap=plt.cm.binary)
    plt.show()
    print("Ожидание: ", class_names[y_test[n]],
          "Результат: ", class_names[np.argmax(predictions[n])])
'''
