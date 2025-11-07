import tensorflow as tf

# Cargar modelo entrenado
model = tf.keras.models.load_model("src/food_model.keras")

# Ver resumen del modelo (opcional)
model.summary()

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np

# Tamaño de entrada del modelo
IMG_SIZE = (224, 224)

# Cargar imagen desde disco
img_path = "src/arroz.jpg"  # ruta local a la imagen
img = image.load_img(img_path, target_size=IMG_SIZE)

# Convertir a array y escalar
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # añadir dimensión batch
img_array = preprocess_input(img_array)

preds = model.predict(img_array)
pred_class = np.argmax(preds, axis=1)[0]
confidence = np.max(preds) * 100


class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 
               'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

print(f"Predicción: {class_names[pred_class]} ({confidence:.2f}%)")
import gradio as gr

from PIL import Image
import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)  

def predict_food(img):
    # Convertir el array numpy (de Gradio) en una imagen PIL
    img = Image.fromarray(np.uint8(img))

    # Redimensionar la imagen
    img = img.resize(IMG_SIZE)

    # Convertir a tensor para el modelo
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)

    preds = model.predict(x)
    top3_idx = np.argsort(preds[0])[::-1][:3]
    top3_labels = [(class_names[i], float(preds[0][i])) for i in top3_idx]

    return {label: prob for label, prob in top3_labels}

import gradio as gr

# Prueba del modelo en Gradio
demo = gr.Interface(
    fn=predict_food,
    inputs=gr.Image(type="numpy", label="Sube una imagen de comida"),
    outputs=gr.Label(num_top_classes=3),
    title="Clasificador de Comidas"
)

demo.launch()

