import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import time

# -----------------------------------------------------------------------------
# CONFIGURACIÓN DE LA PÁGINA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Food Health Analyzer",
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    .stButton>button {
        width: 100%;
        border-radius: 25px;
        background: linear-gradient(45deg, #FF4B2B, #FF416C);
        color: white;
        border: none;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(255, 65, 108, 0.4);
    }
    div[data-testid="stFileUploader"] {
        border: 2px dashed #4c4c4c;
        border-radius: 15px;
        padding: 20px;
    }
    .metric-card {
        background-color: #262730;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CONSTANTES Y CONFIGURACIÓN DEL MODELO
# -----------------------------------------------------------------------------

IMG_SIZE = (224, 224)
MODEL_PATH = "models/food_model.keras"

# Definición de las clases y sus "Health Values" subjetivos (0-100)
FOOD_CLASSES = {
    'Pan': {'index': 0, 'health_val': 40, 'emoji': '🍞'},
    'Producto lácteo': {'index': 1, 'health_val': 55, 'emoji': '🥛'},
    'Postre': {'index': 2, 'health_val': 10, 'emoji': '🧁'},
    'Huevos': {'index': 3, 'health_val': 75, 'emoji': '🥚'},
    'Comida frita': {'index': 4, 'health_val': 15, 'emoji': '🍟'},
    'Carne': {'index': 5, 'health_val': 50, 'emoji': '🥩'},
    'Pasta': {'index': 6, 'health_val': 45, 'emoji': '🍝'},
    'Arroz': {'index': 7, 'health_val': 60, 'emoji': '🍚'},
    'Marisco': {'index': 8, 'health_val': 85, 'emoji': '🦐'},
    'Sopa': {'index': 9, 'health_val': 80, 'emoji': '🍲'},
    'Frutas-Verduras': {'index': 10, 'health_val': 95, 'emoji': '🥦'}
}

CLASS_NAMES = list(FOOD_CLASSES.keys())

# -----------------------------------------------------------------------------
# FUNCIONES DE CARGA Y PREDICCIÓN
# -----------------------------------------------------------------------------

@st.cache_resource
def load_prediction_model():
    """
    Intenta cargar el modelo real. Si falla, devuelve un Mock para demostración.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model, True
    except Exception as e:
        return None, False

def process_image(image_file):
    """Preprocesa la imagen para el modelo"""
    img = Image.open(image_file).convert('RGB')
    img_resized = img.resize(IMG_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img, img_array

def calculate_health_score(probabilities):
    """
    Calcula el score ponderado.
    Health Score = Sum(Probabilidad_Clase * Valor_Salud_Clase)
    """
    weighted_score = 0
    details = []
    
    for i, prob in enumerate(probabilities):
        class_name = CLASS_NAMES[i]
        health_val = FOOD_CLASSES[class_name]['health_val']
        contribution = prob * health_val
        weighted_score += contribution
        
        details.append({
            'Class': class_name,
            'Probability': prob,
            'Health Value': health_val,
            'Contribution': contribution
        })
        
    return weighted_score, details

# -----------------------------------------------------------------------------
# INTERFAZ PRINCIPAL
# -----------------------------------------------------------------------------

def main():
    # Sidebar
    with st.sidebar:
        st.title("🧩 Información")
        st.info("Este sistema usa un modelo modificado de una red neuronal convolucional pre-entrenada (MobileNetV2).")
        st.write("---")
        st.write("**Categorías soportadas:**")
        st.caption(", ".join(CLASS_NAMES))
        st.write("---")
        st.write("**DataSet para el entrenamiento:**")
        st.caption("[**Food-11 Dataset (Kaggle)**](https://www.kaggle.com/datasets/trolukovich/food11-image-dataset)")
        
        # Cargar modelo
        model, model_loaded = load_prediction_model()
        
        st.write("---")
        if model_loaded:
            st.success("✅ Modelo cargado: food_model.keras")
        else:
            st.warning("⚠️ Modelo no encontrado en 'models/'.")
            st.caption("Usando modo demostración (simulación aleatoria) para probar la UI.")

    # Cabecera
    col1, col2 = st.columns([5, 1])
    with col1:
        st.title("AI Food Health Analyzer 🥗🍎")
        st.markdown("### Descubre como de saludable es tu comida con Deep Learning")
        st.markdown("Esta app utiliza un modelo de Deep Learning para clasificar imágenes de alimentos programado con TensorFlow keras, el notebook con el entrenamiento del modelo está disponible en [**GitHub**](https://github.com/DiegoAladren/cnn-food-classifier).")
        st.markdown("Sube una imagen de cualquier comida y obtén una predicción instantánea junto con un 'Health Score' para determinar como de saludable es el alimento.")
        st.markdown("Es importante tener en cuenta que este modelo solo hace una predicción sencilla y que está entrenado con un dataset limitado. No debe usarse para decisiones nutricionales serias. " \
        "Tampoco es recomendable subir imágenes en las que no aparezca comida, el modelo probablemente pensará que todo es un postre 🧁 o una sopa🍲.")
    
    # Área de carga
    uploaded_file = st.file_uploader("Arrastra tu imagen o haz clic para buscar", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Mostrar imagen previa
        col_img, col_data = st.columns([1, 2])
        
        img_pil, img_tensor = process_image(uploaded_file)
        
        with col_img:
            st.image(img_pil, caption="Imagen subida", use_container_width=True) # Updated from use_column_width
            analyze_btn = st.button("🔍 Analizar Comida")

        if analyze_btn:
            with st.spinner('Procesando imagen...'):
                # Simulación de tiempo de cómputo para efecto visual
                time.sleep(1) 
                
                if model_loaded:
                    preds = model.predict(img_tensor)[0]
                else:
                    # MODO DEMO: Generar probabilidades aleatorias que suman 1
                    preds = np.random.dirichlet(np.ones(11), size=1)[0]

                # Cálculos
                health_score, details_list = calculate_health_score(preds)
                df_details = pd.DataFrame(details_list)
                
                # Obtener clase ganadora
                top_idx = np.argmax(preds)
                top_class = CLASS_NAMES[top_idx]
                top_confidence = preds[top_idx] * 100
                top_emoji = FOOD_CLASSES[top_class]['emoji']

            # -----------------------------------------------------------------
            # RESULTADOS
            # -----------------------------------------------------------------
            st.divider()
            
            # 1. Métricas Principales (KPIs)
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.markdown(f"<div class='metric-card'><h3>Predicción</h3><h2>{top_emoji} {top_class}</h2></div>", unsafe_allow_html=True)
            with kpi2:
                color = "#00cc96" if top_confidence > 70 else "#ffa15a"
                st.markdown(f"<div class='metric-card'><h3>Certeza</h3><h2 style='color:{color}'>{top_confidence:.1f}%</h2></div>", unsafe_allow_html=True)
            with kpi3:
                h_color = "#ef553b" # Rojo
                if health_score > 40: h_color = "#ffa15a" # Naranja
                if health_score > 70: h_color = "#00cc96" # Verde
                st.markdown(f"<div class='metric-card'><h3>Health Score</h3><h2 style='color:{h_color}'>{health_score:.1f}/100</h2></div>", unsafe_allow_html=True)

            st.write("")
            st.write("")

            # -----------------------------------------------------------------
            # GRÁFICOS PERSONALIZADOS
            # -----------------------------------------------------------------
            
            c_chart1, c_chart2 = st.columns([1, 1])

            with c_chart1:
                st.subheader("📊 Distribución de Probabilidad")
                # Gráfico de Barras Horizontal Moderno
                df_sorted = df_details.sort_values(by='Probability', ascending=True).tail(5) # Top 5
                fig_bar = px.bar(
                    df_sorted, 
                    x='Probability', 
                    y='Class', 
                    orientation='h',
                    text_auto='.1%',
                    color='Probability',
                    color_continuous_scale='Viridis'
                )
                fig_bar.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                fig_bar.update_xaxes(showgrid=False, range=[0, 1])
                st.plotly_chart(fig_bar, use_container_width=True)

            with c_chart2:
                st.subheader("🎯 Medidor de Salud")
                # Gauge Chart (Velocímetro)
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = health_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Índice Saludable"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "rgba(255, 255, 255, 0.3)"},
                        'steps': [
                            {'range': [0, 40], 'color': "#ef553b"},
                            {'range': [40, 70], 'color': "#ffa15a"},
                            {'range': [70, 100], 'color': "#00cc96"}],
                        'threshold': {
                            'line': {'color': "white", 'width': 4},
                            'thickness': 0.75,
                            'value': health_score}
                    }
                ))
                fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig_gauge, use_container_width=True)

            # -----------------------------------------------------------------
            # ANÁLISIS DE RADAR (Perfil Nutricional Inferido)
            # -----------------------------------------------------------------
            st.subheader("🕸️ Perfil de la Comida")
            
            # Creamos datos para el radar chart
            categories = ['Salud', 'Probabilidad', 'Densidad Calórica (Est)', 'Complejidad']
            
            # Valores estimados para la gráfica (lógica visual)
            r_values = [
                health_score, 
                top_confidence, 
                100 - health_score, # Inverso a salud aprox
                len(df_details[df_details['Probability'] > 0.1]) * 20 # Complejidad basada en confusión del modelo
            ]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=r_values,
                theta=categories,
                fill='toself',
                name=top_class,
                line_color='#FF4B2B'
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # Debug Data Expander
            with st.expander("Ver datos crudos del análisis"):
                st.dataframe(df_details.sort_values(by='Probability', ascending=False).style.format({'Probability': '{:.2%}', 'Contribution': '{:.2f}'}))

if __name__ == "__main__":
    main()