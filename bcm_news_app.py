# Enlace de Streamlit:
# https://bcmnewsapp.streamlit.app

from datetime import datetime  # Librerías para manipulación de fechas y tiempos
import matplotlib.pyplot as plt  # Librería para visualización de datos en 2D
from matplotlib.patches import Patch  # Módulo para definir formas y patrones en gráficos de Matplotlib
import pandas as pd  # Estructuras de datos para análisis de datos
import plotly.express as px  # Librería para gráficos interactivos y visualizaciones (Plotly Express)
import plotly.graph_objects as go  # Librería para crear figuras personalizadas y visualizaciones complejas (Plotly Graph Objects)
import plotly.io as pio  # Módulo para guardar y mostrar figuras de Plotly

# Librerías para aprendizaje automático con scikit-learn
from sklearn import tree  
from sklearn.feature_extraction.text import CountVectorizer  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix  
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold  
from sklearn.tree import DecisionTreeClassifier  
from sklearn.tree import plot_tree

import streamlit as st  # Librería para crear aplicaciones web interactivas
import re  # Módulo para trabajar con expresiones regulares
from collections import Counter  # Módulo para contar elementos en una lista
from wordcloud import WordCloud  # Librería para crear nubes de palabras visualmente atractivas
import itertools  # Módulo para trabajar con combinaciones y permutaciones
from sentiment_analysis_spanish import sentiment_analysis  # Librería para análisis de sentimientos en español
#import spacy  # Procesamiento de lenguaje natural con spaCy

# Módulos para resumen de texto con sumy
from sumy.parsers.plaintext import PlaintextParser  
from sumy.nlp.tokenizers import Tokenizer  
from sumy.summarizers.lsa import LsaSummarizer  

# Módulos de procesamiento de texto con NLTK
import nltk  
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize  
from nltk.stem import WordNetLemmatizer  
from nltk import pos_tag, ne_chunk

# Módulos para realizar solicitudes HTTP y realizar análisis de HTML
import requests  
from bs4 import BeautifulSoup  

from collections import Counter

# Configuración de la página
st.set_page_config(
    page_title="Desarrollo de un Sistema de Análisis de Noticias Salvadoreñas",
    layout="wide",
    initial_sidebar_state="expanded")

# Estilos personalizados para Streamlit
st.markdown("""
<style>
[data-testid="block-container"] {
    padding-left: 2rem;
    padding-right: 2rem;
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-bottom: -7rem;
}
[data-testid="stVerticalBlock"] {
    padding-left: 0rem;
    padding-right: 0rem;
}
[data-testid="stMetric"] {
    background-color: #393939;
    text-align: center;
    padding: 15px 0;
}
[data-testid="stMetricLabel"] {
    display: flex;
    justify-content: center;
    align-items: center;
}
[data-testid="stMetricDeltaIcon-Up"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
[data-testid="stMetricDeltaIcon-Down"] {
    position: relative;
    left: 38%;
    -webkit-transform: translateX(-50%);
    -ms-transform: translateX(-50%);
    transform: translateX(-50%);
}
</style>
""", unsafe_allow_html=True)

# Descargar recursos del tokenizador de NLTK para español
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
#nlp = spacy.load('es_core_news_sm')

# Sidebar
page = st.sidebar.selectbox("Seleccione una página", ("EDA", "ML", "NLP", "NLP Explorer"))
show_raw_data = st.sidebar.checkbox('Show Raw Data')

# Cargar datos
DATE_COLUMN = 'fecha'
DATA_URL = 'data/noticias.csv'

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], format='%b %d, %Y- %H:%M', errors='coerce')
    return data

# Cargar datos en el dataframe
data = load_data()

# Entrenamiento del modelo
X = data['noticia']
y = data['categoria']

# Dividir los datos en conjunto de entrenamiento y evaluación
X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_eval = vectorizer.transform(X_eval)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

def preprocess_text(text):
    # Tokenización de palabras
    tokens = word_tokenize(text.lower())

    # Eliminación de stopwords
    stop_words = set(stopwords.words('spanish'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lematización de palabras
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Unir las palabras preprocesadas en un texto nuevamente
    preprocessed_text = ' '.join(tokens)

    return preprocessed_text

def sentimiento(text):
    if pd.notna(text):
        preprocessed_text = preprocess_text(text)
        return sentiment.sentiment(preprocessed_text)
    else:
        return text

def categorize(number_value):
    if number_value > 0.6:
        return 'Positivo'
    elif number_value < 0.4:
        return 'Negativo'
    else:
        return 'Neutral'

# Funciones
def clean_and_tokenize(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text.split()

def show_wordcloud(word_freq, title):
    # Crear una instancia de WordCloud
    wordcloud = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(word_freq)

    # Crear la figura de Plotly
    fig = px.imshow(wordcloud.to_array(), binary_string=True)
    fig.update_layout(title=title, title_x=0.5, xaxis=dict(visible=False), yaxis=dict(visible=False))

    # Mostrar la figura en Streamlit
    st.plotly_chart(fig, use_container_width=True, clear_figure=True)

def plot_histogram(data, x, color, title, labels):
    fig = px.histogram(data, x=x, color=color, labels=labels, title=title)
    return fig

def plot_pie(data, names, values, title):
    fig = px.pie(names=names, values=values, title=title)
    return fig

def plot_bar(x, y, labels, title):
    fig = px.bar(x=x, y=y, labels=labels, title=title)
    return fig

def plot_scatter(x, y, labels, title, color_sequence):
    fig = px.scatter(x=x, y=y, labels=labels, title=title, color_discrete_sequence=color_sequence)
    return fig

def plot_scatter2(data, x, y, size, color, labels, title):
    fig = px.scatter(data, x=x, y=y, size=size, color=color, labels=labels, title=title)
    return fig

def plot_heatmap(data, x, y, color_continuous_scale, title, xaxis_title, yaxis_title):
    fig = px.imshow(data, x=x, y=y, color_continuous_scale=color_continuous_scale)
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig

def plot_decision_tree(tree_model, vectorizer):
    fig, ax = plt.subplots(figsize=(30, 20))
    ax.set_title('Árbol de decisiones')
    fuente = 14
    tree.plot_tree(tree_model, filled=True, rounded=True, feature_names=list(vectorizer.vocabulary_.keys()), fontsize=fuente, ax=ax)
    class_names = tree_model.classes_
    class_colors = plt.cm.tab10(range(len(class_names)))
    legend_labels = [f"{class_names[i]} - {i}" for i in range(len(class_names))]
    legend_elements = [Patch(facecolor=class_colors[i], label=legend_labels[i]) for i in range(len(class_names))]
    ax.legend(handles=legend_elements, fontsize=fuente)
    return fig

def plot_confusion_matrix(cm, classes):
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale='Viridis',
        text=[[str(val) for val in row] for row in cm],
        texttemplate="%{text}",
        textfont={"size": 14}))

    fig.update_layout(
        xaxis_title='Etiqueta Predicha',
        yaxis_title='Etiqueta Verdadera',
        font=dict(size=14)
    )
    return fig

# Datos del proyecto
st.markdown(
    """
    # Desarrollo de un Sistema de Análisis de Noticias Salvadoreñas
    ## Objetivo General

    El objetivo principal de este proyecto es aplicar los fundamentos de ciencia de datos para desarrollar un sistema integral de análisis de noticias salvadoreñas. Se abordarán diferentes etapas, desde la obtención de datos mediante web scraping hasta la implementación de modelos de aprendizaje automático y procesamiento del lenguaje natural, culminando con la creación de una interfaz interactiva utilizando Streamlit.

    ### Han contribuido a la elaboración de este proyecto:

    1. Nathaly Rebeca Bonilla Morales - UCA
    2. Elmer Elias Chanchan - UFG
    3. Diego Alejandro Manzano Pineda - Lab-Dat
    """
)

# Página de EDA
if page == "EDA":
    st.subheader("Análisis Exploratorio de Datos sobre noticias de El Diario de Hoy")

    if show_raw_data:
        st.subheader('Datos crudos')
        st.write(data)

    st.subheader('Distribución de Noticias a lo Largo del Tiempo')

    # Agrupar datos por día y mes
    data['day'] = data[DATE_COLUMN].dt.day
    data['month'] = data[DATE_COLUMN].dt.month_name()

    # Calcular la cantidad de noticias por día
    news_count = data.groupby(['month', 'day']).size().reset_index(name='count')

    # Agregar una columna con tamaños escalados para el gráfico de dispersión
    news_count['size'] = news_count['count'] / news_count['count'].max() * 100

    # Botón de radio para seleccionar el tipo de gráfico
    chart_type = st.radio('Elige el gráfico a mostrar:', ('Histograma', 'Box Plot', 'Scatter Plot', 'Mapa de Calor'))

    if chart_type == 'Histograma':
        fig = plot_histogram(data, x='day', color='month',
                             labels={'day': 'Día', 'count': 'Cantidad de Noticias'},
                             title='Distribución de Noticias por día y mes')
    elif chart_type == 'Box Plot':
        fig = px.box(data, x='month', y='day', color='month',
                     labels={'month': 'Mes', 'day': 'Día'},
                     title='Distribución de Noticias por día y mes')
    elif chart_type == 'Scatter Plot':
        fig = plot_scatter2(news_count, x='day', y='month', size='size', color='month',
                            labels={'day': 'Día', 'month': 'Mes', 'size': 'Cantidad de Noticias'},
                            title='Distribución de Noticias por día y mes')
    else:
        # Extraer el día de la semana (0 es lunes, 6 es domingo)
        data['dia_semana'] = data[DATE_COLUMN].dt.dayofweek

        # Eliminar filas con valores faltantes en las columnas 'dia_semana' y 'categoria'
        data_clean = data.dropna(subset=['dia_semana', 'categoria']).copy()

        # Convertir 'dia_semana' a tipo entero
        data_clean['dia_semana'] = data_clean['dia_semana'].astype(int)

        # Agrupar por 'dia_semana' y 'categoria' y contar las ocurrencias
        conteo_dias_categoria = data_clean.groupby(['dia_semana', 'categoria']).size().unstack()

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=conteo_dias_categoria.values,
            x=conteo_dias_categoria.columns,
            y=conteo_dias_categoria.index,
            colorscale='Viridis'))

        fig_heatmap.update_layout(
            title='Cantidad de Noticias por Día de la Semana y Categoría',
            xaxis_title='Categoría',
            yaxis_title='Día de la Semana'
        )

        fig = fig_heatmap

    # Mostrar el gráfico seleccionado
    st.plotly_chart(fig, use_container_width=True)
    
# Página de ML
elif page == "ML":
    st.subheader("Machine Learning")

    if show_raw_data:
        st.subheader('Datos crudos')
        st.write(data)

    # Predecir con el modelo de árbol de decisiones
    y_pred = tree_model.predict(X_eval)

    # Calcular métricas de evaluación
    accuracy = accuracy_score(y_eval, y_pred)
    precision = precision_score(y_eval, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_eval, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_eval, y_pred, average='macro', zero_division=1)

    # Definir los umbrales para cada métrica
    umbral_superior = {
        "Exactitud": 0.85,
        "Precisión": 0.90,
        "Recall": 0.80,
        "F1-score": 0.75,
        "Validacion cruzada": 0.80
    }
    fuente = 10

    # Función auxiliar para obtener el delta y color según el umbral
    def get_delta(value, threshold):
        if value > threshold:
            return "+"
        elif value < threshold:
            return "-"
        else:
            return ""

    # Mostrar las métricas de evaluación utilizando columnas en Streamlit
    st.write("Métricas")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Exactitud", "{:.2%}".format(accuracy), delta=get_delta(accuracy, umbral_superior["Exactitud"]))
    col2.metric("Precisión", "{:.2%}".format(precision), delta=get_delta(precision, umbral_superior["Precisión"]))
    col3.metric("Recall", "{:.2%}".format(recall), delta=get_delta(recall, umbral_superior["Recall"]))
    col4.metric("F1-score", "{:.2%}".format(f1), delta=get_delta(f1, umbral_superior["F1-score"]))

    # Realizar validación cruzada y calcular la precisión media
    stratkf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_mean_accuracy = cross_val_score(tree_model, X_eval, y_eval, cv=stratkf, scoring='accuracy').mean()
    col5.metric("Validacion cruzada", "{:.2%}".format(cv_mean_accuracy),
                delta=get_delta(cv_mean_accuracy, umbral_superior["Validacion cruzada"]))

    # Matriz de confusión
    cm = confusion_matrix(y_eval, y_pred)

    # Utilizar st.radio para seleccionar el gráfico a mostrar
    opciones_graficos = ["Árbol de decisiones", "Matriz de confusión"]
    grafico_seleccionado = st.radio("Seleccionar gráfico:", opciones_graficos)

    if grafico_seleccionado == "Árbol de decisiones":
        fig = plot_decision_tree(tree_model, vectorizer)
        st.subheader('Árbol de decisiones')
        st.pyplot(fig, use_container_width=True)

    elif grafico_seleccionado == "Matriz de confusión":
        fig = plot_confusion_matrix(cm, tree_model.classes_)
        st.subheader('Matriz de confusión')
        st.plotly_chart(fig, use_container_width=True)

# Página de NLP
elif page == "NLP":
    st.subheader("Procesamiento del Lenguaje Natural (NLP)")

    if show_raw_data:
        st.subheader('Datos crudos')
        st.dataframe(data)

    # Funciones y configuraciones NLP
    stopwords_es_expanded = set([        
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con", "contra", "cual", "cuando", "de", "del", "desde",
        "donde", "durante", "e", "el", "ella", "ellas", "ello", "ellos", "en", "entre", "era", "erais", "eran", "eras", "eres",
        "es", "esa", "esas", "ese", "eso", "esos", "esta", "estaba", "estabais", "estaban", "estabas", "estad", "estada", "estadas",
        "estado", "estados", "estamos", "estando", "estar", "estaremos", "estará", "estarán", "estarás", "estaré", "estaréis",
        "estaría", "estaríais", "estaríamos", "estarían", "estarías", "estas", "este", "estemos", "esto", "estos", "estoy",
        "estuve", "estuviera", "estuvierais", "estuvieran", "estuvieras", "estuvieron", "estuviese", "estuvieseis", "estuviesen",
        "estuvieses", "estuvimos", "estuviste", "estuvisteis", "estuvo", "fue", "fuera", "fuerais", "fueran", "fueras", "fueron",
        "fuese", "fueseis", "fuesen", "fueses", "fui", "fuimos", "fuiste", "fuisteis", "han", "has", "hasta", "hay", "haya",
        "hayamos", "hayan", "hayas", "hayáis", "he", "hemos", "hube", "hubiera", "hubierais", "hubieran", "hubieras", "hubieron",
        "hubiese", "hubieseis", "hubiesen", "hubieses", "hubimos", "hubiste", "hubisteis", "hubo", "la", "las", "le", "les", "lo",
        "los", "me", "mi", "mis", "mucho", "muchos", "muy", "más", "mí", "mía", "mías", "mío", "míos", "nada", "ni", "no", "nos",
        "nosotras", "nosotros", "nuestra", "nuestras", "nuestro", "nuestros", "o", "os", "otra", "otras", "otro", "otros", "para",
        "pero", "poco", "por", "porque", "que", "quien", "quienes", "qué", "se", "sea", "seamos", "sean", "seas", "seremos",
        "será", "serán", "serás", "seré", "seréis", "sería", "seríais", "seríamos", "serían", "serías", "seáis", "si", "sido",
        "siendo", "sin", "sobre", "sois", "somos", "son", "soy", "su", "sus", "suya", "suyas", "suyo", "suyos", "sí", "también",
        "tanto", "te", "tendremos", "tendrá", "tendrán", "tendrás", "tendré", "tendréis", "tendría", "tendríais", "tendríamos",
        "tendrían", "tendrías", "tened", "tenemos", "tenga", "tengamos", "tengan", "tengas", "tengo", "tengáis", "tenida", "tenidas",
        "tenido", "tenidos", "teniendo", "tenéis", "tenía", "teníais", "teníamos", "tenían", "tenías", "ti", "tiene", "tienen",
        "tienes", "todo", "todos", "tu", "tus", "tuve", "tuviera", "tuvierais", "tuvieran", "tuvieras", "tuvieron", "tuviese",
        "tuvieseis", "tuviesen", "tuvieses", "tuvimos", "tuviste", "tuvisteis", "tuvo", "tuya", "tuyas", "tuyo", "tuyos", "tú",
        "un", "una", "uno", "unos", "vosotras", "vosotros", "vuestra", "vuestras", "vuestro", "vuestros", "y", "ya", "yo", "él",
        "éramos"
    ])

    data['Titulo_tokens'] = data['titulo'].apply(clean_and_tokenize)
    data['Resumen_tokens'] = data['resumen'].apply(clean_and_tokenize)

    all_titulo_tokens = list(itertools.chain(*data['Titulo_tokens']))
    all_resumen_tokens = list(itertools.chain(*data['Resumen_tokens']))

    filtered_titulo_tokens = [w for w in all_titulo_tokens if w not in stopwords_es_expanded]
    filtered_resumen_tokens = [w for w in all_resumen_tokens if w not in stopwords_es_expanded]

    filtered_titulo_freq = Counter(filtered_titulo_tokens)
    filtered_resumen_freq = Counter(filtered_resumen_tokens)

    # Frecuencia de Palabras
    titulo_word_freq = Counter(filtered_titulo_tokens)
    resumen_word_freq = Counter(filtered_resumen_tokens)

    titulo_words, titulo_freqs = zip(*titulo_word_freq.items())
    resumen_words, resumen_freqs = zip(*resumen_word_freq.items())

    # Opciones de gráfico
    graph_options = [
        "Nubes de palabras", 
        "Análisis de Sentimientos", 
        "Longitudes de Palabras", 
        "Frecuencia de palabras - Barras", 
        "Frecuencia de palabras - Dispersión"
    ]

    # Mostrar el selector de opciones de gráfico
    graph_type = st.radio("Selecciona el análisis gráfico", graph_options)

    col1, col2 = st.columns(2)

    if graph_type == "Nubes de palabras":
        # Mostrar nubes de palabras en columnas
        with col1:
            show_wordcloud(filtered_titulo_freq, "Nube de palabras sin stopwords para Título")

        with col2:
            show_wordcloud(filtered_resumen_freq, "Nube de palabras sin stopwords para Resumen")

    elif graph_type == "Análisis de Sentimientos":
        sentiment = sentiment_analysis.SentimentAnalysisSpanish()
        data['sentimiento_summary'] = data['resumen'].apply(lambda x: sentiment.sentiment(x))
        data['sentimiento_summary_cat'] = data['sentimiento_summary'].apply(lambda x: 'Positivo' if x > 0.6 else ('Negativo' if x < 0.4 else 'Neutral'))

        conteo_categorias = data['sentimiento_summary_cat'].value_counts()

        # Histograma de Sentimientos
        fig_histogram = plot_histogram(data, x='sentimiento_summary_cat', color='sentimiento_summary_cat',
                                    title='Histograma de Sentimientos',
                                    labels={'sentimiento_summary_cat': 'Categoría de Sentimiento'})

        # Gráfico de Pastel de Sentimientos
        fig_pie = plot_pie(data, names=conteo_categorias.index, values=conteo_categorias.values,
                        title='Distribución de Sentimientos')

        # Columna 1: Histograma de Sentimientos
        with col1:
            st.plotly_chart(fig_histogram, use_container_width=True)

        # Columna 2: Gráfico de Pastel de Sentimientos
        with col2:
            st.plotly_chart(fig_pie, use_container_width=True)

    elif graph_type == "Longitudes de Palabras":
        # Longitudes de Palabras
        titulo_lengths = [len(word) for word in filtered_titulo_tokens if word]
        resumen_lengths = [len(word) for word in filtered_resumen_tokens if word]

        color_scale = px.colors.qualitative.Set1
        titulo_colors = [color_scale[length % len(color_scale)] for length in titulo_lengths]
        resumen_colors = [color_scale[length % len(color_scale)] for length in resumen_lengths]

        # Histograma de Longitudes de Palabras - Títulos
        try:
            fig_titulos = plot_histogram(pd.DataFrame(titulo_lengths, columns=['Longitud de Palabra']), x='Longitud de Palabra',
                                        color='Longitud de Palabra', title='Longitudes de Palabras - Títulos',
                                        labels={'x': 'Longitud de Palabra'})
        except Exception as e:
            st.error(f"Error al generar el histograma para títulos: {e}")

        # Histograma de Longitudes de Palabras - Resúmenes
        try:
            fig_resumenes = plot_histogram(pd.DataFrame(resumen_lengths, columns=['Longitud de Palabra']),
                                        x='Longitud de Palabra', color='Longitud de Palabra',
                                        title='Longitudes de Palabras - Resúmenes',
                                        labels={'x': 'Longitud de Palabra'})
        except Exception as e:
            st.error(f"Error al generar el histograma para resúmenes: {e}")

        if 'fig_titulos' in locals():
            col1.plotly_chart(fig_titulos, use_container_width=True)

        if 'fig_resumenes' in locals():
            col2.plotly_chart(fig_resumenes, use_container_width=True)

    elif graph_type == "Frecuencia de palabras - Barras":
        fig_frecuencia = None
        try:
            fig_frecuencia_titulos = plot_bar(x=titulo_words, y=titulo_freqs, labels={'x': 'Palabra', 'y': 'Frecuencia'},
                                        title='Frecuencia de Palabras - Títulos')
            fig_frecuencia_resumen = plot_bar(x=resumen_words, y=resumen_freqs, labels={'x': 'Palabra', 'y': 'Frecuencia'},
                                        title='Frecuencia de Palabras - Resúmenes')
        except Exception as e:
            st.error(f"Error al generar el gráfico de barras: {e}")
        
        col1.plotly_chart(fig_frecuencia_titulos, use_container_width=True)
        col2.plotly_chart(fig_frecuencia_resumen, use_container_width=True)

    elif graph_type == "Frecuencia de palabras - Dispersión":
        # Diagrama de Dispersión
        fig_titulo_dispersion = None
        fig_resumen_dispersion = None

        try:
            fig_titulo_dispersion = plot_scatter2(data=titulo_words, x=titulo_words, y=titulo_freqs,
                                        size=None, color=titulo_freqs,
                                        labels={'x': 'Palabra', 'y': 'Frecuencia'},
                                        title='Diagrama de Dispersión - Títulos')

            fig_resumen_dispersion = plot_scatter2(data=resumen_words, x=resumen_words, y=resumen_freqs,
                                        size=None, color=resumen_freqs,
                                        labels={'x': 'Palabra', 'y': 'Frecuencia'},
                                        title='Diagrama de Dispersión - Resúmenes')
        except Exception as e:
            st.error(f"Error al generar el diagrama de dispersión: {e}")

        if 'fig_titulo_dispersion' in locals():
            col1.plotly_chart(fig_titulo_dispersion, use_container_width=True)

        if 'fig_resumen_dispersion' in locals():
            col2.plotly_chart(fig_resumen_dispersion, use_container_width=True)
            
# Página de NLP Explorer
elif page == "NLP Explorer":
    st.write("Esta página te permite ingresar una URL y realizar diversas operaciones en el contenido en español.")

    url = st.text_input("Ingresa la URL:", help="Ingrese la URL de la noticia de El Diario de Hoy")
    get_news_button = st.button("Obtener Noticia")
    data_dict_list = []
    noticias = st.empty()
    entities = st.session_state.get("entities", [])

    if get_news_button:
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            try:
                title = soup.find('article', class_='detail').h1.text
                summary = soup.find('p', class_='summary').text if soup.find('p', class_='summary') else "No se encontró resumen"
                author = soup.find('p', class_='info-article').a.span.text if soup.find('p', class_='info-article') else "No se encontró autor"
                date = soup.find('span', class_='ago').text if soup.find('span', class_='ago') else "No se encontró fecha"
                full_text = soup.find('div', class_='entry-content').text if soup.find('div', class_='entry-content') else "No se encontró noticia completa"
                keyword_list = [a_tag.text for a_tag in soup.find('div', class_='in-this-article').find_all('a', class_='tag')] if soup.find('div', class_='in-this-article') else ["No se encontraron keywords"]

                data_dict = {
                    'Titulo': title,
                    'Resumen': summary,
                    'Autor': author,
                    'Fecha': date,
                    'Noticia': full_text,
                    'Keywords': keyword_list,
                    'URL': url,
                }

                data_dict_list.append(data_dict)

            except AttributeError as e:
                st.error(f"Error al extraer datos de la página: {e}")

        except requests.exceptions.RequestException as e:
            st.error(f'Error al intentar acceder a la página {url}: {e}')

        if data_dict_list:
            text = data_dict_list[0]['Noticia']
        else:
            st.warning("No se encontraron datos de la noticia.")
            text = ""

        st.session_state.text = text

    text = st.session_state.get("text", "")

    st.text_area("Datos de la Noticia:", text, height=200, key="text")

    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Extraer Identidades"):
        text = st.session_state.text

        # Tokenización de palabras
        tokens = word_tokenize(text)

        # Etiquetado gramatical
        tagged_tokens = pos_tag(tokens)

        # Reconocimiento de entidades nombradas (NER)
        named_entities = ne_chunk(tagged_tokens)

        entities = []
        for entity in named_entities:
            if hasattr(entity, 'label'):
                label = entity.label()
                if label == "PERSON" or label == "ORGANIZATION":
                    entities.append((' '.join(child[0] for child in entity.leaves()), label))

        st.session_state.entities = entities

        if entities:
            col1.subheader("Identidades encontradas:")
            for entity, label in entities:
                col1.write("- " + entity + " (" + label + ")")
        else:
            col1.subheader("No se encontraron identidades en el texto.")

    if col2.button("Generar Resumen"):
        # Crear el analizador de texto y el tokenizador
        parser = PlaintextParser.from_string(st.session_state.text, Tokenizer("spanish"))

        # Crear el resumidor LSA y generar el resumen
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, 1)  # Número de frases en el resumen

        # Mostrar el resumen generado
        if summary:
            col2.subheader("Resumen:")
            for sentence in summary:
                col2.write(sentence)
        else:
            col2.subheader("No se pudo generar un resumen para el texto proporcionado.")
        
    if col3.button("Predicción"):
        # Tokenizar y vectorizar el texto de prueba
        texto_vectorizado = vectorizer.transform([st.session_state.text])

        # Realizar la predicción
        categoria_predicha = tree_model.predict(texto_vectorizado)[0]

        # Mostrar la categoría predicha
        col3.write(f"**Categoría Predicha:** {categoria_predicha}")
        
    if col4.button("Sentimiento"):
        sentiment = sentiment_analysis.SentimentAnalysisSpanish()        

        # Obtener el texto de entrada
        text = st.session_state.text
        #text = clean_and_tokenize(text)
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Realizar análisis de sentimiento
        if text:
            score = sentimiento(text)
            category = categorize(score)

            col4.subheader("Análisis de Sentimiento:")
            #col4.write(f"Noticia: {text}")
            col4.write(f"Puntuación: {score * 100:.2f}%")
            col4.write(f"Sentimiento: {category}")
        else:
            col4.warning("No hay texto disponible para realizar el análisis de sentimiento.")
            
#espacio vacío
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")
    st.markdown("")

    #logotipos git y linkedin centrados con hipervínculo y tamaño 30x30
    st.markdown(
        "<p style='text-align: center; margin-bottom: 0px;'>"
        "<a href='https://dagshub.com/echanchan/BCM_Proyecto_Foundation_Data_Science'>"
        "<img src='https://raw.githubusercontent.com/lilicasanova/logo/main/github-mark-white.png' alt='GitHub' width='30' height='30'>"
        "</a>"
        "</p>",
        unsafe_allow_html=True
    )
