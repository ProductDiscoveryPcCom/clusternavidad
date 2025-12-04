"""
🎄 Keyword Semantic Clustering Tool - Navidad Edition
Herramienta para agrupar semánticamente keywords de Google Keyword Planner
con enfoque en campaña de Navidad/Regalos

Desarrollado para PcComponentes
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict
import re
import json
from typing import List, Dict, Tuple, Optional
import anthropic
import openai
from functools import lru_cache
import os

# ============== CONFIGURACIÓN DE PÁGINA ==============
st.set_page_config(
    page_title="🎄 Keyword Clustering - Navidad",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============== ESTILOS CSS PERSONALIZADOS ==============
st.markdown("""
<style>
    /* Estilo general festivo pero profesional */
    .main {
        background: linear-gradient(180deg, #fefefe 0%, #f8f9fa 100%);
    }
    
    /* Headers */
    h1 {
        color: #1a472a !important;
        font-family: 'Georgia', serif !important;
        border-bottom: 3px solid #c41e3a;
        padding-bottom: 10px;
    }
    
    h2, h3 {
        color: #2d5016 !important;
    }
    
    /* Métricas */
    .stMetric {
        background: linear-gradient(135deg, #ffffff 0%, #f0f7f0 100%);
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e8e8e8;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Cards de clusters */
    .cluster-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #c41e3a;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: transform 0.2s;
    }
    
    .cluster-card:hover {
        transform: translateX(5px);
    }
    
    /* URL sugerida */
    .url-suggestion {
        background: linear-gradient(90deg, #1a472a 0%, #2d5016 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-family: monospace;
        font-size: 14px;
        margin: 10px 0;
    }
    
    /* Badges */
    .volume-badge {
        background: #c41e3a;
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 12px;
    }
    
    .intent-badge {
        background: #f0f7f0;
        color: #1a472a;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        border: 1px solid #1a472a;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #fefefe;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: #f8f9fa;
        border-radius: 8px;
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(90deg, #c41e3a 0%, #a01830 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(196, 30, 58, 0.3);
    }
    
    /* Tablas */
    .dataframe {
        border-radius: 8px !important;
        overflow: hidden;
    }
    
    /* Info boxes */
    .info-box {
        background: #e8f4e8;
        border: 1px solid #1a472a;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #fff3e0;
        border: 1px solid #ff9800;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============== CONSTANTES Y CONFIGURACIÓN ==============

# Productos disponibles de PcComponentes (extraídos de las imágenes)
PRODUCTOS_PCCOMPONENTES = {
    "regalos_precio": {
        "hasta_30": ["Productos en campaña entre 20 y 30 €"],
        "hasta_60": ["Productos en campaña entre 30 y 60 €"],
        "hasta_100": ["Productos en campaña entre 60 y 100 €"]
    },
    "tecnologia_acompana": [
        "Smartphones", "Tablets", "EBooks", "Auriculares bluetooth",
        "Patinetes", "Smartwatches"
    ],
    "informatica_todos": [
        "PC no gaming", "Portátiles no gaming", "Periféricos no gaming",
        "Discos duros externos"
    ],
    "esenciales_hogar": [
        "Aspiradoras", "Robots aspirador", "Hidrolimpiadoras",
        "Limpiadoras de vapor", "Planchado", "Tratamiento del aire"
    ],
    "cine_series": [
        "Televisores", "Proyectores", "Altavoces",
        "Altavoces TV y barras de sonido", "Auriculares premium"
    ],
    "gamers": [
        "Portátiles gaming", "PC gaming", "Mesas gaming",
        "Teclados gaming", "Sillas gaming", "Ratones gaming",
        "Auriculares gaming"
    ],
    "consolas": [
        "Mandos de juego", "Accesorios para consolas", "Consolas",
        "Juegos", "Simulación gaming", "Volantes"
    ],
    "amantes_cafe": [
        "Cápsulas para cafeteras", "Cafeteras", "Todas las familias relacionadas"
    ],
    "chefs": [
        "Cocina portátil", "Freidoras", "Robots de cocina",
        "Batidoras", "Planchas, barbacoas y grills"
    ],
    "deportistas": [
        "Smartwatches", "Pulseras actividad", "Auriculares deportivos",
        "Bicicletas elípticas", "Bicicletas estáticas", "Cintas de correr"
    ],
    "belleza_cuidado": [
        "Afeitadoras", "Cortapelo", "Depiladoras",
        "Cepillos de dientes eléctricos", "Planchas de pelo",
        "Secadores de pelo", "Básculas de baño", "Cuidado corporal",
        "Cuidado facial"
    ],
    "gadgets": [
        "Altavoces inteligentes", "Powerbanks"
    ],
    "juguetes": [
        "Juguetes", "Juguetes de imitación", "Juguetes educativos"
    ],
    "lego": ["Lego"]
}

# Patrones para clasificación de intención
INTENT_PATTERNS = {
    "transaccional": [
        r"comprar", r"precio", r"oferta", r"barato", r"descuento",
        r"tienda", r"amazon", r"donde", r"mejor precio"
    ],
    "informacional": [
        r"que es", r"como", r"cual", r"diferencia", r"mejor",
        r"comparativa", r"opinion", r"review", r"guia"
    ],
    "navegacional": [
        r"pccomponentes", r"amazon", r"mediamarkt", r"fnac"
    ],
    "regalo": [
        r"regalo", r"regalar", r"navidad", r"amigo invisible",
        r"detalle", r"obsequio", r"presente"
    ]
}

# ============== FUNCIONES DE UTILIDAD ==============

def clean_keyword(kw: str) -> str:
    """Limpia y normaliza una keyword"""
    kw = str(kw).lower().strip()
    kw = re.sub(r'[^\w\sáéíóúñü]', ' ', kw)
    kw = re.sub(r'\s+', ' ', kw)
    return kw

def extract_seasonal_volume(row: pd.Series, months: List[str]) -> int:
    """Extrae el volumen de búsquedas para meses específicos"""
    total = 0
    for month in months:
        if month in row.index:
            val = row[month]
            if pd.notna(val) and str(val).replace(',', '').replace('.', '').isdigit():
                total += int(str(val).replace(',', ''))
    return total

def classify_intent(keyword: str) -> str:
    """Clasifica la intención de búsqueda de una keyword"""
    keyword_lower = keyword.lower()
    
    scores = {intent: 0 for intent in INTENT_PATTERNS}
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, keyword_lower):
                scores[intent] += 1
    
    if scores["regalo"] > 0:
        return "🎁 Regalo"
    elif max(scores.values()) == 0:
        return "🔍 General"
    else:
        max_intent = max(scores, key=scores.get)
        intent_emojis = {
            "transaccional": "💰 Transaccional",
            "informacional": "📚 Informacional",
            "navegacional": "🧭 Navegacional"
        }
        return intent_emojis.get(max_intent, "🔍 General")

def extract_gift_recipient(keyword: str) -> Optional[str]:
    """Extrae el destinatario del regalo de la keyword"""
    recipients = {
        "hombre": ["hombre", "chico", "novio", "marido", "padre", "papa", "abuelo", "él"],
        "mujer": ["mujer", "chica", "novia", "esposa", "madre", "mama", "abuela", "ella"],
        "niño": ["niño", "hijo", "bebe", "bebé", "infantil", "pequeño"],
        "niña": ["niña", "hija", "princesa"],
        "adolescente": ["adolescente", "joven", "teen"],
        "amigo": ["amigo", "amiga", "amistad"],
        "pareja": ["pareja", "novios", "enamorados"],
        "familia": ["familia", "familiar", "padres", "hermano", "hermana"],
        "compañero": ["compañero", "colega", "jefe", "empleado", "empresa"]
    }
    
    keyword_lower = keyword.lower()
    for recipient, patterns in recipients.items():
        for pattern in patterns:
            if pattern in keyword_lower:
                return recipient
    return None

def extract_price_range(keyword: str) -> Optional[str]:
    """Extrae el rango de precio mencionado en la keyword"""
    patterns = [
        (r"menos de (\d+)", "hasta"),
        (r"hasta (\d+)", "hasta"),
        (r"por (\d+)", "exacto"),
        (r"(\d+)\s*euros?", "aproximado"),
        (r"(\d+)\s*€", "aproximado")
    ]
    
    for pattern, prefix in patterns:
        match = re.search(pattern, keyword.lower())
        if match:
            amount = int(match.group(1))
            if amount <= 30:
                return "hasta_30€"
            elif amount <= 60:
                return "hasta_60€"
            elif amount <= 100:
                return "hasta_100€"
            else:
                return "más_de_100€"
    return None

def match_product_family(keyword: str) -> List[str]:
    """Encuentra las familias de productos que coinciden con la keyword"""
    keyword_lower = keyword.lower()
    matched = []
    
    product_keywords = {
        "tecnologia_acompana": ["smartphone", "telefono", "movil", "tablet", "ebook", "auricular", "patinete", "smartwatch", "reloj inteligente"],
        "informatica_todos": ["pc", "portatil", "ordenador", "laptop", "periferico", "disco duro"],
        "esenciales_hogar": ["aspirador", "robot", "limpia", "plancha", "aire"],
        "cine_series": ["televisor", "tv", "tele", "proyector", "altavoz", "barra de sonido", "home cinema"],
        "gamers": ["gaming", "gamer", "teclado", "raton", "silla", "monitor gaming"],
        "consolas": ["mando", "consola", "juego", "videojuego", "playstation", "xbox", "nintendo", "volante"],
        "amantes_cafe": ["cafe", "cafetera", "capsula", "expreso", "espresso"],
        "chefs": ["cocina", "freidora", "robot cocina", "batidora", "plancha", "barbacoa", "grill"],
        "deportistas": ["deporte", "fitness", "pulsera", "bici", "cinta de correr", "gym"],
        "belleza_cuidado": ["afeitadora", "cortapelo", "depiladora", "cepillo diente", "plancha pelo", "secador", "bascula", "belleza"],
        "gadgets": ["gadget", "powerbank", "bateria externa", "altavoz inteligente", "alexa", "echo"],
        "juguetes": ["juguete", "muñeca", "peluche"],
        "lego": ["lego", "construccion"]
    }
    
    for family, keywords in product_keywords.items():
        for kw in keywords:
            if kw in keyword_lower:
                matched.append(family)
                break
    
    return list(set(matched))

# ============== FUNCIONES DE CLUSTERING ==============

def create_embeddings_tfidf(keywords: List[str]) -> np.ndarray:
    """Crea embeddings usando TF-IDF"""
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=1000,
        stop_words=None  # Mantener palabras en español
    )
    embeddings = vectorizer.fit_transform(keywords)
    return embeddings.toarray(), vectorizer

def cluster_keywords_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agrupa keywords usando K-Means"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

def cluster_keywords_hierarchical(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agrupa keywords usando clustering jerárquico"""
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    clusters = clustering.fit_predict(embeddings)
    return clusters

def calculate_cluster_coherence(embeddings: np.ndarray, clusters: np.ndarray) -> Dict[int, float]:
    """Calcula la coherencia de cada cluster"""
    coherences = {}
    unique_clusters = np.unique(clusters)
    
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_embeddings = embeddings[mask]
        
        if len(cluster_embeddings) > 1:
            similarities = cosine_similarity(cluster_embeddings)
            # Promedio de similitudes (excluyendo diagonal)
            np.fill_diagonal(similarities, 0)
            coherence = similarities.sum() / (len(cluster_embeddings) * (len(cluster_embeddings) - 1))
            coherences[cluster_id] = coherence
        else:
            coherences[cluster_id] = 1.0
    
    return coherences

def suggest_url_for_cluster(cluster_keywords: List[str], cluster_volumes: List[int]) -> str:
    """Sugiere una URL para el cluster basada en las keywords principales"""
    # Encontrar la keyword con mayor volumen
    if cluster_volumes:
        max_idx = cluster_volumes.index(max(cluster_volumes))
        main_kw = cluster_keywords[max_idx]
    else:
        main_kw = cluster_keywords[0] if cluster_keywords else "regalo"
    
    # Limpiar y formatear para URL
    url_slug = clean_keyword(main_kw)
    url_slug = re.sub(r'\s+', '-', url_slug)
    url_slug = re.sub(r'-+', '-', url_slug)
    url_slug = url_slug.strip('-')
    
    return f"/regalos-navidad/{url_slug}/"

# ============== FUNCIONES DE AI ==============

def get_cluster_analysis_claude(keywords: List[str], volumes: List[int], api_key: str) -> Dict:
    """Usa Claude para analizar y conceptualizar un cluster"""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        # Preparar datos del cluster
        kw_data = "\n".join([f"- {kw} (vol: {vol})" for kw, vol in zip(keywords[:20], volumes[:20])])
        
        prompt = f"""Analiza este grupo de keywords de búsqueda relacionadas con regalos de Navidad para una tienda de tecnología (PcComponentes).

Keywords del cluster:
{kw_data}

Responde en formato JSON con esta estructura exacta:
{{
    "nombre_cluster": "Nombre descriptivo corto para el cluster",
    "tema_principal": "Tema o intención principal que une estas keywords",
    "url_sugerida": "/regalos-navidad/slug-descriptivo/",
    "h1_sugerido": "Título H1 para la landing page",
    "meta_description": "Meta description de 150-160 caracteres",
    "productos_recomendados": ["producto1", "producto2", "producto3"],
    "query_fanout": ["búsqueda relacionada 1", "búsqueda relacionada 2", "búsqueda relacionada 3"],
    "nivel_competencia": "bajo/medio/alto",
    "potencial_conversion": "bajo/medio/alto"
}}

Solo responde con el JSON, sin explicaciones adicionales."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = message.content[0].text
        # Limpiar respuesta y parsear JSON
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```json?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        return json.loads(response_text)
    
    except Exception as e:
        st.warning(f"Error al usar Claude API: {str(e)}")
        return None

def get_cluster_analysis_openai(keywords: List[str], volumes: List[int], api_key: str) -> Dict:
    """Usa GPT para analizar y conceptualizar un cluster"""
    try:
        client = openai.OpenAI(api_key=api_key)
        
        # Preparar datos del cluster
        kw_data = "\n".join([f"- {kw} (vol: {vol})" for kw, vol in zip(keywords[:20], volumes[:20])])
        
        prompt = f"""Analiza este grupo de keywords de búsqueda relacionadas con regalos de Navidad para una tienda de tecnología (PcComponentes).

Keywords del cluster:
{kw_data}

Responde en formato JSON con esta estructura exacta:
{{
    "nombre_cluster": "Nombre descriptivo corto para el cluster",
    "tema_principal": "Tema o intención principal que une estas keywords",
    "url_sugerida": "/regalos-navidad/slug-descriptivo/",
    "h1_sugerido": "Título H1 para la landing page",
    "meta_description": "Meta description de 150-160 caracteres",
    "productos_recomendados": ["producto1", "producto2", "producto3"],
    "query_fanout": ["búsqueda relacionada 1", "búsqueda relacionada 2", "búsqueda relacionada 3"],
    "nivel_competencia": "bajo/medio/alto",
    "potencial_conversion": "bajo/medio/alto"
}}

Solo responde con el JSON, sin explicaciones adicionales."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content
        # Limpiar respuesta y parsear JSON
        response_text = response_text.strip()
        if response_text.startswith("```"):
            response_text = re.sub(r'^```json?\s*', '', response_text)
            response_text = re.sub(r'\s*```$', '', response_text)
        
        return json.loads(response_text)
    
    except Exception as e:
        st.warning(f"Error al usar OpenAI API: {str(e)}")
        return None

# ============== FUNCIONES DE VISUALIZACIÓN ==============

def create_cluster_scatter_plot(df: pd.DataFrame, embeddings: np.ndarray) -> go.Figure:
    """Crea un scatter plot 2D de los clusters"""
    # Reducir dimensionalidad con PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    
    df_plot = df.copy()
    df_plot['x'] = coords[:, 0]
    df_plot['y'] = coords[:, 1]
    
    fig = px.scatter(
        df_plot,
        x='x',
        y='y',
        color='cluster_name',
        size='volumen_navidad',
        hover_data=['Keyword', 'volumen_navidad', 'intent'],
        title='Mapa de Clusters Semánticos',
        labels={'x': 'Dimensión 1', 'y': 'Dimensión 2'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Georgia',
        title_font_size=20,
        legend_title_text='Clusters',
        height=600
    )
    
    fig.update_traces(marker=dict(line=dict(width=1, color='white')))
    
    return fig

def create_treemap(df: pd.DataFrame) -> go.Figure:
    """Crea un treemap de clusters por volumen"""
    cluster_data = df.groupby('cluster_name').agg({
        'volumen_navidad': 'sum',
        'Keyword': 'count'
    }).reset_index()
    cluster_data.columns = ['Cluster', 'Volumen Total', 'Num Keywords']
    
    fig = px.treemap(
        cluster_data,
        path=['Cluster'],
        values='Volumen Total',
        color='Volumen Total',
        color_continuous_scale='RdYlGn',
        title='Potencial de Búsqueda por Cluster (Treemap)'
    )
    
    fig.update_layout(
        font_family='Georgia',
        title_font_size=20,
        height=500
    )
    
    return fig

def create_volume_bar_chart(df: pd.DataFrame) -> go.Figure:
    """Crea un gráfico de barras con volumen por cluster"""
    cluster_data = df.groupby('cluster_name').agg({
        'volumen_navidad': 'sum',
        'Keyword': 'count'
    }).reset_index()
    cluster_data.columns = ['Cluster', 'Volumen Total', 'Num Keywords']
    cluster_data = cluster_data.sort_values('Volumen Total', ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=cluster_data['Cluster'],
        x=cluster_data['Volumen Total'],
        orientation='h',
        marker=dict(
            color=cluster_data['Volumen Total'],
            colorscale='Greens',
            line=dict(color='#1a472a', width=1)
        ),
        text=cluster_data['Volumen Total'].apply(lambda x: f'{x:,}'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Volumen: %{x:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Volumen de Búsqueda por Cluster (Estacional)',
        xaxis_title='Volumen Total',
        yaxis_title='',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Georgia',
        title_font_size=20,
        height=max(400, len(cluster_data) * 30),
        margin=dict(l=200)
    )
    
    return fig

def create_url_opportunity_chart(df: pd.DataFrame) -> go.Figure:
    """Crea un gráfico de oportunidades de URL"""
    # Agrupar por cluster y calcular métricas
    cluster_data = df.groupby('cluster_name').agg({
        'volumen_navidad': ['sum', 'mean'],
        'Keyword': 'count'
    }).reset_index()
    cluster_data.columns = ['Cluster', 'Volumen Total', 'Volumen Medio', 'Num Keywords']
    
    # Calcular score de oportunidad
    cluster_data['Score'] = (
        cluster_data['Volumen Total'] * 0.5 +
        cluster_data['Volumen Medio'] * 0.3 +
        cluster_data['Num Keywords'] * 10 * 0.2
    )
    cluster_data = cluster_data.sort_values('Score', ascending=False).head(15)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cluster_data['Num Keywords'],
        y=cluster_data['Volumen Total'],
        mode='markers+text',
        marker=dict(
            size=cluster_data['Score'] / cluster_data['Score'].max() * 50 + 10,
            color=cluster_data['Volumen Medio'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title='Vol. Medio'),
            line=dict(color='white', width=2)
        ),
        text=cluster_data['Cluster'],
        textposition='top center',
        hovertemplate='<b>%{text}</b><br>Keywords: %{x}<br>Volumen Total: %{y:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Matriz de Oportunidades: Volumen vs Cobertura de Keywords',
        xaxis_title='Número de Keywords en Cluster',
        yaxis_title='Volumen Total de Búsqueda',
        plot_bgcolor='white',
        paper_bgcolor='white',
        font_family='Georgia',
        title_font_size=20,
        height=600
    )
    
    return fig

# ============== APLICACIÓN PRINCIPAL ==============

def main():
    # Header
    st.markdown("""
    # 🎄 Keyword Semantic Clustering Tool
    ### Campaña Navidad - PcComponentes
    """)
    
    st.markdown("""
    <div class="info-box">
        <strong>Objetivo:</strong> Identificar clusters semánticos de keywords para crear URLs de landing pages 
        optimizadas para la campaña de Navidad, basándose en volumen de búsqueda estacional (Nov + Dic + Ene configurables).
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar - Configuración
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        # Cargar archivo
        st.markdown("### 📂 Cargar Datos")
        uploaded_file = st.file_uploader(
            "Sube tu CSV de Google Keyword Planner",
            type=['csv'],
            help="El archivo debe contener las columnas de búsquedas mensuales"
        )
        
        st.markdown("---")
        
        # Configuración de clustering
        st.markdown("### 🎯 Parámetros de Clustering")
        
        n_clusters = st.slider(
            "Número de clusters",
            min_value=5,
            max_value=50,
            value=15,
            help="Número de grupos semánticos a crear"
        )
        
        clustering_method = st.selectbox(
            "Método de clustering",
            ["K-Means", "Jerárquico"],
            help="K-Means es más rápido, Jerárquico puede ser más preciso"
        )
        
        min_volume = st.number_input(
            "Volumen mínimo estacional",
            min_value=0,
            value=50,
            help="Filtrar keywords con volumen menor a este valor"
        )
        
        st.markdown("---")
        
        # Selección de meses
        st.markdown("### 📅 Meses para Volumen Estacional")
        
        include_nov = st.checkbox("Noviembre", value=True, help="Black Friday, inicio compras navideñas")
        include_dec = st.checkbox("Diciembre", value=True, help="Pico de búsquedas navideñas")
        include_jan = st.checkbox("Enero", value=True, help="Reyes Magos, rebajas")
        
        st.markdown("---")
        
        # API Keys
        st.markdown("### 🔑 APIs de AI")
        
        api_option = st.selectbox(
            "Proveedor de AI",
            ["Sin AI (solo TF-IDF)", "Claude (Anthropic)", "GPT (OpenAI)"]
        )
        
        api_key = ""
        if api_option != "Sin AI (solo TF-IDF)":
            api_key = st.text_input(
                f"API Key de {api_option.split()[0]}",
                type="password"
            )
        
        st.markdown("---")
        
        # Info de productos
        with st.expander("📦 Productos Disponibles"):
            st.markdown("""
            **Categorías en campaña:**
            - Regalos hasta 30€/60€/100€
            - Tecnología que te acompaña
            - Informática para todos
            - Esenciales hogar
            - Fans cine y series
            - Perfectos para gamers
            - Jugones de consola
            - Amantes del café
            - Chefs en potencia
            - Deportistas y aventureros
            - Belleza y cuidado
            - Gadgets y accesorios
            - Juguetes y juegos
            - LEGO
            """)
    
    # Área principal
    if uploaded_file is not None:
        # Cargar y procesar datos
        try:
            df = pd.read_csv(uploaded_file, encoding='utf-8')
        except:
            df = pd.read_csv(uploaded_file, encoding='latin-1')
        
        st.success(f"✅ Archivo cargado: {len(df)} keywords")
        
        # Identificar columnas de meses
        month_cols = [col for col in df.columns if 'Searches:' in col or 'searches:' in col.lower()]
        
        # Buscar columnas de Noviembre, Diciembre y Enero
        nov_cols = [col for col in month_cols if 'Nov' in col or 'nov' in col.lower()]
        dec_cols = [col for col in month_cols if 'Dec' in col or 'dic' in col.lower()]
        jan_cols = [col for col in month_cols if 'Jan' in col or 'ene' in col.lower()]
        
        # Construir lista de columnas según selección del usuario
        selected_month_cols = []
        selected_months_names = []
        
        if include_nov and nov_cols:
            selected_month_cols.extend(nov_cols)
            selected_months_names.append("Nov")
        if include_dec and dec_cols:
            selected_month_cols.extend(dec_cols)
            selected_months_names.append("Dic")
        if include_jan and jan_cols:
            selected_month_cols.extend(jan_cols)
            selected_months_names.append("Ene")
        
        months_label = "+".join(selected_months_names) if selected_months_names else "Avg"
        
        if not selected_month_cols:
            st.warning("⚠️ No se encontraron columnas de los meses seleccionados. Usando volumen promedio.")
            # Usar columna de volumen promedio
            if 'Avg. monthly searches' in df.columns:
                df['volumen_navidad'] = pd.to_numeric(df['Avg. monthly searches'], errors='coerce').fillna(0).astype(int)
            else:
                df['volumen_navidad'] = 100  # Valor por defecto
        else:
            # Calcular volumen navideño (meses seleccionados)
            df['volumen_navidad'] = 0
            for col in selected_month_cols:
                df['volumen_navidad'] += pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
            st.success(f"✅ Usando volumen de: {', '.join(selected_months_names)}")
        
        # Filtrar por volumen mínimo
        df_filtered = df[df['volumen_navidad'] >= min_volume].copy()
        
        if len(df_filtered) == 0:
            st.error("❌ No hay keywords con el volumen mínimo especificado")
            return
        
        st.info(f"📊 Keywords después de filtrar: {len(df_filtered)}")
        
        # Enriquecer datos
        with st.spinner("🔄 Procesando keywords..."):
            df_filtered['keyword_clean'] = df_filtered['Keyword'].apply(clean_keyword)
            df_filtered['intent'] = df_filtered['Keyword'].apply(classify_intent)
            df_filtered['destinatario'] = df_filtered['Keyword'].apply(extract_gift_recipient)
            df_filtered['rango_precio'] = df_filtered['Keyword'].apply(extract_price_range)
            df_filtered['familias_producto'] = df_filtered['Keyword'].apply(match_product_family)
        
        # Crear embeddings y clusters
        with st.spinner("🧠 Creando clusters semánticos..."):
            keywords_list = df_filtered['keyword_clean'].tolist()
            embeddings, vectorizer = create_embeddings_tfidf(keywords_list)
            
            # Aplicar clustering
            if clustering_method == "K-Means":
                clusters = cluster_keywords_kmeans(embeddings, n_clusters)
            else:
                clusters = cluster_keywords_hierarchical(embeddings, n_clusters)
            
            df_filtered['cluster_id'] = clusters
            
            # Calcular coherencia de clusters
            coherences = calculate_cluster_coherence(embeddings, clusters)
        
        # Nombrar clusters (usando keyword más frecuente o con más volumen)
        cluster_names = {}
        for cluster_id in df_filtered['cluster_id'].unique():
            cluster_kws = df_filtered[df_filtered['cluster_id'] == cluster_id]
            # Keyword con mayor volumen del cluster
            top_kw = cluster_kws.nlargest(1, 'volumen_navidad')['Keyword'].values[0]
            # Simplificar nombre
            name = top_kw[:50] + "..." if len(top_kw) > 50 else top_kw
            cluster_names[cluster_id] = f"C{cluster_id}: {name}"
        
        df_filtered['cluster_name'] = df_filtered['cluster_id'].map(cluster_names)
        
        # ============== DASHBOARD ==============
        
        # Métricas principales
        st.markdown("## 📈 Resumen General")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Keywords",
                f"{len(df_filtered):,}",
                delta=f"{len(df_filtered) - len(df):+,} vs original"
            )
        
        with col2:
            st.metric(
                "Clusters Creados",
                n_clusters,
                delta=f"~{len(df_filtered)//n_clusters} kw/cluster"
            )
        
        with col3:
            total_vol = df_filtered['volumen_navidad'].sum()
            st.metric(
                f"Volumen Total ({months_label})",
                f"{total_vol:,}"
            )
        
        with col4:
            avg_coherence = np.mean(list(coherences.values()))
            st.metric(
                "Coherencia Media",
                f"{avg_coherence:.2%}"
            )
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4 = st.tabs([
            "🗺️ Visualización", 
            "📋 Clusters Detallados",
            "🎯 URLs Recomendadas",
            "📊 Datos Completos"
        ])
        
        with tab1:
            st.markdown("### Visualización de Clusters")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Treemap
                fig_treemap = create_treemap(df_filtered)
                st.plotly_chart(fig_treemap, use_container_width=True)
            
            with col2:
                # Scatter plot
                fig_scatter = create_cluster_scatter_plot(df_filtered, embeddings)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Bar chart de volumen
            fig_bar = create_volume_bar_chart(df_filtered)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Matriz de oportunidades
            fig_opportunity = create_url_opportunity_chart(df_filtered)
            st.plotly_chart(fig_opportunity, use_container_width=True)
        
        with tab2:
            st.markdown("### 📋 Análisis Detallado por Cluster")
            
            # Selector de cluster
            cluster_options = sorted(df_filtered['cluster_name'].unique())
            selected_cluster = st.selectbox("Selecciona un cluster:", cluster_options)
            
            cluster_data = df_filtered[df_filtered['cluster_name'] == selected_cluster]
            
            # Métricas del cluster
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Keywords", len(cluster_data))
            with col2:
                st.metric("Volumen Total", f"{cluster_data['volumen_navidad'].sum():,}")
            with col3:
                st.metric("Volumen Medio", f"{cluster_data['volumen_navidad'].mean():.0f}")
            with col4:
                cluster_id = cluster_data['cluster_id'].iloc[0]
                st.metric("Coherencia", f"{coherences.get(cluster_id, 0):.2%}")
            
            # Intent distribution
            intent_dist = cluster_data['intent'].value_counts()
            st.markdown("**Distribución de Intención:**")
            for intent, count in intent_dist.items():
                st.markdown(f"- {intent}: {count} ({count/len(cluster_data):.1%})")
            
            # Análisis AI si está disponible
            if api_key and api_option != "Sin AI (solo TF-IDF)":
                if st.button("🤖 Analizar con AI", key=f"ai_btn_{selected_cluster}"):
                    with st.spinner("Analizando cluster con AI..."):
                        kws = cluster_data['Keyword'].tolist()
                        vols = cluster_data['volumen_navidad'].tolist()
                        
                        if "Claude" in api_option:
                            analysis = get_cluster_analysis_claude(kws, vols, api_key)
                        else:
                            analysis = get_cluster_analysis_openai(kws, vols, api_key)
                        
                        if analysis:
                            st.markdown("### 🤖 Análisis AI del Cluster")
                            
                            st.markdown(f"**Nombre sugerido:** {analysis.get('nombre_cluster', 'N/A')}")
                            st.markdown(f"**Tema principal:** {analysis.get('tema_principal', 'N/A')}")
                            
                            st.markdown(f"""
                            <div class="url-suggestion">
                                URL Sugerida: {analysis.get('url_sugerida', 'N/A')}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.markdown(f"**H1:** {analysis.get('h1_sugerido', 'N/A')}")
                            st.markdown(f"**Meta Description:** {analysis.get('meta_description', 'N/A')}")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Productos recomendados:**")
                                for prod in analysis.get('productos_recomendados', []):
                                    st.markdown(f"- {prod}")
                            
                            with col2:
                                st.markdown("**Query Fan-Out:**")
                                for query in analysis.get('query_fanout', []):
                                    st.markdown(f"- {query}")
                            
                            st.markdown(f"**Competencia:** {analysis.get('nivel_competencia', 'N/A')} | **Potencial:** {analysis.get('potencial_conversion', 'N/A')}")
            
            # Tabla de keywords del cluster
            st.markdown("### Keywords del Cluster")
            st.dataframe(
                cluster_data[['Keyword', 'volumen_navidad', 'intent', 'destinatario', 'rango_precio']]
                .sort_values('volumen_navidad', ascending=False)
                .head(50),
                use_container_width=True
            )
        
        with tab3:
            st.markdown("### 🎯 URLs Recomendadas por Potencial")
            
            # Calcular métricas por cluster para ranking
            cluster_summary = df_filtered.groupby('cluster_name').agg({
                'volumen_navidad': ['sum', 'mean', 'count'],
                'cluster_id': 'first'
            }).reset_index()
            cluster_summary.columns = ['Cluster', 'Volumen Total', 'Volumen Medio', 'Num Keywords', 'Cluster ID']
            
            # Añadir coherencia
            cluster_summary['Coherencia'] = cluster_summary['Cluster ID'].map(coherences)
            
            # Calcular score de prioridad
            cluster_summary['Score Prioridad'] = (
                cluster_summary['Volumen Total'] * 0.4 +
                cluster_summary['Volumen Medio'] * 0.3 +
                cluster_summary['Num Keywords'] * 50 * 0.2 +
                cluster_summary['Coherencia'] * 1000 * 0.1
            )
            
            cluster_summary = cluster_summary.sort_values('Score Prioridad', ascending=False)
            
            # Mostrar top URLs
            st.markdown("#### 🏆 Top 20 URLs por Potencial")
            
            for idx, row in cluster_summary.head(20).iterrows():
                cluster_kws = df_filtered[df_filtered['cluster_name'] == row['Cluster']]
                top_keywords = cluster_kws.nlargest(5, 'volumen_navidad')['Keyword'].tolist()
                url_suggestion = suggest_url_for_cluster(
                    cluster_kws['Keyword'].tolist(),
                    cluster_kws['volumen_navidad'].tolist()
                )
                
                with st.expander(f"**{row['Cluster'][:60]}** - Vol: {row['Volumen Total']:,.0f} | Score: {row['Score Prioridad']:,.0f}"):
                    st.markdown(f"""
                    <div class="url-suggestion">
                        📍 URL Sugerida: {url_suggestion}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Volumen Total", f"{row['Volumen Total']:,.0f}")
                    with col2:
                        st.metric("Keywords", f"{row['Num Keywords']:,.0f}")
                    with col3:
                        st.metric("Coherencia", f"{row['Coherencia']:.1%}")
                    
                    st.markdown("**Top Keywords:**")
                    for kw in top_keywords:
                        vol = cluster_kws[cluster_kws['Keyword'] == kw]['volumen_navidad'].values[0]
                        st.markdown(f"- {kw} ({vol:,})")
                    
                    # Familias de producto relacionadas
                    all_families = []
                    for fams in cluster_kws['familias_producto']:
                        all_families.extend(fams)
                    if all_families:
                        unique_families = list(set(all_families))
                        st.markdown("**Familias de Producto Relacionadas:**")
                        st.markdown(", ".join(unique_families))
            
            # Exportar recomendaciones
            st.markdown("---")
            st.markdown("### 📥 Exportar Recomendaciones")
            
            export_data = []
            for idx, row in cluster_summary.iterrows():
                cluster_kws = df_filtered[df_filtered['cluster_name'] == row['Cluster']]
                url_suggestion = suggest_url_for_cluster(
                    cluster_kws['Keyword'].tolist(),
                    cluster_kws['volumen_navidad'].tolist()
                )
                
                export_data.append({
                    'Cluster': row['Cluster'],
                    'URL Sugerida': url_suggestion,
                    'Volumen Total Estacional': row['Volumen Total'],
                    'Volumen Medio': row['Volumen Medio'],
                    'Num Keywords': row['Num Keywords'],
                    'Coherencia': row['Coherencia'],
                    'Score Prioridad': row['Score Prioridad'],
                    'Top Keywords': ' | '.join(cluster_kws.nlargest(5, 'volumen_navidad')['Keyword'].tolist())
                })
            
            export_df = pd.DataFrame(export_data)
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Descargar Recomendaciones (CSV)",
                data=csv,
                file_name="urls_recomendadas_navidad.csv",
                mime="text/csv"
            )
        
        with tab4:
            st.markdown("### 📊 Datos Completos")
            
            # Filtros
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_intent = st.multiselect(
                    "Filtrar por Intent",
                    df_filtered['intent'].unique()
                )
            
            with col2:
                filter_recipient = st.multiselect(
                    "Filtrar por Destinatario",
                    [d for d in df_filtered['destinatario'].unique() if d is not None]
                )
            
            with col3:
                filter_price = st.multiselect(
                    "Filtrar por Rango Precio",
                    [p for p in df_filtered['rango_precio'].unique() if p is not None]
                )
            
            # Aplicar filtros
            df_display = df_filtered.copy()
            
            if filter_intent:
                df_display = df_display[df_display['intent'].isin(filter_intent)]
            
            if filter_recipient:
                df_display = df_display[df_display['destinatario'].isin(filter_recipient)]
            
            if filter_price:
                df_display = df_display[df_display['rango_precio'].isin(filter_price)]
            
            st.info(f"Mostrando {len(df_display)} keywords")
            
            # Tabla interactiva
            st.dataframe(
                df_display[[
                    'Keyword', 'cluster_name', 'volumen_navidad', 
                    'intent', 'destinatario', 'rango_precio'
                ]].sort_values('volumen_navidad', ascending=False),
                use_container_width=True,
                height=600
            )
            
            # Exportar datos completos
            csv_full = df_display.to_csv(index=False)
            st.download_button(
                label="⬇️ Descargar Datos Completos (CSV)",
                data=csv_full,
                file_name="keywords_clustering_navidad_completo.csv",
                mime="text/csv"
            )
    
    else:
        # Mostrar instrucciones cuando no hay archivo
        st.markdown("""
        ## 📖 Instrucciones de Uso
        
        ### 1. Preparar el archivo CSV
        
        El archivo debe ser exportado de **Google Keyword Planner** con las siguientes columnas:
        - `Keyword` - Las keywords de búsqueda
        - `Avg. monthly searches` - Volumen promedio mensual
        - `Searches: Dec 2024` - Búsquedas de Diciembre
        - `Searches: Jan 2025` - Búsquedas de Enero
        - `Competition` - Nivel de competencia (opcional)
        
        ### 2. Cargar archivo
        
        Usa el botón **"Browse files"** en la barra lateral para subir tu CSV.
        
        ### 3. Configurar parámetros
        
        - **Número de clusters**: Cuántos grupos semánticos crear (15-25 recomendado)
        - **Método de clustering**: K-Means o Jerárquico
        - **Volumen mínimo**: Filtrar keywords con bajo potencial
        
        ### 4. Opcional: Usar AI
        
        Conecta tu API de **Claude** o **GPT** para obtener:
        - Nombres descriptivos de clusters
        - URLs optimizadas
        - H1 y meta descriptions sugeridos
        - Query Fan-Out para cada cluster
        
        ---
        
        ### 🎯 Metodología de Clustering
        
        Esta herramienta combina:
        
        1. **Clustering Semántico (TF-IDF + NLP)**
           - Agrupa keywords por similitud semántica
           - Considera n-gramas para capturar frases completas
        
        2. **Análisis de Intención de Búsqueda**
           - Transaccional 💰
           - Informacional 📚
           - Navegacional 🧭
           - Regalo 🎁
        
        3. **Enriquecimiento de Datos**
           - Destinatario del regalo
           - Rango de precio mencionado
           - Familias de producto relacionadas
        
        4. **Query Fan-Out**
           - Expansión de queries relacionadas
           - Cobertura completa de intención de búsqueda
        """)
        
        # Mostrar ejemplo de datos esperados
        st.markdown("### 📝 Ejemplo de Datos Esperados")
        
        example_data = pd.DataFrame({
            'Keyword': [
                'regalos navidad hombre',
                'regalos tecnologicos navidad',
                'amigo invisible 30 euros',
                'gadgets para regalar',
                'mejores auriculares para regalar'
            ],
            'Avg. monthly searches': [5400, 3200, 1800, 890, 720],
            'Searches: Dec 2024': [12000, 8500, 4200, 2100, 1800],
            'Searches: Jan 2025': [6500, 4000, 900, 1100, 950],
            'Competition': ['Alta', 'Alta', 'Media', 'Media', 'Alta']
        })
        
        st.dataframe(example_data, use_container_width=True)

if __name__ == "__main__":
    main()
