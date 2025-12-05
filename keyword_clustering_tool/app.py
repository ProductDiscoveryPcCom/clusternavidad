"""
🎄 Keyword Semantic Clustering Tool - Navidad Edition
Herramienta avanzada para agrupar semánticamente keywords de Google Keyword Planner
con enfoque en campaña de Navidad/Regalos

Desarrollado para PcComponentes
v3.0 - Con Sentence Transformers, HDBSCAN y clasificación AI
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from collections import defaultdict, Counter
import re
import json
from typing import List, Dict, Tuple, Optional
from functools import lru_cache
import os

# Imports para AI
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Imports opcionales para NLP avanzado
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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
    /* Tema navideño */
    .stApp {
        background: linear-gradient(180deg, #0a1f0a 0%, #1a2f1a 100%);
    }
    
    .main-header {
        background: linear-gradient(90deg, #1a472a 0%, #2d5a3d 50%, #1a472a 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 2px solid #c41e3a;
        box-shadow: 0 4px 20px rgba(196, 30, 58, 0.3);
    }
    
    .main-header h1 {
        color: #ffd700;
        text-align: center;
        font-size: 2.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-header p {
        color: #e8e8e8;
        text-align: center;
        font-size: 1.1rem;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(135deg, #1a472a 0%, #2d5a3d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #c41e3a;
        margin: 0.5rem 0;
    }
    
    /* Tabs personalizados */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #1a472a;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #2d5a3d;
        border-radius: 8px;
        color: white;
        padding: 0.5rem 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #c41e3a !important;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a472a 0%, #0d2818 100%);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e8e8e8;
    }
    
    /* Botones */
    .stButton > button {
        background: linear-gradient(90deg, #c41e3a 0%, #a01830 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #d42e4a 0%, #b02840 100%);
        box-shadow: 0 4px 15px rgba(196, 30, 58, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ============== DICCIONARIOS DE PRODUCTOS ==============

PRODUCTOS_PCCOMPONENTES = {
    "movil": {
        "nombre": "Tecnología Móvil",
        "emoji": "📱",
        "productos": ["smartphones", "tablets", "smartwatches", "auriculares", "cargadores", "fundas"],
        "keywords_match": [
            "movil", "móvil", "smartphone", "telefono", "teléfono", "iphone", "samsung", 
            "xiaomi", "redmi", "poco", "oneplus", "oppo", "realme", "google pixel",
            "tablet", "ipad", "galaxy tab", "smartwatch", "reloj inteligente", 
            "apple watch", "galaxy watch", "amazfit", "garmin", "fitbit",
            "auriculares", "airpods", "earbuds", "cascos", "headphones",
            "powerbank", "cargador", "cable usb", "funda movil", "protector pantalla",
            "android", "ios", "5g", "dual sim"
        ]
    },
    "informatica": {
        "nombre": "Informática",
        "emoji": "💻",
        "productos": ["portátiles", "sobremesa", "monitores", "periféricos", "almacenamiento"],
        "keywords_match": [
            "portatil", "portátil", "laptop", "notebook", "ordenador", "pc", "computer",
            "sobremesa", "desktop", "workstation", "all in one",
            "monitor", "pantalla", "display", "curved", "gaming monitor",
            "teclado", "keyboard", "raton", "ratón", "mouse", "mousepad", "alfombrilla",
            "webcam", "camara web", "cámara web", "microphone", "microfono",
            "disco duro", "ssd", "hdd", "nvme", "pendrive", "usb", "memoria",
            "hub", "docking", "adaptador", "cable hdmi", "cable displayport",
            "macbook", "imac", "mac mini", "chromebook", "thinkpad", "dell", "hp", "lenovo", "asus", "acer",
            "impresora", "escaner", "scanner", "multifuncion"
        ]
    },
    "hogar": {
        "nombre": "Hogar Inteligente",
        "emoji": "🏠",
        "productos": ["aspiradores", "robots", "domótica", "climatización"],
        "keywords_match": [
            "aspirador", "aspiradora", "robot aspirador", "roomba", "conga", "roborock",
            "dyson", "rowenta", "philips", "cecotec", "xiaomi home",
            "friegasuelos", "mopa", "vaporeta", "limpiador vapor",
            "plancha", "centro planchado", "plancha vapor",
            "ventilador", "aire acondicionado", "calefactor", "radiador", "estufa",
            "purificador", "humidificador", "deshumidificador",
            "domotica", "domótica", "smart home", "alexa", "google home", "echo",
            "bombilla inteligente", "enchufe inteligente", "termostato", "nest",
            "camara seguridad", "cámara seguridad", "vigilancia", "sensor", "alarma"
        ]
    },
    "tv_audio": {
        "nombre": "TV y Audio",
        "emoji": "📺",
        "productos": ["televisores", "barras de sonido", "home cinema", "proyectores"],
        "keywords_match": [
            "television", "televisor", "tv", "smart tv", "oled", "qled", "led",
            "4k", "8k", "uhd", "hdr", "55 pulgadas", "65 pulgadas", "75 pulgadas",
            "samsung tv", "lg tv", "sony tv", "philips tv", "hisense", "tcl",
            "barra sonido", "soundbar", "home cinema", "altavoces", "subwoofer",
            "sonos", "bose", "jbl", "marshall", "harman kardon", "bang olufsen",
            "proyector", "projector", "pantalla proyeccion", "cine casa",
            "chromecast", "fire stick", "apple tv", "roku", "android tv",
            "receptor av", "amplificador", "tocadiscos", "vinilo"
        ]
    },
    "gaming": {
        "nombre": "Gaming",
        "emoji": "🎮",
        "productos": ["PC gaming", "periféricos gaming", "sillas", "streaming"],
        "keywords_match": [
            "gaming", "gamer", "juego", "juegos", "videojuego", "videojuegos",
            "pc gaming", "ordenador gaming", "portatil gaming", "setup gaming",
            "monitor gaming", "144hz", "240hz", "1ms", "gsync", "freesync",
            "teclado gaming", "mecanico", "rgb", "cherry mx", "raton gaming",
            "auriculares gaming", "headset", "microfono streaming",
            "silla gaming", "escritorio gaming", "mesa gaming",
            "streaming", "streamer", "capturadora", "elgato", "obs",
            "razer", "logitech g", "corsair", "steelseries", "hyperx", "asus rog", "msi",
            "nvidia", "rtx", "geforce", "amd radeon", "tarjeta grafica", "gpu"
        ]
    },
    "consolas": {
        "nombre": "Consolas",
        "emoji": "🕹️",
        "productos": ["PlayStation", "Xbox", "Nintendo", "accesorios"],
        "keywords_match": [
            "consola", "playstation", "ps5", "ps4", "sony playstation",
            "xbox", "xbox series", "xbox one", "microsoft xbox",
            "nintendo", "switch", "nintendo switch", "switch oled", "switch lite",
            "mando", "controller", "dualshock", "dualsense", "joy-con",
            "juego ps5", "juego xbox", "juego switch", "exclusivo",
            "ps plus", "game pass", "nintendo online",
            "vr", "psvr", "realidad virtual", "oculus", "meta quest",
            "retro", "mini consola", "arcade"
        ]
    },
    "cafe": {
        "nombre": "Café",
        "emoji": "☕",
        "productos": ["cafeteras", "cápsulas", "accesorios café"],
        "keywords_match": [
            "cafetera", "cafe", "café", "espresso", "cappuccino", "latte",
            "nespresso", "dolce gusto", "senseo", "tassimo",
            "cafetera express", "cafetera italiana", "moka", "french press",
            "cafetera goteo", "cafetera filtro", "chemex", "aeropress",
            "cafetera automatica", "superautomatica", "delonghi", "philips", "krups", "siemens",
            "molinillo", "grinder", "cafe molido", "cafe grano",
            "capsulas", "cápsulas", "compatible nespresso",
            "espumador", "vaporizador leche", "barista"
        ]
    },
    "cocina": {
        "nombre": "Cocina",
        "emoji": "🍳",
        "productos": ["robots cocina", "freidoras", "pequeño electrodoméstico"],
        "keywords_match": [
            "freidora", "freidora aire", "air fryer", "airfryer", "cosori", "ninja",
            "robot cocina", "thermomix", "mambo", "mycook", "monsieur cuisine",
            "batidora", "amasadora", "kitchen aid", "kitchenaid", "bosch",
            "licuadora", "exprimidor", "zumo", "smoothie",
            "tostadora", "sandwichera", "grill", "plancha cocina",
            "microondas", "horno", "mini horno",
            "olla", "olla express", "slow cooker", "crockpot", "instant pot",
            "bascula cocina", "báscula cocina", "termometro cocina",
            "envasadora", "vacio", "deshidratador", "heladera"
        ]
    },
    "deporte": {
        "nombre": "Deporte y Fitness",
        "emoji": "🏃",
        "productos": ["wearables", "accesorios fitness", "movilidad"],
        "keywords_match": [
            "deporte", "fitness", "gym", "gimnasio", "entrenamiento",
            "pulsera actividad", "smartband", "mi band", "fitbit", "garmin",
            "reloj deportivo", "running", "ciclismo", "natacion",
            "gps", "strava", "polar", "suunto", "coros",
            "auriculares deporte", "sport", "resistente agua", "ipx",
            "patinete", "patinete electrico", "scooter", "xiaomi scooter",
            "bicicleta", "bici electrica", "ebike", "ciclocomputador",
            "cinta correr", "eliptica", "bicicleta estatica", "spinning",
            "pesas", "mancuernas", "banco ejercicio", "esterilla", "yoga"
        ]
    },
    "belleza": {
        "nombre": "Belleza y Cuidado Personal",
        "emoji": "💄",
        "productos": ["afeitado", "cuidado capilar", "higiene"],
        "keywords_match": [
            "afeitadora", "maquinilla", "barbero", "cortapelos", "recortadora",
            "braun", "philips", "remington", "panasonic",
            "secador", "secador pelo", "plancha pelo", "rizador", "ghd", "dyson airwrap",
            "cepillo electrico", "oral b", "irrigador", "waterpik",
            "depiladora", "ipl", "laser", "cera", "silk epil",
            "masajeador", "pistola masaje", "electroestimulador",
            "bascula", "báscula", "bascula inteligente", "tanita", "withings",
            "tensiometro", "termometro", "pulsioximetro", "salud"
        ]
    },
    "gadgets": {
        "nombre": "Gadgets y Accesorios",
        "emoji": "🔌",
        "productos": ["altavoces inteligentes", "wearables", "accesorios tech"],
        "keywords_match": [
            "gadget", "gadgets", "accesorio", "accesorios", "tech", "tecnologia",
            "altavoz inteligente", "echo", "alexa", "google nest", "homepod",
            "altavoz bluetooth", "altavoz portatil", "jbl", "bose", "marshall", "sonos",
            "kindle", "ebook", "ereader", "kobo",
            "drone", "dji", "mavic", "mini drone", "fpv",
            "camara", "cámara", "gopro", "action cam", "instantanea", "instax", "polaroid",
            "gafas vr", "realidad aumentada", "ar", "ray ban meta",
            "airtag", "tile", "localizador", "tracker"
        ]
    },
    "juguetes": {
        "nombre": "Juguetes Tech",
        "emoji": "🧸",
        "productos": ["juguetes electrónicos", "educativos", "radiocontrol"],
        "keywords_match": [
            "juguete", "juguetes", "niño", "niña", "infantil",
            "robot", "robot educativo", "programacion niños", "stem",
            "coche rc", "radiocontrol", "drone juguete", "helicoptero",
            "consola niños", "tablet niños", "smartwatch niños",
            "karaoke", "micro", "musical",
            "peluche", "interactivo", "mascota electronica", "tamagotchi",
            "construccion", "electronico", "experimentos", "ciencia"
        ]
    },
    "lego": {
        "nombre": "LEGO",
        "emoji": "🧱",
        "productos": ["LEGO"],
        "keywords_match": [
            "lego", "construccion", "bloques", "technic", "star wars lego",
            "harry potter lego", "marvel lego", "city", "friends", "duplo",
            "creator", "architecture", "ideas", "speed champions",
            "ninjago", "minecraft lego", "super mario lego"
        ]
    }
}

# ============== CATEGORÍAS DE AUDIENCIA ==============

CATEGORIAS_AUDIENCIA = {
    "hombre": {
        "nombre": "Regalos para Hombre",
        "emoji": "👨",
        "tipo": "genero",
        "keywords_match": [
            "hombre", "hombres", "chico", "chicos", "masculino",
            "el", "él", "caballero", "señor", "varon", "varón"
        ]
    },
    "mujer": {
        "nombre": "Regalos para Mujer",
        "emoji": "👩",
        "tipo": "genero",
        "keywords_match": [
            "mujer", "mujeres", "chica", "chicas", "femenino",
            "ella", "dama", "señora"
        ]
    },
    "bebe": {
        "nombre": "Regalos para Bebés",
        "emoji": "👶",
        "tipo": "edad",
        "keywords_match": [
            "bebe", "bebé", "bebes", "bebés", "recien nacido", "recién nacido",
            "0 años", "1 año", "2 años", "lactante"
        ]
    },
    "nino": {
        "nombre": "Regalos para Niños",
        "emoji": "🧒",
        "tipo": "edad",
        "keywords_match": [
            "niño", "niños", "niña", "niñas", "infantil", "infancia",
            "3 años", "4 años", "5 años", "6 años", "7 años", "8 años", 
            "9 años", "10 años", "pequeño", "pequeños", "peque", "peques",
            "hijo", "hija", "hijos", "hijas"
        ]
    },
    "preadolescente": {
        "nombre": "Regalos para Preadolescentes",
        "emoji": "🧑",
        "tipo": "edad",
        "keywords_match": [
            "preadolescente", "11 años", "12 años", "13 años",
            "tweens", "tween"
        ]
    },
    "adolescente": {
        "nombre": "Regalos para Adolescentes",
        "emoji": "🧑‍🎤",
        "tipo": "edad",
        "keywords_match": [
            "adolescente", "adolescentes", "teen", "teens", "teenager",
            "joven", "jovenes", "jóvenes", "juvenil", "juventud",
            "14 años", "15 años", "16 años", "17 años", "18 años"
        ]
    },
    "adulto": {
        "nombre": "Regalos para Adultos",
        "emoji": "🧑‍💼",
        "tipo": "edad",
        "keywords_match": ["adulto", "adultos", "mayor de edad", "mayores"]
    },
    "senior": {
        "nombre": "Regalos para Mayores",
        "emoji": "🧓",
        "tipo": "edad",
        "keywords_match": [
            "abuelo", "abuela", "abuelos", "abuelas", "anciano", "ancianos",
            "mayor", "tercera edad", "senior", "seniors",
            "jubilado", "jubilados", "70 años", "80 años"
        ]
    },
    "padre": {
        "nombre": "Regalos para Padre",
        "emoji": "👨‍👧",
        "tipo": "relacion",
        "keywords_match": [
            "padre", "padres", "papa", "papá", "papi",
            "dia del padre", "día del padre"
        ]
    },
    "madre": {
        "nombre": "Regalos para Madre",
        "emoji": "👩‍👧",
        "tipo": "relacion",
        "keywords_match": [
            "madre", "madres", "mama", "mamá", "mami",
            "dia de la madre", "día de la madre"
        ]
    },
    "hermano": {
        "nombre": "Regalos para Hermanos",
        "emoji": "👫",
        "tipo": "relacion",
        "keywords_match": [
            "hermano", "hermana", "hermanos", "hermanas", "hermanito", "hermanita"
        ]
    },
    "pareja": {
        "nombre": "Regalos para Pareja",
        "emoji": "💑",
        "tipo": "relacion",
        "keywords_match": [
            "pareja", "parejas", "novio", "novia", "novios",
            "marido", "esposo", "esposa", "conyuge", "cónyuge",
            "enamorado", "enamorada", "enamorados", "san valentin", "san valentín",
            "aniversario", "romantico", "romántico"
        ]
    },
    "amigo": {
        "nombre": "Regalos para Amigos",
        "emoji": "🤝",
        "tipo": "relacion",
        "keywords_match": [
            "amigo", "amiga", "amigos", "amigas", "amistad",
            "amigo invisible", "mejor amigo", "mejor amiga",
            "colega", "colegas", "compi", "compis"
        ]
    },
    "companero": {
        "nombre": "Regalos para Compañeros",
        "emoji": "💼",
        "tipo": "relacion",
        "keywords_match": [
            "compañero", "compañera", "compañeros", "trabajo", "oficina",
            "jefe", "jefa", "empleado", "empleada", "empresa", "corporativo"
        ]
    },
    "navidad_general": {
        "nombre": "Navidad General",
        "emoji": "🎄",
        "tipo": "ocasion",
        "keywords_match": [
            "navidad", "navidades", "navideño", "navideña", "navideños",
            "noel", "papa noel", "papá noel", "santa claus",
            "reyes", "reyes magos", "día de reyes",
            "fiestas", "festivo", "festivos", "holidays",
            "diciembre", "enero", "nochebuena", "nochevieja"
        ]
    },
    "amigo_invisible": {
        "nombre": "Amigo Invisible",
        "emoji": "🎁",
        "tipo": "ocasion",
        "keywords_match": [
            "amigo invisible", "amigo secreto", "secret santa",
            "intercambio regalo", "intercambio regalos",
            "sorteo regalo", "kris kringle"
        ]
    },
    "original": {
        "nombre": "Regalos Originales",
        "emoji": "✨",
        "tipo": "estilo",
        "keywords_match": [
            "original", "originales", "creativo", "creativos",
            "unico", "único", "diferente", "especial", "sorpresa",
            "curioso", "divertido", "gracioso", "friki", "geek", "frikis"
        ]
    },
    "economico": {
        "nombre": "Regalos Económicos",
        "emoji": "💰",
        "tipo": "precio",
        "keywords_match": [
            "barato", "baratos", "economico", "económico", "economicos",
            "low cost", "menos de 10", "menos de 20", "menos de 30",
            "hasta 10 euros", "hasta 20 euros", "hasta 30 euros",
            "por poco dinero", "sin gastar mucho", "presupuesto"
        ]
    },
    "premium": {
        "nombre": "Regalos Premium",
        "emoji": "👑",
        "tipo": "precio",
        "keywords_match": [
            "premium", "lujo", "lujoso", "exclusivo", "caro",
            "alta gama", "high end", "top", "mejor", "mejores"
        ]
    }
}

# ============== FUNCIONES DE UTILIDAD ==============

def clean_keyword(keyword: str) -> str:
    """Limpia y normaliza una keyword"""
    kw = str(keyword).lower().strip()
    kw = re.sub(r'[^\w\s]', ' ', kw)
    kw = re.sub(r'\s+', ' ', kw)
    return kw.strip()

def preprocess_keyword_advanced(keyword: str) -> str:
    """Preprocesamiento avanzado de keywords"""
    kw = str(keyword).lower().strip()
    
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u', 'ü': 'u', 'ñ': 'n'
    }
    for old, new in replacements.items():
        kw = kw.replace(old, new)
    
    expansions = {
        'tv': 'television',
        'pc': 'ordenador',
        'portatil': 'ordenador portatil',
        'movil': 'telefono movil',
    }
    words = kw.split()
    expanded = []
    for word in words:
        expanded.append(expansions.get(word, word))
    kw = ' '.join(expanded)
    
    kw = re.sub(r'[^\w\s]', ' ', kw)
    kw = re.sub(r'\s+', ' ', kw).strip()
    
    return kw

def classify_intent(keyword: str) -> str:
    """Clasifica la intención de búsqueda"""
    kw = keyword.lower()
    
    transactional = ['comprar', 'precio', 'oferta', 'descuento', 'barato', 'tienda', 
                     'donde comprar', 'mejor precio', 'black friday', 'rebajas', 'outlet']
    if any(t in kw for t in transactional):
        return "💰 Transaccional"
    
    informational = ['que es', 'qué es', 'como', 'cómo', 'guia', 'guía', 'tutorial',
                    'mejor', 'mejores', 'comparativa', 'vs', 'versus', 'review', 
                    'opinion', 'opinión', 'merece la pena', 'vale la pena']
    if any(i in kw for i in informational):
        return "📚 Informacional"
    
    navigational = ['amazon', 'pccomponentes', 'mediamarkt', 'el corte ingles', 
                   'fnac', 'carrefour', 'aliexpress', 'oficial', 'web']
    if any(n in kw for n in navigational):
        return "🧭 Navegacional"
    
    gift = ['regalo', 'regalos', 'regalar', 'navidad', 'reyes', 'amigo invisible']
    if any(g in kw for g in gift):
        return "🎁 Regalo"
    
    return "🔍 General"

def extract_gift_recipient(keyword: str) -> Optional[str]:
    """Extrae el destinatario del regalo"""
    kw = keyword.lower()
    
    recipients = {
        'padre': ['padre', 'papa', 'papá'],
        'madre': ['madre', 'mama', 'mamá'],
        'hombre': ['hombre', 'chico', 'novio', 'marido', 'él'],
        'mujer': ['mujer', 'chica', 'novia', 'esposa', 'ella'],
        'niño': ['niño', 'niña', 'niños', 'hijo', 'hija', 'pequeño'],
        'adolescente': ['adolescente', 'joven', 'teen'],
        'abuelo': ['abuelo', 'abuela', 'abuelos'],
        'amigo': ['amigo', 'amiga', 'amigo invisible'],
        'pareja': ['pareja', 'novio', 'novia', 'enamorado']
    }
    
    for recipient, keywords in recipients.items():
        if any(k in kw for k in keywords):
            return recipient
    
    return None

def extract_price_range(keyword: str) -> Optional[str]:
    """Extrae el rango de precio mencionado"""
    kw = keyword.lower()
    
    patterns = [
        (r'menos de (\d+)', 'hasta'),
        (r'hasta (\d+)', 'hasta'),
        (r'por debajo de (\d+)', 'hasta'),
        (r'(\d+) euros', 'aprox'),
        (r'(\d+)€', 'aprox'),
    ]
    
    for pattern, prefix in patterns:
        match = re.search(pattern, kw)
        if match:
            amount = int(match.group(1))
            if amount <= 30:
                return "💚 Hasta 30€"
            elif amount <= 60:
                return "💛 Hasta 60€"
            elif amount <= 100:
                return "🧡 Hasta 100€"
            else:
                return "❤️ Más de 100€"
    
    return None

# ============== FUNCIONES DE MATCHING ==============

def match_product_family(keyword: str) -> List[Dict]:
    """Encuentra las familias de producto que coinciden"""
    keyword_lower = keyword.lower()
    keyword_normalized = preprocess_keyword_advanced(keyword)
    
    matches = []
    
    for family_id, family_data in PRODUCTOS_PCCOMPONENTES.items():
        matched_keywords = []
        for kw in family_data["keywords_match"]:
            kw_normalized = preprocess_keyword_advanced(kw)
            if kw_normalized in keyword_normalized or kw.lower() in keyword_lower:
                matched_keywords.append(kw)
        
        if matched_keywords:
            score = len(matched_keywords) + sum(len(k) for k in matched_keywords) / 100
            matches.append({
                "family_id": family_id,
                "family_name": family_data["nombre"],
                "emoji": family_data["emoji"],
                "score": score,
                "matched_keywords": matched_keywords
            })
    
    return sorted(matches, key=lambda x: x["score"], reverse=True)

def get_best_product_family(keyword: str) -> Optional[str]:
    """Retorna el ID de la mejor familia de producto"""
    matches = match_product_family(keyword)
    return matches[0]["family_id"] if matches else None

def get_product_match_score(keyword: str) -> float:
    """Retorna el score de match de producto"""
    matches = match_product_family(keyword)
    return min(matches[0]["score"] / 5, 1.0) if matches else 0.0

def keyword_has_product_match(keyword: str) -> bool:
    """Verifica si una keyword tiene match con productos"""
    return len(match_product_family(keyword)) > 0

def match_audience_category(keyword: str) -> List[Dict]:
    """Encuentra las categorías de audiencia que coinciden"""
    keyword_lower = keyword.lower()
    keyword_normalized = preprocess_keyword_advanced(keyword)
    
    matches = []
    
    for cat_id, cat_data in CATEGORIAS_AUDIENCIA.items():
        matched_terms = []
        for kw in cat_data["keywords_match"]:
            kw_normalized = preprocess_keyword_advanced(kw)
            if kw_normalized in keyword_normalized or kw.lower() in keyword_lower:
                matched_terms.append(kw)
        
        if matched_terms:
            score = len(matched_terms) + sum(len(t) for t in matched_terms) / 100
            matches.append({
                "category_id": cat_id,
                "category_name": cat_data["nombre"],
                "emoji": cat_data["emoji"],
                "tipo": cat_data["tipo"],
                "score": score,
                "matched_terms": matched_terms
            })
    
    return sorted(matches, key=lambda x: x["score"], reverse=True)

def get_audience_by_type(keyword: str) -> Dict[str, Optional[str]]:
    """Retorna la mejor categoría de audiencia por tipo"""
    matches = match_audience_category(keyword)
    
    result = {
        "genero": None, "edad": None, "relacion": None,
        "ocasion": None, "estilo": None, "precio": None
    }
    
    for match in matches:
        tipo = match["tipo"]
        if tipo in result and result[tipo] is None:
            result[tipo] = match["category_name"]
    
    return result

def get_primary_audience(keyword: str) -> Optional[str]:
    """Retorna la categoría de audiencia principal"""
    matches = match_audience_category(keyword)
    return matches[0]["category_name"] if matches else None

def get_audience_emoji(keyword: str) -> str:
    """Retorna el emoji de la audiencia principal"""
    matches = match_audience_category(keyword)
    return matches[0]["emoji"] if matches else "🎁"

# ============== FUNCIONES DE EMBEDDINGS ==============

@st.cache_resource
def load_sentence_transformer(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    """Carga el modelo de Sentence Transformers (cacheado)"""
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        return SentenceTransformer(model_name)
    return None

def create_embeddings_tfidf(keywords: List[str]) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Crea embeddings usando TF-IDF mejorado"""
    processed = [preprocess_keyword_advanced(kw) for kw in keywords]
    
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=2000,
        min_df=1,
        max_df=0.95,
        sublinear_tf=True
    )
    embeddings = vectorizer.fit_transform(processed)
    return embeddings.toarray(), vectorizer

def create_embeddings_sentence_transformer(
    keywords: List[str], 
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    batch_size: int = 64
) -> np.ndarray:
    """Crea embeddings usando Sentence Transformers"""
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        st.warning("⚠️ Sentence Transformers no disponible. Usando TF-IDF.")
        embeddings, _ = create_embeddings_tfidf(keywords)
        return embeddings
    
    model = load_sentence_transformer(model_name)
    
    enhanced_keywords = [f"regalo de navidad: {kw}" for kw in keywords]
    
    embeddings = model.encode(
        enhanced_keywords,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings

def create_embeddings_hybrid(
    keywords: List[str],
    tfidf_weight: float = 0.3,
    semantic_weight: float = 0.7
) -> np.ndarray:
    """Crea embeddings híbridos TF-IDF + Sentence Transformers"""
    tfidf_emb, _ = create_embeddings_tfidf(keywords)
    tfidf_norm = tfidf_emb / (np.linalg.norm(tfidf_emb, axis=1, keepdims=True) + 1e-8)
    
    if SENTENCE_TRANSFORMERS_AVAILABLE:
        st_emb = create_embeddings_sentence_transformer(keywords)
        
        if tfidf_norm.shape[1] > st_emb.shape[1]:
            pca = PCA(n_components=st_emb.shape[1])
            tfidf_reduced = pca.fit_transform(tfidf_norm)
        else:
            tfidf_reduced = tfidf_norm
            if tfidf_reduced.shape[1] < st_emb.shape[1]:
                padding = np.zeros((tfidf_reduced.shape[0], st_emb.shape[1] - tfidf_reduced.shape[1]))
                tfidf_reduced = np.hstack([tfidf_reduced, padding])
        
        embeddings = tfidf_reduced * tfidf_weight + st_emb * semantic_weight
    else:
        embeddings = tfidf_norm
    
    return embeddings

# ============== FUNCIONES DE CLUSTERING ==============

def cluster_keywords_kmeans(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agrupa keywords usando K-Means"""
    n_clusters = min(n_clusters, len(embeddings))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    return kmeans.fit_predict(embeddings)

def cluster_keywords_hierarchical(embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
    """Agrupa keywords usando clustering jerárquico"""
    n_clusters = min(n_clusters, len(embeddings))
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='cosine',
        linkage='average'
    )
    return clustering.fit_predict(embeddings)

def cluster_keywords_hdbscan(
    embeddings: np.ndarray, 
    min_cluster_size: int = 5,
    min_samples: int = 3
) -> np.ndarray:
    """Agrupa keywords usando HDBSCAN"""
    if not HDBSCAN_AVAILABLE:
        st.warning("⚠️ HDBSCAN no disponible. Usando K-Means.")
        n_clusters = max(5, len(embeddings) // 20)
        return cluster_keywords_kmeans(embeddings, n_clusters)
    
    if embeddings.shape[1] > 50:
        if UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=min(50, embeddings.shape[1]), metric='cosine', random_state=42)
            embeddings_reduced = reducer.fit_transform(embeddings)
        else:
            pca = PCA(n_components=min(50, embeddings.shape[1]))
            embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean',
        cluster_selection_method='eom'
    )
    
    return clusterer.fit_predict(embeddings_reduced)

def calculate_cluster_coherence(embeddings: np.ndarray, clusters: np.ndarray) -> Dict[int, float]:
    """Calcula la coherencia de cada cluster"""
    coherences = {}
    
    for cluster_id in np.unique(clusters):
        mask = clusters == cluster_id
        cluster_embeddings = embeddings[mask]
        
        if len(cluster_embeddings) > 1:
            similarities = cosine_similarity(cluster_embeddings)
            np.fill_diagonal(similarities, 0)
            coherence = similarities.sum() / (len(cluster_embeddings) * (len(cluster_embeddings) - 1))
            coherences[cluster_id] = coherence
        else:
            coherences[cluster_id] = 1.0
    
    return coherences

def suggest_url_for_cluster(cluster_keywords: List[str], cluster_volumes: List[int]) -> str:
    """Sugiere una URL para el cluster"""
    if cluster_volumes:
        max_idx = cluster_volumes.index(max(cluster_volumes))
        main_kw = cluster_keywords[max_idx]
    else:
        main_kw = cluster_keywords[0] if cluster_keywords else "regalo"
    
    url_slug = clean_keyword(main_kw)
    url_slug = re.sub(r'\s+', '-', url_slug)
    url_slug = re.sub(r'-+', '-', url_slug).strip('-')
    
    return f"/regalos-navidad/{url_slug}/"

# ============== FUNCIONES DE AI ==============

def cluster_with_ai_classification(
    keywords: List[str],
    volumes: List[int],
    api_key: str,
    provider: str = "claude",
    batch_size: int = 80
) -> Tuple[np.ndarray, Dict[int, str]]:
    """Usa AI para clasificar semánticamente las keywords"""
    
    categories = [
        "Regalos tecnología móvil (smartphones, tablets, smartwatches)",
        "Regalos gaming y videojuegos",
        "Regalos informática (portátiles, periféricos)",
        "Regalos hogar inteligente y electrodomésticos",
        "Regalos imagen y sonido (TV, audio)",
        "Regalos cocina",
        "Regalos belleza y cuidado personal",
        "Regalos deportes y fitness",
        "Regalos para niños y juguetes",
        "Regalos originales y gadgets",
        "Regalos económicos (amigo invisible)",
        "Regalos para hombre",
        "Regalos para mujer",
        "Regalos familiares",
        "Navidad general",
        "Otros"
    ]
    
    clusters = np.zeros(len(keywords), dtype=int)
    cluster_names = {i: cat.split("(")[0].strip() for i, cat in enumerate(categories)}
    
    for batch_start in range(0, len(keywords), batch_size):
        batch_end = min(batch_start + batch_size, len(keywords))
        batch_kws = keywords[batch_start:batch_end]
        batch_vols = volumes[batch_start:batch_end]
        
        kw_list = "\n".join([f"{i+1}. {kw} (vol: {vol})" 
                           for i, (kw, vol) in enumerate(zip(batch_kws, batch_vols))])
        
        categories_list = "\n".join([f"{i}. {cat}" for i, cat in enumerate(categories)])
        
        prompt = f"""Clasifica cada keyword en UNA categoría.

CATEGORÍAS:
{categories_list}

KEYWORDS:
{kw_list}

Responde SOLO con un JSON array de números de categoría (0-{len(categories)-1}).
Ejemplo: [0, 3, 5, 1, 2]"""

        try:
            if provider == "claude" and ANTHROPIC_AVAILABLE:
                client = anthropic.Anthropic(api_key=api_key)
                message = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = message.content[0].text
            elif OPENAI_AVAILABLE:
                client = openai.OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                response_text = response.choices[0].message.content
            else:
                raise Exception("No AI provider available")
            
            response_text = re.sub(r'^```json?\s*', '', response_text.strip())
            response_text = re.sub(r'\s*```$', '', response_text)
            
            batch_clusters = json.loads(response_text)
            
            for i, cluster_id in enumerate(batch_clusters):
                if batch_start + i < len(clusters):
                    clusters[batch_start + i] = int(cluster_id) if cluster_id < len(categories) else len(categories) - 1
                    
        except Exception as e:
            st.warning(f"Error en batch {batch_start}: {str(e)}")
            for i in range(batch_start, batch_end):
                clusters[i] = len(categories) - 1
    
    return clusters, cluster_names

def get_cluster_analysis_claude(keywords: List[str], volumes: List[int], api_key: str) -> Dict:
    """Usa Claude para analizar un cluster"""
    if not ANTHROPIC_AVAILABLE:
        return {"error": "Anthropic no disponible"}
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        
        kw_data = "\n".join([f"- {kw} (vol: {vol})" for kw, vol in zip(keywords[:20], volumes[:20])])
        
        prompt = f"""Analiza este grupo de keywords de búsqueda relacionadas con regalos de Navidad para PcComponentes.

Keywords del cluster:
{kw_data}

Responde en JSON:
{{
    "nombre_cluster": "nombre descriptivo corto",
    "url_sugerida": "/regalos-navidad/slug/",
    "h1_sugerido": "H1 para la página",
    "meta_description": "Meta description SEO",
    "intent_principal": "transaccional/informacional/mixto",
    "productos_recomendados": ["producto1", "producto2", "producto3"],
    "query_fanout": ["query relacionada 1", "query relacionada 2", "query relacionada 3"]
}}"""
        
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text
        response_text = re.sub(r'^```json?\s*', '', response_text.strip())
        response_text = re.sub(r'\s*```$', '', response_text)
        
        return json.loads(response_text)
        
    except Exception as e:
        return {"error": str(e)}

def get_cluster_analysis_openai(keywords: List[str], volumes: List[int], api_key: str) -> Dict:
    """Usa GPT para analizar un cluster"""
    if not OPENAI_AVAILABLE:
        return {"error": "OpenAI no disponible"}
    
    try:
        client = openai.OpenAI(api_key=api_key)
        
        kw_data = "\n".join([f"- {kw} (vol: {vol})" for kw, vol in zip(keywords[:20], volumes[:20])])
        
        prompt = f"""Analiza este grupo de keywords para regalos de Navidad (tienda tech).

Keywords:
{kw_data}

Responde en JSON:
{{
    "nombre_cluster": "nombre corto",
    "url_sugerida": "/regalos-navidad/slug/",
    "h1_sugerido": "H1 SEO",
    "meta_description": "Meta description",
    "intent_principal": "transaccional/informacional/mixto",
    "productos_recomendados": ["prod1", "prod2", "prod3"],
    "query_fanout": ["query1", "query2", "query3"]
}}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content
        response_text = re.sub(r'^```json?\s*', '', response_text.strip())
        response_text = re.sub(r'\s*```$', '', response_text)
        
        return json.loads(response_text)
        
    except Exception as e:
        return {"error": str(e)}

# ============== APLICACIÓN PRINCIPAL ==============

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎄 Keyword Semantic Clustering Tool</h1>
        <p>Agrupa semánticamente keywords de Google Keyword Planner para tu campaña de Navidad</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features disponibles
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.success("✅ Sentence Transformers") if SENTENCE_TRANSFORMERS_AVAILABLE else st.warning("❌ Sentence Transformers")
    with col2:
        st.success("✅ HDBSCAN") if HDBSCAN_AVAILABLE else st.warning("❌ HDBSCAN")
    with col3:
        st.success("✅ Claude API") if ANTHROPIC_AVAILABLE else st.warning("❌ Claude API")
    with col4:
        st.success("✅ OpenAI API") if OPENAI_AVAILABLE else st.warning("❌ OpenAI API")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuración")
        
        st.markdown("### 📁 Datos")
        uploaded_file = st.file_uploader("CSV de Google Keyword Planner", type=['csv'])
        
        st.markdown("---")
        st.markdown("### 🧠 Embeddings")
        
        embedding_options = ["TF-IDF (rápido)"]
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            embedding_options.extend(["Sentence Transformers (semántico)", "Híbrido (TF-IDF + Semántico)"])
        embedding_options.append("Clasificación con AI")
        
        embedding_method = st.selectbox(
            "Método",
            embedding_options,
            index=1 if SENTENCE_TRANSFORMERS_AVAILABLE else 0
        )
        
        st.markdown("### 🎯 Clustering")
        
        clustering_options = ["K-Means", "Jerárquico"]
        if HDBSCAN_AVAILABLE:
            clustering_options.append("HDBSCAN (auto-clusters)")
        
        clustering_method = st.selectbox("Algoritmo", clustering_options)
        
        if "HDBSCAN" in clustering_method:
            min_cluster_size = st.slider("Tamaño mín. cluster", 3, 20, 5)
            n_clusters = None
        else:
            n_clusters = st.slider("Número de clusters", 5, 50, 15)
            min_cluster_size = 5
        
        min_volume = st.number_input("Volumen mínimo", 0, 10000, 50)
        
        st.markdown("---")
        st.markdown("### 📅 Meses")
        include_nov = st.checkbox("Noviembre", True)
        include_dec = st.checkbox("Diciembre", True)
        include_jan = st.checkbox("Enero", True)
        
        st.markdown("---")
        st.markdown("### 📦 Modo de Agrupación")
        
        clustering_mode = st.selectbox(
            "Modo",
            ["Solo semántico", "Guiado por productos", "Guiado por audiencia",
             "Híbrido (Semántico + Productos)", "Híbrido Completo"],
            index=4
        )
        
        filter_by_product = st.checkbox("Solo con match de producto", False)
        
        st.markdown("---")
        st.markdown("### 🔑 APIs")
        
        api_option = st.selectbox("Proveedor AI", ["Sin AI", "Claude (Anthropic)", "GPT (OpenAI)"])
        api_key = st.text_input("API Key", type="password") if api_option != "Sin AI" else ""
    
    # Main content
    if uploaded_file is not None:
        # Cargar datos
        try:
            df = pd.read_csv(uploaded_file, skiprows=2)
        except:
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Error al cargar CSV: {e}")
                return
        
        # Detectar columnas
        keyword_col = None
        for col in df.columns:
            if 'keyword' in col.lower() or 'palabra' in col.lower():
                keyword_col = col
                break
        if keyword_col is None:
            keyword_col = df.columns[0]
        
        df = df.rename(columns={keyword_col: 'Keyword'})
        
        # Detectar meses
        month_columns = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'nov' in col_lower:
                month_columns['nov'] = col
            elif 'dec' in col_lower or 'dic' in col_lower:
                month_columns['dec'] = col
            elif 'jan' in col_lower or 'ene' in col_lower:
                month_columns['jan'] = col
            elif 'avg' in col_lower or 'promedio' in col_lower or 'monthly' in col_lower:
                month_columns['avg'] = col
        
        def safe_convert(val):
            if pd.isna(val):
                return 0
            val_str = str(val).replace(',', '').replace('.', '')
            try:
                return int(val_str)
            except:
                return 0
        
        df['volumen_navidad'] = 0
        if include_nov and 'nov' in month_columns:
            df['volumen_navidad'] += df[month_columns['nov']].apply(safe_convert)
        if include_dec and 'dec' in month_columns:
            df['volumen_navidad'] += df[month_columns['dec']].apply(safe_convert)
        if include_jan and 'jan' in month_columns:
            df['volumen_navidad'] += df[month_columns['jan']].apply(safe_convert)
        
        if df['volumen_navidad'].sum() == 0 and 'avg' in month_columns:
            df['volumen_navidad'] = df[month_columns['avg']].apply(safe_convert)
        
        df_filtered = df[df['volumen_navidad'] >= min_volume].copy()
        
        if len(df_filtered) == 0:
            st.warning("No hay keywords con el volumen mínimo especificado")
            return
        
        # Enriquecer datos
        with st.spinner("🔄 Procesando keywords..."):
            df_filtered['keyword_clean'] = df_filtered['Keyword'].apply(clean_keyword)
            df_filtered['intent'] = df_filtered['Keyword'].apply(classify_intent)
            df_filtered['destinatario'] = df_filtered['Keyword'].apply(extract_gift_recipient)
            df_filtered['rango_precio'] = df_filtered['Keyword'].apply(extract_price_range)
            
            df_filtered['product_matches'] = df_filtered['Keyword'].apply(match_product_family)
            df_filtered['best_product_family'] = df_filtered['Keyword'].apply(get_best_product_family)
            df_filtered['product_match_score'] = df_filtered['Keyword'].apply(get_product_match_score)
            df_filtered['has_product_match'] = df_filtered['Keyword'].apply(keyword_has_product_match)
            df_filtered['familia_producto'] = df_filtered['product_matches'].apply(
                lambda x: x[0]['family_name'] if x else "Sin match"
            )
            
            df_filtered['audience_matches'] = df_filtered['Keyword'].apply(match_audience_category)
            df_filtered['primary_audience'] = df_filtered['Keyword'].apply(get_primary_audience)
            df_filtered['audience_emoji'] = df_filtered['Keyword'].apply(get_audience_emoji)
            
            audience_by_type = df_filtered['Keyword'].apply(get_audience_by_type)
            df_filtered['audiencia_genero'] = audience_by_type.apply(lambda x: x.get('genero'))
            df_filtered['audiencia_edad'] = audience_by_type.apply(lambda x: x.get('edad'))
            df_filtered['audiencia_relacion'] = audience_by_type.apply(lambda x: x.get('relacion'))
            df_filtered['audiencia_ocasion'] = audience_by_type.apply(lambda x: x.get('ocasion'))
            df_filtered['has_audience_match'] = df_filtered['primary_audience'].notna()
        
        if filter_by_product:
            df_filtered = df_filtered[df_filtered['has_product_match']].copy()
            if len(df_filtered) == 0:
                st.warning("No hay keywords con match de producto")
                return
        
        # Crear clusters
        with st.spinner("🧠 Creando clusters semánticos..."):
            keywords_list = df_filtered['keyword_clean'].tolist()
            original_keywords = df_filtered['Keyword'].tolist()
            volumes_list = df_filtered['volumen_navidad'].tolist()
            
            if "Clasificación con AI" in embedding_method:
                if not api_key:
                    st.error("❌ Se requiere API key para clasificación con AI")
                    return
                
                st.info("🤖 Clasificando con AI...")
                provider = "claude" if "Claude" in api_option else "openai"
                clusters, cluster_names = cluster_with_ai_classification(
                    original_keywords, volumes_list, api_key, provider
                )
                embeddings, _ = create_embeddings_tfidf(keywords_list)
                
            else:
                if "Sentence Transformers" in embedding_method:
                    st.info("🧠 Generando embeddings semánticos...")
                    embeddings = create_embeddings_sentence_transformer(keywords_list)
                elif "Híbrido" in embedding_method and "TF-IDF" in embedding_method:
                    st.info("🔀 Generando embeddings híbridos...")
                    embeddings = create_embeddings_hybrid(keywords_list)
                else:
                    st.info("📝 Generando embeddings TF-IDF...")
                    embeddings, _ = create_embeddings_tfidf(keywords_list)
                
                if clustering_mode == "Guiado por productos":
                    family_to_cluster = {fam: i for i, fam in enumerate(PRODUCTOS_PCCOMPONENTES.keys())}
                    family_to_cluster[None] = len(family_to_cluster)
                    
                    clusters = df_filtered['best_product_family'].map(
                        lambda x: family_to_cluster.get(x, family_to_cluster[None])
                    ).values
                    
                    cluster_names = {}
                    for fam_id, clust_id in family_to_cluster.items():
                        if fam_id and fam_id in PRODUCTOS_PCCOMPONENTES:
                            fam_data = PRODUCTOS_PCCOMPONENTES[fam_id]
                            cluster_names[clust_id] = f"{fam_data['emoji']} {fam_data['nombre']}"
                        else:
                            cluster_names[clust_id] = "🔍 Sin match"
                
                elif clustering_mode == "Guiado por audiencia":
                    aud_to_cluster = {aud: i for i, aud in enumerate(CATEGORIAS_AUDIENCIA.keys())}
                    aud_to_cluster[None] = len(aud_to_cluster)
                    
                    def get_aud_id(primary):
                        if primary is None:
                            return None
                        for cat_id, cat_data in CATEGORIAS_AUDIENCIA.items():
                            if cat_data["nombre"] == primary:
                                return cat_id
                        return None
                    
                    df_filtered['audience_id'] = df_filtered['primary_audience'].apply(get_aud_id)
                    clusters = df_filtered['audience_id'].map(
                        lambda x: aud_to_cluster.get(x, aud_to_cluster[None])
                    ).values
                    
                    cluster_names = {}
                    for aud_id, clust_id in aud_to_cluster.items():
                        if aud_id and aud_id in CATEGORIAS_AUDIENCIA:
                            aud_data = CATEGORIAS_AUDIENCIA[aud_id]
                            cluster_names[clust_id] = f"{aud_data['emoji']} {aud_data['nombre']}"
                        else:
                            cluster_names[clust_id] = "🔍 Sin audiencia"
                
                elif "Híbrido" in clustering_mode:
                    family_ids = list(PRODUCTOS_PCCOMPONENTES.keys())
                    product_features = np.zeros((len(df_filtered), len(family_ids) + 1))
                    
                    for i, (_, row) in enumerate(df_filtered.iterrows()):
                        if row['best_product_family'] and row['best_product_family'] in family_ids:
                            idx = family_ids.index(row['best_product_family'])
                            product_features[i, idx] = row['product_match_score'] * 2
                        else:
                            product_features[i, -1] = 0.3
                    
                    if "Completo" in clustering_mode:
                        audience_ids = list(CATEGORIAS_AUDIENCIA.keys())
                        audience_features = np.zeros((len(df_filtered), len(audience_ids) + 1))
                        
                        for i, (_, row) in enumerate(df_filtered.iterrows()):
                            matches = row['audience_matches']
                            if matches:
                                for match in matches[:3]:
                                    if match['category_id'] in audience_ids:
                                        idx = audience_ids.index(match['category_id'])
                                        audience_features[i, idx] = match['score'] / 5
                            else:
                                audience_features[i, -1] = 0.3
                        
                        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
                        embeddings = np.hstack([emb_norm * 0.5, product_features * 0.3, audience_features * 0.2])
                    else:
                        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
                        embeddings = np.hstack([emb_norm * 0.6, product_features * 0.4])
                    
                    if "HDBSCAN" in clustering_method:
                        clusters = cluster_keywords_hdbscan(embeddings, min_cluster_size)
                    elif "Jerárquico" in clustering_method:
                        clusters = cluster_keywords_hierarchical(embeddings, n_clusters)
                    else:
                        clusters = cluster_keywords_kmeans(embeddings, n_clusters)
                    
                    cluster_names = None
                
                else:
                    if "HDBSCAN" in clustering_method:
                        clusters = cluster_keywords_hdbscan(embeddings, min_cluster_size)
                    elif "Jerárquico" in clustering_method:
                        clusters = cluster_keywords_hierarchical(embeddings, n_clusters)
                    else:
                        clusters = cluster_keywords_kmeans(embeddings, n_clusters)
                    
                    cluster_names = None
            
            df_filtered['cluster_id'] = clusters
            coherences = calculate_cluster_coherence(embeddings, clusters)
        
        # Nombrar clusters
        if cluster_names is None:
            cluster_names = {}
            for cluster_id in df_filtered['cluster_id'].unique():
                cluster_kws = df_filtered[df_filtered['cluster_id'] == cluster_id]
                
                family_counts = cluster_kws['familia_producto'].value_counts()
                top_family = family_counts.index[0] if len(family_counts) > 0 else None
                
                audience_counts = cluster_kws['primary_audience'].value_counts()
                top_audience = audience_counts.index[0] if len(audience_counts) > 0 and pd.notna(audience_counts.index[0]) else None
                
                top_kw = cluster_kws.nlargest(1, 'volumen_navidad')['Keyword'].values[0]
                
                emoji = "📦"
                prefix = ""
                
                if top_family and top_family != "Sin match":
                    for fam_id, fam_data in PRODUCTOS_PCCOMPONENTES.items():
                        if fam_data['nombre'] == top_family:
                            emoji = fam_data['emoji']
                            prefix = top_family
                            break
                elif top_audience:
                    for aud_id, aud_data in CATEGORIAS_AUDIENCIA.items():
                        if aud_data['nombre'] == top_audience:
                            emoji = aud_data['emoji']
                            prefix = top_audience
                            break
                
                if prefix:
                    name = f"{emoji} {prefix}: {top_kw[:25]}"
                else:
                    name = f"C{cluster_id}: {top_kw[:35]}"
                
                cluster_names[cluster_id] = name
        
        df_filtered['cluster_name'] = df_filtered['cluster_id'].map(cluster_names)
        df_filtered['coherence'] = df_filtered['cluster_id'].map(coherences)
        
        # ========== RESULTADOS ==========
        
        st.markdown("---")
        st.markdown("## 📊 Resultados")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Keywords", f"{len(df_filtered):,}")
        with col2:
            st.metric("Clusters", f"{len(df_filtered['cluster_id'].unique())}")
        with col3:
            st.metric("Volumen Total", f"{df_filtered['volumen_navidad'].sum():,.0f}")
        with col4:
            avg_coherence = np.mean(list(coherences.values()))
            st.metric("Coherencia Media", f"{avg_coherence:.2f}")
        
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Con Producto", f"{df_filtered['has_product_match'].mean() * 100:.1f}%")
        with col6:
            st.metric("Con Audiencia", f"{df_filtered['has_audience_match'].mean() * 100:.1f}%")
        with col7:
            st.metric("Familias", df_filtered[df_filtered['has_product_match']]['familia_producto'].nunique())
        with col8:
            vol_match = df_filtered[df_filtered['has_product_match']]['volumen_navidad'].sum()
            pct_vol = (vol_match / df_filtered['volumen_navidad'].sum() * 100) if df_filtered['volumen_navidad'].sum() > 0 else 0
            st.metric("Vol. con Match", f"{pct_vol:.1f}%")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Visualización", "📦 Productos", "👥 Audiencias", 
            "📋 Clusters", "🎯 URLs", "📊 Datos"
        ])
        
        with tab1:
            st.markdown("### 🗺️ Visualización de Clusters")
            
            cluster_summary = df_filtered.groupby('cluster_name').agg({
                'volumen_navidad': 'sum',
                'Keyword': 'count',
                'coherence': 'first'
            }).reset_index()
            cluster_summary.columns = ['Cluster', 'Volumen', 'Keywords', 'Coherencia']
            
            fig_treemap = px.treemap(
                cluster_summary, path=['Cluster'], values='Volumen',
                color='Coherencia', color_continuous_scale='RdYlGn',
                title='Distribución de Volumen por Cluster'
            )
            fig_treemap.update_layout(height=500)
            st.plotly_chart(fig_treemap, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if embeddings.shape[1] > 2:
                    pca = PCA(n_components=2)
                    coords = pca.fit_transform(embeddings)
                else:
                    coords = embeddings
                
                df_filtered['x'] = coords[:, 0]
                df_filtered['y'] = coords[:, 1]
                
                fig_scatter = px.scatter(
                    df_filtered, x='x', y='y', color='cluster_name',
                    size='volumen_navidad', hover_data=['Keyword', 'volumen_navidad'],
                    title='Clusters en Espacio 2D (PCA)'
                )
                fig_scatter.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            with col2:
                top_clusters = cluster_summary.nlargest(15, 'Volumen')
                
                fig_bar = go.Figure(go.Bar(
                    y=top_clusters['Cluster'], x=top_clusters['Volumen'],
                    orientation='h', marker_color='#c41e3a'
                ))
                fig_bar.update_layout(
                    title='Top 15 Clusters por Volumen', height=400,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab2:
            st.markdown("### 📦 Análisis por Producto")
            
            product_data = df_filtered[df_filtered['has_product_match']].groupby('familia_producto').agg({
                'volumen_navidad': 'sum', 'Keyword': 'count', 'product_match_score': 'mean'
            }).reset_index()
            product_data.columns = ['Familia', 'Volumen', 'Keywords', 'Score Medio']
            product_data = product_data.sort_values('Volumen', ascending=False)
            
            fig_products = px.bar(
                product_data, x='Familia', y='Volumen', color='Score Medio',
                color_continuous_scale='Greens', title='Volumen por Familia de Producto'
            )
            st.plotly_chart(fig_products, use_container_width=True)
            
            st.dataframe(
                product_data.style.format({'Volumen': '{:,.0f}', 'Keywords': '{:.0f}', 'Score Medio': '{:.2f}'}),
                use_container_width=True
            )
        
        with tab3:
            st.markdown("### 👥 Análisis por Audiencia")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Por Género")
                genero_data = df_filtered[df_filtered['audiencia_genero'].notna()].groupby('audiencia_genero').agg({
                    'volumen_navidad': 'sum', 'Keyword': 'count'
                }).reset_index()
                genero_data.columns = ['Género', 'Volumen', 'Keywords']
                
                if len(genero_data) > 0:
                    fig_genero = px.pie(genero_data, values='Volumen', names='Género', title='Por Género')
                    st.plotly_chart(fig_genero, use_container_width=True)
            
            with col2:
                st.markdown("#### Por Edad")
                edad_data = df_filtered[df_filtered['audiencia_edad'].notna()].groupby('audiencia_edad').agg({
                    'volumen_navidad': 'sum', 'Keyword': 'count'
                }).reset_index()
                edad_data.columns = ['Edad', 'Volumen', 'Keywords']
                
                if len(edad_data) > 0:
                    fig_edad = px.bar(edad_data, x='Edad', y='Volumen', title='Por Grupo de Edad')
                    st.plotly_chart(fig_edad, use_container_width=True)
            
            st.markdown("#### Por Relación")
            relacion_data = df_filtered[df_filtered['audiencia_relacion'].notna()].groupby('audiencia_relacion').agg({
                'volumen_navidad': 'sum', 'Keyword': 'count'
            }).reset_index()
            relacion_data.columns = ['Relación', 'Volumen', 'Keywords']
            relacion_data = relacion_data.sort_values('Volumen', ascending=False)
            
            if len(relacion_data) > 0:
                st.dataframe(
                    relacion_data.style.format({'Volumen': '{:,.0f}', 'Keywords': '{:.0f}'}),
                    use_container_width=True
                )
        
        with tab4:
            st.markdown("### 📋 Análisis por Cluster")
            
            cluster_options = sorted(df_filtered['cluster_name'].unique())
            selected_cluster = st.selectbox("Selecciona cluster:", cluster_options)
            
            cluster_data = df_filtered[df_filtered['cluster_name'] == selected_cluster]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Keywords", len(cluster_data))
            with col2:
                st.metric("Volumen", f"{cluster_data['volumen_navidad'].sum():,.0f}")
            with col3:
                st.metric("Coherencia", f"{cluster_data['coherence'].iloc[0]:.2f}")
            
            st.markdown("**Top Keywords:**")
            top_kws = cluster_data.nlargest(20, 'volumen_navidad')[
                ['Keyword', 'volumen_navidad', 'intent', 'familia_producto', 'primary_audience']
            ]
            st.dataframe(top_kws, use_container_width=True)
            
            if api_key and len(cluster_data) > 0:
                if st.button("🤖 Analizar con AI", key="analyze_cluster"):
                    with st.spinner("Analizando..."):
                        kws = cluster_data['Keyword'].tolist()[:30]
                        vols = cluster_data['volumen_navidad'].tolist()[:30]
                        
                        if "Claude" in api_option:
                            analysis = get_cluster_analysis_claude(kws, vols, api_key)
                        else:
                            analysis = get_cluster_analysis_openai(kws, vols, api_key)
                        
                        if "error" not in analysis:
                            st.success("✅ Análisis completado")
                            st.markdown(f"**Nombre:** {analysis.get('nombre_cluster', 'N/A')}")
                            st.markdown(f"**URL:** `{analysis.get('url_sugerida', 'N/A')}`")
                            st.markdown(f"**H1:** {analysis.get('h1_sugerido', 'N/A')}")
                            st.markdown(f"**Meta:** {analysis.get('meta_description', 'N/A')}")
                            
                            if 'productos_recomendados' in analysis:
                                st.markdown("**Productos:**")
                                for prod in analysis['productos_recomendados']:
                                    st.markdown(f"- {prod}")
                            
                            if 'query_fanout' in analysis:
                                st.markdown("**Query Fan-Out:**")
                                for q in analysis['query_fanout']:
                                    st.markdown(f"- {q}")
                        else:
                            st.error(f"Error: {analysis['error']}")
        
        with tab5:
            st.markdown("### 🎯 URLs Recomendadas")
            
            url_recommendations = []
            
            for cluster_id in df_filtered['cluster_id'].unique():
                cluster_kws = df_filtered[df_filtered['cluster_id'] == cluster_id]
                
                kws_list = cluster_kws['Keyword'].tolist()
                vols_list = cluster_kws['volumen_navidad'].tolist()
                
                total_vol = sum(vols_list)
                avg_vol = np.mean(vols_list)
                n_kws = len(kws_list)
                coherence = cluster_kws['coherence'].iloc[0]
                
                score = (total_vol * 0.4) + (avg_vol * 0.3) + (n_kws * 50 * 0.2) + (coherence * 1000 * 0.1)
                
                url_recommendations.append({
                    'Cluster': cluster_names.get(cluster_id, f"Cluster {cluster_id}"),
                    'URL Sugerida': suggest_url_for_cluster(kws_list, vols_list),
                    'Volumen Total': total_vol,
                    'Keywords': n_kws,
                    'Coherencia': coherence,
                    'Prioridad': score
                })
            
            url_df = pd.DataFrame(url_recommendations).sort_values('Prioridad', ascending=False)
            
            st.dataframe(
                url_df.style.format({
                    'Volumen Total': '{:,.0f}', 'Keywords': '{:.0f}',
                    'Coherencia': '{:.2f}', 'Prioridad': '{:,.0f}'
                }).background_gradient(subset=['Prioridad'], cmap='Greens'),
                use_container_width=True
            )
            
            csv = url_df.to_csv(index=False)
            st.download_button("📥 Descargar URLs", csv, "url_recommendations.csv", "text/csv")
        
        with tab6:
            st.markdown("### 📊 Datos Completos")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_cluster = st.multiselect("Filtrar por cluster", sorted(df_filtered['cluster_name'].unique()))
            with col2:
                filter_family = st.multiselect("Filtrar por familia", sorted(df_filtered['familia_producto'].unique()))
            with col3:
                filter_intent = st.multiselect("Filtrar por intent", sorted(df_filtered['intent'].unique()))
            
            df_display = df_filtered.copy()
            
            if filter_cluster:
                df_display = df_display[df_display['cluster_name'].isin(filter_cluster)]
            if filter_family:
                df_display = df_display[df_display['familia_producto'].isin(filter_family)]
            if filter_intent:
                df_display = df_display[df_display['intent'].isin(filter_intent)]
            
            columns_to_show = [
                'Keyword', 'volumen_navidad', 'cluster_name', 'intent',
                'familia_producto', 'primary_audience', 'destinatario', 'rango_precio'
            ]
            columns_to_show = [c for c in columns_to_show if c in df_display.columns]
            
            st.dataframe(
                df_display[columns_to_show].sort_values('volumen_navidad', ascending=False),
                use_container_width=True, height=600
            )
            
            csv_full = df_display.to_csv(index=False)
            st.download_button("📥 Descargar Datos", csv_full, "keyword_clusters_full.csv", "text/csv")
    
    else:
        st.markdown("""
        ## 🚀 Cómo usar esta herramienta
        
        1. **Exporta keywords de Google Keyword Planner** en formato CSV
        2. **Sube el archivo** usando el selector en la barra lateral
        3. **Configura los parámetros** (Sentence Transformers recomendado)
        4. **Analiza los resultados** en las diferentes pestañas
        5. **Exporta las URLs recomendadas** para implementar
        
        ### 📊 Características
        
        - **Sentence Transformers**: Embeddings semánticos multilingües de alta calidad
        - **HDBSCAN**: Detección automática de clusters sin especificar número
        - **Clasificación AI**: Usa Claude o GPT para clasificar keywords
        - **Análisis de audiencias**: Segmentación por género, edad, relación
        - **Match de productos**: Alineación con catálogo de PcComponentes
        """)

if __name__ == "__main__":
    main()
