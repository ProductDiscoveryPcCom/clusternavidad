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
# Estructura: familia -> {display_name, keywords para matching, emoji}
PRODUCTOS_PCCOMPONENTES = {
    "tecnologia_acompana": {
        "nombre": "Tecnología que te acompaña",
        "emoji": "📱",
        "productos": ["Smartphones", "Tablets", "EBooks", "Auriculares bluetooth", "Patinetes", "Smartwatches"],
        "keywords_match": [
            "smartphone", "telefono", "movil", "iphone", "samsung", "xiaomi", "android",
            "tablet", "ipad", "ebook", "kindle", "libro electronico",
            "auricular", "cascos", "bluetooth", "airpods", "buds",
            "patinete", "scooter", "electrico",
            "smartwatch", "reloj inteligente", "apple watch", "garmin", "fitbit"
        ]
    },
    "informatica_todos": {
        "nombre": "Informática para todos",
        "emoji": "💻",
        "productos": ["PC no gaming", "Portátiles no gaming", "Periféricos no gaming", "Discos duros externos"],
        "keywords_match": [
            "ordenador", "portatil", "laptop", "pc", "sobremesa", "all in one",
            "teclado", "raton", "monitor", "pantalla", "webcam", "impresora",
            "disco duro", "ssd", "usb", "pendrive", "memoria"
        ]
    },
    "esenciales_hogar": {
        "nombre": "Esenciales en el hogar",
        "emoji": "🏠",
        "productos": ["Aspiradoras", "Robots aspirador", "Hidrolimpiadoras", "Limpiadoras de vapor", "Planchado", "Tratamiento del aire"],
        "keywords_match": [
            "aspirador", "aspiradora", "robot aspirador", "roomba", "conga", "roborock",
            "hidrolimpiadora", "karcher", "limpiadora", "vapor", "fregona",
            "plancha", "centro planchado",
            "aire", "purificador", "humidificador", "deshumidificador", "ventilador", "climatizador"
        ]
    },
    "cine_series": {
        "nombre": "Fans de cine y series",
        "emoji": "📺",
        "productos": ["Televisores", "Proyectores", "Altavoces", "Altavoces TV y barras de sonido", "Auriculares premium"],
        "keywords_match": [
            "television", "televisor", "tv", "smart tv", "oled", "qled", "4k", "8k",
            "proyector", "home cinema", "cine en casa",
            "altavoz", "barra de sonido", "soundbar", "subwoofer", "home theater",
            "auriculares", "cascos", "hifi", "premium", "sony", "bose", "sennheiser"
        ]
    },
    "gamers": {
        "nombre": "Perfectos para gamers",
        "emoji": "🎮",
        "productos": ["Portátiles gaming", "PC gaming", "Mesas gaming", "Teclados gaming", "Sillas gaming", "Ratones gaming", "Auriculares gaming"],
        "keywords_match": [
            "gaming", "gamer", "rgb",
            "portatil gaming", "pc gaming", "torre gaming", "setup",
            "mesa gaming", "escritorio gaming",
            "teclado gaming", "mecanico", "teclado rgb",
            "silla gaming", "silla gamer",
            "raton gaming", "mouse gaming",
            "auriculares gaming", "cascos gaming", "headset"
        ]
    },
    "consolas": {
        "nombre": "Jugones de consola",
        "emoji": "🕹️",
        "productos": ["Mandos de juego", "Accesorios para consolas", "Consolas", "Juegos", "Simulación gaming", "Volantes"],
        "keywords_match": [
            "consola", "playstation", "ps5", "ps4", "xbox", "nintendo", "switch",
            "mando", "controller", "dualshock", "dualsense",
            "juego", "videojuego", "game",
            "volante", "simulador", "logitech", "thrustmaster",
            "vr", "realidad virtual", "psvr"
        ]
    },
    "amantes_cafe": {
        "nombre": "Amantes del café",
        "emoji": "☕",
        "productos": ["Cápsulas para cafeteras", "Cafeteras"],
        "keywords_match": [
            "cafe", "cafetera", "espresso", "expreso", "nespresso", "dolce gusto",
            "capsula", "capsulas", "molinillo", "barista",
            "delonghi", "krups", "philips", "superautomatica"
        ]
    },
    "chefs": {
        "nombre": "Chefs en potencia",
        "emoji": "👨‍🍳",
        "productos": ["Cocina portátil", "Freidoras", "Robots de cocina", "Batidoras", "Planchas, barbacoas y grills"],
        "keywords_match": [
            "cocina", "cocinero", "chef",
            "freidora", "freidora aire", "airfryer", "sin aceite",
            "robot cocina", "thermomix", "mambo", "mycook",
            "batidora", "amasadora", "kitchenaid", "procesador alimentos",
            "plancha", "barbacoa", "grill", "sandwichera", "tostadora"
        ]
    },
    "deportistas": {
        "nombre": "Deportistas y aventureros",
        "emoji": "🏃",
        "productos": ["Smartwatches", "Pulseras actividad", "Auriculares deportivos", "Bicicletas elípticas", "Bicicletas estáticas", "Cintas de correr"],
        "keywords_match": [
            "deporte", "deportista", "fitness", "running", "correr", "ciclismo",
            "smartwatch", "pulsera actividad", "garmin", "polar", "suunto",
            "auriculares deporte", "sport", "resistentes sudor",
            "bicicleta estatica", "eliptica", "spinning",
            "cinta correr", "treadmill", "gimnasio", "gym"
        ]
    },
    "belleza_cuidado": {
        "nombre": "Belleza y cuidado",
        "emoji": "💄",
        "productos": ["Afeitadoras", "Cortapelo", "Depiladoras", "Cepillos de dientes eléctricos", "Planchas de pelo", "Secadores de pelo", "Básculas de baño", "Cuidado corporal", "Cuidado facial"],
        "keywords_match": [
            "belleza", "cuidado", "personal",
            "afeitadora", "maquinilla", "braun", "philips", "recortadora", "barba",
            "cortapelo", "cortapelos",
            "depiladora", "depilacion", "laser", "ipl", "cera",
            "cepillo dientes", "oral b", "sonicare", "electrico dental",
            "plancha pelo", "ghd", "rizador", "tenacillas",
            "secador", "secador pelo", "dyson",
            "bascula", "peso", "composicion corporal",
            "masajeador", "spa"
        ]
    },
    "gadgets": {
        "nombre": "Gadgets y accesorios",
        "emoji": "🔌",
        "productos": ["Altavoces inteligentes", "Powerbanks"],
        "keywords_match": [
            "gadget", "accesorio", "tecnologia",
            "altavoz inteligente", "alexa", "echo", "google home", "homepod",
            "powerbank", "bateria externa", "cargador portatil", "carga rapida",
            "dron", "drone", "camara accion", "gopro"
        ]
    },
    "juguetes": {
        "nombre": "Juguetes y juegos",
        "emoji": "🧸",
        "productos": ["Juguetes", "Juguetes de imitación", "Juguetes educativos"],
        "keywords_match": [
            "juguete", "juego", "nino", "nina", "infantil", "bebe",
            "muneca", "peluche", "figura", "accion",
            "educativo", "stem", "aprendizaje", "didactico",
            "imitacion", "cocinita", "supermercado"
        ]
    },
    "lego": {
        "nombre": "LEGO",
        "emoji": "🧱",
        "productos": ["LEGO"],
        "keywords_match": [
            "lego", "construccion", "bloques", "technic", "star wars lego",
            "harry potter lego", "marvel lego", "city", "friends", "duplo",
            "creator", "architecture", "ideas"
        ]
    }
}

# Categorías de audiencia (edad, género, relación)
CATEGORIAS_AUDIENCIA = {
    # Por género
    "hombre": {
        "nombre": "Regalos para Hombre",
        "emoji": "👨",
        "tipo": "genero",
        "keywords_match": [
            "hombre", "hombres", "chico", "chicos", "masculino",
            "el", "él", "caballero", "señor", "varon"
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
    
    # Por edad
    "bebe": {
        "nombre": "Regalos para Bebés",
        "emoji": "👶",
        "tipo": "edad",
        "keywords_match": [
            "bebe", "bebé", "bebes", "bebés", "recien nacido", "recién nacido",
            "0 años", "1 año", "2 años", "lactante", "infantil"
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
            "hijo", "hija", "hijos", "hijas", "crio", "criatura"
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
            "joven", "jovenes", "juvenil", "juventud",
            "14 años", "15 años", "16 años", "17 años", "18 años"
        ]
    },
    "adulto": {
        "nombre": "Regalos para Adultos",
        "emoji": "🧑‍💼",
        "tipo": "edad",
        "keywords_match": [
            "adulto", "adultos", "mayor de edad", "mayores"
        ]
    },
    "senior": {
        "nombre": "Regalos para Mayores",
        "emoji": "🧓",
        "tipo": "edad",
        "keywords_match": [
            "abuelo", "abuela", "abuelos", "abuelas", "anciano", "ancianos",
            "mayor", "mayores", "tercera edad", "senior", "seniors",
            "jubilado", "jubilados", "70 años", "80 años"
        ]
    },
    
    # Por relación familiar
    "padre": {
        "nombre": "Regalos para Padre",
        "emoji": "👨‍👧",
        "tipo": "relacion",
        "keywords_match": [
            "padre", "padres", "papa", "papá", "papi", "progenitor",
            "dia del padre", "día del padre"
        ]
    },
    "madre": {
        "nombre": "Regalos para Madre",
        "emoji": "👩‍👧",
        "tipo": "relacion",
        "keywords_match": [
            "madre", "madres", "mama", "mamá", "mami", "progenitora",
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
            "marido", "esposo", "esposa", "mujer", "conyuge", "cónyuge",
            "enamorado", "enamorada", "enamorados", "san valentin",
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
        "nombre": "Regalos para Compañeros/Trabajo",
        "emoji": "💼",
        "tipo": "relacion",
        "keywords_match": [
            "compañero", "compañera", "compañeros", "trabajo", "oficina",
            "jefe", "jefa", "empleado", "empleada", "empresa", "corporativo",
            "profesional", "cliente", "clientes"
        ]
    },
    
    # Ocasiones especiales
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
            "curioso", "divertido", "gracioso", "friki", "geek"
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
            "por poco dinero", "sin gastar mucho"
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

def match_product_family(keyword: str) -> List[Dict]:
    """
    Encuentra las familias de productos que coinciden con la keyword.
    Retorna lista de dicts con familia, score de match, y keywords que matchearon.
    """
    keyword_lower = keyword.lower()
    # Normalizar acentos comunes
    keyword_normalized = keyword_lower.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    
    matches = []
    
    for family_id, family_data in PRODUCTOS_PCCOMPONENTES.items():
        matched_terms = []
        for kw in family_data["keywords_match"]:
            kw_normalized = kw.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
            if kw_normalized in keyword_normalized:
                matched_terms.append(kw)
        
        if matched_terms:
            # Score basado en cantidad de términos matcheados y longitud del match
            score = len(matched_terms) + sum(len(t) for t in matched_terms) / 100
            matches.append({
                "family_id": family_id,
                "family_name": family_data["nombre"],
                "emoji": family_data["emoji"],
                "score": score,
                "matched_terms": matched_terms
            })
    
    # Ordenar por score descendente
    return sorted(matches, key=lambda x: x["score"], reverse=True)

def get_best_product_family(keyword: str) -> Optional[str]:
    """Retorna el ID de la mejor familia de producto para una keyword"""
    matches = match_product_family(keyword)
    return matches[0]["family_id"] if matches else None

def get_product_match_score(keyword: str) -> float:
    """Retorna el score de match de producto (0-1) para una keyword"""
    matches = match_product_family(keyword)
    if not matches:
        return 0.0
    # Normalizar score a 0-1
    max_possible_score = 5  # Aproximadamente
    return min(matches[0]["score"] / max_possible_score, 1.0)

def keyword_has_product_match(keyword: str) -> bool:
    """Verifica si una keyword tiene al menos un match con productos disponibles"""
    return len(match_product_family(keyword)) > 0

def match_audience_category(keyword: str) -> List[Dict]:
    """
    Encuentra las categorías de audiencia que coinciden con la keyword.
    Retorna lista de dicts con categoría, tipo, score y términos matcheados.
    """
    keyword_lower = keyword.lower()
    keyword_normalized = keyword_lower.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
    
    matches = []
    
    for cat_id, cat_data in CATEGORIAS_AUDIENCIA.items():
        matched_terms = []
        for kw in cat_data["keywords_match"]:
            kw_normalized = kw.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó', 'o').replace('ú', 'u').replace('ñ', 'n')
            if kw_normalized in keyword_normalized:
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
    """
    Retorna un dict con la mejor categoría de audiencia por cada tipo.
    """
    matches = match_audience_category(keyword)
    
    result = {
        "genero": None,
        "edad": None,
        "relacion": None,
        "ocasion": None,
        "estilo": None,
        "precio": None
    }
    
    for match in matches:
        tipo = match["tipo"]
        if tipo in result and result[tipo] is None:
            result[tipo] = match["category_name"]
    
    return result

def get_primary_audience(keyword: str) -> Optional[str]:
    """Retorna la categoría de audiencia principal (la de mayor score)"""
    matches = match_audience_category(keyword)
    if matches:
        return matches[0]["category_name"]
    return None

def get_audience_emoji(keyword: str) -> str:
    """Retorna el emoji de la categoría de audiencia principal"""
    matches = match_audience_category(keyword)
    if matches:
        return matches[0]["emoji"]
    return "🎁"

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
        
        # Configuración de matching de productos
        st.markdown("### 📦 Afinamiento por Productos")
        
        clustering_mode = st.selectbox(
            "Modo de clustering",
            [
                "Semántico puro (TF-IDF)",
                "Guiado por productos",
                "Guiado por audiencia",
                "Híbrido (Semántico + Productos)",
                "Híbrido Completo (Semántico + Productos + Audiencia)"
            ],
            index=4,
            help="""
            - **Semántico puro**: Agrupa solo por similitud de texto
            - **Guiado por productos**: Cada familia de producto = 1 cluster
            - **Guiado por audiencia**: Agrupa por género, edad, relación
            - **Híbrido (Productos)**: Combina semántico + productos
            - **Híbrido Completo**: Combina semántico + productos + audiencia (recomendado)
            """
        )
        
        filter_by_product = st.checkbox(
            "Solo keywords con match de producto",
            value=False,
            help="Elimina keywords que no coinciden con ningún producto disponible"
        )
        
        # Selector de familias de producto
        with st.expander("🎯 Filtrar por familias de producto"):
            st.markdown("Selecciona las familias a incluir:")
            
            selected_families = []
            cols = st.columns(2)
            
            family_list = list(PRODUCTOS_PCCOMPONENTES.keys())
            for i, family_id in enumerate(family_list):
                family_data = PRODUCTOS_PCCOMPONENTES[family_id]
                col = cols[i % 2]
                with col:
                    if st.checkbox(
                        f"{family_data['emoji']} {family_data['nombre']}", 
                        value=True,
                        key=f"family_{family_id}"
                    ):
                        selected_families.append(family_id)
            
            if not selected_families:
                st.warning("⚠️ Selecciona al menos una familia")
        
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
            
            # Matching de productos mejorado
            df_filtered['product_matches'] = df_filtered['Keyword'].apply(match_product_family)
            df_filtered['best_product_family'] = df_filtered['Keyword'].apply(get_best_product_family)
            df_filtered['product_match_score'] = df_filtered['Keyword'].apply(get_product_match_score)
            df_filtered['has_product_match'] = df_filtered['Keyword'].apply(keyword_has_product_match)
            
            # Extraer nombre de familia para display
            df_filtered['familia_producto'] = df_filtered['product_matches'].apply(
                lambda x: x[0]['family_name'] if x else "Sin match"
            )
            
            # NUEVO: Matching de audiencia
            df_filtered['audience_matches'] = df_filtered['Keyword'].apply(match_audience_category)
            df_filtered['primary_audience'] = df_filtered['Keyword'].apply(get_primary_audience)
            df_filtered['audience_emoji'] = df_filtered['Keyword'].apply(get_audience_emoji)
            
            # Extraer audiencia por tipo
            audience_by_type = df_filtered['Keyword'].apply(get_audience_by_type)
            df_filtered['audiencia_genero'] = audience_by_type.apply(lambda x: x.get('genero'))
            df_filtered['audiencia_edad'] = audience_by_type.apply(lambda x: x.get('edad'))
            df_filtered['audiencia_relacion'] = audience_by_type.apply(lambda x: x.get('relacion'))
            df_filtered['audiencia_ocasion'] = audience_by_type.apply(lambda x: x.get('ocasion'))
            df_filtered['audiencia_estilo'] = audience_by_type.apply(lambda x: x.get('estilo'))
            df_filtered['audiencia_precio'] = audience_by_type.apply(lambda x: x.get('precio'))
            
            # Tiene algún match de audiencia
            df_filtered['has_audience_match'] = df_filtered['primary_audience'].notna()
        
        # Filtrar por match de producto si está activado
        if filter_by_product:
            before_filter = len(df_filtered)
            df_filtered = df_filtered[df_filtered['has_product_match']].copy()
            st.info(f"🎯 Filtrado por productos: {before_filter} → {len(df_filtered)} keywords")
        
        # Filtrar por familias seleccionadas
        if selected_families and len(selected_families) < len(PRODUCTOS_PCCOMPONENTES):
            before_filter = len(df_filtered)
            df_filtered = df_filtered[
                df_filtered['best_product_family'].isin(selected_families) | 
                df_filtered['best_product_family'].isna()
            ].copy()
            st.info(f"📦 Filtrado por familias: {before_filter} → {len(df_filtered)} keywords")
        
        if len(df_filtered) == 0:
            st.error("❌ No quedan keywords después de aplicar los filtros")
            return
        
        # Crear embeddings y clusters
        with st.spinner("🧠 Creando clusters semánticos..."):
            keywords_list = df_filtered['keyword_clean'].tolist()
            
            if clustering_mode == "Guiado por productos":
                # Clustering basado en familias de producto
                st.info("🎯 Modo: Clustering guiado por familias de producto")
                
                # Asignar cluster basado en familia de producto
                family_to_cluster = {fam: i for i, fam in enumerate(PRODUCTOS_PCCOMPONENTES.keys())}
                family_to_cluster[None] = len(family_to_cluster)  # Para keywords sin match
                
                clusters = df_filtered['best_product_family'].map(
                    lambda x: family_to_cluster.get(x, family_to_cluster[None])
                ).values
                
                # Crear embeddings para visualización
                embeddings, vectorizer = create_embeddings_tfidf(keywords_list)
                
                # Usar nombres de familia como nombres de cluster
                cluster_names = {}
                for family_id, cluster_id in family_to_cluster.items():
                    if family_id and family_id in PRODUCTOS_PCCOMPONENTES:
                        family_data = PRODUCTOS_PCCOMPONENTES[family_id]
                        cluster_names[cluster_id] = f"{family_data['emoji']} {family_data['nombre']}"
                    else:
                        cluster_names[cluster_id] = "🔍 Sin match de producto"
                
            elif clustering_mode == "Híbrido (Semántico + Productos)":
                # Modo híbrido: combina embeddings TF-IDF con información de producto
                st.info("🔀 Modo: Clustering híbrido (Semántico + Productos)")
                
                # Crear embeddings TF-IDF
                embeddings_tfidf, vectorizer = create_embeddings_tfidf(keywords_list)
                
                # Crear embeddings de producto (one-hot encoding de familias)
                family_ids = list(PRODUCTOS_PCCOMPONENTES.keys())
                product_embeddings = np.zeros((len(df_filtered), len(family_ids) + 1))
                
                for i, (_, row) in enumerate(df_filtered.iterrows()):
                    if row['best_product_family'] and row['best_product_family'] in family_ids:
                        idx = family_ids.index(row['best_product_family'])
                        # Usar el score de match como peso
                        product_embeddings[i, idx] = row['product_match_score'] * 2  # Peso x2 para productos
                    else:
                        product_embeddings[i, -1] = 0.5  # Sin match
                
                # Combinar embeddings (TF-IDF + Productos)
                # Normalizar TF-IDF
                tfidf_norm = embeddings_tfidf / (np.linalg.norm(embeddings_tfidf, axis=1, keepdims=True) + 1e-8)
                
                # Combinar con peso para productos
                product_weight = 0.4  # 40% peso para productos, 60% para semántico
                embeddings = np.hstack([
                    tfidf_norm * (1 - product_weight),
                    product_embeddings * product_weight
                ])
                
                # Aplicar clustering
                if clustering_method == "K-Means":
                    clusters = cluster_keywords_kmeans(embeddings, n_clusters)
                else:
                    clusters = cluster_keywords_hierarchical(embeddings, n_clusters)
                
                cluster_names = None  # Se generarán después
            
            elif clustering_mode == "Guiado por audiencia":
                # Clustering basado en categorías de audiencia
                st.info("👥 Modo: Clustering guiado por audiencia (género, edad, relación)")
                
                # Asignar cluster basado en audiencia principal
                audience_to_cluster = {aud: i for i, aud in enumerate(CATEGORIAS_AUDIENCIA.keys())}
                audience_to_cluster[None] = len(audience_to_cluster)  # Para keywords sin match
                
                # Mapear primary_audience al ID de categoría
                def get_audience_id(primary_audience):
                    if primary_audience is None:
                        return None
                    for cat_id, cat_data in CATEGORIAS_AUDIENCIA.items():
                        if cat_data["nombre"] == primary_audience:
                            return cat_id
                    return None
                
                df_filtered['audience_id'] = df_filtered['primary_audience'].apply(get_audience_id)
                
                clusters = df_filtered['audience_id'].map(
                    lambda x: audience_to_cluster.get(x, audience_to_cluster[None])
                ).values
                
                # Crear embeddings para visualización
                embeddings, vectorizer = create_embeddings_tfidf(keywords_list)
                
                # Usar nombres de audiencia como nombres de cluster
                cluster_names = {}
                for audience_id, cluster_id in audience_to_cluster.items():
                    if audience_id and audience_id in CATEGORIAS_AUDIENCIA:
                        aud_data = CATEGORIAS_AUDIENCIA[audience_id]
                        cluster_names[cluster_id] = f"{aud_data['emoji']} {aud_data['nombre']}"
                    else:
                        cluster_names[cluster_id] = "🔍 Sin audiencia identificada"
            
            elif clustering_mode == "Híbrido Completo (Semántico + Productos + Audiencia)":
                # Modo híbrido completo: combina TF-IDF + productos + audiencia
                st.info("🎯 Modo: Clustering híbrido completo (Semántico + Productos + Audiencia)")
                
                # Crear embeddings TF-IDF
                embeddings_tfidf, vectorizer = create_embeddings_tfidf(keywords_list)
                
                # Crear embeddings de producto
                family_ids = list(PRODUCTOS_PCCOMPONENTES.keys())
                product_embeddings = np.zeros((len(df_filtered), len(family_ids) + 1))
                
                for i, (_, row) in enumerate(df_filtered.iterrows()):
                    if row['best_product_family'] and row['best_product_family'] in family_ids:
                        idx = family_ids.index(row['best_product_family'])
                        product_embeddings[i, idx] = row['product_match_score'] * 2
                    else:
                        product_embeddings[i, -1] = 0.3
                
                # Crear embeddings de audiencia
                audience_ids = list(CATEGORIAS_AUDIENCIA.keys())
                audience_embeddings = np.zeros((len(df_filtered), len(audience_ids) + 1))
                
                for i, (_, row) in enumerate(df_filtered.iterrows()):
                    matches = row['audience_matches']
                    if matches:
                        for match in matches[:3]:  # Top 3 audiencias
                            if match['category_id'] in audience_ids:
                                idx = audience_ids.index(match['category_id'])
                                audience_embeddings[i, idx] = match['score'] / 5  # Normalizar
                    else:
                        audience_embeddings[i, -1] = 0.3
                
                # Normalizar TF-IDF
                tfidf_norm = embeddings_tfidf / (np.linalg.norm(embeddings_tfidf, axis=1, keepdims=True) + 1e-8)
                
                # Combinar con pesos: 50% semántico, 30% productos, 20% audiencia
                embeddings = np.hstack([
                    tfidf_norm * 0.5,
                    product_embeddings * 0.3,
                    audience_embeddings * 0.2
                ])
                
                # Aplicar clustering
                if clustering_method == "K-Means":
                    clusters = cluster_keywords_kmeans(embeddings, n_clusters)
                else:
                    clusters = cluster_keywords_hierarchical(embeddings, n_clusters)
                
                cluster_names = None  # Se generarán después
                
            else:
                # Clustering semántico puro
                st.info("📝 Modo: Clustering semántico puro (TF-IDF)")
                embeddings, vectorizer = create_embeddings_tfidf(keywords_list)
                
                # Aplicar clustering
                if clustering_method == "K-Means":
                    clusters = cluster_keywords_kmeans(embeddings, n_clusters)
                else:
                    clusters = cluster_keywords_hierarchical(embeddings, n_clusters)
                
                cluster_names = None  # Se generarán después
            
            df_filtered['cluster_id'] = clusters
            
            # Calcular coherencia de clusters
            coherences = calculate_cluster_coherence(embeddings, clusters)
        
        # Nombrar clusters (si no están ya nombrados por modo guiado)
        if cluster_names is None:
            cluster_names = {}
            for cluster_id in df_filtered['cluster_id'].unique():
                cluster_kws = df_filtered[df_filtered['cluster_id'] == cluster_id]
                
                # Intentar usar familia de producto más común
                family_counts = cluster_kws['familia_producto'].value_counts()
                top_family = family_counts.index[0] if len(family_counts) > 0 else None
                
                # Intentar usar audiencia más común
                audience_counts = cluster_kws['primary_audience'].value_counts()
                top_audience = audience_counts.index[0] if len(audience_counts) > 0 and pd.notna(audience_counts.index[0]) else None
                
                # Keyword con mayor volumen del cluster
                top_kw = cluster_kws.nlargest(1, 'volumen_navidad')['Keyword'].values[0]
                
                # Construir nombre del cluster
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
        
        # Segunda fila de métricas - Match de productos
        col5, col6, col7, col8 = st.columns(4)
        
        with col5:
            pct_with_match = df_filtered['has_product_match'].mean() * 100
            st.metric(
                "Keywords con Match",
                f"{pct_with_match:.1f}%"
            )
        
        with col6:
            avg_match_score = df_filtered['product_match_score'].mean()
            st.metric(
                "Score Match Medio",
                f"{avg_match_score:.2f}"
            )
        
        with col7:
            n_families = df_filtered['best_product_family'].nunique()
            st.metric(
                "Familias Detectadas",
                f"{n_families}"
            )
        
        with col8:
            vol_with_match = df_filtered[df_filtered['has_product_match']]['volumen_navidad'].sum()
            vol_total = df_filtered['volumen_navidad'].sum()
            pct_vol_match = (vol_with_match / vol_total * 100) if vol_total > 0 else 0
            st.metric(
                "Volumen con Match",
                f"{pct_vol_match:.1f}%"
            )
        
        # Tercera fila de métricas - Audiencia
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            pct_audience = df_filtered['has_audience_match'].mean() * 100
            st.metric(
                "KWs con Audiencia",
                f"{pct_audience:.1f}%"
            )
        
        with col10:
            n_genero = df_filtered['audiencia_genero'].notna().sum()
            st.metric(
                "Con Género",
                f"{n_genero:,}"
            )
        
        with col11:
            n_edad = df_filtered['audiencia_edad'].notna().sum()
            st.metric(
                "Con Edad",
                f"{n_edad:,}"
            )
        
        with col12:
            n_relacion = df_filtered['audiencia_relacion'].notna().sum()
            st.metric(
                "Con Relación",
                f"{n_relacion:,}"
            )
        
        # Tabs para diferentes vistas
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "🗺️ Visualización", 
            "📦 Match Productos",
            "👥 Audiencias",
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
            st.markdown("### 📦 Análisis de Match con Productos Disponibles")
            
            # Resumen por familia de producto
            st.markdown("#### Distribución por Familia de Producto")
            
            family_summary = df_filtered.groupby('familia_producto').agg({
                'volumen_navidad': ['sum', 'mean', 'count'],
                'product_match_score': 'mean'
            }).reset_index()
            family_summary.columns = ['Familia', 'Volumen Total', 'Volumen Medio', 'Num Keywords', 'Score Match']
            family_summary = family_summary.sort_values('Volumen Total', ascending=False)
            
            # Gráfico de barras por familia
            fig_family = go.Figure()
            
            # Colores por familia
            colors = ['#c41e3a', '#1a472a', '#d4a574', '#2d5016', '#8b4513', 
                      '#556b2f', '#8fbc8f', '#cd853f', '#daa520', '#b8860b',
                      '#6b8e23', '#808000', '#9acd32']
            
            fig_family.add_trace(go.Bar(
                y=family_summary['Familia'],
                x=family_summary['Volumen Total'],
                orientation='h',
                marker=dict(
                    color=colors[:len(family_summary)],
                    line=dict(color='white', width=1)
                ),
                text=family_summary.apply(
                    lambda r: f"Vol: {r['Volumen Total']:,.0f} | KWs: {r['Num Keywords']:.0f}", axis=1
                ),
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Volumen: %{x:,}<extra></extra>'
            ))
            
            fig_family.update_layout(
                title='Volumen de Búsqueda por Familia de Producto',
                xaxis_title='Volumen Total',
                yaxis_title='',
                plot_bgcolor='white',
                paper_bgcolor='white',
                font_family='Georgia',
                height=max(400, len(family_summary) * 35),
                margin=dict(l=200, r=150)
            )
            
            st.plotly_chart(fig_family, use_container_width=True)
            
            # Tabla detallada
            st.markdown("#### Métricas por Familia")
            st.dataframe(
                family_summary.style.format({
                    'Volumen Total': '{:,.0f}',
                    'Volumen Medio': '{:,.0f}',
                    'Num Keywords': '{:.0f}',
                    'Score Match': '{:.2f}'
                }),
                use_container_width=True
            )
            
            # Keywords sin match
            st.markdown("---")
            st.markdown("#### ⚠️ Keywords Sin Match de Producto")
            
            no_match_kws = df_filtered[~df_filtered['has_product_match']]
            
            if len(no_match_kws) > 0:
                st.warning(f"Hay **{len(no_match_kws)}** keywords ({len(no_match_kws)/len(df_filtered)*100:.1f}%) sin match con productos disponibles")
                
                # Top keywords sin match por volumen
                st.markdown("**Top 20 keywords sin match (por volumen):**")
                top_no_match = no_match_kws.nlargest(20, 'volumen_navidad')[['Keyword', 'volumen_navidad', 'intent']]
                st.dataframe(top_no_match, use_container_width=True)
                
                st.markdown("""
                <div class="info-box">
                    <strong>💡 Consejo:</strong> Revisa estas keywords - podrías estar perdiendo oportunidades de productos 
                    o necesitas añadir nuevas keywords de matching en la configuración.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✅ Todas las keywords tienen match con productos disponibles")
            
            # Análisis de oportunidades de producto
            st.markdown("---")
            st.markdown("#### 🎯 Oportunidades de Producto")
            
            # Productos con alto volumen pero pocas keywords
            family_opportunity = family_summary[family_summary['Familia'] != 'Sin match'].copy()
            if len(family_opportunity) > 0:
                family_opportunity['Oportunidad'] = family_opportunity['Volumen Total'] / (family_opportunity['Num Keywords'] + 1)
                family_opportunity = family_opportunity.sort_values('Oportunidad', ascending=False)
                
                st.markdown("**Familias con mayor potencial por keyword:**")
                st.dataframe(
                    family_opportunity[['Familia', 'Volumen Total', 'Num Keywords', 'Oportunidad']].head(10).style.format({
                        'Volumen Total': '{:,.0f}',
                        'Num Keywords': '{:.0f}',
                        'Oportunidad': '{:,.0f}'
                    }),
                    use_container_width=True
                )
        
        with tab3:
            st.markdown("### 👥 Análisis de Audiencias (Género, Edad, Relación)")
            
            # Distribución por tipo de audiencia
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 👫 Por Género")
                genero_counts = df_filtered['audiencia_genero'].value_counts().reset_index()
                genero_counts.columns = ['Género', 'Keywords']
                if len(genero_counts) > 0:
                    # Añadir volumen
                    genero_vol = df_filtered.groupby('audiencia_genero')['volumen_navidad'].sum().reset_index()
                    genero_vol.columns = ['Género', 'Volumen']
                    genero_counts = genero_counts.merge(genero_vol, on='Género', how='left')
                    
                    fig_genero = go.Figure(go.Bar(
                        x=genero_counts['Género'],
                        y=genero_counts['Volumen'],
                        text=genero_counts.apply(lambda r: f"{r['Keywords']} KWs<br>{r['Volumen']:,.0f} vol", axis=1),
                        textposition='outside',
                        marker_color=['#3498db', '#e74c3c', '#95a5a6'][:len(genero_counts)]
                    ))
                    fig_genero.update_layout(
                        title='Volumen por Género',
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_genero, use_container_width=True)
                else:
                    st.info("No hay keywords con género identificado")
            
            with col2:
                st.markdown("#### 👶👧🧑 Por Edad")
                edad_counts = df_filtered['audiencia_edad'].value_counts().reset_index()
                edad_counts.columns = ['Edad', 'Keywords']
                if len(edad_counts) > 0:
                    edad_vol = df_filtered.groupby('audiencia_edad')['volumen_navidad'].sum().reset_index()
                    edad_vol.columns = ['Edad', 'Volumen']
                    edad_counts = edad_counts.merge(edad_vol, on='Edad', how='left')
                    
                    fig_edad = go.Figure(go.Bar(
                        x=edad_counts['Edad'],
                        y=edad_counts['Volumen'],
                        text=edad_counts.apply(lambda r: f"{r['Keywords']} KWs", axis=1),
                        textposition='outside',
                        marker_color=['#9b59b6', '#1abc9c', '#f39c12', '#e67e22', '#27ae60', '#3498db'][:len(edad_counts)]
                    ))
                    fig_edad.update_layout(
                        title='Volumen por Grupo de Edad',
                        height=350,
                        showlegend=False
                    )
                    st.plotly_chart(fig_edad, use_container_width=True)
                else:
                    st.info("No hay keywords con edad identificada")
            
            st.markdown("---")
            
            # Relaciones familiares
            st.markdown("#### 👨‍👩‍👧‍👦 Por Relación")
            
            relacion_data = df_filtered[df_filtered['audiencia_relacion'].notna()].groupby('audiencia_relacion').agg({
                'volumen_navidad': ['sum', 'count']
            }).reset_index()
            relacion_data.columns = ['Relación', 'Volumen Total', 'Num Keywords']
            relacion_data = relacion_data.sort_values('Volumen Total', ascending=True)
            
            if len(relacion_data) > 0:
                fig_relacion = go.Figure(go.Bar(
                    y=relacion_data['Relación'],
                    x=relacion_data['Volumen Total'],
                    orientation='h',
                    text=relacion_data.apply(lambda r: f"Vol: {r['Volumen Total']:,.0f} | {r['Num Keywords']:.0f} KWs", axis=1),
                    textposition='outside',
                    marker=dict(
                        color=relacion_data['Volumen Total'],
                        colorscale='Viridis'
                    )
                ))
                fig_relacion.update_layout(
                    title='Volumen por Tipo de Relación',
                    height=max(300, len(relacion_data) * 40),
                    margin=dict(l=200, r=150)
                )
                st.plotly_chart(fig_relacion, use_container_width=True)
            
            st.markdown("---")
            
            # Ocasiones y estilos
            col3, col4 = st.columns(2)
            
            with col3:
                st.markdown("#### 🎄 Por Ocasión")
                ocasion_data = df_filtered[df_filtered['audiencia_ocasion'].notna()].groupby('audiencia_ocasion').agg({
                    'volumen_navidad': 'sum',
                    'Keyword': 'count'
                }).reset_index()
                ocasion_data.columns = ['Ocasión', 'Volumen', 'Keywords']
                ocasion_data = ocasion_data.sort_values('Volumen', ascending=False)
                
                if len(ocasion_data) > 0:
                    st.dataframe(
                        ocasion_data.style.format({'Volumen': '{:,.0f}', 'Keywords': '{:.0f}'}),
                        use_container_width=True
                    )
            
            with col4:
                st.markdown("#### ✨ Por Estilo/Precio")
                estilo_data = df_filtered[df_filtered['audiencia_estilo'].notna() | df_filtered['audiencia_precio'].notna()]
                
                # Combinar estilo y precio
                combined = []
                for _, row in df_filtered.iterrows():
                    if pd.notna(row['audiencia_estilo']):
                        combined.append({'Tipo': row['audiencia_estilo'], 'Volumen': row['volumen_navidad']})
                    if pd.notna(row['audiencia_precio']):
                        combined.append({'Tipo': row['audiencia_precio'], 'Volumen': row['volumen_navidad']})
                
                if combined:
                    combined_df = pd.DataFrame(combined).groupby('Tipo').agg({
                        'Volumen': ['sum', 'count']
                    }).reset_index()
                    combined_df.columns = ['Tipo', 'Volumen', 'Keywords']
                    combined_df = combined_df.sort_values('Volumen', ascending=False)
                    st.dataframe(
                        combined_df.style.format({'Volumen': '{:,.0f}', 'Keywords': '{:.0f}'}),
                        use_container_width=True
                    )
            
            st.markdown("---")
            
            # Matriz de oportunidades: Audiencia x Producto
            st.markdown("#### 🎯 Matriz: Audiencia x Producto")
            st.markdown("Identifica combinaciones de alto potencial (ej: 'Gaming + Adolescente', 'Belleza + Mujer')")
            
            # Crear matriz cruzada
            matrix_data = []
            
            for _, row in df_filtered.iterrows():
                producto = row['familia_producto'] if row['familia_producto'] != 'Sin match' else None
                audiencia = row['primary_audience']
                
                if producto and audiencia:
                    matrix_data.append({
                        'Producto': producto,
                        'Audiencia': audiencia,
                        'Volumen': row['volumen_navidad']
                    })
            
            if matrix_data:
                matrix_df = pd.DataFrame(matrix_data)
                pivot = matrix_df.groupby(['Producto', 'Audiencia'])['Volumen'].sum().reset_index()
                pivot_table = pivot.pivot(index='Producto', columns='Audiencia', values='Volumen').fillna(0)
                
                # Mostrar top combinaciones
                top_combos = pivot.nlargest(15, 'Volumen')
                
                st.markdown("**Top 15 combinaciones Producto + Audiencia:**")
                for _, combo in top_combos.iterrows():
                    st.markdown(f"- **{combo['Producto']}** + **{combo['Audiencia']}**: {combo['Volumen']:,.0f} vol")
            else:
                st.info("No hay suficientes datos para crear la matriz")
        
        with tab4:
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
        
        with tab5:
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
                    family_counts = cluster_kws['familia_producto'].value_counts()
                    families_with_match = family_counts[family_counts.index != 'Sin match']
                    if len(families_with_match) > 0:
                        st.markdown("**Familias de Producto Relacionadas:**")
                        for fam, count in families_with_match.head(5).items():
                            st.markdown(f"- {fam} ({count} keywords)")
            
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
        
        with tab6:
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
