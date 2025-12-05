"""
Utilidades generales para procesamiento de texto y keywords.
"""

import re
import hashlib
import pickle
import logging
from pathlib import Path
from typing import Optional, Any, List
from functools import wraps
from datetime import datetime, timedelta

logger = logging.getLogger("keyword_clustering.utils")

# Stopwords en español para filtrado
SPANISH_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "es", "lo", "como", "más",
    "pero", "sus", "le", "ya", "o", "fue", "este", "ha", "si", "porque", "esta",
    "son", "entre", "está", "cuando", "muy", "sin", "sobre", "ser", "tiene",
    "también", "me", "hasta", "hay", "donde", "han", "quien", "están", "estado",
    "desde", "todo", "nos", "durante", "estados", "todos", "uno", "les", "ni",
    "contra", "otros", "fueron", "ese", "eso", "había", "ante", "ellos", "e",
    "esto", "mi", "antes", "algunos", "qué", "unos", "yo", "otro", "otras",
    "otra", "él", "tanto", "esa", "estos", "mucho", "quienes", "nada", "muchos"
}

# Mapeo de acentos para normalización
ACCENT_MAP = {
    'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
    'ü': 'u', 'ñ': 'n', 'Á': 'A', 'É': 'E', 'Í': 'I',
    'Ó': 'O', 'Ú': 'U', 'Ñ': 'N'
}

# Expansiones de abreviaciones comunes
ABBREVIATION_EXPANSIONS = {
    'tv': 'television',
    'pc': 'ordenador',
    'portatil': 'ordenador portatil',
    'movil': 'telefono movil',
    'wifi': 'wireless',
    'bt': 'bluetooth',
    'gb': 'gigabytes',
    'tb': 'terabytes',
    'hd': 'alta definicion',
    'uhd': 'ultra alta definicion',
}


def clean_keyword(keyword: str) -> str:
    """
    Limpia y normaliza una keyword básicamente.
    
    Args:
        keyword: Keyword original
        
    Returns:
        Keyword limpia y normalizada
    """
    if not keyword or not isinstance(keyword, str):
        return ""
    
    kw = keyword.lower().strip()
    kw = re.sub(r'[^\w\s]', ' ', kw)
    kw = re.sub(r'\s+', ' ', kw)
    return kw.strip()


def normalize_accents(text: str) -> str:
    """
    Normaliza acentos en texto.
    
    Args:
        text: Texto con posibles acentos
        
    Returns:
        Texto sin acentos
    """
    for accented, plain in ACCENT_MAP.items():
        text = text.replace(accented, plain)
    return text


def expand_abbreviations(text: str) -> str:
    """
    Expande abreviaciones comunes.
    
    Args:
        text: Texto con posibles abreviaciones
        
    Returns:
        Texto con abreviaciones expandidas
    """
    words = text.split()
    expanded = []
    
    for word in words:
        expanded.append(ABBREVIATION_EXPANSIONS.get(word, word))
    
    return ' '.join(expanded)


def preprocess_keyword(keyword: str, advanced: bool = True) -> str:
    """
    Preprocesa una keyword con normalización completa.
    
    Args:
        keyword: Keyword original
        advanced: Si aplicar preprocesamiento avanzado
        
    Returns:
        Keyword preprocesada
    """
    if not keyword or not isinstance(keyword, str):
        return ""
    
    kw = keyword.lower().strip()
    
    if advanced:
        kw = normalize_accents(kw)
        kw = expand_abbreviations(kw)
    
    kw = re.sub(r'[^\w\s]', ' ', kw)
    kw = re.sub(r'\s+', ' ', kw).strip()
    
    return kw


def extract_tokens(
    text: str,
    remove_stopwords: bool = True,
    min_length: int = 2
) -> List[str]:
    """
    Extrae tokens de un texto.
    
    Args:
        text: Texto a tokenizar
        remove_stopwords: Si eliminar stopwords
        min_length: Longitud mínima de token
        
    Returns:
        Lista de tokens
    """
    text = preprocess_keyword(text)
    tokens = text.split()
    
    if remove_stopwords:
        tokens = [t for t in tokens if t not in SPANISH_STOPWORDS]
    
    tokens = [t for t in tokens if len(t) >= min_length]
    
    return tokens


def generate_url_slug(text: str, max_length: int = 50) -> str:
    """
    Genera un slug para URL a partir de texto.
    
    Args:
        text: Texto original
        max_length: Longitud máxima del slug
        
    Returns:
        Slug válido para URL
    """
    slug = preprocess_keyword(text, advanced=True)
    slug = re.sub(r'\s+', '-', slug)
    slug = re.sub(r'-+', '-', slug).strip('-')
    
    if len(slug) > max_length:
        # Cortar en palabra completa
        slug = slug[:max_length].rsplit('-', 1)[0]
    
    return slug


def calculate_hash(data: Any) -> str:
    """
    Calcula hash MD5 de cualquier dato serializable.
    
    Args:
        data: Datos a hashear
        
    Returns:
        Hash MD5 como string
    """
    serialized = pickle.dumps(data)
    return hashlib.md5(serialized).hexdigest()


class CacheManager:
    """
    Gestor de caché en disco para embeddings y otros datos pesados.
    """
    
    def __init__(self, cache_dir: Path, ttl_seconds: int = 3600):
        """
        Inicializa el gestor de caché.
        
        Args:
            cache_dir: Directorio para archivos de caché
            ttl_seconds: Tiempo de vida de caché en segundos
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(seconds=ttl_seconds)
        logger.info(f"CacheManager inicializado en {self.cache_dir}")
    
    def _get_cache_path(self, key: str) -> Path:
        """Genera ruta de archivo de caché."""
        return self.cache_dir / f"{key}.pkl"
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene valor de caché si existe y no ha expirado.
        
        Args:
            key: Clave de caché
            
        Returns:
            Valor cacheado o None
        """
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        # Verificar TTL
        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        if datetime.now() - mtime > self.ttl:
            logger.debug(f"Caché expirado para {key}")
            cache_path.unlink()
            return None
        
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            logger.debug(f"Cache hit para {key}")
            return data
        except Exception as e:
            logger.warning(f"Error leyendo caché {key}: {e}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Guarda valor en caché.
        
        Args:
            key: Clave de caché
            value: Valor a cachear
        """
        cache_path = self._get_cache_path(key)
        
        try:
            with open(cache_path, "wb") as f:
                pickle.dump(value, f)
            logger.debug(f"Cacheado {key}")
        except Exception as e:
            logger.warning(f"Error escribiendo caché {key}: {e}")
    
    def invalidate(self, key: str) -> None:
        """
        Invalida una entrada de caché.
        
        Args:
            key: Clave a invalidar
        """
        cache_path = self._get_cache_path(key)
        
        if cache_path.exists():
            cache_path.unlink()
            logger.debug(f"Invalidado caché {key}")
    
    def clear(self) -> int:
        """
        Limpia toda la caché.
        
        Returns:
            Número de archivos eliminados
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
            count += 1
        
        logger.info(f"Limpiados {count} archivos de caché")
        return count


def cached(cache_manager: CacheManager, key_prefix: str = ""):
    """
    Decorador para cachear resultados de funciones.
    
    Args:
        cache_manager: Instancia de CacheManager
        key_prefix: Prefijo para claves de caché
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generar clave única basada en argumentos
            cache_key = f"{key_prefix}_{func.__name__}_{calculate_hash((args, kwargs))}"
            
            # Intentar obtener de caché
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Ejecutar función y cachear resultado
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result)
            
            return result
        return wrapper
    return decorator


def safe_convert_volume(value: Any) -> int:
    """
    Convierte un valor de volumen a entero de forma segura.
    
    Args:
        value: Valor a convertir (puede ser string, float, int, None)
        
    Returns:
        Volumen como entero, 0 si no se puede convertir
    """
    if value is None:
        return 0
    
    if isinstance(value, (int, float)):
        return int(value)
    
    # Limpiar string
    val_str = str(value).strip()
    val_str = val_str.replace(',', '').replace('.', '')
    val_str = re.sub(r'[^\d]', '', val_str)
    
    try:
        return int(val_str) if val_str else 0
    except ValueError:
        return 0


def format_number(num: float, decimals: int = 0) -> str:
    """
    Formatea número con separadores de miles.
    
    Args:
        num: Número a formatear
        decimals: Decimales a mostrar
        
    Returns:
        String formateado
    """
    if decimals == 0:
        return f"{int(num):,}".replace(",", ".")
    else:
        return f"{num:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")
