"""
Configuración central de la aplicación.
Carga settings desde variables de entorno y archivos de configuración.
"""

import os
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Rutas base
BASE_DIR = Path(__file__).parent.parent
CONFIG_DIR = BASE_DIR / "config"
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / ".cache"

# Crear directorio de caché si no existe
CACHE_DIR.mkdir(exist_ok=True)


@dataclass
class EmbeddingConfig:
    """Configuración para generación de embeddings."""
    
    # Sentence Transformers
    st_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    st_batch_size: int = 64
    st_context_prefix: str = "regalo de navidad:"
    
    # TF-IDF
    tfidf_ngram_range: tuple = (1, 3)
    tfidf_max_features: int = 2000
    tfidf_min_df: int = 1
    tfidf_max_df: float = 0.95
    tfidf_sublinear_tf: bool = True
    
    # Híbrido
    hybrid_tfidf_weight: float = 0.3
    hybrid_semantic_weight: float = 0.7


@dataclass
class ClusteringConfig:
    """Configuración para algoritmos de clustering."""
    
    # K-Means
    kmeans_n_init: int = 10
    kmeans_random_state: int = 42
    
    # HDBSCAN
    hdbscan_min_cluster_size: int = 5
    hdbscan_min_samples: int = 3
    hdbscan_metric: str = "euclidean"
    hdbscan_cluster_selection_method: str = "eom"
    
    # UMAP (para reducción dimensional)
    umap_n_components: int = 50
    umap_metric: str = "cosine"
    umap_random_state: int = 42
    
    # Modos híbridos
    hybrid_semantic_weight: float = 0.5
    hybrid_product_weight: float = 0.3
    hybrid_audience_weight: float = 0.2


@dataclass
class AIConfig:
    """Configuración para integraciones de AI."""
    
    # Claude
    claude_model: str = "claude-sonnet-4-20250514"
    claude_max_tokens: int = 2000
    
    # OpenAI
    openai_model: str = "gpt-4o-mini"
    openai_temperature: float = 0.1
    
    # Batch processing
    ai_batch_size: int = 80
    
    # API Keys (desde env)
    @property
    def anthropic_api_key(self) -> Optional[str]:
        return os.getenv("ANTHROPIC_API_KEY")
    
    @property
    def openai_api_key(self) -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")


@dataclass
class AppConfig:
    """Configuración general de la aplicación."""
    
    # UI
    page_title: str = "🎄 Keyword Clustering - Navidad"
    page_icon: str = "🎁"
    layout: str = "wide"
    
    # Defaults
    default_min_volume: int = 50
    default_n_clusters: int = 15
    default_min_cluster_size: int = 5
    
    # Logging
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Cache
    enable_cache: bool = True
    cache_ttl_seconds: int = 3600  # 1 hora


class ConfigLoader:
    """Cargador de configuración desde archivos JSON."""
    
    _products_cache: Optional[Dict] = None
    _audiences_cache: Optional[Dict] = None
    
    @classmethod
    def load_products(cls, reload: bool = False) -> Dict:
        """Carga configuración de productos desde JSON."""
        if cls._products_cache is None or reload:
            products_file = CONFIG_DIR / "products.json"
            
            if not products_file.exists():
                raise FileNotFoundError(
                    f"Archivo de configuración no encontrado: {products_file}"
                )
            
            with open(products_file, "r", encoding="utf-8") as f:
                cls._products_cache = json.load(f)
            
            logging.info(f"Cargadas {len(cls._products_cache)} familias de productos")
        
        return cls._products_cache
    
    @classmethod
    def load_audiences(cls, reload: bool = False) -> Dict:
        """Carga configuración de audiencias desde JSON."""
        if cls._audiences_cache is None or reload:
            audiences_file = CONFIG_DIR / "audiences.json"
            
            if not audiences_file.exists():
                raise FileNotFoundError(
                    f"Archivo de configuración no encontrado: {audiences_file}"
                )
            
            with open(audiences_file, "r", encoding="utf-8") as f:
                cls._audiences_cache = json.load(f)
            
            logging.info(f"Cargadas {len(cls._audiences_cache)} categorías de audiencia")
        
        return cls._audiences_cache
    
    @classmethod
    def save_products(cls, products: Dict) -> None:
        """Guarda configuración de productos a JSON."""
        products_file = CONFIG_DIR / "products.json"
        
        with open(products_file, "w", encoding="utf-8") as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
        
        cls._products_cache = products
        logging.info(f"Guardadas {len(products)} familias de productos")
    
    @classmethod
    def save_audiences(cls, audiences: Dict) -> None:
        """Guarda configuración de audiencias a JSON."""
        audiences_file = CONFIG_DIR / "audiences.json"
        
        with open(audiences_file, "w", encoding="utf-8") as f:
            json.dump(audiences, f, ensure_ascii=False, indent=2)
        
        cls._audiences_cache = audiences
        logging.info(f"Guardadas {len(audiences)} categorías de audiencia")
    
    @classmethod
    def get_product_families(cls) -> List[str]:
        """Retorna lista de IDs de familias de productos."""
        products = cls.load_products()
        return list(products.keys())
    
    @classmethod
    def get_audience_categories(cls) -> List[str]:
        """Retorna lista de IDs de categorías de audiencia."""
        audiences = cls.load_audiences()
        return list(audiences.keys())
    
    @classmethod
    def get_audience_types(cls) -> List[str]:
        """Retorna tipos únicos de audiencia."""
        audiences = cls.load_audiences()
        return list(set(a["tipo"] for a in audiences.values()))


def setup_logging(config: Optional[AppConfig] = None) -> logging.Logger:
    """Configura el sistema de logging."""
    if config is None:
        config = AppConfig()
    
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.log_format
    )
    
    logger = logging.getLogger("keyword_clustering")
    return logger


# Instancias globales de configuración
embedding_config = EmbeddingConfig()
clustering_config = ClusteringConfig()
ai_config = AIConfig()
app_config = AppConfig()

# Configurar logging al importar
logger = setup_logging(app_config)
