"""
Módulo para generación de embeddings de keywords.
Soporta múltiples métodos: TF-IDF, Sentence Transformers, Híbrido.
"""

import logging
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

from .utils import preprocess_keyword, CacheManager, calculate_hash
from config.settings import embedding_config, CACHE_DIR

logger = logging.getLogger("keyword_clustering.embeddings")

# Importaciones opcionales
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("Sentence Transformers no disponible")


class EmbeddingMethod(Enum):
    """Métodos de embedding disponibles."""
    TFIDF = "tfidf"
    SENTENCE_TRANSFORMER = "sentence_transformer"
    HYBRID = "hybrid"


@dataclass
class EmbeddingResult:
    """Resultado de generación de embeddings."""
    embeddings: np.ndarray
    method: EmbeddingMethod
    dimensions: int
    metadata: Dict[str, Any]


class TFIDFEmbedder:
    """
    Generador de embeddings basado en TF-IDF mejorado.
    """
    
    def __init__(
        self,
        ngram_range: Tuple[int, int] = None,
        max_features: int = None,
        min_df: int = None,
        max_df: float = None,
        sublinear_tf: bool = None
    ):
        """
        Inicializa el embedder TF-IDF.
        
        Args:
            ngram_range: Rango de n-gramas
            max_features: Número máximo de features
            min_df: Frecuencia mínima de documento
            max_df: Frecuencia máxima de documento
            sublinear_tf: Usar escalado sublinear
        """
        self.ngram_range = ngram_range or embedding_config.tfidf_ngram_range
        self.max_features = max_features or embedding_config.tfidf_max_features
        self.min_df = min_df or embedding_config.tfidf_min_df
        self.max_df = max_df or embedding_config.tfidf_max_df
        self.sublinear_tf = sublinear_tf if sublinear_tf is not None else embedding_config.tfidf_sublinear_tf
        
        self.vectorizer: Optional[TfidfVectorizer] = None
    
    def fit_transform(self, keywords: List[str]) -> EmbeddingResult:
        """
        Ajusta el vectorizador y transforma keywords a embeddings.
        
        Args:
            keywords: Lista de keywords
            
        Returns:
            Resultado con embeddings
        """
        # Preprocesar keywords
        processed = [preprocess_keyword(kw, advanced=True) for kw in keywords]
        
        # Crear vectorizador
        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf
        )
        
        # Generar embeddings
        embeddings = self.vectorizer.fit_transform(processed).toarray()
        
        logger.info(f"TF-IDF embeddings: {embeddings.shape}")
        
        return EmbeddingResult(
            embeddings=embeddings,
            method=EmbeddingMethod.TFIDF,
            dimensions=embeddings.shape[1],
            metadata={
                "n_keywords": len(keywords),
                "vocabulary_size": len(self.vectorizer.vocabulary_),
                "ngram_range": self.ngram_range
            }
        )
    
    def transform(self, keywords: List[str]) -> np.ndarray:
        """
        Transforma keywords usando vectorizador ya ajustado.
        
        Args:
            keywords: Lista de keywords
            
        Returns:
            Matriz de embeddings
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizador no ajustado. Usa fit_transform primero.")
        
        processed = [preprocess_keyword(kw, advanced=True) for kw in keywords]
        return self.vectorizer.transform(processed).toarray()


class SentenceTransformerEmbedder:
    """
    Generador de embeddings basado en Sentence Transformers.
    """
    
    _model_cache: Dict[str, Any] = {}
    
    def __init__(
        self,
        model_name: str = None,
        batch_size: int = None,
        context_prefix: str = None
    ):
        """
        Inicializa el embedder de Sentence Transformers.
        
        Args:
            model_name: Nombre del modelo a usar
            batch_size: Tamaño de batch para encoding
            context_prefix: Prefijo de contexto para keywords
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Sentence Transformers no está instalado. "
                "Instala con: pip install sentence-transformers"
            )
        
        self.model_name = model_name or embedding_config.st_model_name
        self.batch_size = batch_size or embedding_config.st_batch_size
        self.context_prefix = context_prefix or embedding_config.st_context_prefix
        
        self._model: Optional[SentenceTransformer] = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Carga modelo (con caché de clase)."""
        if self._model is None:
            if self.model_name in self._model_cache:
                self._model = self._model_cache[self.model_name]
                logger.debug(f"Modelo {self.model_name} cargado desde caché")
            else:
                logger.info(f"Cargando modelo {self.model_name}...")
                self._model = SentenceTransformer(self.model_name)
                self._model_cache[self.model_name] = self._model
                logger.info(f"Modelo {self.model_name} cargado")
        
        return self._model
    
    def encode(
        self,
        keywords: List[str],
        add_context: bool = True,
        normalize: bool = True
    ) -> EmbeddingResult:
        """
        Genera embeddings para keywords.
        
        Args:
            keywords: Lista de keywords
            add_context: Añadir prefijo de contexto
            normalize: Normalizar embeddings
            
        Returns:
            Resultado con embeddings
        """
        # Preparar keywords con contexto
        if add_context and self.context_prefix:
            texts = [f"{self.context_prefix} {kw}" for kw in keywords]
        else:
            texts = keywords
        
        # Generar embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize
        )
        
        logger.info(f"Sentence Transformer embeddings: {embeddings.shape}")
        
        return EmbeddingResult(
            embeddings=embeddings,
            method=EmbeddingMethod.SENTENCE_TRANSFORMER,
            dimensions=embeddings.shape[1],
            metadata={
                "n_keywords": len(keywords),
                "model_name": self.model_name,
                "context_prefix": self.context_prefix if add_context else None,
                "normalized": normalize
            }
        )


class HybridEmbedder:
    """
    Generador de embeddings híbridos (TF-IDF + Sentence Transformers).
    """
    
    def __init__(
        self,
        tfidf_weight: float = None,
        semantic_weight: float = None,
        st_model_name: str = None
    ):
        """
        Inicializa el embedder híbrido.
        
        Args:
            tfidf_weight: Peso para embeddings TF-IDF
            semantic_weight: Peso para embeddings semánticos
            st_model_name: Modelo de Sentence Transformers
        """
        self.tfidf_weight = tfidf_weight or embedding_config.hybrid_tfidf_weight
        self.semantic_weight = semantic_weight or embedding_config.hybrid_semantic_weight
        
        self.tfidf_embedder = TFIDFEmbedder()
        self.st_embedder = SentenceTransformerEmbedder(model_name=st_model_name) if SENTENCE_TRANSFORMERS_AVAILABLE else None
    
    def encode(self, keywords: List[str]) -> EmbeddingResult:
        """
        Genera embeddings híbridos.
        
        Args:
            keywords: Lista de keywords
            
        Returns:
            Resultado con embeddings combinados
        """
        # Generar embeddings TF-IDF
        tfidf_result = self.tfidf_embedder.fit_transform(keywords)
        tfidf_emb = tfidf_result.embeddings
        
        # Normalizar TF-IDF
        tfidf_norm = tfidf_emb / (np.linalg.norm(tfidf_emb, axis=1, keepdims=True) + 1e-8)
        
        if self.st_embedder is None:
            logger.warning("Sentence Transformers no disponible, usando solo TF-IDF")
            return EmbeddingResult(
                embeddings=tfidf_norm,
                method=EmbeddingMethod.TFIDF,
                dimensions=tfidf_norm.shape[1],
                metadata={"fallback": "tfidf_only"}
            )
        
        # Generar embeddings semánticos
        st_result = self.st_embedder.encode(keywords)
        st_emb = st_result.embeddings
        
        # Alinear dimensionalidades
        if tfidf_norm.shape[1] > st_emb.shape[1]:
            pca = PCA(n_components=st_emb.shape[1])
            tfidf_aligned = pca.fit_transform(tfidf_norm)
        elif tfidf_norm.shape[1] < st_emb.shape[1]:
            padding = np.zeros((tfidf_norm.shape[0], st_emb.shape[1] - tfidf_norm.shape[1]))
            tfidf_aligned = np.hstack([tfidf_norm, padding])
        else:
            tfidf_aligned = tfidf_norm
        
        # Combinar con pesos
        combined = tfidf_aligned * self.tfidf_weight + st_emb * self.semantic_weight
        
        logger.info(f"Embeddings híbridos: {combined.shape}")
        
        return EmbeddingResult(
            embeddings=combined,
            method=EmbeddingMethod.HYBRID,
            dimensions=combined.shape[1],
            metadata={
                "n_keywords": len(keywords),
                "tfidf_weight": self.tfidf_weight,
                "semantic_weight": self.semantic_weight,
                "tfidf_dims": tfidf_emb.shape[1],
                "st_dims": st_emb.shape[1]
            }
        )


class EmbeddingManager:
    """
    Gestor central de embeddings con caché y selección de método.
    """
    
    def __init__(self, cache_enabled: bool = True, cache_dir: Path = None):
        """
        Inicializa el gestor de embeddings.
        
        Args:
            cache_enabled: Habilitar caché de embeddings
            cache_dir: Directorio de caché
        """
        self.cache_enabled = cache_enabled
        self.cache_dir = cache_dir or CACHE_DIR
        
        if cache_enabled:
            self.cache = CacheManager(self.cache_dir / "embeddings", ttl_seconds=7200)
        else:
            self.cache = None
        
        # Inicializar embedders
        self.tfidf = TFIDFEmbedder()
        self.st = SentenceTransformerEmbedder() if SENTENCE_TRANSFORMERS_AVAILABLE else None
        self.hybrid = HybridEmbedder() if SENTENCE_TRANSFORMERS_AVAILABLE else None
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Retorna métodos disponibles."""
        methods = ["TF-IDF (rápido)"]
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            methods.extend([
                "Sentence Transformers (semántico)",
                "Híbrido (TF-IDF + Semántico)"
            ])
        
        return methods
    
    def generate(
        self,
        keywords: List[str],
        method: str = "Sentence Transformers (semántico)",
        use_cache: bool = True
    ) -> EmbeddingResult:
        """
        Genera embeddings usando el método especificado.
        
        Args:
            keywords: Lista de keywords
            method: Método de embedding a usar
            use_cache: Usar caché si está disponible
            
        Returns:
            Resultado con embeddings
        """
        # Verificar caché
        if use_cache and self.cache:
            cache_key = f"emb_{method}_{calculate_hash(sorted(keywords))}"
            cached = self.cache.get(cache_key)
            if cached is not None:
                logger.info(f"Embeddings cargados desde caché")
                return cached
        
        # Generar embeddings según método
        if "TF-IDF" in method and "Híbrido" not in method:
            result = self.tfidf.fit_transform(keywords)
        
        elif "Sentence Transformers" in method:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("ST no disponible, usando TF-IDF")
                result = self.tfidf.fit_transform(keywords)
            else:
                result = self.st.encode(keywords)
        
        elif "Híbrido" in method:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("ST no disponible para híbrido, usando TF-IDF")
                result = self.tfidf.fit_transform(keywords)
            else:
                result = self.hybrid.encode(keywords)
        
        else:
            logger.warning(f"Método desconocido '{method}', usando TF-IDF")
            result = self.tfidf.fit_transform(keywords)
        
        # Guardar en caché
        if use_cache and self.cache:
            self.cache.set(cache_key, result)
        
        return result
    
    def add_features(
        self,
        base_embeddings: np.ndarray,
        product_features: Optional[np.ndarray] = None,
        audience_features: Optional[np.ndarray] = None,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Añade features adicionales a embeddings base.
        
        Args:
            base_embeddings: Embeddings base
            product_features: Features de producto
            audience_features: Features de audiencia
            weights: Pesos para cada componente
            
        Returns:
            Embeddings combinados
        """
        if weights is None:
            weights = {
                "base": 0.5,
                "product": 0.3,
                "audience": 0.2
            }
        
        # Normalizar base
        base_norm = base_embeddings / (np.linalg.norm(base_embeddings, axis=1, keepdims=True) + 1e-8)
        
        components = [base_norm * weights.get("base", 0.5)]
        
        if product_features is not None:
            components.append(product_features * weights.get("product", 0.3))
        
        if audience_features is not None:
            components.append(audience_features * weights.get("audience", 0.2))
        
        combined = np.hstack(components)
        
        logger.debug(f"Embeddings combinados: {combined.shape}")
        
        return combined


def check_embedding_availability() -> Dict[str, bool]:
    """
    Verifica disponibilidad de métodos de embedding.
    
    Returns:
        Diccionario con disponibilidad de cada método
    """
    return {
        "tfidf": True,
        "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE,
        "hybrid": SENTENCE_TRANSFORMERS_AVAILABLE
    }
