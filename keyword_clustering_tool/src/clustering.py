"""
Módulo para algoritmos de clustering de keywords.
Soporta K-Means, Jerárquico y HDBSCAN.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

from .utils import generate_url_slug
from config.settings import clustering_config, ConfigLoader

logger = logging.getLogger("keyword_clustering.clustering")

# Importaciones opcionales
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN no disponible")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP no disponible")


class ClusteringMethod(Enum):
    """Métodos de clustering disponibles."""
    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    HDBSCAN = "hdbscan"


class ClusteringMode(Enum):
    """Modos de agrupación."""
    SEMANTIC_ONLY = "semantic_only"
    PRODUCT_GUIDED = "product_guided"
    AUDIENCE_GUIDED = "audience_guided"
    HYBRID_PRODUCT = "hybrid_product"
    HYBRID_COMPLETE = "hybrid_complete"


@dataclass
class ClusterInfo:
    """Información de un cluster."""
    id: int
    name: str
    emoji: str
    size: int
    total_volume: int
    avg_volume: float
    coherence: float
    top_keywords: List[str] = field(default_factory=list)
    top_family: Optional[str] = None
    top_audience: Optional[str] = None
    url_suggestion: str = ""


@dataclass
class ClusteringResult:
    """Resultado de clustering."""
    labels: np.ndarray
    n_clusters: int
    method: ClusteringMethod
    mode: ClusteringMode
    cluster_info: Dict[int, ClusterInfo]
    coherences: Dict[int, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class KMeansClusterer:
    """
    Clustering con K-Means.
    """
    
    def __init__(
        self,
        n_clusters: int = 15,
        n_init: int = None,
        random_state: int = None
    ):
        """
        Inicializa K-Means clusterer.
        
        Args:
            n_clusters: Número de clusters
            n_init: Número de inicializaciones
            random_state: Semilla aleatoria
        """
        self.n_clusters = n_clusters
        self.n_init = n_init or clustering_config.kmeans_n_init
        self.random_state = random_state or clustering_config.kmeans_random_state
        
        self.model: Optional[KMeans] = None
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Ajusta modelo y predice clusters.
        
        Args:
            embeddings: Matriz de embeddings
            
        Returns:
            Array de etiquetas de cluster
        """
        n_samples = len(embeddings)
        n_clusters = min(self.n_clusters, n_samples)
        
        self.model = KMeans(
            n_clusters=n_clusters,
            n_init=self.n_init,
            random_state=self.random_state
        )
        
        labels = self.model.fit_predict(embeddings)
        
        logger.info(f"K-Means: {n_clusters} clusters generados")
        
        return labels


class HierarchicalClusterer:
    """
    Clustering jerárquico (Agglomerative).
    """
    
    def __init__(
        self,
        n_clusters: int = 15,
        metric: str = 'cosine',
        linkage: str = 'average'
    ):
        """
        Inicializa clusterer jerárquico.
        
        Args:
            n_clusters: Número de clusters
            metric: Métrica de distancia
            linkage: Método de enlace
        """
        self.n_clusters = n_clusters
        self.metric = metric
        self.linkage = linkage
        
        self.model: Optional[AgglomerativeClustering] = None
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Ajusta modelo y predice clusters.
        
        Args:
            embeddings: Matriz de embeddings
            
        Returns:
            Array de etiquetas de cluster
        """
        n_samples = len(embeddings)
        n_clusters = min(self.n_clusters, n_samples)
        
        self.model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric=self.metric,
            linkage=self.linkage
        )
        
        labels = self.model.fit_predict(embeddings)
        
        logger.info(f"Jerárquico: {n_clusters} clusters generados")
        
        return labels


class HDBSCANClusterer:
    """
    Clustering con HDBSCAN (auto-detección de clusters).
    """
    
    def __init__(
        self,
        min_cluster_size: int = None,
        min_samples: int = None,
        metric: str = None,
        cluster_selection_method: str = None
    ):
        """
        Inicializa HDBSCAN clusterer.
        
        Args:
            min_cluster_size: Tamaño mínimo de cluster
            min_samples: Muestras mínimas
            metric: Métrica de distancia
            cluster_selection_method: Método de selección
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN no está instalado. "
                "Instala con: pip install hdbscan"
            )
        
        self.min_cluster_size = min_cluster_size or clustering_config.hdbscan_min_cluster_size
        self.min_samples = min_samples or clustering_config.hdbscan_min_samples
        self.metric = metric or clustering_config.hdbscan_metric
        self.cluster_selection_method = cluster_selection_method or clustering_config.hdbscan_cluster_selection_method
        
        self.model = None
    
    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Ajusta modelo y predice clusters.
        
        Args:
            embeddings: Matriz de embeddings
            
        Returns:
            Array de etiquetas de cluster (-1 para outliers)
        """
        # Reducir dimensionalidad si es necesario
        if embeddings.shape[1] > 50:
            if UMAP_AVAILABLE:
                reducer = umap.UMAP(
                    n_components=min(50, embeddings.shape[1]),
                    metric='cosine',
                    random_state=42
                )
                embeddings_reduced = reducer.fit_transform(embeddings)
            else:
                pca = PCA(n_components=min(50, embeddings.shape[1]))
                embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
        
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_method=self.cluster_selection_method
        )
        
        labels = self.model.fit_predict(embeddings_reduced)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_outliers = (labels == -1).sum()
        
        logger.info(f"HDBSCAN: {n_clusters} clusters, {n_outliers} outliers")
        
        return labels


class ClusteringManager:
    """
    Gestor central de clustering con múltiples modos y métodos.
    """
    
    def __init__(self):
        """Inicializa el gestor de clustering."""
        self.products = ConfigLoader.load_products()
        self.audiences = ConfigLoader.load_audiences()
    
    @staticmethod
    def get_available_methods() -> List[str]:
        """Retorna métodos disponibles."""
        methods = ["K-Means", "Jerárquico"]
        
        if HDBSCAN_AVAILABLE:
            methods.append("HDBSCAN (auto-clusters)")
        
        return methods
    
    @staticmethod
    def get_available_modes() -> List[str]:
        """Retorna modos disponibles."""
        return [
            "Solo semántico",
            "Guiado por productos",
            "Guiado por audiencia",
            "Híbrido (Semántico + Productos)",
            "Híbrido Completo"
        ]
    
    def cluster(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        method: str = "K-Means",
        mode: str = "Solo semántico",
        n_clusters: int = 15,
        min_cluster_size: int = 5
    ) -> ClusteringResult:
        """
        Ejecuta clustering con el método y modo especificados.
        
        Args:
            embeddings: Embeddings de keywords
            df: DataFrame con datos de keywords
            method: Método de clustering
            mode: Modo de agrupación
            n_clusters: Número de clusters (para K-Means y Jerárquico)
            min_cluster_size: Tamaño mínimo (para HDBSCAN)
            
        Returns:
            Resultado de clustering
        """
        # Determinar modo enum
        mode_map = {
            "Solo semántico": ClusteringMode.SEMANTIC_ONLY,
            "Guiado por productos": ClusteringMode.PRODUCT_GUIDED,
            "Guiado por audiencia": ClusteringMode.AUDIENCE_GUIDED,
            "Híbrido (Semántico + Productos)": ClusteringMode.HYBRID_PRODUCT,
            "Híbrido Completo": ClusteringMode.HYBRID_COMPLETE
        }
        clustering_mode = mode_map.get(mode, ClusteringMode.SEMANTIC_ONLY)
        
        # Aplicar modo guiado si corresponde
        if clustering_mode == ClusteringMode.PRODUCT_GUIDED:
            return self._cluster_by_products(embeddings, df)
        
        elif clustering_mode == ClusteringMode.AUDIENCE_GUIDED:
            return self._cluster_by_audience(embeddings, df)
        
        elif clustering_mode in [ClusteringMode.HYBRID_PRODUCT, ClusteringMode.HYBRID_COMPLETE]:
            embeddings = self._add_hybrid_features(
                embeddings, df,
                include_audience=(clustering_mode == ClusteringMode.HYBRID_COMPLETE)
            )
        
        # Aplicar método de clustering
        if "HDBSCAN" in method and HDBSCAN_AVAILABLE:
            clusterer = HDBSCANClusterer(min_cluster_size=min_cluster_size)
            method_enum = ClusteringMethod.HDBSCAN
        elif "Jerárquico" in method:
            clusterer = HierarchicalClusterer(n_clusters=n_clusters)
            method_enum = ClusteringMethod.HIERARCHICAL
        else:
            clusterer = KMeansClusterer(n_clusters=n_clusters)
            method_enum = ClusteringMethod.KMEANS
        
        labels = clusterer.fit_predict(embeddings)
        
        # Calcular coherencia
        coherences = self._calculate_coherences(embeddings, labels)
        
        # Generar información de clusters
        cluster_info = self._generate_cluster_info(df, labels, coherences)
        
        return ClusteringResult(
            labels=labels,
            n_clusters=len(set(labels)) - (1 if -1 in labels else 0),
            method=method_enum,
            mode=clustering_mode,
            cluster_info=cluster_info,
            coherences=coherences,
            metadata={"n_keywords": len(labels)}
        )
    
    def _cluster_by_products(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame
    ) -> ClusteringResult:
        """Clustering guiado por productos."""
        family_to_cluster = {fam: i for i, fam in enumerate(self.products.keys())}
        family_to_cluster[None] = len(family_to_cluster)
        
        labels = df['best_product_family'].map(
            lambda x: family_to_cluster.get(x, family_to_cluster[None])
        ).values
        
        # Generar nombres de clusters
        cluster_names = {}
        for fam_id, clust_id in family_to_cluster.items():
            if fam_id and fam_id in self.products:
                fam_data = self.products[fam_id]
                cluster_names[clust_id] = (
                    f"{fam_data['emoji']} {fam_data['nombre']}"
                )
            else:
                cluster_names[clust_id] = "🔍 Sin match"
        
        coherences = self._calculate_coherences(embeddings, labels)
        cluster_info = self._generate_cluster_info(
            df, labels, coherences, predefined_names=cluster_names
        )
        
        return ClusteringResult(
            labels=labels,
            n_clusters=len(set(labels)),
            method=ClusteringMethod.KMEANS,
            mode=ClusteringMode.PRODUCT_GUIDED,
            cluster_info=cluster_info,
            coherences=coherences,
            metadata={"guided_by": "products"}
        )
    
    def _cluster_by_audience(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame
    ) -> ClusteringResult:
        """Clustering guiado por audiencia."""
        aud_to_cluster = {aud: i for i, aud in enumerate(self.audiences.keys())}
        aud_to_cluster[None] = len(aud_to_cluster)
        
        def get_aud_id(primary):
            if primary is None:
                return None
            for cat_id, cat_data in self.audiences.items():
                if cat_data["nombre"] == primary:
                    return cat_id
            return None
        
        audience_ids = df['primary_audience'].apply(get_aud_id)
        labels = audience_ids.map(
            lambda x: aud_to_cluster.get(x, aud_to_cluster[None])
        ).values
        
        # Generar nombres de clusters
        cluster_names = {}
        for aud_id, clust_id in aud_to_cluster.items():
            if aud_id and aud_id in self.audiences:
                aud_data = self.audiences[aud_id]
                cluster_names[clust_id] = (
                    f"{aud_data['emoji']} {aud_data['nombre']}"
                )
            else:
                cluster_names[clust_id] = "🔍 Sin audiencia"
        
        coherences = self._calculate_coherences(embeddings, labels)
        cluster_info = self._generate_cluster_info(
            df, labels, coherences, predefined_names=cluster_names
        )
        
        return ClusteringResult(
            labels=labels,
            n_clusters=len(set(labels)),
            method=ClusteringMethod.KMEANS,
            mode=ClusteringMode.AUDIENCE_GUIDED,
            cluster_info=cluster_info,
            coherences=coherences,
            metadata={"guided_by": "audiences"}
        )
    
    def _add_hybrid_features(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        include_audience: bool = False
    ) -> np.ndarray:
        """Añade features de producto y audiencia a embeddings."""
        family_ids = list(self.products.keys())
        product_features = np.zeros((len(df), len(family_ids) + 1))
        
        for i, (_, row) in enumerate(df.iterrows()):
            if row.get('best_product_family') and row['best_product_family'] in family_ids:
                idx = family_ids.index(row['best_product_family'])
                score = row.get('product_match_score', 0.5)
                product_features[i, idx] = score * 2
            else:
                product_features[i, -1] = 0.3
        
        # Normalizar embeddings base
        emb_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        if include_audience:
            audience_ids = list(self.audiences.keys())
            audience_features = np.zeros((len(df), len(audience_ids) + 1))
            
            for i, (_, row) in enumerate(df.iterrows()):
                matches = row.get('audience_matches', [])
                if matches:
                    for match in matches[:3]:
                        if hasattr(match, 'id') and match.id in audience_ids:
                            idx = audience_ids.index(match.id)
                            audience_features[i, idx] = match.score / 5
                        elif isinstance(match, dict) and match.get('category_id') in audience_ids:
                            idx = audience_ids.index(match['category_id'])
                            audience_features[i, idx] = match.get('score', 1) / 5
                else:
                    audience_features[i, -1] = 0.3
            
            combined = np.hstack([
                emb_norm * 0.5,
                product_features * 0.3,
                audience_features * 0.2
            ])
        else:
            combined = np.hstack([
                emb_norm * 0.6,
                product_features * 0.4
            ])
        
        return combined
    
    def _calculate_coherences(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, float]:
        """Calcula coherencia de cada cluster."""
        coherences = {}
        
        for cluster_id in np.unique(labels):
            mask = labels == cluster_id
            cluster_embeddings = embeddings[mask]
            
            if len(cluster_embeddings) > 1:
                similarities = cosine_similarity(cluster_embeddings)
                np.fill_diagonal(similarities, 0)
                coherence = similarities.sum() / (
                    len(cluster_embeddings) * (len(cluster_embeddings) - 1)
                )
                coherences[cluster_id] = float(coherence)
            else:
                coherences[cluster_id] = 1.0
        
        return coherences
    
    def _generate_cluster_info(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        coherences: Dict[int, float],
        predefined_names: Optional[Dict[int, str]] = None
    ) -> Dict[int, ClusterInfo]:
        """Genera información detallada de cada cluster."""
        df = df.copy()
        df['cluster_id'] = labels
        
        cluster_info = {}
        
        for cluster_id in np.unique(labels):
            cluster_df = df[df['cluster_id'] == cluster_id]
            
            # Estadísticas básicas
            size = len(cluster_df)
            total_volume = int(cluster_df.get('volumen_navidad', cluster_df.get('vol_avg', pd.Series([0]))).sum())
            avg_volume = total_volume / size if size > 0 else 0
            
            # Top keywords
            if 'volumen_navidad' in cluster_df.columns:
                top_kws = cluster_df.nlargest(5, 'volumen_navidad')['Keyword'].tolist()
            else:
                top_kws = cluster_df['Keyword'].head(5).tolist()
            
            # Familia dominante
            top_family = None
            if 'familia_producto' in cluster_df.columns:
                family_counts = cluster_df['familia_producto'].value_counts()
                if len(family_counts) > 0:
                    top_family = family_counts.index[0]
                    if top_family == "Sin match":
                        top_family = None
            
            # Audiencia dominante
            top_audience = None
            if 'primary_audience' in cluster_df.columns:
                audience_counts = cluster_df['primary_audience'].value_counts()
                if len(audience_counts) > 0 and pd.notna(audience_counts.index[0]):
                    top_audience = audience_counts.index[0]
            
            # Nombre del cluster
            if predefined_names and cluster_id in predefined_names:
                name = predefined_names[cluster_id]
                emoji = name.split()[0] if name else "📦"
            else:
                name, emoji = self._generate_cluster_name(
                    top_kws[0] if top_kws else "Cluster",
                    top_family,
                    top_audience,
                    cluster_id
                )
            
            # URL sugerida
            url = generate_url_slug(top_kws[0] if top_kws else f"cluster-{cluster_id}")
            url_suggestion = f"/regalos-navidad/{url}/"
            
            cluster_info[cluster_id] = ClusterInfo(
                id=cluster_id,
                name=name,
                emoji=emoji,
                size=size,
                total_volume=total_volume,
                avg_volume=avg_volume,
                coherence=coherences.get(cluster_id, 0.0),
                top_keywords=top_kws,
                top_family=top_family,
                top_audience=top_audience,
                url_suggestion=url_suggestion
            )
        
        return cluster_info
    
    def _generate_cluster_name(
        self,
        top_keyword: str,
        top_family: Optional[str],
        top_audience: Optional[str],
        cluster_id: int
    ) -> Tuple[str, str]:
        """Genera nombre y emoji para un cluster."""
        emoji = "📦"
        prefix = ""
        
        # Prioridad: familia > audiencia > keyword
        if top_family:
            for fam_id, fam_data in self.products.items():
                if fam_data['nombre'] == top_family:
                    emoji = fam_data.get('emoji', '📦')
                    prefix = top_family
                    break
        
        elif top_audience:
            for aud_id, aud_data in self.audiences.items():
                if aud_data['nombre'] == top_audience:
                    emoji = aud_data.get('emoji', '🎁')
                    prefix = top_audience
                    break
        
        if prefix:
            name = f"{emoji} {prefix}: {top_keyword[:25]}"
        else:
            name = f"C{cluster_id}: {top_keyword[:35]}"
        
        return name, emoji


def check_clustering_availability() -> Dict[str, bool]:
    """
    Verifica disponibilidad de métodos de clustering.
    
    Returns:
        Diccionario con disponibilidad de cada método
    """
    return {
        "kmeans": True,
        "hierarchical": True,
        "hdbscan": HDBSCAN_AVAILABLE,
        "umap": UMAP_AVAILABLE
    }
