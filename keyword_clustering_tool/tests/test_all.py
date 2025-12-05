"""
Tests unitarios para Keyword Clustering Tool.
Ejecutar con: pytest tests/ -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Añadir directorio raíz al path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_keywords():
    """Keywords de ejemplo para tests."""
    return [
        "regalo navidad hombre",
        "ideas regalo mujer 30 años",
        "smartphone samsung galaxy",
        "iphone 15 pro max precio",
        "auriculares bluetooth baratos",
        "regalo niño 10 años",
        "playstation 5 oferta",
        "lego star wars",
        "freidora aire cosori",
        "smartwatch garmin"
    ]


@pytest.fixture
def sample_volumes():
    """Volúmenes de ejemplo."""
    return [1000, 800, 5000, 4000, 2000, 1500, 8000, 3000, 2500, 1800]


@pytest.fixture
def sample_dataframe(sample_keywords, sample_volumes):
    """DataFrame de ejemplo para tests."""
    return pd.DataFrame({
        'Keyword': sample_keywords,
        'volumen_navidad': sample_volumes,
        'vol_avg': sample_volumes
    })


# =============================================================================
# TESTS: UTILS
# =============================================================================

class TestUtils:
    """Tests para módulo de utilidades."""
    
    def test_clean_keyword(self):
        """Test limpieza de keywords."""
        from src.utils import clean_keyword
        
        assert clean_keyword("  REGALO Navidad  ") == "regalo navidad"
        assert clean_keyword("precio€100") == "precio 100"
        assert clean_keyword("") == ""
        assert clean_keyword(None) == ""
    
    def test_preprocess_keyword(self):
        """Test preprocesamiento avanzado."""
        from src.utils import preprocess_keyword
        
        result = preprocess_keyword("Móvil SAMSUNG")
        assert "movil" in result.lower()
        assert "samsung" in result.lower()
    
    def test_generate_url_slug(self):
        """Test generación de slugs."""
        from src.utils import generate_url_slug
        
        slug = generate_url_slug("Regalo para Mamá 50€")
        assert " " not in slug
        assert "€" not in slug
        assert "-" in slug
    
    def test_safe_convert_volume(self):
        """Test conversión segura de volúmenes."""
        from src.utils import safe_convert_volume
        
        assert safe_convert_volume(1000) == 1000
        assert safe_convert_volume("1,500") == 1500
        assert safe_convert_volume("2.000") == 2000
        assert safe_convert_volume(None) == 0
        assert safe_convert_volume("invalid") == 0
    
    def test_format_number(self):
        """Test formateo de números."""
        from src.utils import format_number
        
        assert format_number(1000) == "1.000"
        assert format_number(1000000) == "1.000.000"


# =============================================================================
# TESTS: MATCHING
# =============================================================================

class TestProductMatcher:
    """Tests para matching de productos."""
    
    def test_match_smartphone(self):
        """Test match de smartphone."""
        from src.matching import ProductMatcher
        
        matcher = ProductMatcher()
        matches = matcher.match("iphone 15 pro max")
        
        assert len(matches) > 0
        assert matches[0].id == "movil"
    
    def test_match_gaming(self):
        """Test match de gaming."""
        from src.matching import ProductMatcher
        
        matcher = ProductMatcher()
        matches = matcher.match("playstation 5")
        
        assert len(matches) > 0
        assert matches[0].id in ["consolas", "gaming"]
    
    def test_no_match(self):
        """Test sin match."""
        from src.matching import ProductMatcher
        
        matcher = ProductMatcher()
        matches = matcher.match("asdfghjkl")
        
        assert len(matches) == 0
    
    def test_match_score(self):
        """Test score de match."""
        from src.matching import ProductMatcher
        
        matcher = ProductMatcher()
        score = matcher.get_match_score("smartphone samsung galaxy")
        
        assert 0 <= score <= 1
        assert score > 0


class TestAudienceMatcher:
    """Tests para matching de audiencias."""
    
    def test_match_hombre(self):
        """Test match de género hombre."""
        from src.matching import AudienceMatcher
        
        matcher = AudienceMatcher()
        matches = matcher.match("regalo para hombre")
        
        assert len(matches) > 0
        assert any(m.id == "hombre" for m in matches)
    
    def test_match_by_type(self):
        """Test match por tipo."""
        from src.matching import AudienceMatcher
        
        matcher = AudienceMatcher()
        result = matcher.get_by_type("regalo para madre navidad")
        
        assert result['relacion'] is not None
        assert result['ocasion'] is not None
    
    def test_no_audience_match(self):
        """Test sin match de audiencia."""
        from src.matching import AudienceMatcher
        
        matcher = AudienceMatcher()
        matches = matcher.match("ordenador portatil")
        
        # Puede no tener match de audiencia
        assert isinstance(matches, list)


class TestIntentClassifier:
    """Tests para clasificación de intent."""
    
    def test_transactional_intent(self):
        """Test intent transaccional."""
        from src.matching import IntentClassifier
        
        intent = IntentClassifier.classify("comprar iphone barato")
        assert "Transaccional" in intent
    
    def test_informational_intent(self):
        """Test intent informacional."""
        from src.matching import IntentClassifier
        
        intent = IntentClassifier.classify("que es un smartwatch")
        assert "Informacional" in intent
    
    def test_gift_intent(self):
        """Test intent de regalo."""
        from src.matching import IntentClassifier
        
        intent = IntentClassifier.classify("regalo navidad padre")
        assert "Regalo" in intent


# =============================================================================
# TESTS: EMBEDDINGS
# =============================================================================

class TestTFIDFEmbedder:
    """Tests para embeddings TF-IDF."""
    
    def test_basic_embedding(self, sample_keywords):
        """Test generación básica de embeddings."""
        from src.embeddings import TFIDFEmbedder
        
        embedder = TFIDFEmbedder()
        result = embedder.fit_transform(sample_keywords)
        
        assert result.embeddings.shape[0] == len(sample_keywords)
        assert result.embeddings.shape[1] > 0
        assert result.method.value == "tfidf"
    
    def test_embedding_dimensions(self, sample_keywords):
        """Test dimensiones de embeddings."""
        from src.embeddings import TFIDFEmbedder
        
        embedder = TFIDFEmbedder(max_features=100)
        result = embedder.fit_transform(sample_keywords)
        
        assert result.embeddings.shape[1] <= 100


class TestEmbeddingManager:
    """Tests para gestor de embeddings."""
    
    def test_available_methods(self):
        """Test métodos disponibles."""
        from src.embeddings import EmbeddingManager
        
        methods = EmbeddingManager.get_available_methods()
        
        assert len(methods) > 0
        assert "TF-IDF" in methods[0]
    
    def test_generate_tfidf(self, sample_keywords):
        """Test generación con TF-IDF."""
        from src.embeddings import EmbeddingManager
        
        manager = EmbeddingManager(cache_enabled=False)
        result = manager.generate(sample_keywords, method="TF-IDF (rápido)")
        
        assert result.embeddings.shape[0] == len(sample_keywords)


# =============================================================================
# TESTS: CLUSTERING
# =============================================================================

class TestKMeansClusterer:
    """Tests para K-Means."""
    
    def test_basic_clustering(self):
        """Test clustering básico."""
        from src.clustering import KMeansClusterer
        
        # Crear embeddings de prueba
        embeddings = np.random.rand(50, 10)
        
        clusterer = KMeansClusterer(n_clusters=5)
        labels = clusterer.fit_predict(embeddings)
        
        assert len(labels) == 50
        assert len(set(labels)) == 5
    
    def test_auto_reduce_clusters(self):
        """Test reducción automática de clusters."""
        from src.clustering import KMeansClusterer
        
        embeddings = np.random.rand(10, 5)
        
        clusterer = KMeansClusterer(n_clusters=20)  # Más clusters que muestras
        labels = clusterer.fit_predict(embeddings)
        
        assert len(set(labels)) <= 10


class TestHierarchicalClusterer:
    """Tests para clustering jerárquico."""
    
    def test_basic_clustering(self):
        """Test clustering básico."""
        from src.clustering import HierarchicalClusterer
        
        embeddings = np.random.rand(30, 8)
        
        clusterer = HierarchicalClusterer(n_clusters=3)
        labels = clusterer.fit_predict(embeddings)
        
        assert len(labels) == 30
        assert len(set(labels)) == 3


class TestClusteringManager:
    """Tests para gestor de clustering."""
    
    def test_available_methods(self):
        """Test métodos disponibles."""
        from src.clustering import ClusteringManager
        
        methods = ClusteringManager.get_available_methods()
        
        assert "K-Means" in methods
        assert "Jerárquico" in methods
    
    def test_available_modes(self):
        """Test modos disponibles."""
        from src.clustering import ClusteringManager
        
        modes = ClusteringManager.get_available_modes()
        
        assert "Solo semántico" in modes
        assert "Guiado por productos" in modes
    
    def test_semantic_clustering(self, sample_dataframe):
        """Test clustering semántico."""
        from src.clustering import ClusteringManager
        from src.matching import enrich_keywords_full
        
        # Enriquecer datos
        df = enrich_keywords_full(sample_dataframe)
        
        # Crear embeddings simples
        embeddings = np.random.rand(len(df), 10)
        
        manager = ClusteringManager()
        result = manager.cluster(
            embeddings=embeddings,
            df=df,
            method="K-Means",
            mode="Solo semántico",
            n_clusters=3
        )
        
        assert result.n_clusters == 3
        assert len(result.labels) == len(df)
        assert len(result.cluster_info) == 3


# =============================================================================
# TESTS: DATA LOADER
# =============================================================================

class TestKeywordDataLoader:
    """Tests para carga de datos."""
    
    def test_load_from_dataframe(self, sample_dataframe):
        """Test carga desde DataFrame."""
        from src.data_loader import KeywordDataLoader
        
        loader = KeywordDataLoader()
        df = loader.load(sample_dataframe)
        
        assert len(df) == len(sample_dataframe)
        assert 'Keyword' in df.columns
    
    def test_validation(self, sample_dataframe):
        """Test validación de datos."""
        from src.data_loader import KeywordDataLoader
        
        loader = KeywordDataLoader()
        loader.load(sample_dataframe)
        
        result = loader.validate()
        
        assert result.is_valid
        assert result.stats['total_rows'] == len(sample_dataframe)
    
    def test_filter_by_volume(self, sample_dataframe):
        """Test filtrado por volumen."""
        from src.data_loader import KeywordDataLoader
        
        loader = KeywordDataLoader()
        df = loader.load(sample_dataframe)
        df['volumen_navidad'] = df['vol_avg']
        loader.df = df
        
        filtered = loader.filter_by_volume(2000)
        
        assert len(filtered) < len(df)
        assert filtered['volumen_navidad'].min() >= 2000


# =============================================================================
# TESTS: INTEGRATION
# =============================================================================

class TestIntegration:
    """Tests de integración end-to-end."""
    
    def test_full_pipeline(self, sample_dataframe):
        """Test pipeline completo."""
        from src.data_loader import KeywordDataLoader
        from src.matching import enrich_keywords_full
        from src.embeddings import EmbeddingManager
        from src.clustering import ClusteringManager
        
        # 1. Cargar datos
        loader = KeywordDataLoader()
        df = loader.load(sample_dataframe)
        df['volumen_navidad'] = df['vol_avg']
        
        # 2. Enriquecer
        df = enrich_keywords_full(df)
        
        assert 'best_product_family' in df.columns
        assert 'primary_audience' in df.columns
        assert 'intent' in df.columns
        
        # 3. Generar embeddings
        emb_manager = EmbeddingManager(cache_enabled=False)
        emb_result = emb_manager.generate(
            df['Keyword'].tolist(),
            method="TF-IDF (rápido)"
        )
        
        assert emb_result.embeddings.shape[0] == len(df)
        
        # 4. Clustering
        clust_manager = ClusteringManager()
        clust_result = clust_manager.cluster(
            embeddings=emb_result.embeddings,
            df=df,
            method="K-Means",
            mode="Solo semántico",
            n_clusters=3
        )
        
        assert clust_result.n_clusters == 3
        assert len(clust_result.coherences) == 3
        
        # 5. Verificar cluster info
        for cluster_id, info in clust_result.cluster_info.items():
            assert info.size > 0
            assert info.total_volume >= 0
            assert 0 <= info.coherence <= 1


# =============================================================================
# TESTS: CONFIGURATION
# =============================================================================

class TestConfiguration:
    """Tests para configuración."""
    
    def test_load_products(self):
        """Test carga de productos."""
        from config.settings import ConfigLoader
        
        products = ConfigLoader.load_products()
        
        assert len(products) > 0
        assert 'movil' in products
        assert 'nombre' in products['movil']
        assert 'keywords_match' in products['movil']
    
    def test_load_audiences(self):
        """Test carga de audiencias."""
        from config.settings import ConfigLoader
        
        audiences = ConfigLoader.load_audiences()
        
        assert len(audiences) > 0
        assert 'hombre' in audiences
        assert 'tipo' in audiences['hombre']
    
    def test_get_product_families(self):
        """Test lista de familias."""
        from config.settings import ConfigLoader
        
        families = ConfigLoader.get_product_families()
        
        assert len(families) > 0
        assert 'movil' in families
    
    def test_get_audience_types(self):
        """Test tipos de audiencia."""
        from config.settings import ConfigLoader
        
        types = ConfigLoader.get_audience_types()
        
        assert len(types) > 0
        assert 'genero' in types
        assert 'edad' in types


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
