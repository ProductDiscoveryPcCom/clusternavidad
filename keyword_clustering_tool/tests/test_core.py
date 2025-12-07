"""
Tests unitarios para Keyword Clustering Tool.
Ejecutar con: pytest tests/ -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.utils import (
    clean_keyword,
    preprocess_keyword,
    normalize_accents,
    generate_url_slug,
    safe_convert_volume,
    extract_tokens
)
from src.matching import (
    ProductMatcher,
    AudienceMatcher,
    IntentClassifier,
    GiftRecipientExtractor,
    PriceRangeExtractor
)
from src.data_loader import KeywordDataLoader, DataValidationResult
from config.settings import ConfigLoader


# ============================================================================
# TESTS DE UTILIDADES
# ============================================================================

class TestUtils:
    
    def test_clean_keyword_basic(self):
        assert clean_keyword("  Portátil Gaming  ") == "portátil gaming"
        assert clean_keyword("iPhone 15 Pro!!!") == "iphone 15 pro"
        assert clean_keyword("") == ""
        assert clean_keyword(None) == ""
    
    def test_normalize_accents(self):
        assert normalize_accents("portátil") == "portatil"
        assert normalize_accents("móvil") == "movil"
        assert normalize_accents("año") == "ano"
        assert normalize_accents("niño") == "nino"
    
    def test_preprocess_keyword(self):
        assert "portatil" in preprocess_keyword("Portátil", advanced=True)
        assert "movil" in preprocess_keyword("Móvil", advanced=True)
    
    def test_generate_url_slug(self):
        assert generate_url_slug("Regalos para Hombre") == "regalos-para-hombre"
        assert generate_url_slug("PC Gaming €500") == "pc-gaming-500"
        assert "-" not in generate_url_slug("test", max_length=50)[-1]  # No termina en guión
    
    def test_safe_convert_volume(self):
        assert safe_convert_volume("1,000") == 1000
        assert safe_convert_volume("500") == 500
        assert safe_convert_volume(100) == 100
        assert safe_convert_volume(None) == 0
        assert safe_convert_volume("") == 0
        assert safe_convert_volume("abc") == 0
    
    def test_extract_tokens(self):
        tokens = extract_tokens("regalos tecnología hombre")
        assert "regalos" in tokens
        assert "tecnología" not in tokens  # Normalizado
        assert "de" not in tokens  # Stopword


# ============================================================================
# TESTS DE MATCHING
# ============================================================================

class TestProductMatcher:
    
    @pytest.fixture
    def matcher(self):
        return ProductMatcher()
    
    def test_match_mobile(self, matcher):
        matches = matcher.match("iphone 15 pro")
        assert len(matches) > 0
        assert matches[0].id == "movil"
    
    def test_match_gaming(self, matcher):
        matches = matcher.match("teclado gaming rgb")
        assert len(matches) > 0
        assert "gaming" in [m.id for m in matches]
    
    def test_match_no_result(self, matcher):
        matches = matcher.match("zapatos deportivos")
        # Puede no tener match o tener match bajo
        if matches:
            assert matches[0].score < 2  # Score bajo
    
    def test_get_best_family_id(self, matcher):
        assert matcher.get_best_family_id("samsung galaxy") == "movil"
        assert matcher.get_best_family_id("playstation 5") == "consolas"
    
    def test_has_match(self, matcher):
        assert matcher.has_match("portátil gaming") == True
        assert matcher.has_match("libro de cocina") == False  # Probablemente no matchea


class TestAudienceMatcher:
    
    @pytest.fixture
    def matcher(self):
        return AudienceMatcher()
    
    def test_match_gender(self, matcher):
        matches = matcher.match("regalos para hombre")
        assert len(matches) > 0
        assert any(m.id == "hombre" for m in matches)
    
    def test_match_age(self, matcher):
        matches = matcher.match("juguetes para niños")
        assert len(matches) > 0
        # Debe detectar niño
        ids = [m.id for m in matches]
        assert "nino" in ids or "juguetes" in str(matches)
    
    def test_get_by_type(self, matcher):
        result = matcher.get_by_type("regalo padre navidad")
        assert result['relacion'] is not None or result['ocasion'] is not None


class TestIntentClassifier:
    
    def test_transactional(self):
        assert "Transaccional" in IntentClassifier.classify("comprar iphone barato")
        assert "Transaccional" in IntentClassifier.classify("precio ps5")
    
    def test_informational(self):
        assert "Informacional" in IntentClassifier.classify("mejor portátil 2024")
        assert "Informacional" in IntentClassifier.classify("comparativa iphone vs samsung")
    
    def test_gift(self):
        assert "Regalo" in IntentClassifier.classify("regalos navidad hombre")
    
    def test_general(self):
        assert "General" in IntentClassifier.classify("smartphone android")


class TestGiftRecipientExtractor:
    
    def test_extract_father(self):
        assert GiftRecipientExtractor.extract("regalo padre navidad") == "padre"
    
    def test_extract_mother(self):
        assert GiftRecipientExtractor.extract("regalo mama") == "madre"
    
    def test_extract_none(self):
        assert GiftRecipientExtractor.extract("portátil gaming") is None


class TestPriceRangeExtractor:
    
    def test_extract_low(self):
        result = PriceRangeExtractor.extract("regalos menos de 20 euros")
        assert "30€" in result
    
    def test_extract_medium(self):
        result = PriceRangeExtractor.extract("regalo hasta 50 euros")
        assert "60€" in result
    
    def test_extract_none(self):
        assert PriceRangeExtractor.extract("portátil gaming") is None


# ============================================================================
# TESTS DE CONFIG
# ============================================================================

class TestConfigLoader:
    
    def test_load_products(self):
        products = ConfigLoader.load_products()
        assert isinstance(products, dict)
        assert len(products) > 0
        assert "movil" in products
        assert "nombre" in products["movil"]
    
    def test_load_audiences(self):
        audiences = ConfigLoader.load_audiences()
        assert isinstance(audiences, dict)
        assert len(audiences) > 0
        assert "hombre" in audiences
        assert "tipo" in audiences["hombre"]
    
    def test_get_product_families(self):
        families = ConfigLoader.get_product_families()
        assert "movil" in families
        assert "gaming" in families
    
    def test_get_audience_types(self):
        types = ConfigLoader.get_audience_types()
        assert "genero" in types
        assert "edad" in types


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegration:
    
    def test_full_enrichment_pipeline(self):
        """Test del pipeline completo de enriquecimiento."""
        from src.matching import enrich_keywords_full
        
        # Crear DataFrame de prueba
        df = pd.DataFrame({
            'Keyword': [
                'iphone 15 pro',
                'regalo padre navidad',
                'portátil gaming barato',
                'regalos navidad niños'
            ]
        })
        
        # Enriquecer
        df_enriched = enrich_keywords_full(df, keyword_col='Keyword')
        
        # Verificar columnas añadidas
        assert 'best_product_family' in df_enriched.columns
        assert 'primary_audience' in df_enriched.columns
        assert 'intent' in df_enriched.columns
        assert 'has_product_match' in df_enriched.columns
        
        # Verificar valores
        assert df_enriched[df_enriched['Keyword'] == 'iphone 15 pro']['best_product_family'].iloc[0] == 'movil'


# ============================================================================
# EJECUTAR TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
