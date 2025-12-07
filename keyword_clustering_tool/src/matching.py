"""
Módulo para matching de keywords con productos y audiencias.
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pandas as pd

from .utils import preprocess_keyword, clean_keyword
from config.settings import ConfigLoader

logger = logging.getLogger("keyword_clustering.matching")


@dataclass
class MatchResult:
    """Resultado de un match."""
    id: str
    name: str
    emoji: str
    score: float
    matched_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AudienceMatchResult(MatchResult):
    """Resultado de match de audiencia con tipo."""
    tipo: str = ""


class ProductMatcher:
    """
    Matcher de keywords con familias de productos.
    """
    
    def __init__(self, products: Optional[Dict] = None):
        """
        Inicializa el matcher.
        
        Args:
            products: Diccionario de productos (usa config si None)
        """
        self.products = products or ConfigLoader.load_products()
        self._build_index()
    
    def _build_index(self) -> None:
        """Construye índice invertido para búsqueda rápida."""
        self._term_to_families: Dict[str, List[str]] = {}
        
        for family_id, family_data in self.products.items():
            for term in family_data.get("keywords_match", []):
                term_normalized = preprocess_keyword(term)
                
                if term_normalized not in self._term_to_families:
                    self._term_to_families[term_normalized] = []
                
                self._term_to_families[term_normalized].append(family_id)
        
        logger.debug(f"Índice construido con {len(self._term_to_families)} términos")
    
    def match(self, keyword: str) -> List[MatchResult]:
        """
        Encuentra familias de producto que coinciden con la keyword.
        
        Args:
            keyword: Keyword a buscar
            
        Returns:
            Lista de matches ordenados por score descendente
        """
        keyword_lower = keyword.lower()
        keyword_normalized = preprocess_keyword(keyword)
        
        matches: Dict[str, MatchResult] = {}
        
        for family_id, family_data in self.products.items():
            matched_terms = []
            
            for term in family_data.get("keywords_match", []):
                term_normalized = preprocess_keyword(term)
                term_lower = term.lower()
                
                # Match por contenido normalizado o literal
                if term_normalized in keyword_normalized or term_lower in keyword_lower:
                    matched_terms.append(term)
            
            if matched_terms:
                # Score basado en cantidad y longitud de matches
                score = len(matched_terms) + sum(len(t) for t in matched_terms) / 100
                
                matches[family_id] = MatchResult(
                    id=family_id,
                    name=family_data["nombre"],
                    emoji=family_data.get("emoji", "📦"),
                    score=score,
                    matched_terms=matched_terms,
                    metadata={"productos": family_data.get("productos", [])}
                )
        
        # Ordenar por score descendente
        sorted_matches = sorted(
            matches.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_matches
    
    def get_best_match(self, keyword: str) -> Optional[MatchResult]:
        """
        Obtiene el mejor match para una keyword.
        
        Args:
            keyword: Keyword a buscar
            
        Returns:
            Mejor match o None
        """
        matches = self.match(keyword)
        return matches[0] if matches else None
    
    def get_best_family_id(self, keyword: str) -> Optional[str]:
        """
        Obtiene el ID de la mejor familia para una keyword.
        
        Args:
            keyword: Keyword a buscar
            
        Returns:
            ID de familia o None
        """
        best = self.get_best_match(keyword)
        return best.id if best else None
    
    def get_match_score(self, keyword: str) -> float:
        """
        Obtiene el score de match normalizado (0-1).
        
        Args:
            keyword: Keyword a buscar
            
        Returns:
            Score normalizado
        """
        best = self.get_best_match(keyword)
        if not best:
            return 0.0
        return min(best.score / 5, 1.0)
    
    def has_match(self, keyword: str) -> bool:
        """
        Verifica si una keyword tiene match con productos.
        
        Args:
            keyword: Keyword a verificar
            
        Returns:
            True si hay match
        """
        return len(self.match(keyword)) > 0
    
    def enrich_dataframe(self, df: pd.DataFrame, keyword_col: str = 'Keyword') -> pd.DataFrame:
        """
        Enriquece un DataFrame con información de productos.
        
        Args:
            df: DataFrame con keywords
            keyword_col: Nombre de columna de keywords
            
        Returns:
            DataFrame enriquecido
        """
        df = df.copy()
        
        # Calcular matches
        df['product_matches'] = df[keyword_col].apply(self.match)
        df['best_product_family'] = df[keyword_col].apply(self.get_best_family_id)
        df['product_match_score'] = df[keyword_col].apply(self.get_match_score)
        df['has_product_match'] = df[keyword_col].apply(self.has_match)
        
        # Extraer nombre de familia
        df['familia_producto'] = df['product_matches'].apply(
            lambda x: x[0].name if x else "Sin match"
        )
        
        # Extraer emoji
        df['product_emoji'] = df['product_matches'].apply(
            lambda x: x[0].emoji if x else "📦"
        )
        
        logger.info(f"Enriquecido DataFrame con {df['has_product_match'].sum()} matches de producto")
        
        return df


class AudienceMatcher:
    """
    Matcher de keywords con categorías de audiencia.
    """
    
    def __init__(self, audiences: Optional[Dict] = None):
        """
        Inicializa el matcher.
        
        Args:
            audiences: Diccionario de audiencias (usa config si None)
        """
        self.audiences = audiences or ConfigLoader.load_audiences()
        self._build_index()
    
    def _build_index(self) -> None:
        """Construye índice invertido para búsqueda rápida."""
        self._term_to_categories: Dict[str, List[str]] = {}
        
        for cat_id, cat_data in self.audiences.items():
            for term in cat_data.get("keywords_match", []):
                term_normalized = preprocess_keyword(term)
                
                if term_normalized not in self._term_to_categories:
                    self._term_to_categories[term_normalized] = []
                
                self._term_to_categories[term_normalized].append(cat_id)
        
        logger.debug(f"Índice de audiencias construido con {len(self._term_to_categories)} términos")
    
    def match(self, keyword: str) -> List[AudienceMatchResult]:
        """
        Encuentra categorías de audiencia que coinciden con la keyword.
        
        Args:
            keyword: Keyword a buscar
            
        Returns:
            Lista de matches ordenados por score descendente
        """
        keyword_lower = keyword.lower()
        keyword_normalized = preprocess_keyword(keyword)
        
        matches: Dict[str, AudienceMatchResult] = {}
        
        for cat_id, cat_data in self.audiences.items():
            matched_terms = []
            
            for term in cat_data.get("keywords_match", []):
                term_normalized = preprocess_keyword(term)
                term_lower = term.lower()
                
                if term_normalized in keyword_normalized or term_lower in keyword_lower:
                    matched_terms.append(term)
            
            if matched_terms:
                score = len(matched_terms) + sum(len(t) for t in matched_terms) / 100
                
                matches[cat_id] = AudienceMatchResult(
                    id=cat_id,
                    name=cat_data["nombre"],
                    emoji=cat_data.get("emoji", "🎁"),
                    tipo=cat_data.get("tipo", "otro"),
                    score=score,
                    matched_terms=matched_terms
                )
        
        sorted_matches = sorted(
            matches.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_matches
    
    def get_best_match(self, keyword: str) -> Optional[AudienceMatchResult]:
        """Obtiene el mejor match de audiencia."""
        matches = self.match(keyword)
        return matches[0] if matches else None
    
    def get_by_type(self, keyword: str) -> Dict[str, Optional[str]]:
        """
        Obtiene la mejor categoría de audiencia por cada tipo.
        
        Args:
            keyword: Keyword a buscar
            
        Returns:
            Diccionario {tipo: nombre_categoría}
        """
        matches = self.match(keyword)
        
        result = {
            "genero": None,
            "edad": None,
            "relacion": None,
            "ocasion": None,
            "estilo": None,
            "precio": None
        }
        
        for match in matches:
            tipo = match.tipo
            if tipo in result and result[tipo] is None:
                result[tipo] = match.name
        
        return result
    
    def get_primary_audience(self, keyword: str) -> Optional[str]:
        """Obtiene el nombre de la audiencia principal."""
        best = self.get_best_match(keyword)
        return best.name if best else None
    
    def get_emoji(self, keyword: str) -> str:
        """Obtiene el emoji de la audiencia principal."""
        best = self.get_best_match(keyword)
        return best.emoji if best else "🎁"
    
    def has_match(self, keyword: str) -> bool:
        """Verifica si una keyword tiene match de audiencia."""
        return len(self.match(keyword)) > 0
    
    def enrich_dataframe(self, df: pd.DataFrame, keyword_col: str = 'Keyword') -> pd.DataFrame:
        """
        Enriquece un DataFrame con información de audiencias.
        
        Args:
            df: DataFrame con keywords
            keyword_col: Nombre de columna de keywords
            
        Returns:
            DataFrame enriquecido
        """
        df = df.copy()
        
        # Calcular matches
        df['audience_matches'] = df[keyword_col].apply(self.match)
        df['primary_audience'] = df[keyword_col].apply(self.get_primary_audience)
        df['audience_emoji'] = df[keyword_col].apply(self.get_emoji)
        df['has_audience_match'] = df[keyword_col].apply(self.has_match)
        
        # Extraer por tipo
        audience_by_type = df[keyword_col].apply(self.get_by_type)
        df['audiencia_genero'] = audience_by_type.apply(lambda x: x.get('genero'))
        df['audiencia_edad'] = audience_by_type.apply(lambda x: x.get('edad'))
        df['audiencia_relacion'] = audience_by_type.apply(lambda x: x.get('relacion'))
        df['audiencia_ocasion'] = audience_by_type.apply(lambda x: x.get('ocasion'))
        df['audiencia_estilo'] = audience_by_type.apply(lambda x: x.get('estilo'))
        df['audiencia_precio'] = audience_by_type.apply(lambda x: x.get('precio'))
        
        logger.info(f"Enriquecido DataFrame con {df['has_audience_match'].sum()} matches de audiencia")
        
        return df


class IntentClassifier:
    """
    Clasificador de intención de búsqueda.
    """
    
    INTENT_PATTERNS = {
        "💰 Transaccional": [
            'comprar', 'precio', 'oferta', 'descuento', 'barato', 'tienda',
            'donde comprar', 'mejor precio', 'black friday', 'rebajas', 'outlet',
            'promocion', 'promoción', 'cupon', 'cupón', 'chollos'
        ],
        "📚 Informacional": [
            'que es', 'qué es', 'como', 'cómo', 'guia', 'guía', 'tutorial',
            'mejor', 'mejores', 'comparativa', 'vs', 'versus', 'review',
            'opinion', 'opinión', 'merece la pena', 'vale la pena',
            'diferencia', 'caracteristicas', 'características', 'cual elegir'
        ],
        "🧭 Navegacional": [
            'amazon', 'pccomponentes', 'mediamarkt', 'el corte ingles',
            'fnac', 'carrefour', 'aliexpress', 'oficial', 'web',
            'tienda oficial', 'pagina', 'página'
        ],
        "🎁 Regalo": [
            'regalo', 'regalos', 'regalar', 'navidad', 'reyes', 'amigo invisible',
            'cumpleaños', 'aniversario', 'dia del padre', 'dia de la madre'
        ]
    }
    
    @classmethod
    def classify(cls, keyword: str) -> str:
        """
        Clasifica la intención de una keyword.
        
        Args:
            keyword: Keyword a clasificar
            
        Returns:
            Etiqueta de intención
        """
        kw = keyword.lower()
        
        for intent, patterns in cls.INTENT_PATTERNS.items():
            if any(p in kw for p in patterns):
                return intent
        
        return "🔍 General"
    
    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame, keyword_col: str = 'Keyword') -> pd.DataFrame:
        """Enriquece DataFrame con clasificación de intent."""
        df = df.copy()
        df['intent'] = df[keyword_col].apply(cls.classify)
        return df


class GiftRecipientExtractor:
    """
    Extractor de destinatario de regalo.
    """
    
    RECIPIENT_PATTERNS = {
        'padre': ['padre', 'papa', 'papá', 'papi'],
        'madre': ['madre', 'mama', 'mamá', 'mami'],
        'hombre': ['hombre', 'chico', 'él', 'el'],
        'mujer': ['mujer', 'chica', 'ella'],
        'niño': ['niño', 'niña', 'niños', 'hijo', 'hija', 'pequeño'],
        'adolescente': ['adolescente', 'joven', 'teen'],
        'abuelo': ['abuelo', 'abuela', 'abuelos'],
        'amigo': ['amigo', 'amiga', 'amigo invisible'],
        'pareja': ['pareja', 'novio', 'novia', 'enamorado']
    }
    
    @classmethod
    def extract(cls, keyword: str) -> Optional[str]:
        """
        Extrae el destinatario del regalo.
        
        Args:
            keyword: Keyword a analizar
            
        Returns:
            Destinatario o None
        """
        kw = keyword.lower()
        
        for recipient, patterns in cls.RECIPIENT_PATTERNS.items():
            if any(p in kw for p in patterns):
                return recipient
        
        return None
    
    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame, keyword_col: str = 'Keyword') -> pd.DataFrame:
        """Enriquece DataFrame con destinatario."""
        df = df.copy()
        df['destinatario'] = df[keyword_col].apply(cls.extract)
        return df


class PriceRangeExtractor:
    """
    Extractor de rango de precio mencionado.
    """
    
    PRICE_PATTERNS = [
        (r'menos de (\d+)', 'hasta'),
        (r'hasta (\d+)', 'hasta'),
        (r'por debajo de (\d+)', 'hasta'),
        (r'(\d+)\s*euros', 'aprox'),
        (r'(\d+)€', 'aprox'),
        (r'(\d+)\s*eur', 'aprox'),
    ]
    
    @classmethod
    def extract(cls, keyword: str) -> Optional[str]:
        """
        Extrae el rango de precio mencionado.
        
        Args:
            keyword: Keyword a analizar
            
        Returns:
            Rango de precio o None
        """
        kw = keyword.lower()
        
        for pattern, _ in cls.PRICE_PATTERNS:
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
    
    @classmethod
    def enrich_dataframe(cls, df: pd.DataFrame, keyword_col: str = 'Keyword') -> pd.DataFrame:
        """Enriquece DataFrame con rango de precio."""
        df = df.copy()
        df['rango_precio'] = df[keyword_col].apply(cls.extract)
        return df


def enrich_keywords_full(
    df: pd.DataFrame,
    keyword_col: str = 'Keyword',
    products: Optional[Dict] = None,
    audiences: Optional[Dict] = None
) -> pd.DataFrame:
    """
    Enriquece un DataFrame de keywords con toda la información disponible.
    
    Args:
        df: DataFrame con keywords
        keyword_col: Nombre de columna de keywords
        products: Diccionario de productos (opcional)
        audiences: Diccionario de audiencias (opcional)
        
    Returns:
        DataFrame completamente enriquecido
    """
    # Inicializar matchers
    product_matcher = ProductMatcher(products)
    audience_matcher = AudienceMatcher(audiences)
    
    # Enriquecer con productos
    df = product_matcher.enrich_dataframe(df, keyword_col)
    
    # Enriquecer con audiencias
    df = audience_matcher.enrich_dataframe(df, keyword_col)
    
    # Enriquecer con intent
    df = IntentClassifier.enrich_dataframe(df, keyword_col)
    
    # Enriquecer con destinatario
    df = GiftRecipientExtractor.enrich_dataframe(df, keyword_col)
    
    # Enriquecer con rango de precio
    df = PriceRangeExtractor.enrich_dataframe(df, keyword_col)
    
    logger.info(f"Enriquecimiento completo: {len(df)} keywords procesadas")
    
    return df
