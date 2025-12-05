"""
Módulo para análisis y clasificación con AI (Claude/GPT).
"""

import logging
import json
import re
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from config.settings import ai_config

logger = logging.getLogger("keyword_clustering.analysis")

# Importaciones opcionales
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic no disponible")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI no disponible")


class AIProvider(Enum):
    """Proveedores de AI disponibles."""
    CLAUDE = "claude"
    OPENAI = "openai"
    NONE = "none"


@dataclass
class ClusterAnalysis:
    """Resultado de análisis de cluster con AI."""
    nombre_cluster: str
    url_sugerida: str
    h1_sugerido: str
    meta_description: str
    intent_principal: str
    productos_recomendados: List[str]
    query_fanout: List[str]
    raw_response: Optional[Dict] = None
    error: Optional[str] = None


# Categorías predefinidas para clasificación AI
AI_CLASSIFICATION_CATEGORIES = [
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


class AIAnalyzer:
    """
    Analizador de clusters usando AI.
    """
    
    def __init__(
        self,
        provider: AIProvider = AIProvider.CLAUDE,
        api_key: Optional[str] = None
    ):
        """
        Inicializa el analizador.
        
        Args:
            provider: Proveedor de AI a usar
            api_key: API key (usa config si None)
        """
        self.provider = provider
        
        if provider == AIProvider.CLAUDE:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic no está instalado")
            self.api_key = api_key or ai_config.anthropic_api_key
            if not self.api_key:
                raise ValueError("API key de Anthropic no proporcionada")
            self.client = anthropic.Anthropic(api_key=self.api_key)
        
        elif provider == AIProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI no está instalado")
            self.api_key = api_key or ai_config.openai_api_key
            if not self.api_key:
                raise ValueError("API key de OpenAI no proporcionada")
            self.client = openai.OpenAI(api_key=self.api_key)
        
        else:
            self.client = None
            self.api_key = None
    
    def analyze_cluster(
        self,
        keywords: List[str],
        volumes: List[int],
        max_keywords: int = 20
    ) -> ClusterAnalysis:
        """
        Analiza un cluster de keywords.
        
        Args:
            keywords: Keywords del cluster
            volumes: Volúmenes correspondientes
            max_keywords: Máximo de keywords a enviar
            
        Returns:
            Análisis del cluster
        """
        if self.client is None:
            return ClusterAnalysis(
                nombre_cluster="",
                url_sugerida="",
                h1_sugerido="",
                meta_description="",
                intent_principal="",
                productos_recomendados=[],
                query_fanout=[],
                error="No hay proveedor de AI configurado"
            )
        
        # Limitar keywords
        keywords = keywords[:max_keywords]
        volumes = volumes[:max_keywords]
        
        # Preparar datos
        kw_data = "\n".join([
            f"- {kw} (vol: {vol})"
            for kw, vol in zip(keywords, volumes)
        ])
        
        prompt = f"""Analiza este grupo de keywords de búsqueda relacionadas con regalos de Navidad para PcComponentes (tienda de tecnología).

Keywords del cluster:
{kw_data}

Responde ÚNICAMENTE con un JSON válido con esta estructura exacta:
{{
    "nombre_cluster": "nombre descriptivo corto del cluster",
    "url_sugerida": "/regalos-navidad/slug-descriptivo/",
    "h1_sugerido": "H1 optimizado para SEO",
    "meta_description": "Meta description de 150-160 caracteres",
    "intent_principal": "transaccional/informacional/mixto",
    "productos_recomendados": ["producto1", "producto2", "producto3"],
    "query_fanout": ["query relacionada 1", "query relacionada 2", "query relacionada 3"]
}}

No incluyas explicaciones, solo el JSON."""

        try:
            response_text = self._call_api(prompt)
            parsed = self._parse_json_response(response_text)
            
            return ClusterAnalysis(
                nombre_cluster=parsed.get("nombre_cluster", ""),
                url_sugerida=parsed.get("url_sugerida", ""),
                h1_sugerido=parsed.get("h1_sugerido", ""),
                meta_description=parsed.get("meta_description", ""),
                intent_principal=parsed.get("intent_principal", ""),
                productos_recomendados=parsed.get("productos_recomendados", []),
                query_fanout=parsed.get("query_fanout", []),
                raw_response=parsed
            )
            
        except Exception as e:
            logger.error(f"Error en análisis AI: {e}")
            return ClusterAnalysis(
                nombre_cluster="",
                url_sugerida="",
                h1_sugerido="",
                meta_description="",
                intent_principal="",
                productos_recomendados=[],
                query_fanout=[],
                error=str(e)
            )
    
    def classify_keywords(
        self,
        keywords: List[str],
        volumes: List[int],
        batch_size: int = None,
        categories: List[str] = None
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Clasifica keywords en categorías usando AI.
        
        Args:
            keywords: Lista de keywords
            volumes: Volúmenes correspondientes
            batch_size: Tamaño de batch
            categories: Categorías a usar
            
        Returns:
            Tupla de (labels, nombres de clusters)
        """
        if self.client is None:
            raise ValueError("No hay proveedor de AI configurado")
        
        batch_size = batch_size or ai_config.ai_batch_size
        categories = categories or AI_CLASSIFICATION_CATEGORIES
        
        clusters = np.zeros(len(keywords), dtype=int)
        cluster_names = {i: cat.split("(")[0].strip() for i, cat in enumerate(categories)}
        
        categories_list = "\n".join([f"{i}. {cat}" for i, cat in enumerate(categories)])
        
        for batch_start in range(0, len(keywords), batch_size):
            batch_end = min(batch_start + batch_size, len(keywords))
            batch_kws = keywords[batch_start:batch_end]
            batch_vols = volumes[batch_start:batch_end]
            
            kw_list = "\n".join([
                f"{i+1}. {kw} (vol: {vol})"
                for i, (kw, vol) in enumerate(zip(batch_kws, batch_vols))
            ])
            
            prompt = f"""Clasifica cada keyword en UNA categoría.

CATEGORÍAS:
{categories_list}

KEYWORDS:
{kw_list}

Responde SOLO con un JSON array de números de categoría (0-{len(categories)-1}).
Ejemplo para 5 keywords: [0, 3, 5, 1, 2]

No incluyas explicaciones, solo el array JSON."""

            try:
                response_text = self._call_api(prompt)
                batch_clusters = self._parse_json_response(response_text)
                
                if isinstance(batch_clusters, list):
                    for i, cluster_id in enumerate(batch_clusters):
                        if batch_start + i < len(clusters):
                            cid = int(cluster_id) if isinstance(cluster_id, (int, float)) else len(categories) - 1
                            clusters[batch_start + i] = min(cid, len(categories) - 1)
                
            except Exception as e:
                logger.warning(f"Error en batch {batch_start}: {e}")
                for i in range(batch_start, batch_end):
                    clusters[i] = len(categories) - 1
        
        logger.info(f"Clasificación AI completada: {len(keywords)} keywords")
        
        return clusters, cluster_names
    
    def _call_api(self, prompt: str, max_tokens: int = None) -> str:
        """Llama a la API correspondiente."""
        max_tokens = max_tokens or ai_config.claude_max_tokens
        
        if self.provider == AIProvider.CLAUDE:
            message = self.client.messages.create(
                model=ai_config.claude_model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        
        elif self.provider == AIProvider.OPENAI:
            response = self.client.chat.completions.create(
                model=ai_config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=ai_config.openai_temperature
            )
            return response.choices[0].message.content
        
        else:
            raise ValueError("Proveedor no configurado")
    
    def _parse_json_response(self, response: str) -> Any:
        """Parsea respuesta JSON de la API."""
        # Limpiar markdown
        response = re.sub(r'^```json?\s*', '', response.strip())
        response = re.sub(r'\s*```$', '', response)
        response = response.strip()
        
        return json.loads(response)


class AIClassifier:
    """
    Clasificador de keywords con AI.
    """
    
    def __init__(self, api_key: str, provider: str = "claude"):
        """
        Inicializa el clasificador.
        
        Args:
            api_key: API key
            provider: "claude" o "openai"
        """
        provider_enum = AIProvider.CLAUDE if provider == "claude" else AIProvider.OPENAI
        self.analyzer = AIAnalyzer(provider=provider_enum, api_key=api_key)
    
    def classify(
        self,
        keywords: List[str],
        volumes: List[int]
    ) -> Tuple[np.ndarray, Dict[int, str]]:
        """
        Clasifica keywords.
        
        Args:
            keywords: Lista de keywords
            volumes: Volúmenes
            
        Returns:
            Tupla de (labels, nombres)
        """
        return self.analyzer.classify_keywords(keywords, volumes)
    
    def analyze(
        self,
        keywords: List[str],
        volumes: List[int]
    ) -> ClusterAnalysis:
        """
        Analiza un cluster.
        
        Args:
            keywords: Keywords del cluster
            volumes: Volúmenes
            
        Returns:
            Análisis del cluster
        """
        return self.analyzer.analyze_cluster(keywords, volumes)


def check_ai_availability() -> Dict[str, bool]:
    """
    Verifica disponibilidad de proveedores de AI.
    
    Returns:
        Diccionario con disponibilidad
    """
    return {
        "anthropic": ANTHROPIC_AVAILABLE,
        "openai": OPENAI_AVAILABLE,
        "any": ANTHROPIC_AVAILABLE or OPENAI_AVAILABLE
    }


def get_available_providers() -> List[str]:
    """
    Retorna lista de proveedores disponibles.
    
    Returns:
        Lista de proveedores
    """
    providers = ["Sin AI"]
    
    if ANTHROPIC_AVAILABLE:
        providers.append("Claude (Anthropic)")
    
    if OPENAI_AVAILABLE:
        providers.append("GPT (OpenAI)")
    
    return providers
