"""
Módulo para carga y validación de datos de Google Keyword Planner.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO

from .utils import safe_convert_volume, clean_keyword

logger = logging.getLogger("keyword_clustering.data_loader")


class DataLoadError(Exception):
    """Error durante la carga de datos."""
    pass


class ValidationError(Exception):
    """Error de validación de datos."""
    pass


class CSVFormat(Enum):
    """Formatos de CSV soportados."""
    GOOGLE_KWP_ES = "google_kwp_es"
    GOOGLE_KWP_EN = "google_kwp_en"
    GENERIC = "generic"
    UNKNOWN = "unknown"


@dataclass
class ColumnMapping:
    """Mapeo de columnas del CSV."""
    keyword: str
    avg_monthly_searches: Optional[str] = None
    competition: Optional[str] = None
    monthly_columns: Dict[str, str] = field(default_factory=dict)
    
    def has_monthly_data(self) -> bool:
        """Verifica si hay datos mensuales."""
        return len(self.monthly_columns) > 0


@dataclass
class DataValidationResult:
    """Resultado de validación de datos."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class KeywordDataLoader:
    """
    Cargador de datos de keywords con validación robusta.
    Soporta múltiples formatos de CSV de Google Keyword Planner.
    """
    
    # Patrones para detectar columnas de keywords
    KEYWORD_PATTERNS = [
        'keyword', 'palabra clave', 'palabra', 'query', 'término', 'termino'
    ]
    
    # Patrones para columnas de volumen mensual
    MONTH_PATTERNS = {
        'jan': ['jan', 'ene', 'enero', 'january'],
        'feb': ['feb', 'febrero', 'february'],
        'mar': ['mar', 'marzo', 'march'],
        'apr': ['apr', 'abr', 'abril', 'april'],
        'may': ['may', 'mayo'],
        'jun': ['jun', 'junio', 'june'],
        'jul': ['jul', 'julio', 'july'],
        'aug': ['aug', 'ago', 'agosto', 'august'],
        'sep': ['sep', 'sept', 'septiembre', 'september'],
        'oct': ['oct', 'octubre', 'october'],
        'nov': ['nov', 'noviembre', 'november'],
        'dec': ['dec', 'dic', 'diciembre', 'december']
    }
    
    # Patrones para volumen promedio
    AVG_VOLUME_PATTERNS = [
        'avg. monthly searches', 'búsquedas mensuales promedio',
        'promedio de búsquedas', 'average monthly', 'monthly searches',
        'búsquedas mensuales'
    ]
    
    # Patrones para competencia
    COMPETITION_PATTERNS = [
        'competition', 'competencia', 'competition (indexed)'
    ]
    
    def __init__(self):
        """Inicializa el cargador de datos."""
        self.df: Optional[pd.DataFrame] = None
        self.column_mapping: Optional[ColumnMapping] = None
        self.format_detected: CSVFormat = CSVFormat.UNKNOWN
        self._raw_df: Optional[pd.DataFrame] = None
    
    def load(
        self,
        source: Union[str, Path, BytesIO, pd.DataFrame],
        skip_rows: Optional[int] = None,
        encoding: str = "utf-8"
    ) -> pd.DataFrame:
        """
        Carga datos desde múltiples fuentes.
        
        Args:
            source: Ruta de archivo, BytesIO, o DataFrame existente
            skip_rows: Filas a saltar (auto-detecta si es None)
            encoding: Codificación del archivo
            
        Returns:
            DataFrame cargado y procesado
            
        Raises:
            DataLoadError: Si hay error en la carga
        """
        try:
            if isinstance(source, pd.DataFrame):
                self._raw_df = source.copy()
            elif isinstance(source, (str, Path)):
                self._raw_df = self._load_from_file(Path(source), skip_rows, encoding)
            elif isinstance(source, BytesIO):
                self._raw_df = self._load_from_bytes(source, skip_rows, encoding)
            else:
                raise DataLoadError(f"Tipo de fuente no soportado: {type(source)}")
            
            # Detectar formato y mapear columnas
            self.format_detected = self._detect_format()
            self.column_mapping = self._detect_columns()
            
            # Procesar y normalizar datos
            self.df = self._process_dataframe()
            
            logger.info(
                f"Cargados {len(self.df)} keywords en formato {self.format_detected.value}"
            )
            
            return self.df
            
        except Exception as e:
            logger.error(f"Error cargando datos: {e}")
            raise DataLoadError(f"Error cargando datos: {e}") from e
    
    def _load_from_file(
        self,
        path: Path,
        skip_rows: Optional[int],
        encoding: str
    ) -> pd.DataFrame:
        """Carga desde archivo."""
        if not path.exists():
            raise DataLoadError(f"Archivo no encontrado: {path}")
        
        # Intentar diferentes configuraciones
        attempts = [
            {"skiprows": skip_rows or 0, "encoding": encoding},
            {"skiprows": 2, "encoding": encoding},
            {"skiprows": 0, "encoding": encoding},
            {"skiprows": 2, "encoding": "latin-1"},
            {"skiprows": 0, "encoding": "latin-1"},
        ]
        
        last_error = None
        for params in attempts:
            try:
                df = pd.read_csv(path, **params)
                if self._is_valid_dataframe(df):
                    return df
            except Exception as e:
                last_error = e
                continue
        
        raise DataLoadError(f"No se pudo cargar el archivo: {last_error}")
    
    def _load_from_bytes(
        self,
        data: BytesIO,
        skip_rows: Optional[int],
        encoding: str
    ) -> pd.DataFrame:
        """Carga desde BytesIO."""
        attempts = [
            {"skiprows": skip_rows or 0, "encoding": encoding},
            {"skiprows": 2, "encoding": encoding},
            {"skiprows": 0, "encoding": encoding},
        ]
        
        last_error = None
        for params in attempts:
            try:
                data.seek(0)
                df = pd.read_csv(data, **params)
                if self._is_valid_dataframe(df):
                    return df
            except Exception as e:
                last_error = e
                continue
        
        raise DataLoadError(f"No se pudo cargar los datos: {last_error}")
    
    def _is_valid_dataframe(self, df: pd.DataFrame) -> bool:
        """Verifica si el DataFrame es válido."""
        if df is None or df.empty:
            return False
        if len(df.columns) < 2:
            return False
        # Verificar que no sean todas filas de encabezado
        if df.iloc[0].astype(str).str.contains('keyword|palabra', case=False).any():
            return False
        return True
    
    def _detect_format(self) -> CSVFormat:
        """Detecta el formato del CSV."""
        if self._raw_df is None:
            return CSVFormat.UNKNOWN
        
        columns_lower = [c.lower() for c in self._raw_df.columns]
        columns_str = ' '.join(columns_lower)
        
        # Google Keyword Planner español
        if any(p in columns_str for p in ['palabra clave', 'búsquedas mensuales']):
            return CSVFormat.GOOGLE_KWP_ES
        
        # Google Keyword Planner inglés
        if any(p in columns_str for p in ['keyword', 'monthly searches']):
            return CSVFormat.GOOGLE_KWP_EN
        
        return CSVFormat.GENERIC
    
    def _detect_columns(self) -> ColumnMapping:
        """Detecta y mapea las columnas del CSV."""
        if self._raw_df is None:
            raise DataLoadError("No hay datos cargados")
        
        columns = self._raw_df.columns.tolist()
        columns_lower = [c.lower() for c in columns]
        
        mapping = ColumnMapping(keyword=columns[0])  # Default a primera columna
        
        # Buscar columna de keywords
        for i, col in enumerate(columns_lower):
            if any(p in col for p in self.KEYWORD_PATTERNS):
                mapping.keyword = columns[i]
                break
        
        # Buscar columna de volumen promedio
        for i, col in enumerate(columns_lower):
            if any(p in col for p in self.AVG_VOLUME_PATTERNS):
                mapping.avg_monthly_searches = columns[i]
                break
        
        # Buscar columnas de meses
        for month_key, patterns in self.MONTH_PATTERNS.items():
            for i, col in enumerate(columns_lower):
                if any(p in col for p in patterns):
                    mapping.monthly_columns[month_key] = columns[i]
                    break
        
        # Buscar columna de competencia
        for i, col in enumerate(columns_lower):
            if any(p in col for p in self.COMPETITION_PATTERNS):
                mapping.competition = columns[i]
                break
        
        logger.debug(f"Mapping detectado: keyword={mapping.keyword}, "
                    f"monthly_cols={len(mapping.monthly_columns)}")
        
        return mapping
    
    def _process_dataframe(self) -> pd.DataFrame:
        """Procesa y normaliza el DataFrame."""
        if self._raw_df is None or self.column_mapping is None:
            raise DataLoadError("No hay datos para procesar")
        
        df = self._raw_df.copy()
        
        # Renombrar columna de keywords
        df = df.rename(columns={self.column_mapping.keyword: 'Keyword'})
        
        # Eliminar filas sin keyword
        df = df.dropna(subset=['Keyword'])
        df = df[df['Keyword'].astype(str).str.strip() != '']
        
        # Limpiar keywords
        df['keyword_clean'] = df['Keyword'].apply(clean_keyword)
        
        # Procesar columnas de volumen mensual
        for month_key, col_name in self.column_mapping.monthly_columns.items():
            new_col = f'vol_{month_key}'
            df[new_col] = df[col_name].apply(safe_convert_volume)
        
        # Procesar volumen promedio si existe
        if self.column_mapping.avg_monthly_searches:
            df['vol_avg'] = df[self.column_mapping.avg_monthly_searches].apply(
                safe_convert_volume
            )
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def calculate_seasonal_volume(
        self,
        months: List[str],
        fallback_to_avg: bool = True
    ) -> pd.Series:
        """
        Calcula volumen estacional sumando meses seleccionados.
        
        Args:
            months: Lista de meses a incluir (ej: ['nov', 'dec', 'jan'])
            fallback_to_avg: Usar promedio si no hay datos mensuales
            
        Returns:
            Serie con volumen estacional
        """
        if self.df is None:
            raise DataLoadError("No hay datos cargados")
        
        volume = pd.Series(0, index=self.df.index)
        
        for month in months:
            col_name = f'vol_{month}'
            if col_name in self.df.columns:
                volume += self.df[col_name]
        
        # Fallback a promedio si no hay datos mensuales
        if volume.sum() == 0 and fallback_to_avg and 'vol_avg' in self.df.columns:
            volume = self.df['vol_avg']
            logger.info("Usando volumen promedio (sin datos mensuales)")
        
        return volume
    
    def validate(self) -> DataValidationResult:
        """
        Valida los datos cargados.
        
        Returns:
            Resultado de validación con errores, warnings y estadísticas
        """
        result = DataValidationResult(is_valid=True)
        
        if self.df is None:
            result.is_valid = False
            result.errors.append("No hay datos cargados")
            return result
        
        # Estadísticas básicas
        result.stats['total_rows'] = len(self.df)
        result.stats['columns'] = list(self.df.columns)
        result.stats['format_detected'] = self.format_detected.value
        
        # Validar keywords
        empty_keywords = self.df['Keyword'].isna().sum()
        if empty_keywords > 0:
            result.warnings.append(f"{empty_keywords} keywords vacías")
        
        duplicate_keywords = self.df['Keyword'].duplicated().sum()
        if duplicate_keywords > 0:
            result.warnings.append(f"{duplicate_keywords} keywords duplicadas")
        
        # Validar volúmenes
        vol_columns = [c for c in self.df.columns if c.startswith('vol_')]
        if not vol_columns:
            result.errors.append("No se detectaron columnas de volumen")
            result.is_valid = False
        else:
            result.stats['volume_columns'] = vol_columns
            
            total_volume = sum(self.df[c].sum() for c in vol_columns)
            result.stats['total_volume'] = total_volume
            
            if total_volume == 0:
                result.warnings.append("Todos los volúmenes son 0")
        
        # Verificar datos mínimos
        if len(self.df) < 10:
            result.warnings.append("Muy pocas keywords (<10)")
        
        return result
    
    def filter_by_volume(
        self,
        min_volume: int,
        volume_column: str = 'volumen_navidad'
    ) -> pd.DataFrame:
        """
        Filtra keywords por volumen mínimo.
        
        Args:
            min_volume: Volumen mínimo requerido
            volume_column: Columna de volumen a usar
            
        Returns:
            DataFrame filtrado
        """
        if self.df is None:
            raise DataLoadError("No hay datos cargados")
        
        if volume_column not in self.df.columns:
            raise ValidationError(f"Columna {volume_column} no existe")
        
        filtered = self.df[self.df[volume_column] >= min_volume].copy()
        
        logger.info(
            f"Filtrado por volumen >= {min_volume}: "
            f"{len(self.df)} -> {len(filtered)} keywords"
        )
        
        return filtered
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Obtiene resumen de los datos cargados.
        
        Returns:
            Diccionario con estadísticas
        """
        if self.df is None:
            return {"error": "No hay datos cargados"}
        
        vol_columns = [c for c in self.df.columns if c.startswith('vol_')]
        
        return {
            "total_keywords": len(self.df),
            "unique_keywords": self.df['Keyword'].nunique(),
            "format": self.format_detected.value,
            "has_monthly_data": self.column_mapping.has_monthly_data() if self.column_mapping else False,
            "volume_columns": vol_columns,
            "sample_keywords": self.df['Keyword'].head(5).tolist()
        }
