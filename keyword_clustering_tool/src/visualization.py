"""
Módulo para generación de visualizaciones con Plotly.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from sklearn.decomposition import PCA

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger("keyword_clustering.visualization")


class ClusterVisualizer:
    """
    Generador de visualizaciones para clusters de keywords.
    """
    
    # Paleta de colores navideña
    CHRISTMAS_COLORS = [
        "#c41e3a",  # Rojo navideño
        "#1a472a",  # Verde pino
        "#ffd700",  # Dorado
        "#228b22",  # Verde bosque
        "#8b0000",  # Rojo oscuro
        "#2e8b57",  # Verde mar
        "#daa520",  # Dorado viejo
        "#006400",  # Verde oscuro
        "#b22222",  # Rojo ladrillo
        "#3cb371",  # Verde medio
    ]
    
    @classmethod
    def create_treemap(
        cls,
        df: pd.DataFrame,
        cluster_col: str = 'cluster_name',
        volume_col: str = 'volumen_navidad',
        coherence_col: str = 'coherence',
        title: str = "Distribución de Volumen por Cluster"
    ) -> go.Figure:
        """
        Crea un treemap de clusters.
        
        Args:
            df: DataFrame con datos
            cluster_col: Columna de cluster
            volume_col: Columna de volumen
            coherence_col: Columna de coherencia
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        cluster_summary = df.groupby(cluster_col).agg({
            volume_col: 'sum',
            'Keyword': 'count',
            coherence_col: 'first'
        }).reset_index()
        
        cluster_summary.columns = ['Cluster', 'Volumen', 'Keywords', 'Coherencia']
        
        fig = px.treemap(
            cluster_summary,
            path=['Cluster'],
            values='Volumen',
            color='Coherencia',
            color_continuous_scale='RdYlGn',
            title=title,
            hover_data=['Keywords', 'Coherencia']
        )
        
        fig.update_layout(
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    @classmethod
    def create_scatter_2d(
        cls,
        df: pd.DataFrame,
        embeddings: np.ndarray,
        cluster_col: str = 'cluster_name',
        volume_col: str = 'volumen_navidad',
        title: str = "Clusters en Espacio 2D (PCA)"
    ) -> go.Figure:
        """
        Crea scatter plot 2D con PCA.
        
        Args:
            df: DataFrame con datos
            embeddings: Matriz de embeddings
            cluster_col: Columna de cluster
            volume_col: Columna de volumen
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        # Reducir a 2D
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2)
            coords = pca.fit_transform(embeddings)
            variance_explained = sum(pca.explained_variance_ratio_) * 100
        else:
            coords = embeddings
            variance_explained = 100
        
        df_plot = df.copy()
        df_plot['x'] = coords[:, 0]
        df_plot['y'] = coords[:, 1]
        
        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color=cluster_col,
            size=volume_col,
            hover_data=['Keyword', volume_col],
            title=f"{title} ({variance_explained:.1f}% varianza)"
        )
        
        fig.update_layout(
            height=400,
            showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,47,42,0.5)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    @classmethod
    def create_top_clusters_bar(
        cls,
        df: pd.DataFrame,
        cluster_col: str = 'cluster_name',
        volume_col: str = 'volumen_navidad',
        top_n: int = 15,
        title: str = "Top Clusters por Volumen"
    ) -> go.Figure:
        """
        Crea gráfico de barras horizontales.
        
        Args:
            df: DataFrame con datos
            cluster_col: Columna de cluster
            volume_col: Columna de volumen
            top_n: Número de clusters a mostrar
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        cluster_summary = df.groupby(cluster_col).agg({
            volume_col: 'sum',
            'Keyword': 'count'
        }).reset_index()
        
        cluster_summary.columns = ['Cluster', 'Volumen', 'Keywords']
        top_clusters = cluster_summary.nlargest(top_n, 'Volumen')
        
        fig = go.Figure(go.Bar(
            y=top_clusters['Cluster'],
            x=top_clusters['Volumen'],
            orientation='h',
            marker_color=cls.CHRISTMAS_COLORS[0],
            text=top_clusters['Keywords'].apply(lambda x: f"{x} kws"),
            textposition='inside'
        ))
        
        fig.update_layout(
            title=title,
            height=400,
            yaxis={'categoryorder': 'total ascending'},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,47,42,0.5)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    @classmethod
    def create_product_distribution(
        cls,
        df: pd.DataFrame,
        family_col: str = 'familia_producto',
        volume_col: str = 'volumen_navidad',
        title: str = "Distribución por Familia de Producto"
    ) -> go.Figure:
        """
        Crea gráfico de distribución por productos.
        
        Args:
            df: DataFrame con datos
            family_col: Columna de familia
            volume_col: Columna de volumen
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        df_filtered = df[df.get('has_product_match', True) == True]
        
        if len(df_filtered) == 0:
            df_filtered = df
        
        product_data = df_filtered.groupby(family_col).agg({
            volume_col: 'sum',
            'Keyword': 'count'
        }).reset_index()
        
        product_data.columns = ['Familia', 'Volumen', 'Keywords']
        product_data = product_data.sort_values('Volumen', ascending=False)
        
        fig = px.bar(
            product_data,
            x='Familia',
            y='Volumen',
            color='Keywords',
            color_continuous_scale='Greens',
            title=title,
            text='Keywords'
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,47,42,0.5)',
            font=dict(color='white'),
            xaxis=dict(tickangle=45),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    @classmethod
    def create_audience_pie(
        cls,
        df: pd.DataFrame,
        audience_col: str = 'audiencia_genero',
        volume_col: str = 'volumen_navidad',
        title: str = "Distribución por Audiencia"
    ) -> go.Figure:
        """
        Crea gráfico de pastel de audiencias.
        
        Args:
            df: DataFrame con datos
            audience_col: Columna de audiencia
            volume_col: Columna de volumen
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        df_filtered = df[df[audience_col].notna()]
        
        if len(df_filtered) == 0:
            return cls._create_empty_chart(title, "Sin datos de audiencia")
        
        audience_data = df_filtered.groupby(audience_col).agg({
            volume_col: 'sum',
            'Keyword': 'count'
        }).reset_index()
        
        audience_data.columns = ['Audiencia', 'Volumen', 'Keywords']
        
        fig = px.pie(
            audience_data,
            values='Volumen',
            names='Audiencia',
            title=title,
            color_discrete_sequence=cls.CHRISTMAS_COLORS
        )
        
        fig.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    @classmethod
    def create_audience_bar(
        cls,
        df: pd.DataFrame,
        audience_col: str = 'audiencia_edad',
        volume_col: str = 'volumen_navidad',
        title: str = "Distribución por Edad"
    ) -> go.Figure:
        """
        Crea gráfico de barras de audiencias.
        
        Args:
            df: DataFrame con datos
            audience_col: Columna de audiencia
            volume_col: Columna de volumen
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        df_filtered = df[df[audience_col].notna()]
        
        if len(df_filtered) == 0:
            return cls._create_empty_chart(title, "Sin datos")
        
        audience_data = df_filtered.groupby(audience_col).agg({
            volume_col: 'sum',
            'Keyword': 'count'
        }).reset_index()
        
        audience_data.columns = ['Audiencia', 'Volumen', 'Keywords']
        audience_data = audience_data.sort_values('Volumen', ascending=False)
        
        fig = px.bar(
            audience_data,
            x='Audiencia',
            y='Volumen',
            title=title,
            color='Volumen',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,47,42,0.5)',
            font=dict(color='white'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    @classmethod
    def create_intent_distribution(
        cls,
        df: pd.DataFrame,
        intent_col: str = 'intent',
        volume_col: str = 'volumen_navidad',
        title: str = "Distribución por Intent"
    ) -> go.Figure:
        """
        Crea gráfico de distribución de intent.
        
        Args:
            df: DataFrame con datos
            intent_col: Columna de intent
            volume_col: Columna de volumen
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        intent_data = df.groupby(intent_col).agg({
            volume_col: 'sum',
            'Keyword': 'count'
        }).reset_index()
        
        intent_data.columns = ['Intent', 'Volumen', 'Keywords']
        
        fig = px.sunburst(
            intent_data,
            path=['Intent'],
            values='Volumen',
            title=title,
            color='Keywords',
            color_continuous_scale='Greens'
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        
        return fig
    
    @classmethod
    def create_coherence_histogram(
        cls,
        coherences: Dict[int, float],
        title: str = "Distribución de Coherencia"
    ) -> go.Figure:
        """
        Crea histograma de coherencia de clusters.
        
        Args:
            coherences: Diccionario {cluster_id: coherence}
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        values = list(coherences.values())
        
        fig = go.Figure(data=[
            go.Histogram(
                x=values,
                nbinsx=20,
                marker_color=cls.CHRISTMAS_COLORS[0]
            )
        ])
        
        fig.add_vline(
            x=np.mean(values),
            line_dash="dash",
            line_color=cls.CHRISTMAS_COLORS[2],
            annotation_text=f"Media: {np.mean(values):.2f}"
        )
        
        fig.update_layout(
            title=title,
            xaxis_title="Coherencia",
            yaxis_title="Frecuencia",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,47,42,0.5)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    @classmethod
    def create_volume_vs_keywords(
        cls,
        cluster_info: Dict[int, Any],
        title: str = "Volumen vs Keywords por Cluster"
    ) -> go.Figure:
        """
        Crea scatter de volumen vs número de keywords.
        
        Args:
            cluster_info: Información de clusters
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        data = []
        for cid, info in cluster_info.items():
            data.append({
                'Cluster': info.name if hasattr(info, 'name') else f"Cluster {cid}",
                'Keywords': info.size if hasattr(info, 'size') else 0,
                'Volumen': info.total_volume if hasattr(info, 'total_volume') else 0,
                'Coherencia': info.coherence if hasattr(info, 'coherence') else 0
            })
        
        df = pd.DataFrame(data)
        
        fig = px.scatter(
            df,
            x='Keywords',
            y='Volumen',
            size='Coherencia',
            color='Coherencia',
            hover_name='Cluster',
            title=title,
            color_continuous_scale='RdYlGn'
        )
        
        fig.update_layout(
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,47,42,0.5)',
            font=dict(color='white'),
            xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
        )
        
        return fig
    
    @classmethod
    def _create_empty_chart(cls, title: str, message: str) -> go.Figure:
        """Crea un gráfico vacío con mensaje."""
        fig = go.Figure()
        
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=14, color='white')
        )
        
        fig.update_layout(
            title=title,
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,47,42,0.5)',
            font=dict(color='white'),
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        
        return fig


def create_dashboard_metrics(
    df: pd.DataFrame,
    cluster_info: Dict[int, Any],
    coherences: Dict[int, float]
) -> Dict[str, Any]:
    """
    Genera métricas para el dashboard.
    
    Args:
        df: DataFrame con datos
        cluster_info: Información de clusters
        coherences: Coherencias de clusters
        
    Returns:
        Diccionario con métricas
    """
    metrics = {
        "total_keywords": len(df),
        "total_clusters": len(cluster_info),
        "total_volume": int(df.get('volumen_navidad', df.get('vol_avg', pd.Series([0]))).sum()),
        "avg_coherence": float(np.mean(list(coherences.values()))) if coherences else 0,
    }
    
    # Métricas de productos
    if 'has_product_match' in df.columns:
        metrics["pct_product_match"] = float(df['has_product_match'].mean() * 100)
        metrics["n_families"] = int(df[df['has_product_match']]['familia_producto'].nunique())
        vol_match = df[df['has_product_match']].get('volumen_navidad', pd.Series([0])).sum()
        total_vol = df.get('volumen_navidad', pd.Series([1])).sum()
        metrics["pct_volume_match"] = float(vol_match / total_vol * 100) if total_vol > 0 else 0
    
    # Métricas de audiencia
    if 'has_audience_match' in df.columns:
        metrics["pct_audience_match"] = float(df['has_audience_match'].mean() * 100)
    
    return metrics
