"""
🎄 Keyword Clustering Tool - Navidad PcComponentes
Aplicación profesional para clustering semántico de keywords.

Versión: 2.0.0
"""

import sys
from pathlib import Path

# Añadir directorio raíz al path
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

# Importaciones locales
from config.settings import ConfigLoader, app_config, logger
from src.data_loader import KeywordDataLoader, DataLoadError
from src.matching import enrich_keywords_full, ProductMatcher, AudienceMatcher
from src.embeddings import EmbeddingManager, check_embedding_availability
from src.clustering import ClusteringManager, check_clustering_availability
from src.analysis import AIAnalyzer, AIProvider, check_ai_availability, get_available_providers
from src.visualization import ClusterVisualizer, create_dashboard_metrics


# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title=app_config.page_title,
    page_icon=app_config.page_icon,
    layout=app_config.layout,
    initial_sidebar_state="expanded"
)


# ============================================================================
# CSS PERSONALIZADO - TEMA LIMPIO Y PROFESIONAL
# ============================================================================

st.markdown("""
<style>
    /* Botones principales */
    .stButton > button {
        background: linear-gradient(135deg, #c41e3a 0%, #a01830 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #d42a4a 0%, #c41e3a 100%);
        box-shadow: 0 4px 12px rgba(196, 30, 58, 0.3);
    }
    
    /* Tabs con estilo navideño sutil */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f8f9fa;
        padding: 8px;
        border-radius: 10px;
        border: 1px solid #e9ecef;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #c41e3a !important;
        color: white !important;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #1f2937 !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #c41e3a !important;
    }
    
    /* Headers principales */
    .main h1 {
        color: #1f2937 !important;
    }
    
    .main h2, .main h3 {
        color: #374151 !important;
    }
    
    /* Footer navideño */
    .christmas-footer {
        text-align: center;
        padding: 20px;
        font-size: 1.5rem;
        opacity: 0.8;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 8px;
    }
    
    /* Selectbox y inputs más limpios */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# FUNCIONES DE UTILIDAD
# ============================================================================

@st.cache_data
def load_and_process_data(uploaded_file_bytes, selected_months, min_volume):
    """Carga y procesa datos del archivo subido."""
    try:
        loader = KeywordDataLoader()
        df = loader.load(BytesIO(uploaded_file_bytes))
        
        df['volumen_navidad'] = loader.calculate_seasonal_volume(selected_months)
        df = df[df['volumen_navidad'] >= min_volume].copy()
        
        if len(df) == 0:
            return None, "No hay keywords con el volumen mínimo especificado"
        
        df = enrich_keywords_full(df, keyword_col='Keyword')
        return df, None
        
    except Exception as e:
        logger.error(f"Error cargando datos: {e}")
        return None, str(e)


@st.cache_resource
def get_embedding_manager():
    return EmbeddingManager(cache_enabled=True)


@st.cache_resource
def get_clustering_manager():
    return ClusteringManager()


def display_availability_badges():
    """Muestra badges de disponibilidad de features."""
    emb_avail = check_embedding_availability()
    clust_avail = check_clustering_availability()
    ai_avail = check_ai_availability()
    
    cols = st.columns(4)
    with cols[0]:
        st.caption(f"Sentence Transformers {'✅' if emb_avail['sentence_transformers'] else '❌'}")
    with cols[1]:
        st.caption(f"HDBSCAN {'✅' if clust_avail['hdbscan'] else '❌'}")
    with cols[2]:
        st.caption(f"Claude API {'✅' if ai_avail['anthropic'] else '❌'}")
    with cols[3]:
        st.caption(f"OpenAI API {'✅' if ai_avail['openai'] else '❌'}")


def format_number(num, decimals=0):
    if decimals == 0:
        return f"{int(num):,}".replace(",", ".")
    return f"{num:,.{decimals}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("# 🎄 Configuración")
        st.markdown("---")
        
        # Datos
        st.markdown("### 📁 Datos")
        uploaded_file = st.file_uploader("CSV de Google Keyword Planner", type=['csv'])
        
        # Meses
        st.markdown("### 📅 Temporada")
        months_options = {'Noviembre': 'nov', 'Diciembre': 'dec', 'Enero': 'jan'}
        selected_months_names = st.multiselect(
            "Meses a incluir",
            options=list(months_options.keys()),
            default=['Noviembre', 'Diciembre']
        )
        selected_months = [months_options[m] for m in selected_months_names]
        
        min_volume = st.number_input("Volumen mínimo", min_value=0, value=50, step=10)
        
        st.markdown("---")
        
        # Embeddings
        st.markdown("### 🧠 Embeddings")
        emb_methods = EmbeddingManager.get_available_methods()
        embedding_method = st.selectbox("Método de embedding", options=emb_methods, index=0)
        
        st.markdown("---")
        
        # Clustering
        st.markdown("### 🎯 Clustering")
        clustering_modes = ClusteringManager.get_available_modes()
        clustering_mode = st.selectbox("Modo de agrupación", options=clustering_modes, index=0)
        
        clustering_methods = ClusteringManager.get_available_methods()
        clustering_method = st.selectbox("Algoritmo", options=clustering_methods, index=0)
        
        if "HDBSCAN" in clustering_method:
            min_cluster_size = st.slider("Tamaño mínimo cluster", 3, 50, 5)
            n_clusters = None
        else:
            n_clusters = st.slider("Número de clusters", 3, 50, 15)
            min_cluster_size = 5
        
        st.markdown("---")
        
        # AI
        st.markdown("### 🤖 Análisis AI")
        ai_providers = get_available_providers()
        ai_provider = st.selectbox("Proveedor AI", options=ai_providers, index=0)
        
        api_key = None
        if ai_provider != "Sin AI":
            api_key = st.text_input("API Key", type="password")
        
        st.markdown("---")
        
        # Filtros
        st.markdown("### 🔍 Filtros")
        filter_products = st.checkbox("Solo con match de producto", value=False)
        
        return {
            'uploaded_file': uploaded_file,
            'selected_months': selected_months,
            'min_volume': min_volume,
            'embedding_method': embedding_method,
            'clustering_mode': clustering_mode,
            'clustering_method': clustering_method,
            'n_clusters': n_clusters,
            'min_cluster_size': min_cluster_size,
            'ai_provider': ai_provider,
            'api_key': api_key,
            'filter_products': filter_products
        }


# ============================================================================
# TABS
# ============================================================================

def render_visualization_tab(df, embeddings, clustering_result):
    col1, col2 = st.columns(2)
    
    with col1:
        fig = ClusterVisualizer.create_treemap(df, 'cluster_name', 'volumen_navidad', 'coherence')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = ClusterVisualizer.create_scatter_2d(df, embeddings, 'cluster_name', 'volumen_navidad')
        st.plotly_chart(fig, use_container_width=True)
    
    fig = ClusterVisualizer.create_top_clusters_bar(df, 'cluster_name', 'volumen_navidad', 15)
    st.plotly_chart(fig, use_container_width=True)


def render_products_tab(df):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = ClusterVisualizer.create_product_distribution(df, 'familia_producto', 'volumen_navidad')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'familia_producto' in df.columns and 'has_product_match' in df.columns:
            df_products = df[df['has_product_match']]
            summary = df_products.groupby('familia_producto').agg({
                'volumen_navidad': 'sum',
                'Keyword': 'count',
                'product_match_score': 'mean'
            }).reset_index()
            summary.columns = ['Familia', 'Volumen', 'Keywords', 'Score']
            summary = summary.sort_values('Volumen', ascending=False)
            summary['Score'] = summary['Score'].round(2)
            st.dataframe(summary, use_container_width=True, hide_index=True)


def render_audiences_tab(df):
    col1, col2 = st.columns(2)
    
    with col1:
        fig = ClusterVisualizer.create_audience_pie(df, 'audiencia_genero', 'volumen_navidad', "📊 Por Género")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = ClusterVisualizer.create_audience_bar(df, 'audiencia_edad', 'volumen_navidad', "📊 Por Edad")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### 👥 Por Relación")
    if 'audiencia_relacion' in df.columns:
        df_rel = df[df['audiencia_relacion'].notna()]
        if len(df_rel) > 0:
            rel_summary = df_rel.groupby('audiencia_relacion').agg({
                'volumen_navidad': 'sum', 'Keyword': 'count'
            }).reset_index()
            rel_summary.columns = ['Relación', 'Volumen', 'Keywords']
            st.dataframe(rel_summary.sort_values('Volumen', ascending=False), hide_index=True)


def render_clusters_tab(df, clustering_result, config):
    cluster_names = df['cluster_name'].unique().tolist()
    selected_cluster = st.selectbox("Selecciona un cluster", options=cluster_names)
    
    if selected_cluster:
        cluster_df = df[df['cluster_name'] == selected_cluster]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Keywords", len(cluster_df))
        with col2:
            st.metric("Volumen", format_number(cluster_df['volumen_navidad'].sum()))
        with col3:
            coh = cluster_df['coherence'].iloc[0] if 'coherence' in cluster_df.columns else 0
            st.metric("Coherencia", f"{coh:.2f}")
        with col4:
            st.metric("Vol. Medio", format_number(cluster_df['volumen_navidad'].mean()))
        
        st.markdown("#### 🔑 Top Keywords")
        top_kws = cluster_df.nlargest(20, 'volumen_navidad')[
            ['Keyword', 'volumen_navidad', 'intent', 'familia_producto', 'primary_audience']
        ]
        st.dataframe(top_kws, use_container_width=True, hide_index=True)
        
        # AI Analysis
        if config['ai_provider'] != "Sin AI" and config['api_key']:
            st.markdown("#### 🤖 Análisis AI")
            if st.button("Analizar con AI", key="analyze_cluster"):
                with st.spinner("Analizando..."):
                    try:
                        provider = AIProvider.CLAUDE if "Claude" in config['ai_provider'] else AIProvider.OPENAI
                        analyzer = AIAnalyzer(provider=provider, api_key=config['api_key'])
                        
                        analysis = analyzer.analyze_cluster(
                            cluster_df['Keyword'].tolist()[:20],
                            cluster_df['volumen_navidad'].tolist()[:20]
                        )
                        
                        if analysis.error:
                            st.error(f"Error: {analysis.error}")
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown(f"**Nombre:** {analysis.nombre_cluster}")
                                st.markdown(f"**URL:** `{analysis.url_sugerida}`")
                                st.markdown(f"**H1:** {analysis.h1_sugerido}")
                            with col2:
                                st.info(analysis.meta_description)
                                st.markdown("**Productos:**")
                                for p in analysis.productos_recomendados[:5]:
                                    st.markdown(f"- {p}")
                    except Exception as e:
                        st.error(f"Error: {e}")


def render_urls_tab(df, clustering_result):
    st.markdown("### 🔗 URLs Recomendadas")
    
    url_data = []
    for cluster_id, info in clustering_result.cluster_info.items():
        cluster_df = df[df['cluster_id'] == cluster_id]
        total_vol = cluster_df['volumen_navidad'].sum()
        n_kws = len(cluster_df)
        priority = total_vol * 0.4 + (total_vol/n_kws if n_kws else 0) * 0.3 + n_kws * 100 * 0.2 + info.coherence * 10000 * 0.1
        
        url_data.append({
            'Cluster': info.name,
            'URL': info.url_suggestion,
            'Volumen': int(total_vol),
            'Keywords': n_kws,
            'Coherencia': round(info.coherence, 2),
            'Prioridad': int(priority)
        })
    
    url_df = pd.DataFrame(url_data).sort_values('Prioridad', ascending=False)
    st.dataframe(url_df, use_container_width=True, hide_index=True)
    
    csv = url_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar URLs (CSV)", csv, "url_recommendations.csv", "text/csv")


def render_data_tab(df):
    st.markdown("### 📊 Datos Completos")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        clusters = ['Todos'] + df['cluster_name'].unique().tolist()
        filter_cluster = st.selectbox("Cluster", clusters)
    with col2:
        families = ['Todas'] + df['familia_producto'].unique().tolist()
        filter_family = st.selectbox("Familia", families)
    with col3:
        intents = ['Todos'] + df['intent'].unique().tolist()
        filter_intent = st.selectbox("Intent", intents)
    
    df_filtered = df.copy()
    if filter_cluster != 'Todos':
        df_filtered = df_filtered[df_filtered['cluster_name'] == filter_cluster]
    if filter_family != 'Todas':
        df_filtered = df_filtered[df_filtered['familia_producto'] == filter_family]
    if filter_intent != 'Todos':
        df_filtered = df_filtered[df_filtered['intent'] == filter_intent]
    
    st.info(f"Mostrando {len(df_filtered)} de {len(df)} keywords")
    
    display_cols = ['Keyword', 'volumen_navidad', 'cluster_name', 'intent', 'familia_producto', 'primary_audience']
    display_cols = [c for c in display_cols if c in df_filtered.columns]
    
    st.dataframe(df_filtered[display_cols].sort_values('volumen_navidad', ascending=False), hide_index=True)
    
    csv = df_filtered.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Descargar datos (CSV)", csv, "keywords_clustered.csv", "text/csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown("# 🎄 Keyword Clustering Tool")
    st.markdown("### Clustering Semántico para Campaña de Navidad")
    
    display_availability_badges()
    st.markdown("---")
    
    config = render_sidebar()
    
    if config['uploaded_file'] is None:
        st.info("👈 Sube un archivo CSV de Google Keyword Planner para comenzar")
        with st.expander("📋 Formato esperado"):
            st.markdown("""
            - **Keyword** o **Palabra clave**: La keyword de búsqueda
            - Columnas mensuales (Nov, Dec, Jan) o **Avg. monthly searches**
            """)
        return
    
    # Procesar datos
    with st.spinner("Cargando datos..."):
        df, error = load_and_process_data(
            config['uploaded_file'].getvalue(),
            config['selected_months'],
            config['min_volume']
        )
    
    if error:
        st.error(f"Error: {error}")
        return
    
    if df is None or len(df) == 0:
        st.warning("No hay datos para procesar")
        return
    
    if config['filter_products'] and 'has_product_match' in df.columns:
        df = df[df['has_product_match']].copy()
        if len(df) == 0:
            st.warning("No hay keywords con match de producto")
            return
    
    # Embeddings
    with st.spinner("Generando embeddings..."):
        emb_manager = get_embedding_manager()
        emb_result = emb_manager.generate(df['Keyword'].tolist(), method=config['embedding_method'])
        embeddings = emb_result.embeddings
    
    # Clustering
    with st.spinner("Ejecutando clustering..."):
        clust_manager = get_clustering_manager()
        clustering_result = clust_manager.cluster(
            embeddings=embeddings,
            df=df,
            method=config['clustering_method'],
            mode=config['clustering_mode'],
            n_clusters=config['n_clusters'] or 15,
            min_cluster_size=config['min_cluster_size']
        )
    
    # Añadir resultados
    df['cluster_id'] = clustering_result.labels
    df['cluster_name'] = df['cluster_id'].map(
        lambda x: clustering_result.cluster_info[x].name if x in clustering_result.cluster_info else f"Cluster {x}"
    )
    df['coherence'] = df['cluster_id'].map(clustering_result.coherences)
    
    # Métricas
    metrics = create_dashboard_metrics(df, clustering_result.cluster_info, clustering_result.coherences)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📝 Keywords", format_number(metrics['total_keywords']))
    with col2:
        st.metric("🎯 Clusters", metrics['total_clusters'])
    with col3:
        st.metric("📊 Volumen", format_number(metrics['total_volume']))
    with col4:
        st.metric("🎖️ Coherencia", f"{metrics['avg_coherence']:.2f}")
    
    if 'pct_product_match' in metrics:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🏷️ Match Producto", f"{metrics['pct_product_match']:.1f}%")
        with col2:
            st.metric("👥 Match Audiencia", f"{metrics.get('pct_audience_match', 0):.1f}%")
        with col3:
            st.metric("📦 Familias", metrics.get('n_families', 0))
        with col4:
            st.metric("💰 Vol Match", f"{metrics.get('pct_volume_match', 0):.1f}%")
    
    st.markdown("---")
    
    # Tabs
    tabs = st.tabs(["📊 Visualización", "🏷️ Productos", "👥 Audiencias", "🎯 Clusters", "🔗 URLs", "📋 Datos"])
    
    with tabs[0]:
        render_visualization_tab(df, embeddings, clustering_result)
    with tabs[1]:
        render_products_tab(df)
    with tabs[2]:
        render_audiences_tab(df)
    with tabs[3]:
        render_clusters_tab(df, clustering_result, config)
    with tabs[4]:
        render_urls_tab(df, clustering_result)
    with tabs[5]:
        render_data_tab(df)
    
    st.markdown("---")
    st.markdown("<div class='christmas-footer'>🎄 ❄️ 🎁 ⭐ 🎅</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
