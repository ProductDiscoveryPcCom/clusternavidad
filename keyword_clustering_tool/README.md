# 🎄 Keyword Clustering Tool v2.0

Herramienta profesional de clustering semántico de keywords para campañas de Navidad.

## ✨ Características

- **Múltiples métodos de embedding**: TF-IDF, Sentence Transformers, Híbrido
- **Algoritmos de clustering**: K-Means, Jerárquico, HDBSCAN
- **Matching inteligente**: Productos y audiencias PcComponentes
- **Análisis con AI**: Claude (Anthropic) y GPT (OpenAI)
- **Visualizaciones**: Treemaps, scatter plots, gráficos de barras
- **Exportación**: CSV con recomendaciones de URLs

## 🚀 Instalación

```bash
cd keyword_clustering_tool
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Estructura

```
keyword_clustering_tool/
├── app.py                 # App principal Streamlit
├── config/
│   ├── settings.py       # Configuración
│   ├── products.json     # Familias productos
│   └── audiences.json    # Audiencias
├── src/
│   ├── data_loader.py    # Carga datos
│   ├── matching.py       # Matching
│   ├── embeddings.py     # Embeddings
│   ├── clustering.py     # Clustering
│   ├── analysis.py       # AI
│   └── visualization.py  # Gráficos
└── tests/
    └── test_core.py      # Tests
```

## 🧪 Tests

```bash
pytest tests/ -v
```

---
Desarrollado para PcComponentes 🎄
