# 🎄 Keyword Semantic Clustering Tool - Navidad Edition

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Descripción

Herramienta de **clustering semántico de keywords** diseñada para identificar oportunidades de creación de URLs/landing pages para campañas de **Navidad y Regalos**. 

Procesa datos de **Google Keyword Planner** y utiliza técnicas de NLP + AI (Claude/GPT) para:
- Agrupar keywords por similitud semántica
- Calcular volumen estacional (Diciembre + Enero)
- Sugerir URLs optimizadas para SEO
- Generar análisis de Query Fan-Out
- Identificar intención de búsqueda y destinatarios de regalo

## 🎯 Caso de Uso Principal

Esta herramienta está diseñada para equipos de **SEO y Content** que necesitan:

1. **Identificar oportunidades de contenido** basadas en volumen de búsqueda estacional
2. **Agrupar keywords semánticamente** para evitar canibalización de URLs
3. **Priorizar la creación de landing pages** por potencial de tráfico
4. **Generar Query Fan-Out** para cobertura completa de intención

## 🛠️ Metodología

### Clustering Semántico

La herramienta utiliza una combinación de técnicas:

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE CLUSTERING                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. PREPROCESAMIENTO                                            │
│     └── Limpieza de keywords → Normalización → Stopwords        │
│                                                                  │
│  2. VECTORIZACIÓN (TF-IDF)                                      │
│     └── N-gramas (1-3) → Features (1000) → Embeddings           │
│                                                                  │
│  3. CLUSTERING                                                   │
│     ├── K-Means (rápido, clusters esféricos)                    │
│     └── Jerárquico (más preciso, dendrograma)                   │
│                                                                  │
│  4. ENRIQUECIMIENTO                                             │
│     ├── Intención de búsqueda (transaccional/info/regalo)       │
│     ├── Destinatario del regalo (hombre/mujer/niño/etc)         │
│     ├── Rango de precio mencionado                              │
│     └── Familias de producto relacionadas                        │
│                                                                  │
│  5. ANÁLISIS AI (opcional)                                      │
│     └── Claude/GPT → Nombre cluster, URL, H1, Meta, Fan-Out     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Query Fan-Out

El **Query Fan-Out** es una técnica donde se expande una query principal en múltiples sub-queries relacionadas para capturar diferentes intenciones de búsqueda:

```
Query: "regalos tecnológicos navidad"
    │
    ├── "mejores regalos tecnológicos 2024"
    ├── "gadgets para regalar en navidad"
    ├── "regalos tech para hombre"
    ├── "regalos tecnología menos de 50 euros"
    ├── "ideas regalo tecnológico original"
    └── "dispositivos electrónicos para regalar"
```

Esto es crucial para:
- **Google AI Mode** y **AI Overviews** que utilizan esta técnica
- Crear contenido que cubra toda la intención del usuario
- Mejorar la autoridad tópica de las URLs

## 🚀 Instalación

### Opción 1: Local

```bash
# Clonar repositorio
git clone https://github.com/tu-usuario/keyword-clustering-navidad.git
cd keyword-clustering-navidad

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

### Opción 2: Streamlit Cloud

1. Fork este repositorio
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Deploy 🚀

## 📊 Formato de Datos de Entrada

El archivo CSV debe provenir de **Google Keyword Planner** con las siguientes columnas:

| Columna | Descripción | Requerida |
|---------|-------------|-----------|
| `Keyword` | Término de búsqueda | ✅ |
| `Avg. monthly searches` | Volumen promedio mensual | ✅ |
| `Searches: Dec 2024` | Búsquedas en Diciembre | ✅ |
| `Searches: Jan 2025` | Búsquedas en Enero | ✅ |
| `Competition` | Nivel de competencia | ❌ |
| `Top of page bid` | CPC estimado | ❌ |

### Ejemplo de CSV:

```csv
Keyword,Avg. monthly searches,Searches: Dec 2024,Searches: Jan 2025,Competition
regalos navidad hombre,5400,12000,6500,Alta
regalos tecnologicos navidad,3200,8500,4000,Alta
amigo invisible 30 euros,1800,4200,900,Media
```

## 🎨 Características de la UI

### Dashboard Principal

- **Métricas globales**: Total keywords, clusters, volumen, coherencia
- **Visualizaciones interactivas**:
  - Treemap de volumen por cluster
  - Scatter plot PCA de clusters
  - Gráfico de barras de volumen
  - Matriz de oportunidades

### Análisis de Clusters

- Vista detallada por cluster
- Distribución de intención de búsqueda
- Análisis AI con:
  - Nombre descriptivo del cluster
  - URL sugerida
  - H1 y Meta Description
  - Productos recomendados
  - Query Fan-Out

### URLs Recomendadas

- Ranking por score de prioridad
- Export a CSV para implementación
- Familias de producto relacionadas

## 🔑 Configuración de APIs

### Claude (Anthropic)

```bash
export ANTHROPIC_API_KEY="tu-api-key"
```

### GPT (OpenAI)

```bash
export OPENAI_API_KEY="tu-api-key"
```

También puedes introducir las API keys directamente en la interfaz de la aplicación.

## 📈 Score de Prioridad

El score de prioridad para ranking de URLs se calcula:

```python
Score = (Volumen_Total × 0.4) + 
        (Volumen_Medio × 0.3) + 
        (Num_Keywords × 50 × 0.2) + 
        (Coherencia × 1000 × 0.1)
```

Esto prioriza:
1. **Alto volumen total**: Más tráfico potencial
2. **Alto volumen medio**: Keywords de calidad
3. **Muchas keywords**: Cobertura temática
4. **Alta coherencia**: Cluster bien definido

## 📁 Estructura del Proyecto

```
keyword-clustering-navidad/
├── app.py                 # Aplicación principal Streamlit
├── requirements.txt       # Dependencias Python
├── README.md             # Este archivo
├── .streamlit/
│   └── config.toml       # Configuración Streamlit
├── data/
│   └── sample.csv        # Datos de ejemplo
└── docs/
    ├── methodology.md    # Documentación técnica
    └── screenshots/      # Capturas de pantalla
```

## 🔧 Personalización

### Añadir nuevos productos/categorías

Edita el diccionario `PRODUCTOS_PCCOMPONENTES` en `app.py`:

```python
PRODUCTOS_PCCOMPONENTES = {
    "nueva_categoria": [
        "Producto 1", "Producto 2", "Producto 3"
    ],
    # ...
}
```

### Modificar patrones de intención

Edita `INTENT_PATTERNS` para añadir nuevos patrones regex:

```python
INTENT_PATTERNS = {
    "transaccional": [r"comprar", r"precio", ...],
    "informacional": [r"que es", r"como", ...],
    # Añadir nueva intención
    "comparativa": [r"vs", r"comparar", r"mejor entre"]
}
```

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Añade nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

## 🙏 Créditos

Desarrollado para **PcComponentes** - Equipo de Product Discovery & Content

### Referencias

- [Keyword Clustering Guide - Keyword Insights](https://www.keywordinsights.ai/blog/keyword-clustering-guide/)
- [Query Fan-Out - Semrush](https://www.semrush.com/blog/query-fan-out/)
- [Semantic Keyword Clustering - SEO.ai](https://seo.ai/blog/semantic-keyword-clustering)

---

<p align="center">
  Made with ❤️ for Christmas SEO campaigns 🎄
</p>
