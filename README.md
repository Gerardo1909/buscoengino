# Buscoengino - Search Engine from Scratch

Un motor de búsqueda minimalista implementado desde cero para comprender los fundamentos de Information Retrieval (IR). Este proyecto cubre parsing de texto, tokenización, índices invertidos, scoring (TF/TF-IDF) y ranking básico.

## Propósito

Buscoengino es un proyecto diseñado para:

- Entender el pipeline completo de un motor de búsqueda
- Implementar conceptos clave de IR sin abstracciones innecesarias
- Servir como base para experimentos con ranking y relevancia
- Ser escalable sin sobreingeniería inicial

## Instalación

> Notar que se usa uv como gestor de dependencias, sin embargo se deja un archivo requirements.txt para quien quiera instalar dependencias usando venv.

```bash
# Clonar el repositorio
git clone https://github.com/Gerardo1909/buscoengino.git
cd buscoengino

# Instalar dependencias
uv sync

# Ejecutar tests
uv run pytest
```

## Estructura del Proyecto

```
buscoengino/
├── pyproject.toml              # Configuración del proyecto 
├── uv.lock                     # Lock file para reproducibilidad
├── README.md
├── .gitignore
│
├── data/
│   ├── raw/documents/          # Textos originales (entrada del pipeline)
│   ├── processed/corpus_clean/ # Textos normalizados
│   └── sample_queries/         # Queries de prueba
│
├── src/search_engine/          # Código principal del motor de búsqueda
│   ├── __init__.py
│   │
│   ├── config/
│   │   └── settings.py         # Paths, configuraciones, flags globales
│   │
│   ├── ingestion/
│   │   └── loader.py           # Lectura de documentos desde archivo
│   │
│   ├── preprocessing/
│   │   ├── tokenizer.py        # Conversión de texto a tokens
│   │   ├── normalizer.py       # Normalización (minúsculas, puntuación, etc.)
│   │   └── stopwords.py        # Gestión de palabras vacías
│   │
│   ├── indexing/
│   │   └── inverted_index.py   # Construcción e implementación del índice invertido
│   │
│   ├── ranking/
│   │   ├── tf.py               # Cálculo de Term Frequency
│   │   ├── tfidf.py            # Cálculo de TF-IDF
│   │   └── scorer.py           # Scoring de documentos
│   │
│   ├── search/
│   │   └── engine.py           # Pipeline completo: consulta → preprocesamiento → ranking
│   │
│   ├── evaluation/
│   │   └── metrics.py          # Precision, Recall, MRR (opcional)
│   │
│   └── utils/
│       └── io.py               # Funciones de entrada/salida
│
├── tests/
│   ├── conftest.py             # Fixtures compartidas
│   ├── test_tokenizer.py
│   ├── test_index.py
│   └── test_search.py
│
└── notebooks/
    └── exploration.ipynb       # Experimentación opcional
```

## Pipeline de IR

El motor funciona en cinco etapas:

### 1. Ingesta (Ingestion)
- Leer documentos desde archivos
- Asignar IDs únicos a cada documento
- Guardar metadatos (timestamp, fuente, etc.)

### 2. Preprocesamiento (Preprocessing)
- **Tokenización:** Dividir texto en palabras
- **Normalización:** Convertir a minúsculas, remover puntuación
- **Stopwords:** Eliminar palabras comunes (the, a, and, etc.)

### 3. Indexado (Indexing)
- Construir índice invertido: término → [doc1, doc2, ...]
- Estructura eficiente para búsquedas rápidas

### 4. Ranking (Ranking)
- **TF (Term Frequency):** Frecuencia del término en el documento
- **TF-IDF:** Importancia relativa de un término en la colección
- **Scoring:** Asignar score a cada documento para una query

### 5. Búsqueda (Search)
- Procesar query con el mismo pipeline de preprocesamiento
- Buscar términos en el índice invertido
- Rankear documentos por relevancia
- Retornar resultados ordenados

## Datos de Prueba

Se proporcionan tres niveles de complejidad:

### Dataset 1: Mini corpus controlado
```
doc1.txt → "El gato duerme en la casa"
doc2.txt → "El perro corre en el parque"
doc3.txt → "El gato juega con el perro"
```

Queries de prueba:
```
"gato"
"perro"
"gato perro"
```

### Dataset 2: Corpus temático
Documentos categorizados (tecnología, deportes, cocina) para evaluar ranking.

### Dataset 3: Corpus real (futuro)
- Wikipedia dumps pequeños
- News datasets (AG News, etc.)
- 20 Newsgroups

## Testing

Cobertura básica de funcionalidad:

### Tokenización
```
Input: "Hola mundo!"
Expected: ["hola", "mundo"]
```

### Índice Invertido
```
doc_a: "gato negro"
doc_b: "gato blanco"

Result: {
  "gato": [doc_a, doc_b],
  "negro": [doc_a],
  "blanco": [doc_b]
}
```

### Búsqueda
```
Query: "gato"
Expected: [doc_a, doc_b] ordenados por score
```

Ejecutar tests:
```bash
uv run pytest -v
uv run pytest --cov=src/search_engine tests/
```

---

**Stack:** Python 3.13 · NLTK · Scikit-learn · Pytest

**Autor:** Gerardo Toboso · [gerardotoboso1909@gmail.com](mailto:gerardotoboso1909@gmail.com)

**Licencia:** MIT
