import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ollama import Client
import io, sys
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
from docx import Document

warnings.filterwarnings('ignore')

plt.style.use('default')
sns.set_palette("husl")

class DataAnalysisApp:
    def __init__(self):
        self.client = None
        self.df = None
        self.df_dict = None
        self.df_examples = None
        self.chroma_client = chromadb.Client()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_collection = None

    def init_client(self):
    
    # Lista de puertos a probar
        ports_to_try = [11435]  # Puerto estándar primero
        
        for port in ports_to_try:
            try:
                st.sidebar.info(f"🔄 Intentando conectar en puerto {port}...")
                self.client = Client(host=f"http://localhost:{port}")
                
                # Verificar conexión con timeout corto
                models = self.client.list()
                
                if models and models.models:
                    st.sidebar.success(f"✅ Conectado a Ollama puerto {port} - {len(models.models)} modelos disponibles")
                    
                    # Mostrar modelos disponibles
                    model_names = [model.model for model in models.models]
                    st.sidebar.info(f"📦 Modelos: {', '.join(model_names[:3])}...")
                    
                    return True
                else:
                    st.sidebar.warning(f"⚠️ Puerto {port} responde pero sin modelos")
                    
            except Exception as e:
                st.sidebar.warning(f"❌ Puerto {port}: {str(e)[:50]}...")
                continue
        
        # Si ningún puerto funciona
        st.sidebar.error("❌ No se pudo conectar a Ollama en ningún puerto")
        
        with st.sidebar.expander("🔧 Soluciones", expanded=True):
            st.markdown("""
            **Pasos para solucionar:**
            
            1. **Verificar Ollama:**
            ```bash
            ollama serve
            ```
            
            2. **Verificar modelos:**
            ```bash
            ollama list
            ollama pull llama3.2:3b
            ```
            
            3. **Reiniciar Ollama:**
            - Windows: Ctrl+C en terminal, luego `ollama serve`
            - Cerrar desde Task Manager si es necesario
            
            4. **Verificar puerto:**
            - Ollama usa puerto 11434 por defecto
            - Verifica que no esté ocupado por otra app
            """)
        
        return False

    # MÉTODO ADICIONAL: Verificar estado del sistema
    def check_ollama_health(self):
        """Verificar estado de salud de Ollama"""
        try:
            import requests
            import psutil
            
            # Verificar memoria disponible
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 4:
                st.sidebar.error(f"⚠️ Memoria baja: {available_gb:.1f}GB disponible")
                st.sidebar.info("💡 Cierra otras aplicaciones o usa modelo más pequeño")
                return False
            
            # Verificar si el servicio responde
            for port in [11434, 11435]:
                try:
                    response = requests.get(f"http://localhost:{port}/api/tags", timeout=5)
                    if response.status_code == 200:
                        st.sidebar.success(f"✅ Ollama saludable en puerto {port}")
                        return True
                except:
                    continue
            
            st.sidebar.error("❌ Ollama no responde en ningún puerto")
            return False
            
        except Exception as e:
            st.sidebar.error(f"❌ Error verificando salud: {e}")
            return False

    # MÉTODO MEJORADO: Manejo de errores en chat
    def safe_chat(self, model, messages, options):
        """Chat con manejo seguro de errores"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options=options
                )
                return response.message.content
                
            except Exception as e:
                retry_count += 1
                error_msg = str(e).lower()
                
                if "terminated" in error_msg or "exit status" in error_msg:
                    st.error(f"❌ Modelo {model} falló. Intento {retry_count}/{max_retries}")
                    
                    if retry_count < max_retries:
                        st.info("🔄 Reintentando en 3 segundos...")
                        time.sleep(3)
                        
                        # Reintentar con cliente nuevo
                        try:
                            self.init_client()
                        except:
                            pass
                    else:
                        st.error("❌ Máximo de reintentos alcanzado")
                        st.info("💡 Prueba con un modelo más ligero o reinicia Ollama")
                        
                        # Sugerir modelos alternativos
                        suggested_models = ["llama3.2:3b", "qwen3:8b", "llama3.2:1b"]
                        st.info(f"🔄 Modelos alternativos: {', '.join(suggested_models)}")
                        
                        return "Error: No se pudo completar la consulta. Prueba con un modelo más ligero."
                else:
                    raise e
        
        return "Error: Máximo de reintentos alcanzado"

    def load_interface(self):
        st.set_page_config(page_title="🧠 Análisis con IA", layout="wide", initial_sidebar_state="expanded")
        st.sidebar.markdown("""
            <style>
            [data-testid="stSidebar"] > div {
                background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .stSelectbox label, .stSlider label, .stFileUploader label {
                color: white !important;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
        try:
            st.sidebar.image(r"D:/Julio/ExperienciaAnalitica/IA/EASinFondo.png", width=250)
        except Exception:
            st.sidebar.markdown("### 🧠 Análisis con IA")
        with st.sidebar.expander("📁 Archivos de datos", expanded=True):
            datos_file = st.file_uploader("Datos principales (CSV)", type="csv", key="datos")
            dict_file = st.file_uploader("Diccionario de variables (CSV)", type="csv", key="diccionario")
            ejemplos_file = st.file_uploader("Ejemplos de análisis (XLSX)", type="xlsx", key="ejemplos")
        with st.sidebar.expander("⚙️ Configuración del modelo", expanded=True):
            model = st.selectbox("Modelo de IA", ["deepseek-coder:6.7b","qwen3:14b", "deepseek-coder-v2:16b", "codellama", "llama3", "qwen3:8b", "deepseek-r1:7b"], index=0)
            temperature = st.slider("Temperatura", 0.0, 1.0, 0.1, 0.1)
            max_tokens = st.slider("Máximo de tokens", 50, 4000, 800, 50)
        return datos_file, dict_file, ejemplos_file, model, temperature, max_tokens

    def load_data(self, datos_file, dict_file, ejemplos_file):
        if datos_file:
            try:
                self.df = pd.read_csv(datos_file)
                st.sidebar.success(f"✅ Datos cargados: {self.df.shape[0]:,} filas, {self.df.shape[1]} columnas")
            except Exception as e:
                st.sidebar.error(f"❌ Error cargando datos: {e}")
                self.df = None
        if dict_file:
            try:
                self.df_dict = pd.read_csv(dict_file)
                st.sidebar.success(f"✅ Diccionario cargado: {len(self.df_dict)} variables")
            except Exception as e:
                st.sidebar.error(f"❌ Error cargando diccionario: {e}")
                self.df_dict = None
        if ejemplos_file:
            try:
                self.df_examples = pd.read_excel(ejemplos_file)
                st.sidebar.success(f"✅ Ejemplos cargados: {len(self.df_examples)} casos")
            except Exception as e:
                st.sidebar.error(f"❌ Error cargando ejemplos: {e}")
                self.df_examples = None

    def index_metadata_dictionary(self):
        if self.df_dict is not None and not self.vector_collection:
            self.vector_collection = self.chroma_client.get_or_create_collection("diccionario_metadatos")
            for i, fila in self.df_dict.iterrows():
                descripcion = str(fila.get("Descripción", ""))
                if descripcion:
                    self.vector_collection.add(documents=[descripcion], ids=[f"id_{i}"])

    def retrieve_relevant_fragments(self, query):
        if not self.vector_collection:
            return ""
        query_embedding = self.embedder.encode([query])[0]
        result = self.vector_collection.query(query_embeddings=[query_embedding], n_results=5)
        fragmentos = result.get("documents", [[]])[0]
        return "\n".join(fragmentos)

    def generate_enhanced_context(self):
        if self.df is None:
            return "No hay DataFrame disponible."
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                sample_values = self.df[col].dropna().head(10)
                if len(sample_values) > 0:
                    try:
                        pd.to_datetime(sample_values, dayfirst=True, errors='raise')
                        datetime_cols.append(col)
                        categorical_cols.remove(col)
                    except:
                        pass
        context = f"""
INFORMACIÓN DETALLADA DEL DATAFRAME 'df':
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 ESTRUCTURA GENERAL:
- Dimensiones: {self.df.shape[0]:,} filas × {self.df.shape[1]} columnas
- Memoria utilizada: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB
- Valores nulos totales: {self.df.isnull().sum().sum():,}
📈 COLUMNAS NUMÉRICAS ({len(numeric_cols)}):
{numeric_cols}
🏷️ COLUMNAS CATEGÓRICAS ({len(categorical_cols)}):
{categorical_cols}
📅 COLUMNAS DE FECHA ({len(datetime_cols)}):
{datetime_cols}
📋 MUESTRA DE DATOS (primeras 3 filas):
{self.df.head(3).to_string()}
📊 ESTADÍSTICAS BÁSICAS:
{self.df.describe(include='all').fillna('N/A').to_string()}
💾 INFORMACIÓN DE VALORES ÚNICOS:
{pd.DataFrame({'Columna': self.df.columns, 'Valores_únicos': self.df.nunique(), 'Tipo': self.df.dtypes}).to_string()}
"""
        if self.df_dict is not None and not self.df_dict.empty:
            context += f"\n\n📚 DICCIONARIO DE VARIABLES DISPONIBLE:\n{self.df_dict.to_string()}"
        return context

    def generate_prompt_with_vectorstore(self, context, user_query):
        self.index_metadata_dictionary()
        fragmentos_relevantes = self.retrieve_relevant_fragments(user_query)
        return f"""
Contexto del DataFrame:
{context}

Fragmentos relevantes del diccionario de variables:
{fragmentos_relevantes}

Consulta del usuario:
{user_query}
"""

    def clean_generated_code(self, codigo):
        """Limpiar código generado eliminando DataFrames ficticios y carga de archivos"""
        if self.df is None:
            return codigo
        
        # Patrones problemáticos a eliminar
        problematic_patterns = [
            "pd.read_csv(",
            "pd.read_excel(",
            "pd.read_json(",
            "pd.DataFrame(",
            "df = pd.DataFrame",
            "data = pd.DataFrame",
            "datos = pd.DataFrame",
            ".read_csv(",
            ".read_excel(",
            "pd.read_",
            "import pandas as pd",
            "import numpy as np",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns"
        ]
        
        lines = codigo.split('\n')
        cleaned_lines = []
        skip_block = False
        bracket_count = 0
        
        for line in lines:
            line_stripped = line.strip()
            
            # Saltar líneas de importación (ya están disponibles)
            if line_stripped.startswith(('import pandas', 'import numpy', 'import matplotlib', 'import seaborn')):
                continue
            
            # Detectar patrones problemáticos
            if any(pattern in line for pattern in problematic_patterns):
                st.warning(f"⚠️ Línea problemática eliminada: {line_stripped}")
                skip_block = True
                bracket_count = line.count('(') - line.count(')')
                if bracket_count <= 0:
                    skip_block = False
                continue
            
            # Si estamos saltando un bloque
            if skip_block:
                bracket_count += line.count('(') - line.count(')')
                if bracket_count <= 0:
                    skip_block = False
                continue
            
            # Corregir errores comunes con fechas
            if '.days' in line and not '.dt.days' in line:
                line = line.replace('.days', '.dt.days')
                st.info(f"🔧 Corregido error de fecha: {line_stripped}")

            # Asegurar dayfirst=True en conversiones de fecha
            if 'pd.to_datetime(' in line and 'dayfirst=' not in line:
                line = line.replace('pd.to_datetime(', 'pd.to_datetime(').replace(')', ', dayfirst=True)')
                st.info(f"🔧 Agregado dayfirst=True: {line_stripped}")
            
            # Mantener líneas válidas
            cleaned_lines.append(line)
        
        cleaned_code = '\n'.join(cleaned_lines)
        
        # Verificación adicional para asegurar que no hay cargas de archivos
        if any(pattern in cleaned_code for pattern in ["read_csv", "read_excel", "read_json"]):
            st.error("❌ El código generado intenta cargar archivos. Esto ha sido bloqueado.")
            return "print('Error: El código intentaba cargar archivos cuando el DataFrame ya está disponible')"
        
        return cleaned_code

    def execute_code_safely(self, codigo):
        """Ejecutar código de forma segura con mejor manejo de errores"""
        codigo = self.clean_generated_code(codigo)
        
        # Preparar variables locales
        local_vars = {
            "df": self.df,
            "df_dict": self.df_dict,
            "df_examples": self.df_examples,
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "px": px,
            "go": go,
            "make_subplots": make_subplots,
            "datetime": datetime,
            "st": st  # Agregar st para que el código pueda usar st.pyplot() si es necesario
        }
        
        # Limpiar figuras previas
        plt.close('all')
        
        # Capturar output
        old_stdout = sys.stdout
        buffer = io.StringIO()
        sys.stdout = buffer
        
        try:
            # Verificar y corregir código problemático con Series y .days
            codigo_corregido = self.fix_series_days_operations(codigo)
            exec(codigo_corregido, {}, local_vars)
            success = True
            error_msg = None
            
            # Capturar figuras de matplotlib si existen
            figures = []
            if plt.get_fignums():
                for fignum in plt.get_fignums():
                    fig = plt.figure(fignum)
                    figures.append(fig)
            
            # Almacenar figuras en local_vars para acceso posterior
            local_vars['_figures'] = figures
            
        except Exception as e:
            success = False
            error_msg = str(e)
            print(f"❌ Error ejecutando código: {e}")
            local_vars['_figures'] = []
        finally:
            sys.stdout = old_stdout
        
        output = buffer.getvalue()
        return output, local_vars, success, error_msg

    def fix_series_days_operations(self, codigo):
        """Corregir operaciones problemáticas con Series, fechas y .days"""
        import re
        
        # Patrones problemáticos comunes con .days en Series
        patterns_to_fix = [
            # Patrón: (fecha1 - fecha2).days - Solo agregar .dt si no existe
            (r'\(([^)]+)\)\.days(?![\w.])', r'(\1).dt.days'),
            # Patrón: variable.days donde variable es una Serie - Solo si no tiene .dt
            (r'([a-zA-Z_][a-zA-Z0-9_]*(?:\[[^\]]+\])?)(?<!\.dt)\.days(?!\w)', r'\1.dt.days'),
            # Patrón: df['col'].days - Solo si no tiene .dt
            (r'(df\[[\'"][^\'"]+[\'"]\])(?<!\.dt)\.days(?!\w)', r'\1.dt.days'),
        ]
        
        codigo_corregido = codigo
        
        # Aplicar correcciones de .days evitando duplicar .dt
        for pattern, replacement in patterns_to_fix:
            # Verificar que no estemos duplicando .dt
            if re.search(pattern, codigo_corregido) and '\.dt\.dt\.' not in codigo_corregido:
                codigo_corregido = re.sub(pattern, replacement, codigo_corregido)
                st.info(f"🔧 Corrigiendo operación .days en Serie")
        
        # Corregir operaciones de fechas incompatibles
        codigo_corregido = self.fix_datetime_operations(codigo_corregido)
        
        # Limpiar cualquier .dt.dt duplicado que pueda haberse creado
        codigo_corregido = self.clean_duplicate_dt(codigo_corregido)
        
        return codigo_corregido

    def clean_duplicate_dt(self, codigo):
        """Limpiar operaciones .dt.dt duplicadas"""
        import re
        
        # Remover .dt.dt duplicados
        duplicates_fixed = [
            (r'\.dt\.dt\.', r'.dt.'),
            (r'\.dt\.dt\.dt\.', r'.dt.'),
        ]
        
        codigo_corregido = codigo
        
        for pattern, replacement in duplicates_fixed:
            if re.search(pattern, codigo_corregido):
                codigo_corregido = re.sub(pattern, replacement, codigo_corregido)
                st.info("🔧 Corrigiendo .dt duplicados")
        
        return codigo_corregido

    def fix_datetime_operations(self, codigo):
        """Corregir operaciones incompatibles entre DatetimeArray y datetime.date"""
        import re
        
        # Patrones comunes de operaciones problemáticas con fechas
        datetime_fixes = [
            # datetime.date.today() en operaciones con Series
            (r'datetime\.date\.today\(\)', r'pd.Timestamp.today()'),
            # date.today() en operaciones con Series  
            (r'date\.today\(\)', r'pd.Timestamp.today()'),
            # datetime.now().date() en operaciones con Series
            (r'datetime\.now\(\)\.date\(\)', r'pd.Timestamp.now()'),
        ]
        
        codigo_corregido = codigo
        
        for pattern, replacement in datetime_fixes:
            if re.search(pattern, codigo_corregido):
                codigo_corregido = re.sub(pattern, replacement, codigo_corregido)
                st.info(f"🔧 Corrigiendo operación de fecha: {pattern} → {replacement}")
        
        # Corregir operaciones aritméticas con Timestamp y enteros
        codigo_corregido = self.fix_timestamp_arithmetic(codigo_corregido)
        
        # Si detectamos operaciones de resta con fechas, agregar conversión explícita
        if re.search(r'df\[[^\]]+\]\s*-\s*(datetime\.|date\.)', codigo_corregido):
            st.info("🔧 Se detectó resta entre Series de fechas y objetos datetime. Usando pd.Timestamp para compatibilidad.")
        
        return codigo_corregido

    def fix_timestamp_arithmetic(self, codigo):
        """Corregir operaciones aritméticas problemáticas con Timestamp"""
        import re
        
        # Patrones de operaciones aritméticas problemáticas
        arithmetic_fixes = [
            # Timestamp + número → Timestamp + pd.Timedelta(days=número)
            (r'(pd\.Timestamp[^+\-]*)\s*\+\s*(\d+)(?!\s*\*)', r'\1 + pd.Timedelta(days=\2)'),
            # Timestamp - número → Timestamp - pd.Timedelta(days=número)  
            (r'(pd\.Timestamp[^+\-]*)\s*-\s*(\d+)(?!\s*\*)', r'\1 - pd.Timedelta(days=\2)'),
            # Series de fechas + número → Series + pd.Timedelta(days=número)
            (r'(df\[[^\]]+\])\s*\+\s*(\d+)(?!\s*\*)', r'\1 + pd.Timedelta(days=\2)'),
            # Series de fechas - número → Series - pd.Timedelta(days=número)
            (r'(df\[[^\]]+\])\s*-\s*(\d+)(?!\s*\*)', r'\1 - pd.Timedelta(days=\2)'),
        ]
        
        codigo_corregido = codigo
        
        for pattern, replacement in arithmetic_fixes:
            if re.search(pattern, codigo_corregido):
                codigo_corregido = re.sub(pattern, replacement, codigo_corregido)
                st.info(f"🔧 Corrigiendo aritmética de fechas: usando pd.Timedelta()")
        
        # Casos especiales: detectar si hay variables que representan números de días
        # y están siendo sumadas/restadas a fechas
        lines = codigo_corregido.split('\n')
        for i, line in enumerate(lines):
            # Buscar patrones como: fecha + variable_numerica
            if re.search(r'(pd\.Timestamp|df\[[^\]]+\])\s*[+\-]\s*[a-zA-Z_][a-zA-Z0-9_]*(?!\()', line):
                # Si no ya tiene pd.Timedelta, sugerir la corrección
                if 'pd.Timedelta' not in line and ('+ ' in line or '- ' in line):
                    st.warning(f"⚠️ Línea {i+1}: Posible operación aritmética problemática con fechas. Considera usar pd.Timedelta()")
        
        return codigo_corregido

    def generate_quick_insights(self):
        """Generar insights rápidos del dataset"""
        if self.df is None:
            return
        
        st.markdown("### 🔍 Insights Rápidos del Dataset")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Registros", f"{len(self.df):,}")
        
        with col2:
            st.metric("📈 Columnas Numéricas", len(self.df.select_dtypes(include=[np.number]).columns))
        
        with col3:
            missing_pct = (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100)
            st.metric("❓ % Datos Faltantes", f"{missing_pct:.1f}%")
        
        with col4:
            duplicates = self.df.duplicated().sum()
            st.metric("🔄 Duplicados", f"{duplicates:,}")
        
        # Gráfico de valores faltantes si existen
        if self.df.isnull().sum().sum() > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            missing_data = self.df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            if len(missing_data) > 0:
                missing_data.plot(kind='bar', ax=ax, color='salmon')
                ax.set_title('Valores Faltantes por Columna')
                ax.set_xlabel('Columnas')
                ax.set_ylabel('Cantidad de Valores Faltantes')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

    def run_app(self):
        """Ejecutar aplicación principal"""
        # Cargar interfaz
        datos_file, dict_file, ejemplos_file, model, temperature, max_tokens = self.load_interface()
        
        # Inicializar cliente
        if not self.init_client():
            st.error("❌ No se pudo conectar con Ollama. Verifica que esté ejecutándose.")
            return
        
        # Cargar datos
        self.load_data(datos_file, dict_file, ejemplos_file)
        
        # Título principal
        st.title("🧠 Análisis Inteligente de Datos con IA")
        st.markdown("---")
        
        # Mostrar insights rápidos si hay datos
        if self.df is not None:
            self.generate_quick_insights()
            st.markdown("---")
        
        # Initialize user_query in session state if not exists
        if "user_query" not in st.session_state:
            st.session_state.user_query = ""
        
        # Área principal de consulta
        with st.form("prompt_form"):
            objetivo = st.text_input("Objetivo del análisis (ej. analizar ventas mensuales)", key="objetivo")
            variable_clave = st.selectbox(
                "Variable clave del análisis", 
                options=self.df.columns.tolist() if self.df is not None else [], 
                key="var_clave"
            )
            nivel = st.radio("Nivel de detalle", options=["General", "Por categoría", "Temporal"], key="nivel")
            tipo_salida = st.selectbox("Formato de salida esperado", options=["Gráfico", "Tabla", "Ambos"], key="formato")
            submitted = st.form_submit_button("Actualizar prompt")
            
            if submitted and self.df is not None:
                st.session_state.user_query = f"Analiza {objetivo} usando la variable '{variable_clave}', con un nivel de detalle {nivel.lower()}, mostrando el resultado como {tipo_salida.lower()}."
                st.success(f"✅ Consulta generada: {st.session_state.user_query}")

        # Botón para generar explicación
        if st.button("💡 Generar Explicación y Código", type="primary"):
            if self.df is None:
                st.error("❌ Necesitas cargar un archivo CSV primero.")
                return
            
            if not st.session_state.user_query:
                st.error("❌ Primero debes generar una consulta usando el formulario de arriba.")
                return
            
            with st.spinner("🤖 Generando explicación..."):
                # Generar contexto completo
                context = self.generate_enhanced_context()
                prompt_explicacion = context + f"\n\nCONSULTA DEL USUARIO:\n{st.session_state.user_query}"
                
                # Obtener explicación en lenguaje natural
                respuesta = self.client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "system", 
                            "content": """Eres un analista de datos experto. Analiza la consulta del usuario y proporciona:
                            1. Una explicación clara de qué análisis se puede realizar
                            2. Qué insights específicos se pueden obtener
                            3. Qué visualizaciones serían útiles
                            4. Consideraciones importantes sobre los datos
                            
                            Responde en español, sé conciso pero informativo. Usa bullet points para organizar la información."""
                        },
                        {"role": "user", "content": prompt_explicacion}
                    ],
                    options={"temperature": temperature, "num_predict": max_tokens}
                ).message.content
                
                st.session_state["respuesta"] = respuesta
            
            with st.spinner("🔧 Generando código optimizado..."):
                # Generar código Python
                prompt_codigo = self.generate_prompt_with_vectorstore(context, st.session_state.user_query)
                
                SYSTEM_PROMPT = f"""INSTRUCCIONES CRÍTICAS PARA GENERACIÓN DE CÓDIGO:

ASEGÚRATE DE USAR EL NOMBRE 'df' COMO DATAFRAME PRINCIPAL. No uses nombres como 'data', 'datos', ni otros.

DATAFRAME YA DISPONIBLE:
- El DataFrame 'df' YA EXISTE con {self.df.shape[0]} filas y {self.df.shape[1]} columnas
- NO CARGAR ARCHIVOS: No uses pd.read_csv(), pd.read_excel() ni similares
- NO CREAR DataFrames: No uses pd.DataFrame() para crear datos ficticios
- Columnas disponibles: {list(self.df.columns)}

MANEJO DE FECHAS - MUY IMPORTANTE:
- El formato de fecha es DD/MM/YYYY (ej: 02/05/2025)
- SIEMPRE usa pd.to_datetime(df['columna_fecha'], dayfirst=True, errors='coerce')
- Para calcular diferencias de días: (fecha2 - fecha1).dt.days
- Para calcular años: (fecha2 - fecha1).dt.days / 365.25
- NO uses .days directamente en Series, usa .dt.days
- Para extraer componentes: df['fecha'].dt.year, df['fecha'].dt.month, df['fecha'].dt.day
- Para filtrar por fecha: df[df['fecha'] >= pd.to_datetime('01/01/2024', dayfirst=True)]

ANTES DE HACER TRANSFORMACIONES DE VARIABLES CATEGÓRICAS:
- Verifica si la variable existe en 'df_dict'
- Usa la columna 'Valores' en 'df_dict' para identificar cómo deben recodificarse los valores
- Si en 'df_dict' la variable tiene 'Valores' como '0=Masculino,1=Femenino', usa exactamente ese mapeo
- RESPETA MAYÚSCULAS Y MINÚSCULAS tal como aparecen en los valores del DataFrame
- NO HAGAS supuestos como {{'M': 0, 'F': 1}} a menos que esté claramente definido en df_dict

PARA MODELOS CON COEFICIENTES:
- SIEMPRE muestra los coeficientes con el nombre del campo correspondiente
- Usa pd.DataFrame para crear una tabla con columnas 'Variable' y 'Coeficiente'
- Para regresión lineal: pd.DataFrame({{'Variable': X.columns, 'Coeficiente': modelo.coef_}})
- Para regresión logística: pd.DataFrame({{'Variable': X.columns, 'Coeficiente': modelo.coef_[0]}})
- Incluye el intercepto si existe: agregar fila con 'Intercepto' y modelo.intercept_
- Ordena por valor absoluto del coeficiente

PARA MODELOS PREDICTIVOS:
- NO generes pronósticos automáticamente a menos que se solicite explícitamente
- Enfócate en mostrar las métricas del modelo (R cuadrado, accuracy, etc.) y los coeficientes
- Solo incluye predicciones si el usuario específicamente pide "predecir", "pronosticar" o "hacer predicciones"

REGLAS DE CÓDIGO:
1. RESPONDE ÚNICAMENTE CON CÓDIGO PYTHON PURO, sin texto explicativo
2. NO INCLUYAS frases como 'Aquí está el código', 'Este código' o comentarios introductorios o finales
3. El código debe empezar directamente con sintaxis Python válida
4. NO uses triple comillas, bloques markdown ni explicaciones
5. El DataFrame 'df' YA ESTÁ CARGADO, NO INTENTES cargarlo de nuevo
6. Usa 4 espacios para indentación
7. Para visualizaciones usa matplotlib (plt) o seaborn (sns)
8. Solo usa columnas que existen realmente: {list(self.df.columns)}
9. El código debe terminar mostrando el resultado
10. Para fechas usa dayfirst=True: pd.to_datetime(df['fecha'], dayfirst=True)
11. NUNCA uses Series directamente en condicionales, usa .any(), .all(), etc.
12. Usa operadores de comparación explícitos: df['col'] == valor
13. Devuelve resultados como DataFrames o Series, no texto
14. NO incluyas comentarios, texto explicativo ni markdown
15. Evita errores de sintaxis
16. El código debe ser completo y ejecutable inmediatamente
17. NO GENERES COMENTARIOS EN EL CÓDIGO: Sin # explicativos, sin texto descriptivo
18. NO INCLUYAS EXPLICACIONES AL FINAL: Sin texto después del código ejecutable
19. EL CÓDIGO DEBE TERMINAR CON LA ÚLTIMA LÍNEA EJECUTABLE (print, display, etc.)
20. PROHIBIDO AGREGAR TEXTO EXPLICATIVO DESPUÉS DEL CÓDIGO

CRÍTICO: df YA CONTIENE DATOS REALES, NO CREAR DATOS FICTICIOS NI CARGAR ARCHIVOS.
FUNDAMENTAL: RESPUESTA DEBE SER ÚNICAMENTE CÓDIGO PYTHON EJECUTABLE, SIN COMENTARIOS NI EXPLICACIONES.
"""
                
                codigo = self.client.chat(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": SYSTEM_PROMPT
                        },
                        {"role": "user", "content": prompt_codigo}
                    ],
                    options={"temperature": temperature, "num_predict": max_tokens}
                ).message.content.strip().replace("```python", "").replace("```", "")
                
                st.session_state["codigo"] = codigo
        
        # Mostrar resultados
        if "respuesta" in st.session_state:
            st.markdown("### 2️⃣ Explicación del Análisis")
            st.write(st.session_state["respuesta"])
        
        if "codigo" in st.session_state:
            st.markdown("### 3️⃣ Código Generado")
            
            # Mostrar código en columnas
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Hacer el código editable usando text_area
                codigo_editado = st.text_area(
                    "Código Python (editable):",
                    value=st.session_state["codigo"],
                    height=400,
                    key="codigo_editor",
                    help="Puedes editar el código aquí antes de ejecutarlo"
                )
                
                # Actualizar el código en session_state si se ha modificado
                if codigo_editado != st.session_state["codigo"]:
                    st.session_state["codigo_modificado"] = codigo_editado
                    st.info("✏️ Código modificado. Presiona 'Ejecutar Código' para usar la versión editada.")
            
            with col2:
                if st.button("▶️ Ejecutar Código", type="secondary"):
                    with st.spinner("⚡ Ejecutando análisis..."):
                        # Usar código modificado si existe, sino usar el original
                        codigo_a_ejecutar = st.session_state.get("codigo_modificado", st.session_state["codigo"])
                        
                        output, variables, success, error_msg = self.execute_code_safely(codigo_a_ejecutar)
                        
                        st.session_state["output"] = output
                        st.session_state["variables"] = variables  # LÍNEA AGREGADA
                        st.session_state["success"] = success
                        st.session_state["error_msg"] = error_msg
                        st.session_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["codigo_ejecutado"] = codigo_a_ejecutar
        
        # Mostrar resultados de ejecución
        if "output" in st.session_state:
            st.markdown("### 4️⃣ Resultados del Análisis")
            
            if st.session_state["success"]:
                # Mostrar output de texto
                if st.session_state["output"].strip():
                    st.text(st.session_state["output"])
                
                # Mostrar gráficos si existen - SECCIÓN CORREGIDA
                if "variables" in st.session_state and "_figures" in st.session_state["variables"]:
                    figures = st.session_state["variables"]["_figures"]
                    if figures:
                        st.markdown("#### 📊 Visualizaciones")
                        for i, fig in enumerate(figures):
                            st.pyplot(fig)
                            st.markdown(f"*Gráfico {i+1}*")
                        # Limpiar figuras después de mostrarlas
                        plt.close('all')
                
                # Verificar si hay figuras activas de matplotlib (fallback)
                elif plt.get_fignums():
                    st.markdown("#### 📊 Visualizaciones")
                    for fignum in plt.get_fignums():
                        fig = plt.figure(fignum)
                        st.pyplot(fig)
                    plt.close('all')
                
                # Generar interpretación
                st.markdown("### 5️⃣ Interpretación de Resultados")
                with st.spinner("🧠 Interpretando resultados..."):
                    interpretacion = self.client.chat(
                        model=model,
                        messages=[
                            {
                                "role": "system",
                                "content": """Interpreta los resultados del análisis de datos de manera clara y profesional.
                                Incluye:
                                - Qué muestran los números/estadísticas
                                - Insights clave y patrones identificados
                                - Implicaciones prácticas
                                - Recomendaciones basadas en los hallazgos
                                - Limitaciones o consideraciones importantes
                                
                                IMPORTANTE: Si ves coeficientes en los resultados, interpreta correctamente su signo:
                                - Coeficiente POSITIVO = cuando la variable aumenta, la variable dependiente también aumenta
                                - Coeficiente NEGATIVO = cuando la variable aumenta, la variable dependiente disminuye
                                - No contradigas el signo del coeficiente en tu interpretación
                                - Usa lenguaje técnico pero accesible."""
                            },
                            {"role": "user", "content": f"Resultados del análisis:\n{st.session_state['output']}"}
                        ],
                        options={"temperature": temperature, "num_predict": max_tokens}
                    ).message.content
                    
                    st.write(interpretacion)
                
                # Guardar en historial
                if "historial" not in st.session_state:
                    st.session_state.historial = []
                
                st.session_state.historial.append({
                    "timestamp": st.session_state["timestamp"],
                    "consulta": st.session_state.user_query,
                    "respuesta": st.session_state["respuesta"],
                    "codigo": st.session_state["codigo"],
                    "output": st.session_state["output"],
                    "interpretacion": interpretacion
                })
                
            else:
                st.error(f"❌ Error en la ejecución: {st.session_state['error_msg']}")
                st.markdown("**Código que causó el error:**")
                st.code(st.session_state["codigo"], language="python")
                
                # Sugerencias para solucionar el error
                st.markdown("**💡 Posibles soluciones:**")
                if "No such file or directory" in st.session_state['error_msg']:
                    st.warning("🚨 El código está intentando cargar un archivo. El DataFrame 'df' ya está disponible en memoria.")
                elif "KeyError" in st.session_state['error_msg']:
                    st.warning(f"🚨 El código está usando una columna que no existe. Columnas disponibles: {list(self.df.columns)}")
                elif "NameError" in st.session_state['error_msg']:
                    st.warning("🚨 El código está usando una variable no definida. Revisa los nombres de variables.")
                else:
                    st.info("🔍 Intenta reformular tu consulta o ser más específico sobre lo que necesitas analizar.")
        
        # Historial de análisis
        if "historial" in st.session_state and st.session_state.historial:
            st.markdown("## 📜 Historial de Análisis")

            for i, h in enumerate(reversed(st.session_state.historial)):
                num = len(st.session_state.historial) - i
                with st.expander(f"🕒 Análisis #{num} - {h['timestamp']}", expanded=False):
                    st.markdown(f"**Consulta:** {h['consulta']}")
                    st.markdown("**Código:**")
                    st.code(h['codigo'], language="python")
                    st.markdown("**Resultados:**")
                    st.text(h['output'])
                    st.markdown("**Interpretación:**")
                    st.write(h['interpretacion'])
                    st.markdown("---")

# Ejecutar aplicación
if __name__ == "__main__":
    app = DataAnalysisApp()
    app.run_app()