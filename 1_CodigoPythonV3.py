import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ollama import Client
import io, sys
from datetime import datetime

# Inicializar cliente Ollama
def init_client():
    try:
        client = Client(host="http://localhost:11435")
        client.list()
    except Exception:
        client = None
    return client

# Cargar interfaz de usuario
def load_interface():
    st.set_page_config(page_title="Análisis con IA", layout="wide")
    try:
        st.sidebar.image(r"D:/Julio/ExperienciaAnalitica/IA/EASinFondo.png", width=250)
    except Exception as e:
        st.sidebar.error(f"❌ No se pudo cargar la imagen: {e}")
    st.sidebar.markdown("""
        <style>[data-testid=\"stSidebar\"] > div { background-color: #d4d5d9; color: black; }</style>
    """, unsafe_allow_html=True)
    with st.sidebar.expander("Archivos de datos", expanded=True):
        datos_file = st.file_uploader("Datos (CSV)", type="csv", key="datos")
        dict_file = st.file_uploader("Diccionario (CSV)", type="csv", key="diccionario")
        ejemplos_file = st.file_uploader("Ejemplos (XLSX)", type="xlsx", key="ejemplos")
    model = st.sidebar.selectbox("Modelo de IA", ["deepseek-coder-v2:16b", "llama3", "qwen3:8b", "deepseek-r1:7b"], index=0)
    temperature = st.sidebar.slider("Temperatura", 0.0, 1.0, 0.1)
    max_tokens = st.sidebar.slider("Máximo de predicciones", 50, 4000, 800)
    return datos_file, dict_file, ejemplos_file, model, temperature, max_tokens

# Cargar archivos cargados por el usuario
def load_data(datos_file, dict_file, ejemplos_file):
    df = df_dict = df_examples = None
    if datos_file:
        try:
            df = pd.read_csv(datos_file)
            st.sidebar.success(f"✅ 'df' cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando 'df': {e}")
    if dict_file:
        try:
            df_dict = pd.read_csv(dict_file)
            st.sidebar.success(f"✅ 'df_dict' cargado")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando 'df_dict': {e}")
    if ejemplos_file:
        try:
            df_examples = pd.read_excel(ejemplos_file)
            st.sidebar.success(f"✅ 'df_examples' cargado")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando 'df_examples': {e}")
    return df, df_dict, df_examples

# Ejecutar código generado y mostrar resultados
def ejecutar_codigo(codigo, df, df_dict, df_examples):
    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer
    local_vars = {"df": df, "df_dict": df_dict, "df_examples": df_examples, "pd": pd, "np": np, "plt": plt, "sns": sns}
    try:
        exec(codigo, {}, local_vars)
    except Exception as e:
        print(f"❌ Error ejecutando el código: {e}")
    sys.stdout = old_stdout
    return buffer.getvalue(), local_vars

# Ejecutar aplicación principal
def main():
    datos_file, dict_file, ejemplos_file, model, temperature, max_tokens = load_interface()
    client = init_client()
    if not client:
        st.error("❌ No se pudo conectar con Ollama")
        return
    df, df_dict, df_examples = load_data(datos_file, dict_file, ejemplos_file)

    st.title("🧠 Consulta y generación de código con IA")

    user_query = st.text_area("1️⃣ Consulta (Prompt inicial):", height=150, key="consulta")
    if not user_query:
        return

    if st.button("💬 Generar explicación"):
        columnas_df = ', '.join(df.columns) if df is not None else ''
        resumen = f"Columnas disponibles en el DataFrame: {columnas_df}\n"
        prompt_base = resumen + user_query

        respuesta = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Responde en lenguaje natural. No generes código. Usa únicamente las columnas existentes en 'df'. Sé concreto pero claro. Para cada propuesta de análisis, presenta la idea en forma de bullet e incluye una explicación concisa que la justifique."},
                {"role": "user", "content": prompt_base}
            ],
            options={"temperature": temperature, "num_predict": max_tokens}
        ).message.content

        st.session_state["respuesta"] = respuesta

        prompt_opt = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": f"Mejora el siguiente prompt para generar código Python usando exclusivamente las columnas disponibles en el DataFrame: {columnas_df}. No inventes columnas. Si vas a crear un ejemplo, asegúrate de que todos los nombres de columnas sean válidos y coincidan exactamente con los nombres existentes en 'df' (respetando mayúsculas y minúsculas)."},
                {"role": "user", "content": user_query}
            ],
            options={"temperature": temperature, "num_predict": max_tokens}
        ).message.content.strip()

        st.session_state["prompt_opt"] = prompt_opt

    if "respuesta" in st.session_state:
        st.markdown("### 2️⃣ Respuesta en lenguaje natural")
        st.write(st.session_state["respuesta"])

        if df is not None:
            with st.expander("### 🧾 Estadísticas del DataFrame", expanded=False):
                # Aplicar tipos categóricos según df_dict
                if df_dict is not None and 'Variable' in df_dict.columns and 'Tipo' in df_dict.columns:
                    cat_vars = df_dict[df_dict['Tipo'].str.lower().isin(['categórica', 'categorica', 'category'])]['Variable'].values
                    for col in cat_vars:
                        if col in df.columns:
                            df[col] = df[col].astype('category')

                # Convertir automáticamente columnas con fechas en formato adecuado
                for col in df.columns:
                    if df[col].dtype == object:
                        try:
                            df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                        except:
                            pass
                # Descripción numérica
                desc = df.describe(include='all').transpose()
                desc["Valores perdidos"] = df.isnull().sum()
                desc["Valores atípicos"] = ((df.select_dtypes(include=[np.number]) > (df.select_dtypes(include=[np.number]).mean() + 3 * df.select_dtypes(include=[np.number]).std())) | (df.select_dtypes(include=[np.number]) < (df.select_dtypes(include=[np.number]).mean() - 3 * df.select_dtypes(include=[np.number]).std()))).sum()
                st.dataframe(desc)
                # Información categórica
                # Panel simple sin nesting para variables categóricas
                st.markdown("#### 📊 Variables categóricas")
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                for col in cat_cols:
                    st.markdown(f"**{col}**")
                    st.dataframe(df[col].value_counts(dropna=False).rename('Frecuencia').to_frame())
                    st.write("Valores perdidos:", df[col].isnull().sum())

        nuevo_prompt = st.text_area("### 3️⃣ Prompt optimizado para código:", value=st.session_state["prompt_opt"], key="prompt_editable")

        if st.button("▶ Ejecutar Prompt optimizado"):
            resumen = f"Columnas disponibles en el DataFrame: {', '.join(df.columns) if df is not None else ''}\n"
            prompt_final = resumen + nuevo_prompt

            instrucciones = """
ASEGÚRATE DE USAR EL NOMBRE 'df' COMO DATAFRAME PRINCIPAL. No uses nombres como 'data', 'datos', ni otros.

ANTES DE HACER TRANSFORMACIONES DE VARIABLES CATEGÓRICAS:
- Verifica si la variable existe en 'df_dict'.
- Usa la columna 'Valores' en 'df_dict' para identificar cómo deben recodificarse los valores.
- Si en 'df_dict' la variable 'sexo' tiene como 'Valores' la cadena '0=Masculino,1=Femenino', usa exactamente ese mapeo.
- RESPETA MAYÚSCULAS Y MINÚSCULAS tal como aparecen en los valores del DataFrame. No asumas que 'F' es igual a 'f', ni 'M' igual a 'm'.
- NO HAGAS supuestos como {'M': 0, 'F': 1} a menos que esté claramente definido en df_dict.

ANTES DE HACER TRANSFORMACIONES DE VARIABLES CATEGÓRICAS, verifica si la variable existe en 'df_dict'.
Si existe, usa la columna 'Valores' en 'df_dict' para identificar cómo deben recodificarse los valores.
Ejemplo: si en df_dict la variable 'sexo' tiene como 'Valores' la cadena '0=Masculino,1=Femenino', debes usar ese mapeo al transformar la variable 'sexo'.
NO HAGAS suposiciones como {'M': 0, 'F': 1} a menos que esté claramente definido en df_dict.

PARA MODELOS CON COEFICIENTES:
- SIEMPRE muestra los coeficientes con el nombre del campo correspondiente.
- Usa pd.DataFrame para crear una tabla con columnas 'Variable' y 'Coeficiente'.
- Para modelos de regresión lineal: pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_})
- Para modelos logísticos: pd.DataFrame({'Variable': X.columns, 'Coeficiente': modelo.coef_[0]})
- Incluye el intercepto si existe: agregar fila con 'Intercepto' y modelo.intercept_
- Ordena por valor absoluto del coeficiente para mostrar las variables más importantes primero.

PARA MODELOS PREDICTIVOS:
- NO generes pronósticos automáticamente a menos que se solicite explícitamente.
- Enfócate en mostrar las métricas del modelo (R², accuracy, etc.) y los coeficientes.
- Solo incluye predicciones si el usuario específicamente pide "predecir", "pronosticar" o "hacer predicciones".
- Muestra estadísticas de evaluación del modelo en lugar de predicciones por defecto.

RESPONDE ÚNICAMENTE CON CÓDIGO PYTHON PURO, sin texto explicativo ni comentarios introductorios.
2. NO INCLUYAS frases como 'Aquí está el código' o 'Para calcular...'.
3. El código debe empezar directamente con sintaxis Python válida (sin texto antes).
4. NO uses triple comillas, bloques markdown ni explicaciones antes o después del código.
5. Asume que el DataFrame YA está cargado como 'df', NO INTENTES cargarlo de nuevo.
6. Asegúrate de que el código tenga una indentación correcta y consistente.
7. Usa 4 espacios para la indentación, no tabulaciones.
8. Si necesitas generar visualizaciones, usa matplotlib o seaborn (ya importados como plt y sns).
9. NO ALUCINES: solo usa las columnas y datos que existen realmente en el DataFrame.
10. El código debe terminar imprimiendo o mostrando el resultado final.
11. Si trabajas con fechas, SIEMPRE usa dayfirst=True al convertir: pd.to_datetime(df['fecha'], dayfirst=True).
12. Al procesar fechas, asume formato europeo (día/mes/año) a menos que se indique lo contrario.
13. NUNCA uses Series de pandas directamente en condicionales como 'if df['columna']:', usa métodos específicos como .any(), .all(), .empty o compara con valores específicos.
14. Si necesitas comprobar valores en una Serie, usa operadores de comparación explícitos: df['columna'] == valor, df['columna'] > valor, etc.
15. SIEMPRE devuelve los resultados como DataFrames o Series de pandas en lugar de imprimir texto.
17. Si requieres más contexto, pregúntalo antes de generar el código de Python.
18. Si tienes sugerencias para mejorar el prompt, hazlas.
19. Asegúrate de generar el código completo y sin truncarlo en ninguna parte.
20. NO INCLUYAS ningún comentario, texto explicativo, encabezado ni markdown.
21. El código debe comenzar directamente con sintaxis Python válida.
22. Evita errores de sintaxis por cadenas de texto u observaciones adicionales.
"""

            codigo = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": instrucciones},
                    {"role": "user", "content": prompt_final}
                ],
                options={"temperature": temperature, "num_predict": max_tokens}
            ).message.content.strip().replace("```python", "").replace("```", "")

            st.session_state["codigo"] = codigo

            salida, variables = ejecutar_codigo(codigo, df, df_dict, df_examples)
            st.session_state["salida"] = salida
            st.session_state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if "codigo" in st.session_state:
            st.markdown("### 4️⃣ Código generado")
            st.code(st.session_state["codigo"], language="python")

        if "salida" in st.session_state:
            st.markdown("### 5️⃣ Resultado de la ejecución")
            if st.session_state["salida"].strip():
                st.text(st.session_state["salida"])
            elif plt.get_fignums():
                st.pyplot(plt.gcf()); plt.clf()

            if "historial" not in st.session_state:
                st.session_state.historial = []
            st.session_state.historial.append({
                "consulta": user_query,
                "respuesta": st.session_state["respuesta"],
                "prompt": nuevo_prompt,
                "codigo": st.session_state["codigo"],
                "salida": st.session_state["salida"],
                "timestamp": st.session_state["timestamp"]
            })

            st.markdown("### 6️⃣ Interpretación de resultados")
            interpretacion = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": "Eres un consultor de negocios explicando resultados de análisis de datos a ejecutivos sin conocimientos técnicos. Explica los resultados de manera clara y práctica, enfocándote en: 1) Qué significan los números para el negocio, 2) Cuáles son los factores más importantes, 3) Qué acciones se pueden tomar basándose en estos resultados. Usa un lenguaje sencillo, evita jerga técnica, y conecta siempre los hallazgos con implicaciones de negocio. Si hay coeficientes de modelo, explica cuáles variables tienen mayor impacto y en qué dirección."},
                    {"role": "user", "content": st.session_state["salida"]}
                ],
                options={"temperature": temperature, "num_predict": max_tokens}
            ).message.content
            st.write(interpretacion)

            with st.expander("📜 Historial de ejecuciones"):
                for h in reversed(st.session_state.historial):
                    st.markdown(f"**🕒 {h['timestamp']}**")
                    st.markdown(f"- Prompt: {h['consulta']}")
                    st.markdown(f"- Prompt optimizado: {h['prompt']}")
                    st.markdown(f"- Código generado:")
                    st.code(h['codigo'], language="python")
                    st.markdown(f"- Resultado: {h['salida']}")

if __name__ == "__main__":
    main()