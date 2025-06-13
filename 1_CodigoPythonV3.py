import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ollama import Client

def init_client():
    try:
        client = Client(host="http://localhost:11435")
        client.list()
    except Exception:
        client = None
    return client

def load_interface():
    st.set_page_config(page_title="Análisis de información con IA", layout="wide")
    try:
        st.sidebar.image(r"D:/Julio/ExperienciaAnalitica/IA/EASinFondo.png", use_container_width=False, width=250)
    except Exception as e:
        st.sidebar.error(f"❌ No se pudo cargar la imagen de logo: {e}")
    st.sidebar.markdown(
        """
        <style>
        [data-testid="stSidebar"] > div { background-color: #d4d5d9; color: black; }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.sidebar.expander("Archivos de datos", expanded=True):
        datos_file = st.file_uploader("Datos (CSV)", type="csv", key="datos")
        dict_file = st.file_uploader("Diccionario (CSV)", type="csv", key="diccionario")
        ejemplos_file = st.file_uploader("Ejemplos (XLSX)", type="xlsx", key="ejemplos")
    with st.sidebar.expander("Configuración del modelo", expanded=False):
        model_options = ["deepseek-coder-v2:16b", "deepseek-r1:7b", "codellama", "qwen3:8b", "llama3", "starcoder2:15b"]
        model = st.selectbox("Modelo Ollama", options=model_options, index=0)
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.1)
        max_tokens = st.slider("Máximo de predicciones", 50, 1000, 200)
    return datos_file, dict_file, ejemplos_file, model, temperature, max_tokens

def load_data(datos_file, dict_file, ejemplos_file):
    df = df_dict = df_examples = None
    if datos_file:
        try:
            df = pd.read_csv(datos_file)
            st.sidebar.success(f"✅ DataFrame 'df' cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando Datos: {e}")
    if dict_file:
        try:
            df_dict = pd.read_csv(dict_file)
            st.sidebar.success(f"✅ DataFrame 'df_dict' cargado: {df_dict.shape[0]} filas x {df_dict.shape[1]} columnas")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando Diccionario: {e}")
    if ejemplos_file:
        try:
            df_examples = pd.read_excel(ejemplos_file)
            st.sidebar.success(f"✅ DataFrame 'df_examples' cargado: {df_examples.shape[0]} filas x {df_examples.shape[1]} columnas")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando Ejemplos: {e}")
    if df is not None and df_dict is not None and df_examples is not None:
        st.sidebar.success("ℹ️ Información cargada completamente")
    if df is not None:
        with st.expander("Vista previa de datos", expanded=False):
            st.dataframe(df.head(5))
    if df_dict is not None:
        with st.expander("Diccionario de datos", expanded=False):
            width = min(len(df_dict.columns), 5) * 200
            st.dataframe(df_dict, width=width)
    return df, df_dict, df_examples

def build_prompt(user_query: str, df: pd.DataFrame, df_dict: pd.DataFrame, df_examples: pd.DataFrame) -> str:
    summary = (
        "# Resumen de DataFrame 'df'\n"
        f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n"
        f"Columnas: {', '.join(df.columns)}\n\n"
        "# DataFrame 'df_dict' (diccionario) y 'df_examples' (ejemplos) disponibles para referencia.\n\n"
    )
    return summary + user_query

def main():
    datos_file, dict_file, ejemplos_file, model, temperature, max_tokens = load_interface()
    client = init_client()
    if client:
        st.success("✅ Conexión con Ollama (puerto 11435) establecida")
    else:
        st.error("❌ No se pudo conectar con Ollama (puerto 11435)")
    df, df_dict, df_examples = load_data(datos_file, dict_file, ejemplos_file)
    st.subheader("Consulta al LLM")
    user_query = st.text_area("Escriba su consulta aquí:", height=200)
    if st.button("🚀 Enviar consulta"):
        if not client:
            st.error("❌ Cliente Ollama no disponible")
            return
        if df is None or df_dict is None or df_examples is None:
            st.warning("⚠️ Asegúrese de cargar 'df', 'df_dict' y 'df_examples' antes de enviar")
            return
        if not user_query.strip():
            st.warning("⚠️ Escriba su consulta antes de enviar")
            return
        prompt = build_prompt(user_query, df, df_dict, df_examples)
        try:
            instructions = (
                "1. RESPONDE ÚNICAMENTE CON CÓDIGO PYTHON PURO, sin texto explicativo ni comentarios introductorios.\n"
                "2. NO INCLUYAS frases como \"Aquí está el código\" o \"Para calcular...\".\n"
                "3. El código debe empezar directamente con sintaxis Python válida (sin texto antes).\n"
                "4. NO uses triple comillas, bloques markdown ni explicaciones antes o después del código.\n"
                "5. Asume que el DataFrame YA está cargado como 'df', NO INTENTES cargarlo de nuevo.\n"
                "6. Asegúrate de que el código tenga una indentación correcta y consistente.\n"
                "7. Usa 4 espacios para la indentación, no tabulaciones.\n"
                "8. Si necesitas generar visualizaciones, usa matplotlib o seaborn (ya importados como plt y sns).\n"
                "9. NO ALUCINES: solo usa las columnas y datos que existen realmente en el DataFrame.\n"
                "10. El código debe terminar imprimiendo o mostrando el resultado final.\n"
                "11. Si trabajas con fechas, SIEMPRE usa dayfirst=True al convertir: pd.to_datetime(df['fecha'], dayfirst=True)\n"
                "12. Al procesar fechas, asume formato europeo (día/mes/año) a menos que se indique lo contrario.\n"
                "13. NUNCA uses Series de pandas directamente en condicionales como 'if df['columna']:', usa métodos específicos como .any(), .all(), .empty, o compara con valores específicos.\n"
                "14. Si necesitas comprobar valores en una Serie, usa operadores de comparación explícitos: df['columna'] == valor, df['columna'] > valor, etc.\n"
                "15. SIEMPRE devuelve los resultados como DataFrames o Series de pandas en lugar de imprimir texto.\n"
                "16. Si usas scikit-learn y específicamente OneHotEncoder, usa pd.get_dummies() si no estás seguro de la versión.\n"
                "17. Si requieres más contexto, pregúntalo antes de generar el código de Python.\n"
                "18. Si tienes sugerencias para mejorar el prompt, hazlas.\n"
            )
            system_prompt = (
                "Genere solo código Python ejecutable usando el DataFrame 'df'. "
                "No incluya lecturas de archivos ni definiciones de 'df'. "
                "Use 'df_dict' para mapear nombres de columnas y 'df_examples' para referencias de código.\n\n"
                f"{instructions}"   
            )
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            raw_content = response.message.content
            clean_response = raw_content.lstrip()
            code = clean_response.replace("```python", "").replace("```", "").strip()
            st.subheader("Código generado por Ollama")
            st.code(code, language="python")
            st.subheader("Resultado de la ejecución")
            try:
                import io, sys
                buf = io.StringIO()
                sys_stdout = sys.stdout
                sys.stdout = buf
                local_vars = {"df": df, "df_dict": df_dict, "df_examples": df_examples, "pd": pd, "np": np, "plt": plt, "sns": sns}
                exec(code, {}, local_vars)
                sys.stdout = sys_stdout
                output = buf.getvalue()
                if plt.get_fignums():
                    st.pyplot(plt.gcf())
                    plt.clf()
                if output:
                    st.text(output)
                elif "result" in local_vars:
                    st.write(local_vars["result"])
                elif not output and not plt.get_fignums():
                    st.write(local_vars)
            except Exception as e:
                st.error(f"❌ Error al ejecutar el código: {e}")
        except Exception as e:
            st.error(f"❌ Error al generar el código: {e}")

if __name__ == "__main__":
    main()
