import streamlit as st
import pandas as pd
from ollama import Client

# 1. Inicializar cliente Ollama en puerto 11435
def init_client():
    try:
        client = Client(host="http://localhost:11435")
        client.list()
    except Exception as e:
        client = None
    return client

# 2. Configuración de interfaz: carga de archivos y parámetros en expanders
def load_interface():
    st.set_page_config(page_title="Análisis de información con IA", layout="wide")
    inject_css()

    with st.sidebar.expander("Archivos de datos", expanded=True):
        datos_file = st.file_uploader("Datos (CSV)", type="csv", key="datos")
        dict_file = st.file_uploader("Diccionario (CSV)", type="csv", key="diccionario")
        ejemplos_file = st.file_uploader("Ejemplos (XLSX)", type="xlsx", key="ejemplos")

    with st.sidebar.expander("Configuración del modelo", expanded=False):
        model_options = [
            "deepseek-coder-v2:16b",
            "deepseek-r1:7b",
            "codellama",
            "qwen3:8b",
            "llama3",
            "starcoder2:15b"
        ]
        model = st.selectbox("Modelo Ollama", options=model_options, index=0)
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.1)
        max_tokens = st.slider("Máximo de predicciones", 50, 1000, 200)

    return datos_file, dict_file, ejemplos_file, model, temperature, max_tokens

# 3. Lectura y validación de los tres archivos
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

    # Previews en expanders
    if df is not None:
        with st.expander("Vista previa de 'df'", expanded=False):
            st.dataframe(df)
    if df_examples is not None:
        with st.expander("Vista previa de 'df_examples'", expanded=False):
            st.dataframe(df_examples)
    if df_dict is not None:
        with st.expander("Diccionario de datos", expanded=False):
            width = min(len(df_dict.columns), 5) * 200
            st.dataframe(df_dict, width=width)

    return df, df_dict, df_examples

# 4. Construir prompt incorporando df y auxiliares
def build_prompt(user_query: str, df: pd.DataFrame, df_dict: pd.DataFrame, df_examples: pd.DataFrame) -> str:
    summary = (
        "# Resumen de DataFrame 'df'\n"
        f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}\n"
        f"Columnas: {', '.join(df.columns)}\n\n"
        "# DataFrame 'df_dict' (diccionario) y 'df_examples' (ejemplos) disponibles para referencia.\n\n"
    )
    return summary + user_query

# 5. Función principal
def main():
    datos_file, dict_file, ejemplos_file, model, temperature, max_tokens = load_interface()
    client = init_client()

    # Mensaje de conexión visible en panel principal
    if client is not None:
        st.success("✅ Conexión con Ollama (puerto 11435) establecida")
    else:
        st.error("❌ No se pudo conectar con Ollama (puerto 11435)")

    df, df_dict, df_examples = load_data(datos_file, dict_file, ejemplos_file)

    st.subheader("Consulta al LLM")
    user_query = st.text_area("Escriba su consulta aquí:", height=200)

    if st.button("🚀 Enviar consulta"):
        if client is None:
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
            response = client.chat(
                model=model,
                messages=[
                    {"role": "system", "content": (
                        "Genere solo código Python ejecutable usando el DataFrame 'df'. "
                        "No incluya lecturas de archivos ni definiciones de 'df'. "
                        "Use 'df_dict' para mapear nombres de columnas y 'df_examples' para referencias de código." )},
                    {"role": "user",   "content": prompt}
                ],
                options={"temperature": temperature, "num_predict": max_tokens}
            )
            raw_content = response.message.content
            clean_response = raw_content.lstrip()
            code = clean_response.replace("```python", "").replace("```", "").strip()

            st.subheader("Código generado por Ollama")
            st.code(code, language="python")

            st.subheader("Respuesta del LLM")
            st.text_area("", value=clean_response, height=200)

        except Exception as e:
            st.error(f"❌ Error al generar el código: {e}")

if __name__ == "__main__":
    main()