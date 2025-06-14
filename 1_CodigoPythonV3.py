import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ollama import Client
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

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
        [data-testid=\"stSidebar\"] > div { background-color: #d4d5d9; color: black; }
        </style>
        """,
        unsafe_allow_html=True
    )
    with st.sidebar.expander("Archivos de datos", expanded=True):
        datos_file = st.file_uploader("Datos (CSV)", type="csv", key="datos")
        dict_file = st.file_uploader("Diccionario (CSV)", type="csv", key="diccionario")
        ejemplos_file = st.file_uploader("Ejemplos (XLSX)", type="xlsx", key="ejemplos")
    mode = st.sidebar.selectbox("Modo de operación", ["Consulta general", "Generación de código"] )
    with st.sidebar.expander("Configuración del modelo", expanded=False):
        if mode == "Consulta general":
            model_options = ["llama3", "qwen3:8b", "deepseek-r1:7b"]
        else:
            model_options = ["deepseek-coder-v2:16b", "codellama", "starcoder2:15b"]
        model = st.selectbox("Modelo Ollama", model_options)
        temperature = st.slider("Temperatura", 0.0, 1.0, 0.1)
        max_tokens = st.slider("Máximo de predicciones", 50, 1000, 200)
    return datos_file, dict_file, ejemplos_file, mode, model, temperature, max_tokens

def load_data(datos_file, dict_file, ejemplos_file):
    df = df_dict = df_examples = None
    if datos_file:
        try:
            df = pd.read_csv(datos_file)
            st.sidebar.success(f"✅ DataFrame 'df' cargado: {df.shape[0]}x{df.shape[1]}")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando Datos: {e}")
    if dict_file:
        try:
            df_dict = pd.read_csv(dict_file)
            st.sidebar.success(f"✅ DataFrame 'df_dict' cargado: {df_dict.shape[0]}x{df_dict.shape[1]}")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando Diccionario: {e}")
    if ejemplos_file:
        try:
            df_examples = pd.read_excel(ejemplos_file)
            st.sidebar.success(f"✅ DataFrame 'df_examples' cargado: {df_examples.shape[0]}x{df_examples.shape[1]}")
        except Exception as e:
            st.sidebar.error(f"❌ Error cargando Ejemplos: {e}")
    return df, df_dict, df_examples

def main():
    datos_file, dict_file, ejemplos_file, mode, model, temperature, max_tokens = load_interface()
    client = init_client()
    if not client:
        st.error("❌ No se pudo conectar con Ollama")
        return
    st.success("✅ Conexión con Ollama establecida")
    df, df_dict, df_examples = load_data(datos_file, dict_file, ejemplos_file)
    st.subheader("Consulta al LLM")
    user_query = st.text_area("Escriba su consulta aquí:", height=200)
    if not st.button("🚀 Enviar consulta"):
        return
    if mode == "Consulta general":
        if df is None or df_dict is None:
            st.warning("⚠️ Cargue 'df' y 'df_dict' para este modo")
            return
        summary = f"DataFrame 'df': {df.shape[0]} filas, {df.shape[1]} columnas. Columnas: {', '.join(df.columns)}\n"
        summary += f"Diccionario 'df_dict' columnas: {', '.join(df_dict.columns)}\n\n"
        prompt = summary + user_query
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": "Responde con texto, no generes código. Los datos están en 'df' y el diccionario en 'df_dict'."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        st.markdown(response.message.content)
    else:
        if df is None or df_dict is None or df_examples is None:
            st.warning("⚠️ Cargue 'df', 'df_dict' y 'df_examples' para generación de código")
            return
        summary = (
            f"# Resumen de DataFrame 'df'\nFilas: {df.shape[0]}, Columnas: {df.shape[1]}\n"
            f"Columnas: {', '.join(df.columns)}\n\n"
            "# DataFrame 'df_dict' y 'df_examples' disponibles.\n\n"
        )
        prompt = summary + user_query
        instructions = (
            "1. RESPONDE ÚNICAMENTE CON CÓDIGO PYTHON PURO, sin texto explicativo ni comentarios introductorios."
            "2. NO INCLUYAS frases como 'Aquí está el código' o 'Para calcular...'."
            "3. El código debe empezar directamente con sintaxis Python válida (sin texto antes)."
            "4. NO uses triple comillas, bloques markdown ni explicaciones antes o después del código."
            "5. Asume que el DataFrame YA está cargado como 'df', NO INTENTES cargarlo de nuevo."
            "6. Asegúrate de que el código tenga una indentación correcta y consistente."
            "7. Usa 4 espacios para la indentación, no tabulaciones."
            "8. Si necesitas generar visualizaciones, usa matplotlib o seaborn (ya importados como plt y sns)."
            "9. NO ALUCINES: solo usa las columnas y datos que existen realmente en el DataFrame."
            "10. El código debe terminar imprimiendo o mostrando el resultado final."
            "11. Si trabajas con fechas, SIEMPRE usa dayfirst=True al convertir: pd.to_datetime(df['fecha'], dayfirst=True)"
            "12. Al procesar fechas, asume formato europeo (día/mes/año) a menos que se indique lo contrario."
            "13. NUNCA uses Series de pandas directamente en condicionales como 'if df['columna']:', usa métodos específicos como .any(), .all(), .empty o compara con valores específicos."
            "14. Si necesitas comprobar valores en una Serie, usa operadores de comparación explícitos: df['columna'] == valor, df['columna'] > valor, etc."
            "15. SIEMPRE devuelve los resultados como DataFrames o Series de pandas en lugar de imprimir texto."
            "16. Para regresión, usa `LinearRegression` en lugar de `LogisticRegression`, e incluye validación con `train_test_split`."
            "17. Si requieres más contexto, pregúntalo antes de generar el código de Python."
            "18. Si tienes sugerencias para mejorar el prompt, hazlas."
            "19. Asegúrate de generar el código completo y sin truncarlo en ninguna parte."
        )
        system_prompt = (
            "Genere solo código Python ejecutable usando 'df'. Use 'df_dict' y 'df_examples' para referencia." + instructions
        )
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature, "num_predict": max_tokens}
        )
        code = response.message.content.lstrip().replace("```python", "").replace("```", "").strip()
        st.subheader("Código generado")
        st.code(code, language="python")
        st.subheader("Resultado de la ejecución")
        import io, sys
        buf = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        local_vars = {"df": df, "df_dict": df_dict, "df_examples": df_examples, "pd": pd, "np": np, "plt": plt, "sns": sns}
        exec(code, {}, local_vars)
        sys.stdout = old_stdout
        if plt.get_fignums():
            st.pyplot(plt.gcf()); plt.clf()
        output = buf.getvalue()
        if output:
            st.markdown(output.replace("\n", "  \n"))
        elif "result" in local_vars:
            res = local_vars["result"]
            if isinstance(res, (pd.DataFrame, pd.Series)):
                st.dataframe(res)
            else:
                st.markdown(str(res))

if __name__ == "__main__":
    main()
