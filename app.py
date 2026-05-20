import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
import numpy as np

# ------------------------------------------------
# CONFIGURACIÓN
# ------------------------------------------------

st.set_page_config(
    page_title="AI Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------
# CSS PROFESIONAL
# ------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
    color: white;
}

.metric-card {
    background: linear-gradient(135deg, #1f2937, #111827);
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.4);
}

h1, h2, h3 {
    color: #00E5FF;
}

</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------

st.sidebar.title("⚙️ Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Sube un archivo CSV",
    type=["csv"]
)

# ------------------------------------------------
# CARGA DE DATOS
# ------------------------------------------------

@st.cache_data
def load_data(file):
    df = pd.read_csv(file)
    return df

if uploaded_file:

    df = load_data(uploaded_file)

    st.title("📊 AI Financial Dashboard")

    st.write("Vista previa de datos")
    st.dataframe(df)

    # ------------------------------------------------
    # FILTROS
    # ------------------------------------------------

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

    selected_column = st.sidebar.selectbox(
        "Selecciona columna numérica",
        numeric_columns
    )

    # ------------------------------------------------
    # KPIs
    # ------------------------------------------------

    total = df[selected_column].sum()
    promedio = df[selected_column].mean()
    maximo = df[selected_column].max()
    minimo = df[selected_column].min()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total", f"{total:,.2f}")
    col2.metric("📈 Promedio", f"{promedio:,.2f}")
    col3.metric("🚀 Máximo", f"{maximo:,.2f}")
    col4.metric("📉 Mínimo", f"{minimo:,.2f}")

    st.divider()

    # ------------------------------------------------
    # GRÁFICA INTERACTIVA
    # ------------------------------------------------

    st.subheader("📈 Visualización Interactiva")

    fig = px.line(
        df,
        y=selected_column,
        title=f"Análisis de {selected_column}",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ------------------------------------------------
    # DETECCIÓN DE ANOMALÍAS
    # ------------------------------------------------

    st.subheader("🧠 Detección de anomalías")

    model = IsolationForest(contamination=0.05)

    df["anomaly"] = model.fit_predict(
        df[[selected_column]]
    )

    anomaly_df = df[df["anomaly"] == -1]

    fig2 = px.scatter(
        df,
        y=selected_column,
        color=df["anomaly"].astype(str),
        template="plotly_dark",
        title="Detección de anomalías"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.write("Registros anómalos")
    st.dataframe(anomaly_df)

    # ------------------------------------------------
    # HISTOGRAMA
    # ------------------------------------------------

    st.subheader("📊 Distribución")

    hist = px.histogram(
        df,
        x=selected_column,
        nbins=30,
        template="plotly_dark"
    )

    st.plotly_chart(hist, use_container_width=True)

    # ------------------------------------------------
    # EXPORTACIÓN
    # ------------------------------------------------

    csv = df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="⬇️ Descargar análisis",
        data=csv,
        file_name="analisis.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Sube un archivo CSV para comenzar")