
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Regresión lineal simple", page_icon="📈", layout="centered")

st.title("Regresión lineal simple")

# 1) Cargar datos
st.header("1 Cargar datos")
uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"No se pudo leer el CSV: {e}")
        st.stop()

    st.write("Vista previa de los datos:")
    st.dataframe(df.head())

    # columnas numéricas
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) < 2:
        st.error("Se requieren al menos 2 columnas numéricas (X e Y).")
        st.stop()

    x_col = st.selectbox("Selecciona la variable independiente (X)", numeric_cols, index=0)
    y_col = st.selectbox("Selecciona la variable dependiente (Y)", [c for c in numeric_cols if c != x_col], index=0)

    # Entrenar modelo
    X = df[[x_col]].values
    y = df[y_col].values

    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    st.success("Modelo entrenado correctamente")

    #  Ecuación del modelo (en LaTeX)
    b1 = float(model.coef_[0])
    b0 = float(model.intercept_)
    st.subheader("Ecuación del modelo:")
    st.latex(rf"Y = {b1:0.2f}\, X + {b0:0.2f}")

    #  Mostrar R^2
    st.subheader("Mostrar el R^2")
    st.write("Coeficiente de determinación (R²):")
    st.markdown(f"**{r2:0.4f}**")
    st.write(f"El valor de R² es: **{r2:0.4f}**")
    st.latex(rf"R^2 = {r2:0.4f}")

    # 2) Predicción
    st.header("2 Realiza una predicción")
    x_new = st.number_input(f"Introduce un valor para {x_col}:", value=float(np.nan if df[x_col].isna().all() else round(df[x_col].median(),2)))
    if st.button("Predecir"):
        y_hat = model.predict(np.array([[x_new]], dtype=float))[0]
        st.markdown(f"🔹 **Predicción para {x_col} = {x_new}: {y_hat:0.2f}**")

    # 3) Visualización
    st.header("3 Visualización del modelo")
    fig, ax = plt.subplots(figsize=(7,4))
    # datos reales
    ax.scatter(df[x_col], df[y_col], label="Datos reales", alpha=0.8)
    # línea de regresión
    x_line = np.linspace(df[x_col].min(), df[x_col].max(), 200).reshape(-1,1)
    y_line = model.predict(x_line)
    ax.plot(x_line, y_line, label="Línea de regresión")
    # punto de predicción, si se ha presionado
    if "y_hat" in locals():
        ax.scatter([x_new], [y_hat], s=100, marker="o", label="Predicción")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Visualización del modelo")
    ax.legend()
    st.pyplot(fig)
else:
    st.info("Sube un CSV para continuar. Ejemplo de columnas: `horas, calificacion`.")
