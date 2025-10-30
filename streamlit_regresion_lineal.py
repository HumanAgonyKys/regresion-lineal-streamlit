#refactorizado para que pueda funcionar con streamlit
#el archivo original esta en main
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import streamlit as st

st.title("Aprendizaje no supervizado: k-means")
st.subheader("By Leonardo Garcia Bernal")
st.subheader("cargar datos")

uploaded_file = st.file_uploader("Sube un archivo CSV con tus datos", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Datos")
    st.dataframe(df.head())

    escalador = MinMaxScaler().fit(df.values)
    df_normalizado = pd.DataFrame(escalador.transform(df.values), columns=["Saldo", "transacciones"])

    st.dataframe(df_normalizado)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(df_normalizado.values)

    df_normalizado["cluster"] = kmeans.labels_

    st.write(f"{kmeans.cluster_centers_}")
    st.write(f"{kmeans.inertia_}")

    fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=100)
    colores = ["red", "blue", "orange", "black", "purple", "pink", "brown"]

    for cluster in range(kmeans.n_clusters):
        ax1.scatter(df_normalizado[df_normalizado["cluster"] == cluster]["Saldo"],
                    df_normalizado[df_normalizado["cluster"] == cluster]["transacciones"],
                    marker="o", s=180, color=colores[cluster], alpha=0.5)
        ax1.scatter(kmeans.cluster_centers_[cluster][0],
                    kmeans.cluster_centers_[cluster][1],
                    marker="P", s=280, color=colores[cluster])

    ax1.set_title("clientes", fontsize=20)
    ax1.set_xlabel("saldo en cuenta de ahorros", fontsize=15)
    ax1.set_ylabel("veces que uso tarjeta de credito", fontsize=15)
    ax1.text(1.15, 0.2, "k=%i" % kmeans.n_clusters, fontsize=15)
    ax1.text(1.15, 0, "Inercia = %0.2f" % kmeans.inertia_, fontsize=15)
    ax1.set_xlim(-0.1, 1.15)
    ax1.set_ylim(-0.1, 1.15)
    plt.tight_layout()
    st.pyplot(fig1)


    inercias = []
    for k in range(2, 10):
        kmeans_temp = KMeans(n_clusters=k).fit(df_normalizado[["Saldo", "transacciones"]].values)
        inercias.append(kmeans_temp.inertia_)

    fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=100)
    ax2.scatter(range(2, 10), inercias, marker="o", s=180, color="purple")
    ax2.set_xlabel("numero de clusters", fontsize=25)
    ax2.set_ylabel("inercia", fontsize=25)
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("Por favor, sube un archivo CSV para comenzar el análisis")