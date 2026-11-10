import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ---------------------------------------------------------
# Configuración general
# ---------------------------------------------------------
st.set_page_config(
    page_title="Simulador - Regresión logística multinomial",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Simulador de regresión logística multinomial")
st.write(
    "Predice la clase de un pasajero del Titanic a partir de su edad "
    "(`Age`) y tarifa (`Fare`), utilizando la variable objetivo `Class`."
)

st.info(
    "El simulador entrena el modelo con el archivo Titanic.csv que se carga "
    "a continuación. Las variables predictoras son las mismas que definiste "
    "en tu notebook: Age y Fare."
)

# ---------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------
st.sidebar.header("1. Datos de entrenamiento")
archivo = st.sidebar.file_uploader(
    "Carga el archivo Titanic.csv",
    type=["csv"]
)

df = None

if archivo is not None:
    try:
        df = pd.read_csv(archivo)
    except Exception as e:
        st.error(f"No fue posible leer el archivo CSV: {e}")
        st.stop()
else:
    # Permite ejecutar la app si Titanic.csv está en la misma carpeta.
    try:
        df = pd.read_csv("Titanic.csv")
        st.sidebar.success("Se encontró Titanic.csv en la carpeta de la aplicación.")
    except FileNotFoundError:
        st.warning(
            "Carga el archivo Titanic.csv desde el panel lateral para comenzar."
        )
        st.stop()

# ---------------------------------------------------------
# Validación y preparación de datos
# ---------------------------------------------------------
columnas_requeridas = ["Age", "Fare", "Class"]
faltantes = [col for col in columnas_requeridas if col not in df.columns]

if faltantes:
    st.error(
        "El archivo no contiene las columnas requeridas: "
        + ", ".join(faltantes)
    )
    st.stop()

datos = df[columnas_requeridas].copy()

# Convertimos las variables predictoras a formato numérico.
datos["Age"] = pd.to_numeric(datos["Age"], errors="coerce")
datos["Fare"] = pd.to_numeric(datos["Fare"], errors="coerce")

# La variable objetivo no puede estar vacía.
datos = datos.dropna(subset=["Class"]).copy()
datos["Class"] = datos["Class"].astype(str).str.strip()
datos = datos[datos["Class"] != ""]

# Se eliminan registros sin información en ambas variables predictoras.
datos = datos.dropna(subset=["Age", "Fare"], how="all")

if datos["Class"].nunique() < 2:
    st.error("Se requieren al menos dos clases distintas en la columna Class.")
    st.stop()

if datos["Class"].value_counts().min() < 2:
    st.error(
        "Cada clase debe tener al menos dos registros para realizar "
        "la división estratificada de entrenamiento y prueba."
    )
    st.stop()

X = datos[["Age", "Fare"]]
y = datos["Class"]

# ---------------------------------------------------------
# División de datos y entrenamiento
# ---------------------------------------------------------
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )
except ValueError as e:
    st.error(f"No fue posible dividir los datos: {e}")
    st.stop()

# Pipeline: imputación, escalamiento y regresión logística.
# El imputador y el escalador se ajustan exclusivamente con entrenamiento.
modelo = Pipeline(steps=[
    ("imputador", SimpleImputer(strategy="median")),
    ("escalador", StandardScaler()),
    ("clasificador", LogisticRegression(
        solver="lbfgs",
        max_iter=1000
    ))
])

modelo.fit(X_train, y_train)

# Evaluación con el conjunto de prueba.
y_pred = modelo.predict(X_test)
clases = modelo.named_steps["clasificador"].classes_

exactitud = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
sensibilidad = recall_score(y_test, y_pred, average="weighted", zero_division=0)
puntaje_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

# ---------------------------------------------------------
# Resumen del conjunto de datos y métricas
# ---------------------------------------------------------
st.subheader("2. Información del modelo")

c1, c2, c3 = st.columns(3)
c1.metric("Registros utilizados", f"{len(datos):,}")
c2.metric("Número de clases", len(clases))
c3.metric("Variables independientes", "2")

st.caption(
    f"Entrenamiento: {len(X_train):,} registros | "
    f"Prueba: {len(X_test):,} registros | "
    f"Clases: {', '.join(map(str, clases))}"
)

st.subheader("3. Desempeño del modelo")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Exactitud (Accuracy)", f"{exactitud:.2%}")
m2.metric("Precisión ponderada", f"{precision:.2%}")
m3.metric("Sensibilidad ponderada", f"{sensibilidad:.2%}")
m4.metric("Puntaje F1 ponderado", f"{puntaje_f1:.2%}")

with st.expander("Consultar reporte de clasificación"):
    reporte = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True
    )
    st.dataframe(
        pd.DataFrame(reporte).transpose().style.format(precision=3),
        use_container_width=True
    )

with st.expander("Consultar matriz de confusión"):
    matriz = confusion_matrix(y_test, y_pred, labels=clases)
    fig_cm, ax_cm = plt.subplots(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=matriz,
        display_labels=clases
    )
    disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
    ax_cm.set_title("Matriz de confusión")
    st.pyplot(fig_cm)
    plt.close(fig_cm)

# ---------------------------------------------------------
# Simulador interactivo
# ---------------------------------------------------------
st.divider()
st.header("4. Simulador de predicción")
st.write(
    "Modifica los valores de las variables independientes y presiona "
    "**Predecir clase** para obtener la clasificación estimada."
)

col_entrada, col_resultado = st.columns([1, 1.25])

# Límites basados en los datos disponibles.
edad_min = float(datos["Age"].min()) if datos["Age"].notna().any() else 0.0
edad_max = float(datos["Age"].max()) if datos["Age"].notna().any() else 100.0
tarifa_min = float(datos["Fare"].min()) if datos["Fare"].notna().any() else 0.0
tarifa_max = float(datos["Fare"].max()) if datos["Fare"].notna().any() else 600.0

if edad_min == edad_max:
    edad_max = edad_min + 1.0
if tarifa_min == tarifa_max:
    tarifa_max = tarifa_min + 1.0

with col_entrada:
    st.subheader("Variables independientes")

    edad = st.number_input(
        "Edad del pasajero (Age)",
        min_value=max(0.0, edad_min),
        max_value=max(edad_max, edad_min + 1.0),
        value=float(np.clip(30.0, max(0.0, edad_min), max(edad_max, edad_min + 1.0))),
        step=1.0,
        help="Edad del pasajero en años."
    )

    tarifa = st.number_input(
        "Tarifa del boleto (Fare)",
        min_value=max(0.0, tarifa_min),
        max_value=max(tarifa_max, tarifa_min + 1.0),
        value=float(np.clip(30.0, max(0.0, tarifa_min), max(tarifa_max, tarifa_min + 1.0))),
        step=1.0,
        help="Tarifa pagada por el pasajero."
    )

    predecir = st.button(
        "🔍 Predecir clase",
        type="primary",
        use_container_width=True
    )

with col_resultado:
    st.subheader("Resultado de la predicción")

    if predecir:
        nuevo_pasajero = pd.DataFrame({
            "Age": [edad],
            "Fare": [tarifa]
        })

        clase_predicha = modelo.predict(nuevo_pasajero)[0]
        probabilidades = modelo.predict_proba(nuevo_pasajero)[0]

        st.success(f"Clase predicha: {clase_predicha}")

        df_prob = pd.DataFrame({
            "Clase": clases,
            "Probabilidad": probabilidades
        }).sort_values("Probabilidad", ascending=False)

        st.write("**Probabilidades estimadas por categoría:**")
        for _, fila in df_prob.iterrows():
            st.write(f"{fila['Clase']}: {fila['Probabilidad']:.2%}")
            st.progress(float(fila["Probabilidad"]))

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(
            df_prob["Clase"].astype(str),
            df_prob["Probabilidad"] * 100
        )
        ax.set_ylabel("Probabilidad (%)")
        ax.set_xlabel("Clase")
        ax.set_title("Probabilidad estimada para cada clase")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.25)
        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "La clase mostrada es la categoría con la mayor probabilidad "
            "estimada por el modelo. Las probabilidades son estimaciones "
            "del clasificador, no garantías del resultado real."
        )
    else:
        st.info(
            "Introduce la edad y la tarifa del pasajero y selecciona "
            "'Predecir clase' para visualizar el resultado."
        )

# ---------------------------------------------------------
# Vista de las clases disponibles
# ---------------------------------------------------------
with st.expander("Consultar distribución de las clases"):
    distribucion = datos["Class"].value_counts().rename_axis("Clase").reset_index(name="Registros")
    st.dataframe(distribucion, use_container_width=True)

st.divider()
st.caption(
    "Modelo basado en las variables Age y Fare y en la variable objetivo "
    "Class, de acuerdo con el notebook de regresión logística multinomial."
)
