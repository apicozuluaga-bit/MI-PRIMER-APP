#!pip install streamlit pandas numpy yfinance matplotlib

#Importación las librerías

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.markdown("<h1 style='text-align: center; color:#004aad;'>Smart Portafolio - Simulación de Escenarios</h1>", unsafe_allow_html=True)
st.write("""
Esta aplicación realiza una *simulación de escenarios de inversión, aplicando la *Teoría Moderna de Portafolios de Markowitz.

Se analizan tres tipos de portafolios según el perfil de riesgo del inversionista:

- 🟩 *Conservador:* prioriza la estabilidad, minimizando el riesgo.  
- 🟨 *Moderado:* busca equilibrio entre riesgo y rentabilidad.  
- 🟥 *Agresivo:* asume un riesgo alto para intentar maximizar las ganancias.

Los datos se obtienen directamente desde *Yahoo Finance*, permitiendo analizar empresas reales del mercado financiero.
""")

# Configuración de entradas

st.sidebar.markdown("## ⚙ Configuración del Análisis")

# Entrada libre de tickers
tickers_input = st.sidebar.text_input(
    "Empresas (separa por comas):",
    value="AAPL, META"
)

# Convertir texto en lista
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip() != ""]


# Rango de fechas
fecha_inicio = st.sidebar.date_input("📅 Fecha Inicial", pd.to_datetime("2020-01-01"))
fecha_fin = st.sidebar.date_input("📅 Fecha Final", pd.to_datetime("2023-12-31"))

# Inversión inicial
inversion_inicial = st.sidebar.number_input("💰 Inversión Inicial (USD)", min_value=1000, value=10000, step=500)

# Frecuencia temporal
frecuencia = st.sidebar.selectbox("⏱ Frecuencia Temporal", ["Diaria", "Semanal", "Mensual"])

# Tipo de escenario
escenario = st.sidebar.selectbox("💰 Escenario de Inversión", ["Conservador", "Moderado", "Agresivo"])

# Botón para ejecutar
descargar = st.sidebar.button("📥 Descargar y Analizar")


# Descarga de datos

data = yf.download(tickers, start=fecha_inicio, end=fecha_fin)["Close"]
st.subheader("📊 Datos Descargados")
st.dataframe(data.tail())

# Ajuste según frecuencia

if frecuencia == "Semanal":
    data = data.resample('W').last()
elif frecuencia == "Mensual":
    data = data.resample('M').last()
    
# Funciones de exportación

# Visualización de Precios

st.subheader("📈 Evolución de Precios")
fig1, ax1 = plt.subplots(figsize=(10, 4))
data.plot(ax=ax1)
plt.title("Evolución de Precios Ajustados")
plt.xlabel("Fecha")
plt.ylabel("Precio (USD)")
st.pyplot(fig1)

# Cálculo de rendimientos

returns = data.pct_change().dropna()
mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

# Estadísticas generales
st.dataframe(returns.describe().T)

# Escenario de inversión

escenarios = {
    "Conservador": np.linspace(0.6, 0.1, len(tickers)),
    "Moderado": np.linspace(0.4, 0.2, len(tickers)),
    "Agresivo": np.linspace(0.2, 0.6, len(tickers))
}

weights = escenarios[escenario]
weights = weights / np.sum(weights)  # normalizamos

# Cálculos del portafolio

port_return = np.dot(weights, mean_returns)
port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
sharpe_ratio = port_return / port_volatility

# Retorno acumulado y evolución monetaria

returns["Portfolio"] = (returns[tickers] * weights).sum(axis=1)
valor_portafolio = (1 + returns["Portfolio"]).cumprod() * inversion_inicial

# Resultados

st.subheader(f"📊 Resultados del Portafolio ({escenario})")
st.write("*Pesos del Portafolio:*", dict(zip(tickers, weights.round(2))))
st.write(f"*Rendimiento Esperado:* {port_return:.2%}")
st.write(f"*Volatilidad Esperada:* {port_volatility:.2%}")
st.write(f"*Sharpe Ratio:* {sharpe_ratio:.2f}")

st.markdown("---")
st.subheader("🧠 Interpretación del Escenario Seleccionado")

if escenario == "Conservador":
    st.info("🔹 Este portafolio busca minimizar el riesgo, con un enfoque en estabilidad. Su rendimiento esperado es menor, pero ofrece menor volatilidad y pérdidas potenciales.")
elif escenario == "Moderado":
    st.info("🟨 Este portafolio equilibra riesgo y rendimiento. Es ideal para inversores con tolerancia media al riesgo que buscan un crecimiento sostenido.")
else:
    st.info("🔺 Este portafolio asume mayor riesgo con el objetivo de maximizar el rendimiento. Es adecuado para inversionistas con alta tolerancia a la volatilidad y posibles pérdidas.")

# Evolución del valor monetario

st.subheader("💵 Evolución del Valor del Portafolio")
fig2, ax2 = plt.subplots(figsize=(10, 4))
valor_portafolio.plot(ax=ax2, color='green')
plt.title("Evolución del valor monetario del portafolio")
plt.xlabel("Fecha")
plt.ylabel("Valor (USD)")
st.pyplot(fig2)

# Diagrama riesgo - retorno

st.subheader("📊 Diagrama Riesgo - Retorno")

# Asegurar que solo se usen los tickers seleccionados
asset_returns = mean_returns[tickers]
asset_risk = returns[tickers].std() * np.sqrt(252)

# Convertir a listas para graficar
x_riesgo = asset_risk.values
y_retorno = asset_returns.values

# Crear el gráfico
fig3, ax3 = plt.subplots(figsize=(7, 5))

# Graficar los activos individuales (solo puntos)
ax3.scatter(x_riesgo, y_retorno, c='blue', s=80)

# Etiquetar cada punto con su ticker
for i, ticker in enumerate(tickers):
    ax3.text(x_riesgo[i] + 0.002, y_retorno[i], ticker, fontsize=9, ha='left', va='center')

# Etiquetas y estilo
ax3.set_xlabel("Volatilidad (Riesgo)")
ax3.set_ylabel("Rendimiento Esperado")
ax3.set_title("Diagrama Riesgo - Retorno")
ax3.grid(True, linestyle='--', alpha=0.6)

st.pyplot(fig3)

#  Correlaciones

st.subheader("🔥 Correlaciones entre Activos")
corr_matrix = returns[tickers].corr()
st.dataframe(corr_matrix)

fig4, ax4 = plt.subplots()
cax = ax4.imshow(corr_matrix, cmap="coolwarm", interpolation="nearest")
plt.title("Matriz de Correlaciones")
plt.colorbar(cax)
ax4.set_xticks(range(len(corr_matrix)))
ax4.set_xticklabels(corr_matrix.columns, rotation=45)
ax4.set_yticks(range(len(corr_matrix)))
ax4.set_yticklabels(corr_matrix.columns)
st.pyplot(fig4)

# Visualización del portafolio

st.subheader("🥧 Distribución del Portafolio por Escenario")

fig, ax = plt.subplots()
ax.pie(weights, labels=tickers, autopct="%1.1f%%", startangle=90)
ax.set_title(f"Distribución del Portafolio ({escenario})")
st.pyplot(fig)

# Distribución de pesos por escenario

st.subheader("📊 Comparación de Escenarios de Inversión")

fig_all, axs = plt.subplots(1, 3, figsize=(12, 4))
for i, (nombre, base_pesos) in enumerate({
    "Conservador": np.linspace(0.6, 0.1, len(tickers)),
    "Moderado": np.linspace(0.4, 0.2, len(tickers)),
    "Agresivo": np.linspace(0.2, 0.6, len(tickers))
}.items()):
    w = base_pesos / np.sum(base_pesos)
    # Aseguramos que las etiquetas coincidan con la cantidad de pesos
    labels = tickers[:len(w)]
    axs[i].pie(w, labels=labels, autopct='%1.1f%%', startangle=90)
    axs[i].set_title(nombre)

plt.suptitle("Distribución de Pesos por Tipo de Portafolio")
st.pyplot(fig_all)

# Evaluación y recomendación de escenarios

st.subheader("🤖 Recomendación de Escenario Óptimo")

# Calcular métricas para cada escenario
resultados = {}
for nombre, pesos in {
    "Conservador": np.linspace(0.6, 0.1, len(tickers)),
    "Moderado": np.linspace(0.4, 0.2, len(tickers)),
    "Agresivo": np.linspace(0.2, 0.6, len(tickers))
}.items():
    w = pesos / np.sum(pesos)
    rendimiento = np.dot(w, mean_returns)
    riesgo = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    sharpe = rendimiento / riesgo
    resultados[nombre] = {"rendimiento": rendimiento, "riesgo": riesgo, "sharpe": sharpe}

# Crear DataFrame ordenado
df_resultados = pd.DataFrame(resultados).T
df_resultados = df_resultados.sort_values("sharpe", ascending=False)

st.dataframe(df_resultados.style.format({
    "rendimiento": "{:.2%}",
    "riesgo": "{:.2%}",
    "sharpe": "{:.2f}"
}))

# Determinar el escenario óptimo
mejor_escenario = df_resultados.index[0]
st.success(f"✅ El escenario más eficiente según el Ratio de Sharpe es: *{mejor_escenario}* 🎯")

# Comentario interpretativo
if mejor_escenario == "Conservador":
    st.info("💡 Recomendación: Este portafolio ofrece mayor estabilidad y menor riesgo. Ideal para perfiles que priorizan seguridad sobre rentabilidad.")
elif mejor_escenario == "Moderado":
    st.info("💡 Recomendación: Este portafolio equilibra riesgo y rendimiento, siendo adecuado para inversores con tolerancia media al riesgo.")
else:
    st.info("💡 Recomendación: Este portafolio maximiza el rendimiento a costa de mayor volatilidad. Ideal para perfiles arriesgados que buscan crecimiento a largo plazo.")
