#!pip install streamlit pandas numpy yfinance matplotlib

#Importación las librerías

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

st.markdown("<h1 style='text-align: center; color:#004aad;'>Smart Portafolio - Simulación de Escenarios</h1>", unsafe_allow_html=True)

# app.py (versión corregida y con fallback visible)
import streamlit as st
from io import BytesIO
import datetime

# importa componentes de forma explícita
import streamlit.components.v1 as components

# intentar importar cairosvg, pero no morir si falta
try:
    import cairosvg
    CAIROSVG_OK = True
except Exception:
    CAIROSVG_OK = False

st.set_page_config(page_title="Smart Portafolio - Logo Maker", layout="centered")

# -----------------------
# SVG builder
# -----------------------
def build_svg(symbol_color="#38FFB0", text_color="#0D0D0D", bg_color="#FFFFFF",
              tilt_deg=0, brand_text="Smart Portafolio", tagline="Optimiza. Decide. Escala.",
              caps=False):
    display_text = brand_text.upper() if caps else brand_text
    width = 900
    height = 360

    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="{bg_color}" rx="20" />
  <g transform="translate(80,40) rotate({tilt_deg} 0 0)">
    <text x="0" y="150" font-family="Montserrat, Inter, Arial, sans-serif" font-weight="700" font-size="220" fill="{symbol_color}" stroke="{symbol_color}" >
      S
    </text>
    <text x="42" y="120" font-family="Montserrat, Inter, Arial, sans-serif" font-weight="800" font-size="140" fill="none" stroke="{text_color}" stroke-width="6" opacity="0.12">
      $
    </text>
    <polygon points="210,90 250,110 210,130" fill="{text_color}" opacity="0.9" transform="translate(0,22) rotate(8 230 110)"/>
  </g>
  <g transform="translate(320,120)">
    <text x="0" y="0" font-family="Montserrat, Inter, Arial, sans-serif" font-weight="700" font-size="40" fill="{text_color}">{display_text}</text>
    <text x="0" y="48" font-family="Inter, Arial, sans-serif" font-weight="500" font-size="18" fill="#6B7280">{tagline}</text>
  </g>
  <rect x="320" y="90" width="120" height="4" rx="2" fill="{symbol_color}" opacity="0.9"/>
</svg>
'''
    return svg

def svg_to_png_bytes(svg_str):
    if not CAIROSVG_OK:
        raise RuntimeError("cairosvg no está instalado en el entorno.")
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode('utf-8'))
    return png_bytes

def make_download_button(data_bytes, filename, label, mime):
    st.download_button(label=label, data=data_bytes, file_name=filename, mime=mime)

# -----------------------
# UI
# -----------------------
st.title("🎨 Logo Maker — Smart Portafolio (versión C)")
st.markdown("Genera un logo dinámico, juvenil y con un guiño subliminal. Ajusta y descarga.")

with st.sidebar:
    scheme = st.selectbox("Tema visual", ["Claro (texto oscuro)", "Oscuro (texto claro)"])
    inclination = st.selectbox("Inclinación", ["Recto (0°)", "Inclinado 8°"])
    caps = st.checkbox("Usar TODO EN MAYÚSCULAS", value=False)
    brand_text = st.text_input("Texto de marca", "Smart Portafolio")
    tagline = st.text_input("Tagline (pequeño)", "Optimiza. Decide. Escala.")
    fmt = st.multiselect("Formatos", ["SVG", "PNG"], default=["SVG","PNG"])

if scheme.startswith("Claro"):
    bg_color = "#FFFFFF"; text_color = "#0D0D0D"
else:
    bg_color = "#0D0D0D"; text_color = "#FFFFFF"

symbol_color = "#38FFB0"
tilt_deg = 8 if inclination.startswith("Inclinado") else 0

svg_str = build_svg(symbol_color=symbol_color, text_color=text_color,
                    bg_color=bg_color, tilt_deg=tilt_deg,
                    brand_text=brand_text, tagline=tagline, caps=caps)

st.subheader("Preview")

# Mostrar SVG con components.html (fallback a texto si falla)
try:
    svg_html = f'<div style="width:100%; display:flex; justify-content:center;">{svg_str}</div>'
    components.html(svg_html, height=420)
except Exception as e:
    st.error("No se pudo renderizar SVG en componentes HTML. Mostrando código SVG crudo.")
    st.code(svg_str[:1000] + ("\n... (truncado)" if len(svg_str)>1000 else ""))

st.subheader("Descargas")
now = datetime.datetime.now().strftime("%Y%m%d_%H%M")

if "SVG" in fmt:
    svg_bytes = svg_str.encode("utf-8")
    make_download_button(svg_bytes, f"smart_portafolio_{now}.svg", "Descargar SVG", "image/svg+xml")

if "PNG" in fmt:
    if CAIROSVG_OK:
        try:
            png_bytes = svg_to_png_bytes(svg_str)
            # mostrar inline como imagen (esto asegura que el usuario vea algo)
            st.image(png_bytes, caption="Preview PNG (convertido desde SVG)", use_column_width=False)
            make_download_button(png_bytes, f"smart_portafolio_{now}.png", "Descargar PNG", "image/png")
        except Exception as e:
            st.error("Error al convertir SVG→PNG.")
            st.write("Detalle:", e)
    else:
        st.warning("cairosvg no instalado — no se puede generar PNG. Instala 'cairosvg' en requirements.")
        # como ayuda, mostramos un PNG muy básico generado por PIL (si está disponible)
        try:
            from PIL import Image, ImageDraw, ImageFont
            # fallback visual sencillo
            img = Image.new("RGB", (900,360), color=bg_color)
            draw = ImageDraw.Draw(img)
            draw.text((330,120), brand_text, fill=text_color)
            bio = BytesIO()
            img.save(bio, format="PNG")
            bio.seek(0)
            st.image(bio.read(), caption="Fallback PNG (PIL) mostrado")
        except Exception:
            pass

st.markdown("---")
st.caption("Si no ves cambios: guarda el archivo y reinicia la app con `streamlit run app.py`.")

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

from io import BytesIO

st.subheader("📥 Descarga de Resultados")

# Exportar datos a Excel
excel_buffer = BytesIO()

# Combinar datos y retornos para exportar todo junto
with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
    data.to_excel(writer, sheet_name='Precios')
    returns.to_excel(writer, sheet_name='Rendimientos')
    df_resultados.to_excel(writer, sheet_name='Escenarios')

st.download_button(
    label="📊 Descargar en Excel",
    data=excel_buffer.getvalue(),
    file_name="analisis_portafolio.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

# Generar reporte PDF simple (texto) 

from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# --- Crear PDF con formato ---
st.subheader("📄 Generar Reporte en PDF")

pdf_buffer = BytesIO()

# Crear documento
doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
styles = getSampleStyleSheet()
elements = []

# --- Título ---
title = Paragraph("<b><font size=18 color='#004aad'>SMART PORTAFOLIO - REPORTE DE INVERSIÓN</font></b>", styles["Title"])
elements.append(title)
elements.append(Spacer(1, 0.2 * inch))

# --- Datos generales ---
intro = Paragraph(f"""
<font size=12>
<b>Escenario seleccionado:</b> {escenario}<br/>
<b>Activos analizados:</b> {', '.join(tickers)}<br/>
<b>Inversión inicial:</b> ${inversion_inicial:,.2f}
</font>
""", styles["Normal"])
elements.append(intro)
elements.append(Spacer(1, 0.2 * inch))

# --- Resultados ---
resumen_data = [
    ["Métrica", "Valor"],
    ["Rendimiento esperado", f"{port_return:.2%}"],
    ["Volatilidad esperada", f"{port_volatility:.2%}"],
    ["Ratio de Sharpe", f"{sharpe_ratio:.2f}"],
    ["Escenario recomendado", mejor_escenario]
]

table = Table(resumen_data, hAlign='LEFT')
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
    ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
]))
elements.append(table)
elements.append(Spacer(1, 0.3 * inch))

# --- Conclusión ---
conclusion = Paragraph(f"""
<font size=12>
La simulación de escenarios permite observar cómo el riesgo y el rendimiento están estrechamente relacionados.<br/>
El portafolio <b>{mejor_escenario}</b> presenta la mejor eficiencia según el Ratio de Sharpe.<br/><br/>
<b>Interpretación:</b><br/>
{("Este portafolio prioriza la estabilidad, ideal para perfiles conservadores." if mejor_escenario == "Conservador" 
else "Este portafolio equilibra riesgo y rendimiento, ideal para inversores moderados." 
if mejor_escenario == "Moderado" 
else "Este portafolio busca maximizar ganancias, ideal para perfiles arriesgados.")}
</font>
""", styles["Normal"])
elements.append(conclusion)

# --- Guardar PDF ---
doc.build(elements)
pdf_buffer.seek(0)

st.download_button(
    label="📑 Descargar Reporte en PDF (formateado)",
    data=pdf_buffer,
    file_name="Reporte_Portafolio.pdf",
    mime="application/pdf"
)
