# 📦 Importaciones
import time, io, zipfile, sqlite3
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
from tensorflow.keras.models import load_model

# ⚙️ Configuración inicial
st.set_page_config(page_title="Reconocimiento de Personas", page_icon="🧠", layout="wide")

# 📁 Rutas de recursos
MODEL_PATH = "keras_model.h5"
LABELS_PATH = "labels.txt"
DB_PATH = "db.sqlite"

# 🧠 Cargar modelo y etiquetas
@st.cache_resource
def load_model_cached(path): return load_model(path, compile=False)

@st.cache_data
def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

try:
    model = load_model_cached(MODEL_PATH)
    labels = load_labels(LABELS_PATH)
except Exception as e:
    st.error(f"Error al cargar modelo/etiquetas: {e}")
    st.stop()

# 🗃️ Inicializar base de datos
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS predicciones (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT, fuente TEXT, etiqueta TEXT, confianza REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS personas (
        etiqueta TEXT PRIMARY KEY, nombre TEXT, correo TEXT, rol TEXT, umbral REAL, notas TEXT)""")
    conn.commit(); conn.close()

init_db()

# 📷 Clase para transformar video
class VideoTransformer(VideoTransformerBase):
    def __init__(self): self.latest = {"class": None, "confidence": 0.0}
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = (resized.astype(np.float32).reshape(1, 224, 224, 3) / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx] if idx < len(labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        self.latest = {"class": label, "confidence": conf}
        overlay = img.copy()
        text = f"{label} | {conf*100:.1f}%"
        cv2.rectangle(overlay, (5, 5), (5 + 8*len(text), 45), (0, 0, 0), -1)
        cv2.putText(overlay, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return overlay

# 📌 Menú lateral
menu = st.sidebar.radio("📂 Navegación", ["En vivo", "Administración", "Analítica"])

# 🌐 Configuración WebRTC
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# 📸 Sección: En vivo
if menu == "En vivo":
    st.title("🎥 Clasificación en tiempo real")
    quality = st.sidebar.selectbox("Calidad de video", ["640x480", "1280x720"], index=1)
    w, h = map(int, quality.split("x"))
    media_constraints = {"video": {"width": w, "height": h}, "audio": False}
    fuente = "cámara"

    webrtc_ctx = webrtc_streamer(
        key="live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints=media_constraints,
        video_transformer_factory=VideoTransformer,
        async_processing=True,
    )

    if webrtc_ctx and webrtc_ctx.state.playing:
        st.subheader("Resultados")
        result_placeholder = st.empty()
        for _ in range(300000):
            if not webrtc_ctx.state.playing: break
            vt = webrtc_ctx.video_transformer
            if vt and vt.latest["class"]:
                cls, conf = vt.latest["class"], vt.latest["confidence"]
                result_placeholder.markdown(f"**Etiqueta:** `{cls}`\n**Confianza:** `{conf*100:.2f}%`")
                # Guardar en DB
                conn = sqlite3.connect(DB_PATH)
                conn.execute("INSERT INTO predicciones (timestamp, fuente, etiqueta, confianza) VALUES (?, ?, ?, ?)",
                             (datetime.utcnow().isoformat(), fuente, cls, conf))
                conn.commit(); conn.close()
            time.sleep(1)

    st.markdown("---")
    st.subheader("📷 Alternativa: captura por foto")
    snap = st.camera_input("Captura una imagen")
    if snap:
        fuente = "imagen"
        file_bytes = np.asarray(bytearray(snap.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        resized = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        x = (resized.astype(np.float32).reshape(1, 224, 224, 3) / 127.5) - 1.0
        pred = model.predict(x, verbose=0)
        idx = int(np.argmax(pred))
        label = labels[idx] if idx < len(labels) else f"Clase {idx}"
        conf = float(pred[0][idx])
        st.image(img, caption=f"{label} | {conf*100:.2f}%")
        st.success(f"Predicción: **{label}** ({conf*100:.2f}%)")
        conn = sqlite3.connect(DB_PATH)
        conn.execute("INSERT INTO predicciones (timestamp, fuente, etiqueta, confianza) VALUES (?, ?, ?, ?)",
                     (datetime.utcnow().isoformat(), fuente, label, conf))
        conn.commit(); conn.close()

# 🧑 Sección: Administración
elif menu == "Administración":
    st.title("👥 Gestión de personas")
    with st.form("form"):
        etiqueta = st.selectbox("Etiqueta del modelo", labels)
        nombre = st.text_input("Nombre completo")
        correo = st.text_input("Correo electrónico")
        rol = st.selectbox("Rol", ["Estudiante", "Docente", "Visitante"])
        umbral = st.slider("Umbral de confianza", 0.0, 1.0, 0.5)
        notas = st.text_area("Notas")
        if st.form_submit_button("Guardar"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""INSERT OR REPLACE INTO personas (etiqueta, nombre, correo, rol, umbral, notas)
                            VALUES (?, ?, ?, ?, ?, ?)""", (etiqueta, nombre, correo, rol, umbral, notas))
            conn.commit(); conn.close()
            st.success("Persona guardada correctamente.")

    st.subheader("📋 Personas registradas")
    df = pd.read_sql_query("SELECT * FROM personas", sqlite3.connect(DB_PATH))
    st.dataframe(df)

# 📊 Sección: Analítica
elif menu == "Analítica":
    st.title("📈 Panel analítico")
    df = pd.read_sql_query("SELECT * FROM predicciones", sqlite3.connect(DB_PATH))

    st.subheader("Frecuencia por etiqueta")
    fig1, ax1 = plt.subplots()
    df["etiqueta"].value_counts().plot(kind="bar", ax=ax1)
    st.pyplot(fig1)

    st.subheader("Confianza promedio por etiqueta")
    fig2, ax2 = plt.subplots()
    df.groupby("etiqueta")["confianza"].mean().plot(kind="barh", ax=ax2)
    st.pyplot(fig2)

    st.subheader("Predicciones por fuente")
    fig3, ax3 = plt.subplots()
    df["fuente"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax3)
    st.pyplot(fig3)

    st.subheader("Evolución temporal")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    fig4, ax4 = plt.subplots()
    df.set_index("timestamp").resample("1min")["confianza"].mean().plot(ax=ax4)
    st.pyplot(fig4)

    st.subheader("Exportar datos")
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar CSV", data=csv_bytes, file_name="predicciones.csv", mime="text/csv")

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zf:
        for i, fig in enumerate([fig1, fig2, fig3, fig4]):
            img_bytes = io.BytesIO()
            fig.savefig(img_bytes, format="png")
            zf.writestr(f"grafica_{i+1}.png", img_bytes.getvalue())
    st.download_button("📦 Descargar ZIP con gráficas",
        data=zip_buffer.getvalue(),
        file_name="graficas_predicciones.zip",
        mime="application/zip"
    )
