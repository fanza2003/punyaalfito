from pathlib import Path
import PIL
from datetime import datetime
import pytz
import streamlit as st
from sqlalchemy import create_engine, Column, Integer, String, Float, Table, MetaData, DateTime, LargeBinary
from sqlalchemy.orm import sessionmaker
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, VideoTransformerBase
import settings
import helper
import io

# Mengatur layout halaman
st.set_page_config(
    page_title="Deteksi Kematangan Buah Kopi menggunakan YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Setup database
engine = create_engine('sqlite:///detection_results.db')
metadata = MetaData()

detections_table = Table(
    'detections', metadata,
    Column('id', Integer, primary_key=True),
    Column('username', String),
    Column('image_name', String),
    Column('detection_confidence', Float),
    Column('detection_data', String),
    Column('detection_time', DateTime, default=datetime.utcnow),
    Column('image_data', LargeBinary)  # New column for storing image as BLOB
)

metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Manajemen login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'page' not in st.session_state:
    st.session_state.page = "Deteksi"

def login(username, password):
    # Login sederhana, gunakan metode autentikasi yang tepat di aplikasi nyata
    if username == "admin" and password == "admin":
        st.session_state.logged_in = True
        st.session_state.username = username
        st.success("Login berhasil!")
    else:
        st.error("Username atau password salah")

def logout():
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        login(username, password)
else:
    st.title("Deteksi Kematangan Buah Kopi menggunakan YOLOv8")
    st.sidebar.button("Logout", on_click=logout)

    # Navigasi utama dengan tombol
    if st.session_state.page == "Deteksi":
        st.sidebar.button("Lihat Hasil Deteksi", on_click=lambda: st.session_state.update(page="Lihat Hasil Deteksi"))

    if st.session_state.page == "Lihat Hasil Deteksi":
        st.sidebar.button("Deteksi", on_click=lambda: st.session_state.update(page="Deteksi"))

    if st.session_state.page == "Deteksi":
        st.sidebar.subheader("Konfigurasi Model ML")
        model_type = st.sidebar.radio("Pilih Tugas", ['Deteksi'])
        confidence = float(st.sidebar.slider("Pilih Tingkat Keyakinan Model (%)", 0, 100, 40)) / 100
        model_path = Path(settings.DETECTION_MODEL)

        try:
            model = helper.load_model(model_path)
        except Exception as ex:
            st.error(f"Tidak dapat memuat model. Periksa jalur yang ditentukan: {model_path}")
            st.error(ex)

        st.sidebar.subheader("Konfigurasi Gambar/Video")
        source_radio = st.sidebar.radio("Pilih Sumber", [settings.IMAGE, settings.WEBCAM])

        if source_radio == settings.IMAGE:
            source_img = st.sidebar.file_uploader("Pilih gambar...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

            col1, col2 = st.columns(2)
            with col1:
                try:
                    if source_img is None:
                        default_image_path = str(settings.DEFAULT_IMAGE)
                        st.image(default_image_path, caption="Gambar Default", use_column_width=True)
                    else:
                        uploaded_image = PIL.Image.open(source_img)
                        st.image(source_img, caption="Gambar yang Diunggah", use_column_width=True)
                except Exception as ex:
                    st.error("Terjadi kesalahan saat membuka gambar.")
                    st.error(ex)

            with col2:
                if source_img is None:
                    default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                    st.image(default_detected_image_path, caption='Gambar yang Dideteksi', use_column_width=True)
                else:
                    if st.sidebar.button('Deteksi Objek'):
                        res = model.predict(uploaded_image, conf=confidence)
                        boxes = res[0].boxes
                        res_plotted = res[0].plot()[:, :, ::-1]

                        # Convert image to bytes
                        img_byte_arr = io.BytesIO()
                        PIL.Image.fromarray(res_plotted).save(img_byte_arr, format='PNG')
                        img_byte_arr = img_byte_arr.getvalue()

                        st.image(res_plotted, caption='Gambar yang Dideteksi', use_column_width=True)
                        try:
                            detection_data = []
                            with st.expander("Hasil Deteksi"):
                                for box in boxes:
                                    st.write(box.data)
                                    detection_data.append(str(box.data))

                            # Save detection results to the database
                            detection_record = {
                                'username': st.session_state.username,
                                'image_name': f"processed_{source_img.name}",
                                'detection_confidence': confidence,
                                'detection_data': ','.join(detection_data),
                                'detection_time': datetime.utcnow(),
                                'image_data': img_byte_arr  # Save image as BLOB
                            }
                            ins = detections_table.insert().values(detection_record)
                            session.execute(ins)
                            session.commit()
                            st.success("Hasil deteksi telah disimpan ke database.")
                        except Exception as ex:
                            st.error("Terjadi kesalahan saat menyimpan hasil deteksi.")
                            st.error(ex)

        elif source_radio == settings.WEBCAM:
            class VideoTransformer(VideoTransformerBase):
                def __init__(self, confidence, model):
                    self.confidence = confidence
                    self.model = model

                def transform(self, frame):
                    image = frame.to_ndarray(format="bgr24")
                    res = self.model.predict(image, conf=self.confidence)
                    res_plotted = res[0].plot()
                    return res_plotted

            webrtc_ctx = webrtc_streamer(
                key="example",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                media_stream_constraints={
                    "video": True,
                    "audio": False
                },
                video_transformer_factory=lambda: VideoTransformer(confidence, model),
                async_transform=True,
            )

    elif st.session_state.page == "Lihat Hasil Deteksi":
        st.header("Hasil Deteksi yang Disimpan")
        with st.expander("Klik untuk melihat hasil deteksi"):
            try:
                results = session.query(detections_table).all()
                if results:
                    local_tz = pytz.timezone('Asia/Jakarta')
                    for row in results:
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            local_time = row.detection_time.replace(tzinfo=pytz.utc).astimezone(local_tz) if row.detection_time else "N/A"
                            st.write(f"Username: {row.username}")
                            st.write(f"Image Name: {row.image_name}")
                            st.write(f"Detection Confidence: {row.detection_confidence}")
                            st.write(f"Detection Time: {local_time.strftime('%Y-%m-%d %H:%M:%S') if row.detection_time else 'N/A'}")

                            # Display image from BLOB
                            try:
                                if row.image_data:
                                    img = PIL.Image.open(io.BytesIO(row.image_data))
                                    st.image(img, caption=f'Gambar yang Dideteksi: {row.image_name}', use_column_width=True)
                                else:
                                    st.warning("Gambar tidak ditemukan di server.")
                            except Exception as ex:
                                st.error("Terjadi kesalahan saat membuka gambar deteksi.")
                                st.error(ex)
                        with col2:
                            if st.button("Delete", key=row.id):
                                session.query(detections_table).filter_by(id=row.id).delete()
                                session.commit()
                                st.experimental_rerun()
                        st.write("---")
                else:
                    st.write("Tidak ada hasil deteksi yang disimpan.")
            except Exception as ex:
                st.error("Terjadi kesalahan saat mengambil hasil deteksi.")
                st.error(ex)
