import streamlit as st
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import json

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Dashboard Gempa BMKG",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# FUNGSI UNTUK FETCH DATA DARI BMKG
# ============================================

@st.cache_data(ttl=300)  # Cache 5 menit
def fetch_gempa_terkini():
    """Fetch data gempa terkini dari BMKG (JSON)"""
    try:
        url = "https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        gempa = data['Infogempa']['gempa']
        
        # Parse koordinat (format: lat,lon)
        coords = gempa['Coordinates'].split(',')
        
        return {
            'Tanggal': gempa['Tanggal'],
            'Jam': gempa['Jam'],
            'DateTime': gempa['DateTime'],
            'Magnitude': float(gempa['Magnitude']),
            'Kedalaman': gempa['Kedalaman'],
            'Lintang': float(coords[0]),
            'Bujur': float(coords[1]),
            'Wilayah': gempa['Wilayah'],
            'Potensi': gempa['Potensi'],
            'Dirasakan': gempa.get('Dirasakan', '-'),
            'Shakemap': gempa['Shakemap'] # Tetap ambil data untuk jaga-jaga, tapi tidak akan ditampilkan
        }
    except Exception as e:
        st.error(f"Gagal mengambil data gempa terkini: {e}")
        return None

@st.cache_data(ttl=300)
def fetch_gempa_m5():
    """Fetch 15 gempa M 5.0+ dari BMKG (JSON)"""
    try:
        url = "https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        gempa_list = []
        for gempa in data['Infogempa']['gempa']:
            coords = gempa['Coordinates'].split(',')
            gempa_list.append({
                'Tanggal': gempa['Tanggal'],
                'Jam': gempa['Jam'],
                'DateTime': gempa['DateTime'],
                'Magnitude': float(gempa['Magnitude']),
                'Kedalaman': gempa['Kedalaman'],
                'Lintang': float(coords[0]),
                'Bujur': float(coords[1]),
                'Wilayah': gempa['Wilayah'],
                'Potensi': gempa['Potensi']
            })
        
        return pd.DataFrame(gempa_list)
    except Exception as e:
        st.error(f"Gagal mengambil data gempa M5+: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_gempa_dirasakan():
    """Fetch 15 gempa dirasakan dari BMKG (JSON)"""
    try:
        url = "https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json"
        response = requests.get(url, timeout=10)
        data = response.json()
        
        gempa_list = []
        for gempa in data['Infogempa']['gempa']:
            coords = gempa['Coordinates'].split(',')
            gempa_list.append({
                'Tanggal': gempa['Tanggal'],
                'Jam': gempa['Jam'],
                'DateTime': gempa['DateTime'],
                'Magnitude': float(gempa['Magnitude']),
                'Kedalaman': gempa['Kedalaman'],
                'Lintang': float(coords[0]),
                'Bujur': float(coords[1]),
                'Wilayah': gempa['Wilayah'],
                'Dirasakan': gempa.get('Dirasakan', '-')
            })
        
        return pd.DataFrame(gempa_list)
    except Exception as e:
        st.error(f"Gagal mengambil data gempa dirasakan: {e}")
        return pd.DataFrame()

# ============================================
# FUNGSI HELPER
# ============================================

def parse_kedalaman(kedalaman_str):
    """Parse string kedalaman menjadi float (contoh: '10 km' -> 10.0)"""
    try:
        return float(kedalaman_str.replace(' km', '').replace(',', '.'))
    except:
        return 10.0

def kategori_gempa(magnitude):
    """Kategorikan gempa berdasarkan magnitude"""
    if magnitude < 3:
        return "Mikro", "🟢"
    elif magnitude < 4:
        return "Minor", "🟡"
    elif magnitude < 5:
        return "Ringan", "🟠"
    elif magnitude < 6:
        return "Sedang", "🔴"
    elif magnitude < 7:
        return "Kuat", "🔴"
    else:
        return "Sangat Kuat", "⚫"

# ============================================
# HEADER & SIDEBAR
# ============================================

st.title("🌋 Dashboard Analisis & Prediksi Gempa Bumi")
st.markdown("### Data Real-time dari BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.image("https://cdn.bmkg.go.id/Web/Logo-BMKG-new.png", width=200)
    st.markdown("## 📊 Pengaturan")
    
    refresh = st.button("🔄 Refresh Data", use_container_width=True)
    if refresh:
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### Sumber Data")
    st.info("Data dari BMKG Portal:\nhttps://data.bmkg.go.id")
    
    st.markdown("---")
    st.markdown("### ⚠️ Disclaimer")
    st.caption("""
    Prediksi yang ditampilkan hanya untuk keperluan 
    pembelajaran dan tidak dapat dijadikan acuan resmi. 
    Untuk informasi resmi, kunjungi BMKG.
    """)

# ============================================
# TABS NAVIGATION
# ============================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔴 Gempa Terkini",
    "📊 Data Gempa M5+",
    "👥 Gempa Dirasakan",
    "🗺️ Peta Sebaran",
    "🤖 Prediksi AI"
])

# ============================================
# TAB 1: GEMPA TERKINI
# ============================================

with tab1:
    st.header("🔴 Gempabumi Terkini")
    
    gempa_terkini = fetch_gempa_terkini()
    
    if gempa_terkini:
        col1, col2, col3, col4 = st.columns(4)
        
        kategori, emoji = kategori_gempa(gempa_terkini['Magnitude'])
        
        with col1:
            st.metric("Magnitude", f"{gempa_terkini['Magnitude']} SR")
        with col2:
            st.metric("Kedalaman", gempa_terkini['Kedalaman'])
        with col3:
            st.metric("Tanggal", gempa_terkini['Tanggal'])
        with col4:
            st.metric("Jam", gempa_terkini['Jam'])
        
        st.markdown("---")
        
        # Menggunakan 3 kolom untuk menampilkan detail (Menghapus Shakemap)
        col1, col2, col3 = st.columns(3) 
        
        with col1:
            st.subheader("📍 Informasi Lokasi")
            st.write(f"**Wilayah:** {gempa_terkini['Wilayah']}")
            st.write(f"**Koordinat:** {gempa_terkini['Lintang']}, {gempa_terkini['Bujur']}")
        
        with col2:
            st.subheader("Classification")
            st.write(f"**Kategori:** {emoji} {kategori}")
            
            if gempa_terkini['Potensi']:
                st.warning(f"**⚠️ Potensi Tsunami:** {gempa_terkini['Potensi']}")
            
        with col3:
            st.subheader("Dampak")
            if gempa_terkini['Dirasakan'] != '-':
                st.info(f"**👥 Dirasakan:** {gempa_terkini['Dirasakan']}")
            else:
                st.caption("Belum ada informasi dirasakan atau tidak dirasakan.")


        # Peta lokasi
        st.subheader("🗺️ Lokasi Episentrum")
        
        # 
        
        map_data = pd.DataFrame({
            'lat': [gempa_terkini['Lintang']],
            'lon': [gempa_terkini['Bujur']],
            'magnitude': [gempa_terkini['Magnitude']],
            'location': [gempa_terkini['Wilayah']]
        })
        
        fig = px.scatter_mapbox(
            map_data,
            lat='lat',
            lon='lon',
            size='magnitude',
            color='magnitude',
            hover_data=['location'],
            color_continuous_scale='Reds',
            size_max=30,
            zoom=5,
            height=500
        )
        fig.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Gagal memuat data gempa terkini")

# ============================================
# TAB 2: GEMPA M5+
# ============================================

with tab2:
    st.header("📊 15 Gempabumi Magnitude 5.0+")
    
    df_m5 = fetch_gempa_m5()
    
    if not df_m5.empty:
        # Statistik
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Gempa M5+", len(df_m5))
        with col2:
            st.metric("Magnitude Tertinggi", f"{df_m5['Magnitude'].max():.1f} SR")
        with col3:
            st.metric("Magnitude Rata-rata", f"{df_m5['Magnitude'].mean():.2f} SR")
        with col4:
            rata_kedalaman = df_m5['Kedalaman'].apply(parse_kedalaman).mean()
            st.metric("Kedalaman Rata-rata", f"{rata_kedalaman:.0f} km")
        
        st.markdown("---")
        
        # Visualisasi
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Distribusi Magnitude")
            fig1 = px.histogram(
                df_m5,
                x='Magnitude',
                nbins=15,
                color_discrete_sequence=['#FF4B4B']
            )
            fig1.update_layout(
                xaxis_title="Magnitude",
                yaxis_title="Frekuensi",
                showlegend=False
            )
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            st.subheader("Gempa per Wilayah")
            fig2 = px.bar(
                x=df_m5['Wilayah'].value_counts().head(10).values,
                y=df_m5['Wilayah'].value_counts().head(10).index,
                orientation='h',
                color=df_m5['Wilayah'].value_counts().head(10).values,
                color_continuous_scale='Reds'
            )
            fig2.update_layout(
                xaxis_title="Jumlah Gempa",
                yaxis_title="Wilayah",
                showlegend=False
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Tabel data
        st.subheader("📋 Daftar Lengkap")
        
        # Format display
        df_display = df_m5.copy()
        df_display['Kategori'] = df_display['Magnitude'].apply(lambda x: kategori_gempa(x)[0])
        
        st.dataframe(
            df_display[['Tanggal', 'Jam', 'Magnitude', 'Kedalaman', 'Wilayah', 'Kategori', 'Potensi']],
            use_container_width=True,
            height=400
        )
    else:
        st.error("Gagal memuat data gempa M5+")

# ============================================
# TAB 3: GEMPA DIRASAKAN
# ============================================

with tab3:
    st.header("👥 15 Gempabumi Dirasakan")
    
    df_dirasakan = fetch_gempa_dirasakan()
    
    if not df_dirasakan.empty:
        # Statistik
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Gempa Dirasakan", len(df_dirasakan))
        with col2:
            st.metric("Magnitude Tertinggi", f"{df_dirasakan['Magnitude'].max():.1f} SR")
        with col3:
            st.metric("Magnitude Terendah", f"{df_dirasakan['Magnitude'].min():.1f} SR")
        
        st.markdown("---")
        
        # Tabel dengan highlight
        st.subheader("📋 Daftar Gempa yang Dirasakan")
        
        df_display = df_dirasakan.copy()
        df_display['Kategori'] = df_display['Magnitude'].apply(lambda x: kategori_gempa(x)[0])
        
        st.dataframe(
            df_display[['Tanggal', 'Jam', 'Magnitude', 'Kedalaman', 'Wilayah', 'Dirasakan', 'Kategori']],
            use_container_width=True,
            height=500
        )
        
        # Visualisasi wilayah yang sering merasakan gempa
        st.subheader("📈 Korelasi Tanggal dan Magnitude")
        
        st.info("""
        Grafik menunjukkan distribusi magnitude gempa yang dirasakan dari waktu ke waktu. 
        Titik yang lebih besar mewakili gempa dengan magnitude yang lebih tinggi.
        """)
        
        fig3 = px.scatter(
            df_dirasakan,
            x='Tanggal',
            y='Magnitude',
            size='Magnitude',
            color='Magnitude',
            hover_data=['Wilayah', 'Dirasakan'],
            color_continuous_scale='Reds'
        )
        fig3.update_layout(
            xaxis_title="Tanggal",
            yaxis_title="Magnitude",
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)
        
    else:
        st.error("Gagal memuat data gempa dirasakan")

# ============================================
# TAB 4: PETA SEBARAN
# ============================================

with tab4:
    st.header("🗺️ Peta Sebaran Semua Gempa")
    
    # Gabungkan semua data
    df_m5 = fetch_gempa_m5()
    df_dirasakan = fetch_gempa_dirasakan()
    
    if not df_m5.empty and not df_dirasakan.empty:
        # Gabung data
        df_m5['Tipe'] = 'M5+'
        df_dirasakan['Tipe'] = 'Dirasakan'
        
        # Menghapus duplikasi baris dengan kombinasi Lintang, Bujur, dan Magnitude yang sama
        df_all = pd.concat([df_m5, df_dirasakan]).drop_duplicates(
            subset=['Lintang', 'Bujur', 'Magnitude'], 
            keep='first'
        ).reset_index(drop=True)
        
        # Filter
        col1, col2 = st.columns(2)
        
        with col1:
            tipe_filter = st.multiselect(
                "Filter Tipe Gempa",
                options=['M5+', 'Dirasakan'],
                default=['M5+', 'Dirasakan']
            )
        
        with col2:
            mag_range = st.slider(
                "Filter Magnitude",
                float(df_all['Magnitude'].min()),
                float(df_all['Magnitude'].max()),
                (float(df_all['Magnitude'].min()), float(df_all['Magnitude'].max()))
            )
        
        # Apply filter
        df_filtered = df_all[
            (df_all['Tipe'].isin(tipe_filter)) &
            (df_all['Magnitude'] >= mag_range[0]) &
            (df_all['Magnitude'] <= mag_range[1])
        ]
        
        st.info(f"Menampilkan {len(df_filtered)} dari {len(df_all)} gempa")
        
        # Peta
        fig_map = px.scatter_mapbox(
            df_filtered,
            lat='Lintang',
            lon='Bujur',
            color='Magnitude',
            size='Magnitude',
            hover_data=['Tanggal', 'Jam', 'Wilayah', 'Kedalaman', 'Tipe'],
            color_continuous_scale='Reds',
            size_max=20,
            zoom=4,
            height=700,
            title="Sebaran Gempabumi di Indonesia"
        )
        
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 40, "l": 0, "b": 0}
        )
        
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Statistik wilayah
        st.subheader("📊 Statistik per Wilayah")
        
        wilayah_stats = df_filtered.groupby('Wilayah').agg({
            'Magnitude': ['count', 'mean', 'max']
        }).round(2)
        
        wilayah_stats.columns = ['Jumlah', 'Magnitude Rata-rata', 'Magnitude Maksimal']
        wilayah_stats = wilayah_stats.sort_values('Jumlah', ascending=False).head(10)
        
        st.dataframe(wilayah_stats, use_container_width=True)
        
    else:
        st.error("Gagal memuat data untuk peta")

# ============================================
# TAB 5: PREDIKSI AI
# ============================================

with tab5:
    st.header("🤖 Prediksi Magnitude Gempa dengan Machine Learning")
    
    st.info("""
    **Model Prediksi** menggunakan Random Forest Regressor untuk memprediksi magnitude 
    berdasarkan data historis gempa dari BMKG.
    
    ⚠️ **Catatan:** Prediksi ini hanya untuk pembelajaran dan tidak dapat digunakan 
    sebagai peringatan dini resmi.
    """)
    
    # Gabungkan data untuk training
    df_m5 = fetch_gempa_m5()
    df_dirasakan = fetch_gempa_dirasakan()
    
    if not df_m5.empty and not df_dirasakan.empty:
        df_train = pd.concat([df_m5, df_dirasakan]).drop_duplicates(
            subset=['Lintang', 'Bujur', 'Magnitude'], 
            keep='first'
        ).reset_index(drop=True)
        
        # Parse kedalaman
        df_train['Kedalaman_num'] = df_train['Kedalaman'].apply(parse_kedalaman)
        
        # Prepare features
        X = df_train[['Lintang', 'Bujur', 'Kedalaman_num']]
        y = df_train['Magnitude']
        
        # Training controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_estimators = st.slider("Jumlah Trees", 50, 200, 100, 25)
        with col2:
            max_depth = st.slider("Max Depth", 5, 20, 10, 5)
        with col3:
            test_size_percent = st.slider("Test Size (%)", 10, 40, 20, 5)
            test_size = test_size_percent / 100
        
        if st.button("🚀 Latih Model", type="primary", use_container_width=True):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            with st.spinner("Melatih model..."):
                # Train model
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Predict
                y_pred = model.predict(X_test)
                
                # Metrics
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                st.success("✅ Model berhasil dilatih!")
                
                # Display metrics
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("RMSE", f"{rmse:.3f}")
                col2.metric("MAE", f"{mae:.3f}")
                col3.metric("R² Score (Akurasi)", f"{r2:.3f}")
                col4.metric("Jumlah Data Training", len(X_train))
                
                # Visualisasi prediksi
                st.subheader("📈 Hasil Prediksi vs Aktual")
                
                # 
                
                fig_pred = go.Figure()
                
                fig_pred.add_trace(go.Scatter(
                    x=y_test,
                    y=y_pred,
                    mode='markers',
                    marker=dict(
                        color='#FF4B4B',
                        size=10,
                        opacity=0.6,
                        line=dict(width=1, color='white')
                    ),
                    name='Prediksi'
                ))
                
                # Perfect line
                min_val = min(y_test.min(), y_pred.min())
                max_val = max(y_test.max(), y_pred.max())
                
                fig_pred.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='blue', dash='dash', width=2),
                    name='Prediksi Sempurna'
                ))
                
                fig_pred.update_layout(
                    xaxis_title="Magnitude Aktual",
                    yaxis_title="Magnitude Prediksi",
                    hovermode='closest',
                    height=500
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Feature importance
                st.subheader("🎯 Feature Importance")
                
                importance_df = pd.DataFrame({
                    'Feature': ['Lintang', 'Bujur', 'Kedalaman'],
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=True)
                
                fig_imp = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Save model untuk prediksi manual
                st.session_state['trained_model'] = model
        
        # Prediksi Manual
        st.markdown("---")
        st.subheader("🔮 Prediksi Manual")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            input_lat = st.number_input(
                "Lintang (Latitude)",
                float(X['Lintang'].min()), float(X['Lintang'].max()), float(X['Lintang'].mean()), 0.1,
                help="Range: -11 (Selatan) hingga 6 (Utara)"
            )
        with col2:
            input_lon = st.number_input(
                "Bujur (Longitude)",
                float(X['Bujur'].min()), float(X['Bujur'].max()), float(X['Bujur'].mean()), 0.1,
                help="Range: 95 (Barat) hingga 141 (Timur)"
            )
        with col3:
            input_depth = st.number_input(
                "Kedalaman (km)",
                float(X['Kedalaman_num'].min()), float(X['Kedalaman_num'].max()), float(X['Kedalaman_num'].mean()), 1.0,
                help="Kedalaman episentrum gempa"
            )
        
        if st.button("🎯 Prediksi Magnitude", use_container_width=True):
            if 'trained_model' in st.session_state:
                model = st.session_state['trained_model']
                
                prediction = model.predict([[input_lat, input_lon, input_depth]])[0]
                
                st.success(f"### Prediksi Magnitude: **{prediction:.2f} SR**")
                
                kategori, emoji = kategori_gempa(prediction)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Kategori:** {emoji} {kategori}")
                
                with col2:
                    if prediction < 4:
                        st.success("Tingkat kerusakan: Rendah")
                    elif prediction < 6:
                        st.warning("Tingkat kerusakan: Sedang (Berpotensi merusak bangunan tidak kokoh)")
                    else:
                        st.error("Tingkat kerusakan: Tinggi (Berpotensi merusak bangunan dan infrastruktur)")
            else:
                st.warning("⚠️ Silakan latih model terlebih dahulu!")
    
    else:
        st.error("Gagal memuat data untuk training model")

# ============================================
# FOOTER
# ============================================

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <p><strong>Dashboard Gempabumi Indonesia</strong></p>
    <p>Sumber Data: <a href='https://data.bmkg.go.id' target='_blank'>BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)</a></p>
    <p style='color: gray; font-size: 12px;'>
        Data diperbarui secara berkala dari server BMKG.<br>
        Untuk informasi resmi dan peringatan dini, kunjungi 
        <a href='https://www.bmkg.go.id' target='_blank'>www.bmkg.go.id</a>
    </p>
</div>
""", unsafe_allow_html=True)