# 🌋 Dashboard Gempabumi Indonesia

Dashboard real-time untuk monitoring dan analisis gempa bumi di Indonesia menggunakan data dari BMKG.

## Features
✅ Data gempa real-time dari BMKG
✅ Visualisasi peta interaktif
✅ Analisis statistik gempa
✅ Prediksi magnitude dengan Machine Learning
✅ Informasi gempa terkini, M5+, dan yang dirasakan

## Data Source
Data diperoleh dari **BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)**
- Portal: https://data.bmkg.go.id/gempabumi
- Format: JSON
- Update: Real-time

## Installation

### Local Development
```bash
# Clone repository
git clone https://github.com/USERNAME/dashboard-gempa-bmkg.git
cd dashboard-gempa-bmkg

# Install dependencies
pip install -r requirements.txt

# Run aplikasi
streamlit run app.py
```

## Live Demo
🔗 [https://USERNAME-dashboard-gempa-bmkg.streamlit.app](https://USERNAME-dashboard-gempa-bmkg.streamlit.app)

## Screenshots
[Tambahkan screenshot nanti]

## Tech Stack
- **Frontend:** Streamlit
- **Data Visualization:** Plotly
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy

## API Endpoints BMKG
1. Gempa Terkini: `https://data.bmkg.go.id/DataMKG/TEWS/autogempa.json`
2. Gempa M5+: `https://data.bmkg.go.id/DataMKG/TEWS/gempaterkini.json`
3. Gempa Dirasakan: `https://data.bmkg.go.id/DataMKG/TEWS/gempadirasakan.json`

## Disclaimer
⚠️ Aplikasi ini dibuat untuk keperluan edukasi dan pembelajaran. Untuk informasi resmi dan peringatan dini gempa, kunjungi [www.bmkg.go.id](https://www.bmkg.go.id)

## Attribution
Sumber data: **BMKG (Badan Meteorologi, Klimatologi, dan Geofisika)**

## License
MIT License

## Author
[Nama Anda]

## Contact
- GitHub: [@username](https://github.com/username)
- Email: aurakkireina@upi.edu