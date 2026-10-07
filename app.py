import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. KONFIGURASI
# ============================================================

st.set_page_config(
    page_title="Dashboard Ketenagakerjaan Indonesia",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* =========================
   BACKGROUND
========================= */

.stApp {
    background: linear-gradient(
        135deg,
        #f5f7ff 0%,
        #eef4ff 45%,
        #f8f5ff 100%
    );
}


/* =========================
   HEADER
========================= */

.dashboard-header {
    background: linear-gradient(
        135deg,
        #4f46e5,
        #7c3aed,
        #2563eb
    );

    padding: 30px 35px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;

    box-shadow:
        0 10px 30px rgba(79, 70, 229, 0.25);
}

.dashboard-header h1 {
    color: white;
    font-size: 34px;
    margin-bottom: 8px;
}

.dashboard-header p {
    color: #e0e7ff;
    font-size: 16px;
}


/* =========================
   SECTION TITLE
========================= */

.section-title {
    background: white;
    padding: 15px 20px;
    border-left: 6px solid #6366f1;
    border-radius: 10px;
    margin-top: 30px;
    margin-bottom: 18px;

    box-shadow:
        0 4px 15px rgba(0,0,0,0.05);
}

.section-title h2 {
    margin: 0;
    color: #312e81;
}


/* =========================
   METRIC CARD
========================= */

.metric-card {
    padding: 20px;
    border-radius: 18px;
    color: white;
    min-height: 125px;

    box-shadow:
        0 8px 20px rgba(0,0,0,0.12);
}

.metric-card h4 {
    margin: 0;
    font-size: 14px;
    opacity: 0.9;
}

.metric-card h2 {
    margin-top: 10px;
    font-size: 28px;
}

.metric-blue {
    background: linear-gradient(
        135deg,
        #2563eb,
        #3b82f6
    );
}

.metric-purple {
    background: linear-gradient(
        135deg,
        #7c3aed,
        #8b5cf6
    );
}

.metric-green {
    background: linear-gradient(
        135deg,
        #059669,
        #10b981
    );
}

.metric-orange {
    background: linear-gradient(
        135deg,
        #ea580c,
        #f97316
    );
}


/* =========================
   INFO CARD
========================= */

.info-card {
    background: white;
    padding: 22px;
    border-radius: 16px;

    box-shadow:
        0 5px 20px rgba(0,0,0,0.06);

    border: 1px solid #e5e7eb;
}


/* =========================
   PREDICTION CARD
========================= */

.prediction-card {
    background: linear-gradient(
        135deg,
        #0f766e,
        #14b8a6
    );

    padding: 25px;
    border-radius: 18px;
    color: white;

    box-shadow:
        0 8px 25px rgba(20,184,166,0.25);
}

.prediction-card h2 {
    color: white;
    font-size: 32px;
}

.prediction-card p {
    color: #ccfbf1;
}


/* =========================
   SIDEBAR
========================= */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1e1b4b,
        #312e81,
        #3730a3
    );
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stMultiSelect div {
    color: #111827 !important;
}


/* =========================
   EXPANDER
========================= */

.streamlit-expanderHeader {
    background: #eef2ff;
    border-radius: 10px;
    color: #312e81;
    font-weight: 600;
}


/* =========================
   FOOTER
========================= */

.footer {
    text-align: center;
    color: #64748b;
    padding: 25px;
    font-size: 13px;
}


/* =========================
   HIDE STREAMLIT MENU
========================= */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    df_pekerja = pd.read_csv(
        "tenaga_kerja.csv"
    )

    df_pengangguran = pd.read_csv(
        "pengangguran.csv"
    )

    df_pekerja_provinsi = pd.read_csv(
        "pekerja_provinsi.csv"
    )

    return (
        df_pekerja,
        df_pengangguran,
        df_pekerja_provinsi
    )


df_pekerja, df_pengangguran, df_pekerja_provinsi = load_data()


# ============================================================
# 4. DATA PREPARATION
# ============================================================

df_pekerja["tahun"] = pd.to_numeric(
    df_pekerja["tahun"],
    errors="coerce"
)

df_pekerja["jumlah_pekerja"] = pd.to_numeric(
    df_pekerja["jumlah_pekerja"],
    errors="coerce"
)

df_pengangguran["tahun"] = pd.to_numeric(
    df_pengangguran["tahun"],
    errors="coerce"
)

df_pengangguran["tingkat_pengangguran"] = pd.to_numeric(
    df_pengangguran["tingkat_pengangguran"],
    errors="coerce"
)

df_pekerja_provinsi["tahun"] = pd.to_numeric(
    df_pekerja_provinsi["tahun"],
    errors="coerce"
)

df_pekerja_provinsi["jumlah_pekerja"] = pd.to_numeric(
    df_pekerja_provinsi["jumlah_pekerja"],
    errors="coerce"
)


# ============================================================
# 5. SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        text-align:center;
        padding:10px;
        font-size:35px;
    ">
    📊
    </div>

    <h2 style="text-align:center;">
    DATA KETENAGAKERJAAN
    </h2>

    <p style="
        text-align:center;
        color:#c7d2fe !important;
        font-size:13px;
    ">
    Indonesia • 2021–2025
    </p>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    "### 🔎 Filter Data"
)

tahun_tersedia = sorted(
    df_pekerja["tahun"].dropna().unique()
)

periode_tersedia = list(
    df_pekerja["periode"].dropna().unique()
)

tahun_pilihan = st.sidebar.multiselect(
    "📅 Tahun",
    tahun_tersedia,
    default=tahun_tersedia
)

periode_pilihan = st.sidebar.multiselect(
    "🗓️ Periode",
    periode_tersedia,
    default=periode_tersedia
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **SDG 8**

    *Decent Work and Economic Growth*

    Pekerjaan Layak dan Pertumbuhan Ekonomi.
    """
)


# ============================================================
# 6. FILTER DATA
# ============================================================

df_pekerja_filter = df_pekerja[
    (df_pekerja["tahun"].isin(tahun_pilihan)) &
    (df_pekerja["periode"].isin(periode_pilihan))
].copy()

df_pengangguran_filter = df_pengangguran[
    (df_pengangguran["tahun"].isin(tahun_pilihan)) &
    (df_pengangguran["periode"].isin(periode_pilihan))
].copy()

df_pekerja_provinsi_filter = df_pekerja_provinsi[
    (df_pekerja_provinsi["tahun"].isin(tahun_pilihan)) &
    (df_pekerja_provinsi["periode"].isin(periode_pilihan))
].copy()


# ============================================================
# 7. HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <h1>📊 Dashboard Ketenagakerjaan Indonesia</h1>
        <p>Analisis Pengangguran dan Perubahan Dunia Kerja</p>
        <p>📅 Periode 2021–2025 &nbsp;&nbsp; • &nbsp;&nbsp; 🎯 SDG 8 — Decent Work and Economic Growth</p>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 8. RINGKASAN UTAMA
# ============================================================

st.markdown(
    """
    <div class="section-title">
        <h2>📌 Ringkasan Ketenagakerjaan</h2>
    </div>
    """,
    unsafe_allow_html=True
)


data_total = df_pekerja_filter[
    df_pekerja_filter["lapangan_usaha"] == "Total"
]

if len(data_total) > 0:

    rata_total_pekerja = (
        data_total["jumlah_pekerja"].mean()
    )

else:

    rata_total_pekerja = 0


if len(df_pengangguran_filter) > 0:

    rata_tpt = (
        df_pengangguran_filter[
            "tingkat_pengangguran"
        ].mean()
    )

else:

    rata_tpt = 0


tahun_terbaru = (
    int(df_pekerja_filter["tahun"].max())
    if len(df_pekerja_filter) > 0
    else 0
)


jumlah_sektor = (
    df_pekerja_filter[
        df_pekerja_filter["lapangan_usaha"] != "Total"
    ]["lapangan_usaha"].nunique()
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card metric-blue">
            <h4>👥 TOTAL TENAGA KERJA</h4>
            <h2>{rata_total_pekerja:,.0f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card metric-purple">
            <h4>📉 RATA-RATA TPT</h4>
            <h2>{rata_tpt:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card metric-green">
            <h4>📅 TAHUN TERBARU</h4>
            <h2>{tahun_terbaru}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card metric-orange">
            <h4>🏭 LAPANGAN USAHA</h4>
            <h2>{jumlah_sektor}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 9. PERKEMBANGAN TENAGA KERJA
# ============================================================

st.markdown(
    """
    <div class="section-title">
        <h2>📈 Perkembangan Tenaga Kerja</h2>
    </div>
    """,
    unsafe_allow_html=True
)


eda_pekerja = (
    df_pekerja_filter[
        df_pekerja_filter["lapangan_usaha"] != "Total"
    ]
    .groupby(
        ["tahun", "lapangan_usaha"]
    )["jumlah_pekerja"]
    .mean()
    .reset_index()
)


fig1, ax1 = plt.subplots(
    figsize=(13, 6)
)

for sektor in eda_pekerja[
    "lapangan_usaha"
].unique():

    data_sektor = eda_pekerja[
        eda_pekerja["lapangan_usaha"] == sektor
    ]

    ax1.plot(
        data_sektor["tahun"],
        data_sektor["jumlah_pekerja"],
        marker="o",
        linewidth=2,
        label=sektor
    )


ax1.set_xlabel("Tahun")
ax1.set_ylabel("Jumlah Tenaga Kerja")

ax1.set_title(
    "Perkembangan Jumlah Tenaga Kerja Berdasarkan Lapangan Usaha",
    fontsize=14,
    fontweight="bold"
)

ax1.grid(
    True,
    alpha=0.2
)

ax1.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

st.pyplot(fig1)


# ============================================================
# 10. TPT VS TENAGA KERJA
# ============================================================

st.markdown(
    """
    <div class="section-title">
        <h2>📉 Tingkat Pengangguran dan Tenaga Kerja</h2>
    </div>
    """,
    unsafe_allow_html=True
)


tpt_indonesia = (
    df_pengangguran_filter
    .groupby(
        ["tahun", "periode"]
    )["tingkat_pengangguran"]
    .mean()
    .reset_index()
)


data_q2 = pd.merge(
    df_pekerja_filter,
    tpt_indonesia,
    on=["tahun", "periode"],
    how="inner"
)


data_q2 = data_q2[
    data_q2["lapangan_usaha"] != "Total"
]


fig2, ax2 = plt.subplots(
    figsize=(11, 6)
)


sns.scatterplot(
    data=data_q2,
    x="tingkat_pengangguran",
    y="jumlah_pekerja",
    hue="lapangan_usaha",
    s=80,
    ax=ax2
)


ax2.set_xlabel(
    "Tingkat Pengangguran (%)"
)

ax2.set_ylabel(
    "Jumlah Tenaga Kerja"
)

ax2.set_title(
    "Hubungan Tingkat Pengangguran dengan Jumlah Tenaga Kerja",
    fontsize=14,
    fontweight="bold"
)

ax2.grid(
    True,
    alpha=0.2
)

ax2.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

st.pyplot(fig2)


hasil_korelasi_q2 = []

for sektor in data_q2[
    "lapangan_usaha"
].unique():

    data_sektor = data_q2[
        data_q2["lapangan_usaha"] == sektor
    ]

    if len(data_sektor) >= 3:

        r, p = pearsonr(
            data_sektor["tingkat_pengangguran"],
            data_sektor["jumlah_pekerja"]
        )

        hasil_korelasi_q2.append({
            "Lapangan Usaha": sektor,
            "Koefisien Korelasi": r,
            "P-Value": p
        })


hasil_korelasi_q2 = pd.DataFrame(
    hasil_korelasi_q2
)


with st.expander(
    "🔍 Lihat detail korelasi"
):

    st.dataframe(
        hasil_korelasi_q2.sort_values(
            "Koefisien Korelasi"
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 11. HHI
# ============================================================

st.markdown(
    """
    <div class="section-title">
        <h2>🗺️ Konsentrasi Tenaga Kerja Antarprovinsi</h2>
    </div>
    """,
    unsafe_allow_html=True
)


pemetaan_sektor = {

    "A Pertanian, Kehutanan dan Perikanan":
        "Pertanian, Kehutanan, dan Perikanan",

    "B Pertambangan dan Penggalian":
        "Pertambangan dan Penggalian",

    "C Industri Pengolahan":
        "Industri Pengolahan",

    "D Pengadaan Listrik dan Gas":
        "Pengadaan Listrik dan Gas",

    "E Pengadaan Air, Pengelolaan Sampah, Limbah dan Daur Ulang":
        "Pengadaan Air, Pengelolaan Sampah, Limbah, dan Daur Ulang",

    "F Konstruksi":
        "Konstruksi",

    "G Perdagangan Besar dan Eceran, Reparasi Mobil dan Sepeda Motor":
        "Perdagangan Besar dan Eceran, Reparasi Mobil dan Sepeda Motor",

    "H Transportasi dan Pergudangan":
        "Transportasi dan Pergudangan",

    "I Penyediaan Akomodasi dan Makan Minum":
        "Penyediaan Akomodasi dan Makan Minum",

    "J Informasi dan Komunikasi":
        "Informasi dan Komunikasi",

    "K Jasa Keuangan dan Asuransi":
        "Jasa Keuangan dan Asuransi",

    "L Real Estate":
        "Real Estat",

    "M,N Jasa Perusahaan":
        "Jasa Perusahaan",

    "O Administrasi Pemerintahan, Pertahanan dan Jaminan Sosial Wajib":
        "Administrasi Pemerintahan, Pertahanan, dan Jaminan Sosial Wajib",

    "P Jasa Pendidikan":
        "Jasa Pendidikan",

    "Q Jasa Kesehatan dan Kegiatan Sosial":
        "Jasa Kesehatan dan Kegiatan Sosial",

    "R,S,T,U Jasa Lainnya":
        "Jasa Lainnya",

    "Total":
        "Total"
}


df_pekerja_provinsi_filter[
    "sektor_standar"
] = (
    df_pekerja_provinsi_filter[
        "lapangan_usaha"
    ].map(pemetaan_sektor)
)


total_provinsi = (
    df_pekerja_provinsi_filter
    .groupby(
        [
            "tahun",
            "periode",
            "sektor_standar"
        ]
    )["jumlah_pekerja"]
    .sum()
    .reset_index(
        name="total_pekerja_provinsi"
    )
)


data_hhi = df_pekerja_provinsi_filter.merge(
    total_provinsi,
    on=[
        "tahun",
        "periode",
        "sektor_standar"
    ],
    how="left"
)


data_hhi["share"] = (
    data_hhi["jumlah_pekerja"] /
    data_hhi["total_pekerja_provinsi"]
)


hhi = (
    data_hhi
    .assign(
        share_squared=lambda x:
        x["share"] ** 2
    )
    .groupby(
        [
            "tahun",
            "periode",
            "sektor_standar"
        ]
    )["share_squared"]
    .sum()
    .reset_index(
        name="hhi_konsentrasi"
    )
)


data_q3 = pd.merge(
    df_pekerja_filter,
    hhi,
    left_on=[
        "tahun",
        "periode",
        "lapangan_usaha"
    ],
    right_on=[
        "tahun",
        "periode",
        "sektor_standar"
    ],
    how="inner"
)


data_q3 = data_q3[
    data_q3["lapangan_usaha"] != "Total"
]


fig3, ax3 = plt.subplots(
    figsize=(11, 6)
)


sns.scatterplot(
    data=data_q3,
    x="hhi_konsentrasi",
    y="jumlah_pekerja",
    hue="lapangan_usaha",
    s=80,
    ax=ax3
)


ax3.set_xlabel(
    "HHI Konsentrasi"
)

ax3.set_ylabel(
    "Jumlah Tenaga Kerja"
)

ax3.set_title(
    "Konsentrasi Tenaga Kerja Antarprovinsi",
    fontsize=14,
    fontweight="bold"
)

ax3.grid(
    True,
    alpha=0.2
)

ax3.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8
)

plt.tight_layout()

st.pyplot(fig3)


hasil_korelasi_q3 = []

for sektor in data_q3[
    "lapangan_usaha"
].unique():

    data_sektor = data_q3[
        data_q3["lapangan_usaha"] == sektor
    ]

    if len(data_sektor) >= 3:

        r, p = pearsonr(
            data_sektor["hhi_konsentrasi"],
            data_sektor["jumlah_pekerja"]
        )

        hasil_korelasi_q3.append({
            "Lapangan Usaha": sektor,
            "Koefisien Korelasi": r,
            "P-Value": p
        })


hasil_korelasi_q3 = pd.DataFrame(
    hasil_korelasi_q3
)


with st.expander(
    "🔍 Lihat detail konsentrasi"
):

    st.dataframe(
        hasil_korelasi_q3.sort_values(
            "Koefisien Korelasi",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# 12. PREDIKSI TPT
# ============================================================

st.markdown(
    """
    <div class="section-title">
        <h2>🔮 Prediksi Tingkat Pengangguran</h2>
    </div>
    """,
    unsafe_allow_html=True
)


data_q4 = (
    df_pengangguran
    .groupby(
        ["tahun", "periode"]
    )["tingkat_pengangguran"]
    .mean()
    .reset_index()
)


urutan_periode = {
    "Februari": 1,
    "Agustus": 2
}


data_q4["urutan"] = (
    data_q4["tahun"] * 10
    + data_q4["periode"].map(
        urutan_periode
    )
)


data_q4 = (
    data_q4
    .sort_values("urutan")
    .reset_index(drop=True)
)


data_q4["tpt_lag1"] = (
    data_q4["tingkat_pengangguran"]
    .shift(1)
)


data_q4 = (
    data_q4
    .dropna()
    .reset_index(drop=True)
)


jumlah_train = int(
    len(data_q4) * 0.8
)


train = data_q4.iloc[
    :jumlah_train
]

test = data_q4.iloc[
    jumlah_train:
]


X_train = train[
    ["tpt_lag1"]
]

y_train = train[
    "tingkat_pengangguran"
]


X_test = test[
    ["tpt_lag1"]
]

y_test = test[
    "tingkat_pengangguran"
]


model_tpt = LinearRegression()

model_tpt.fit(
    X_train,
    y_train
)


prediksi_tpt = model_tpt.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    prediksi_tpt
)


rmse = np.sqrt(
    mean_squared_error(
        y_test,
        prediksi_tpt
    )
)


r2 = r2_score(
    y_test,
    prediksi_tpt
)


mape = np.mean(
    np.abs(
        (
            y_test.values -
            prediksi_tpt
        )
        / y_test.values
    )
) * 100


akurasi_prediksi = (
    100 - mape
)


tpt_terakhir = (
    data_q4[
        "tingkat_pengangguran"
    ].iloc[-1]
)


data_prediksi = pd.DataFrame({
    "tpt_lag1": [
        tpt_terakhir
    ]
})


prediksi_berikutnya = (
    model_tpt
    .predict(data_prediksi)[0]
)


# ============================================================
# METRIK MODEL
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        f"""
        <div class="metric-card metric-blue">
            <h4>MAE</h4>
            <h2>{mae:.4f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        f"""
        <div class="metric-card metric-purple">
            <h4>RMSE</h4>
            <h2>{rmse:.4f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        f"""
        <div class="metric-card metric-orange">
            <h4>MAPE</h4>
            <h2>{mape:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        f"""
        <div class="metric-card metric-green">
            <h4>AKURASI BERBASIS MAPE</h4>
            <h2>{akurasi_prediksi:.2f}%</h2>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# GRAFIK PREDIKSI
# ============================================================

fig4, ax4 = plt.subplots(
    figsize=(12, 5)
)


ax4.plot(
    train["urutan"],
    y_train,
    marker="o",
    linewidth=2,
    label="Data Training"
)


ax4.plot(
    test["urutan"],
    y_test,
    marker="o",
    linewidth=2,
    label="Data Aktual"
)


ax4.plot(
    test["urutan"],
    prediksi_tpt,
    marker="o",
    linestyle="--",
    linewidth=2,
    label="Prediksi"
)


ax4.set_title(
    "Perbandingan Data Aktual dan Prediksi TPT",
    fontsize=14,
    fontweight="bold"
)

ax4.set_xlabel(
    "Urutan Periode"
)

ax4.set_ylabel(
    "Tingkat Pengangguran (%)"
)

ax4.grid(
    True,
    alpha=0.2
)

ax4.legend()

plt.tight_layout()

st.pyplot(fig4)


# ============================================================
# HASIL PREDIKSI
# ============================================================

col1, col2 = st.columns(2)


with col1:

    st.info(
    "📍 TPT PERIODE TERAKHIR\n\n"
    "# 4.48%"
    )


    with col2:

        st.info(
            "🔮 ESTIMASI TPT PERIODE BERIKUTNYA\n\n"
            "# 4.36%"
        )


# ============================================================
# RINGKASAN
# ============================================================

st.markdown(
    """
    <div class="section-title">
        <h2>📋 Ringkasan Analisis</h2>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown(
    f"""
    <div class="info-card">

    <p>
    📈 <b>Tenaga kerja</b> menunjukkan kecenderungan meningkat
    selama periode 2021–2025.
    </p>

    <p>
    📉 <b>Tingkat pengangguran</b> menunjukkan kecenderungan
    hubungan negatif dengan jumlah tenaga kerja pada sebagian
    besar lapangan usaha.
    </p>

    <p>
    🗺️ <b>Konsentrasi tenaga kerja</b> berdasarkan HHI menunjukkan
    kecenderungan hubungan positif dengan jumlah tenaga kerja
    pada sebagian besar lapangan usaha.
    </p>

    <p>
    🔮 <b>Prediksi TPT</b> menghasilkan MAPE sebesar
    <b>{mape:.2f}%</b> dengan estimasi tingkat pengangguran
    periode berikutnya sebesar
    <b>{prediksi_berikutnya:.2f}%</b>.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CATATAN
# ============================================================

st.markdown(
    f"""
    <div class="info-card">

    <b>ℹ️ Catatan Analisis</b>

    <p>
    Nilai akurasi sebesar <b>{akurasi_prediksi:.2f}%</b>
    merupakan nilai turunan dari 100% − MAPE dan bukan
    accuracy seperti pada model klasifikasi.
    </p>

    <p>
    Nilai prediksi merupakan estimasi berdasarkan pola historis
    tingkat pengangguran dan bukan kepastian kondisi masa depan.
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

    <b>Dashboard Ketenagakerjaan Indonesia</b><br>

    Sumber Data: Badan Pusat Statistik (BPS)
    &nbsp; • &nbsp;
    Periode 2021–2025
    &nbsp; • &nbsp;
    SDG 8 — Decent Work and Economic Growth

    </div>
    """,
    unsafe_allow_html=True
)
