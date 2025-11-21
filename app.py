# ===============================================
# Streamlit App: Mental Health of Students
# Tim: Kelompok Bjorkanism
# Dataset: https://www.kaggle.com/datasets/aminasalamat/mental-health-of-students-dataset
# Jalankan: streamlit run app.py
# ===============================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Mental Health of Students – Kelompok Bjorkanism",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sedikit styling agar lebih menarik
st.markdown("""
<style>
/* rapikan lebar semua elemen */
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
/* badge kecil */
.badge {display:inline-block; padding:3px 8px; border-radius:8px; background:#E0F2FE; color:#0369A1; font-weight:600; font-size:0.8rem;}
/* card tipis */
div[data-testid="stMetricValue"] {font-size: 1.6rem;}
</style>
""", unsafe_allow_html=True)

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.title("Kelompok Bjorkanism 🧠")
    st.caption("Interactive Website – Mental Health of Students")
    st.markdown("""
**Anggota:**
- Muhammad Dimas Sudirman  
- Ari Wahyu Patriangga  
- Lola Aritasari
""")
    st.markdown("---")
    page = st.radio("Menu",
                    ["🏠 Beranda",
                     "🔎 Explorer",
                     "📊 Dashboard",
                     "🧮 Modeling",
                     "📝 Insight & Laporan",
                     "ℹ️ About"])
    st.markdown("---")
    st.markdown("**Dataset Input**")
    st.caption("1) Letakkan CSV Kaggle di `data/mental_health_students.csv` **atau** 2) Upload di sini.")
    up = st.file_uploader("Upload CSV", type=["csv"])

# ---------------------- HELPERS ----------------------
COMMON_BOOL = [
    "Depression","depression",
    "Anxiety","anxiety",
    "Panic attack","panic attack","Panic","panic",
    "Did you seek treatment?","Treatment","treatment"
]

def normalize_yesno(series: pd.Series):
    """Map berbagai variasi Yes/No -> 1/0, pertahankan NaN."""
    mapping = {"yes":1,"y":1,"true":1,"1":1,1:1,
               "no":0,"n":0,"false":0,"0":0,0:0}
    return series.astype(str).str.strip().str.lower().map(mapping)

def read_default_csv():
    for p in ["data/mental_health_students.csv",
              "data/mental_health.csv",
              "data/sample_mental_health.csv"]:
        try:
            return pd.read_csv(p)
        except Exception:
            continue
    return None

@st.cache_data(show_spinner=False)
def load_data(uploaded):
    if uploaded is not None:
        try:
            return pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Gagal membaca file upload: {e}")
            return None
    return read_default_csv()

def smart_cast(df: pd.DataFrame):
    """Coba konversi object->numeric jika memungkinkan."""
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="ignore")
    return df

def safe_col(df, candidates):
    """Ambil kolom pertama yang ada dari daftar kandidat."""
    return next((c for c in candidates if c in df.columns), None)

# ---------------------- LOAD DATA ----------------------
df = load_data(up)
if df is None or df.empty:
    st.error("Dataset belum tersedia. Upload CSV atau letakkan file Kaggle pada `data/mental_health_students.csv`.")
    st.stop()
df = smart_cast(df.copy())

# ---------------------- HEADER ----------------------
st.title("Mental Health of Students – Kelompok Bjorkanism")
st.caption("Sumber data: Kaggle (Amina Salamat). Aplikasi ini untuk keperluan edukasi dan eksplorasi data.")

# ---------------------- PAGES ----------------------
# ========== BERANDA ==========
if page == "🏠 Beranda":
    st.subheader("Selamat datang! 👋")
    st.write("""
Aplikasi ini membantu menjelajah data **kesehatan mental mahasiswa** melalui:
- **Explorer:** filter dinamis & statistik ringkas.  
- **Dashboard:** visualisasi interaktif (prevalensi, korelasi, perbandingan).  
- **Modeling:** pemodelan klasifikasi untuk memprediksi target biner (misal *Depression/Anxiety*).  
- **Insight & Laporan:** ringkasan temuan otomatis + unduhan.
""")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    bool_like = [c for c in df.columns if c in COMMON_BOOL]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observasi", f"{len(df):,}")
    c2.metric("Kolom Numerik", f"{len(num_cols)}")
    c3.metric("Kolom Kategorikal", f"{len(cat_cols)}")
    c4.metric("Kandidat Target (Yes/No)", f"{len(bool_like)}")

    st.markdown("### Pratinjau Data")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)

    st.markdown("### Catatan Data")
    st.markdown('<span class="badge">Tip</span> Gunakan menu **Explorer** untuk menyaring Gender/Course/Year/CGPA dan unduh data terfilter.',
                unsafe_allow_html=True)

# ========== EXPLORER ==========
elif page == "🔎 Explorer":
    st.subheader("Data Explorer 🔍")

    left, right = st.columns([1,2], gap="large")

    with left:
        gcol = safe_col(df, ["Gender","gender"])
        if gcol:
            g_vals = sorted(df[gcol].dropna().astype(str).unique().tolist())
            g_sel = st.multiselect("Filter Gender", g_vals, default=g_vals)

        ccol = safe_col(df, ["Course","course"])
        if ccol:
            c_vals = sorted(df[ccol].dropna().astype(str).unique().tolist())
            c_sel = st.multiselect("Filter Course", c_vals, default=c_vals[:6] if len(c_vals)>6 else c_vals)

        ycol = safe_col(df, ["Year of Study","Year","year"])
        if ycol:
            y_vals = sorted([str(x) for x in df[ycol].dropna().astype(str).unique()])
            y_sel = st.multiselect("Filter Year", y_vals, default=y_vals)

        cgcol = safe_col(df, ["CGPA","cgpa"])
        if cgcol:
            cmin, cmax = float(np.nanmin(df[cgcol])), float(np.nanmax(df[cgcol]))
            rmin, rmax = st.slider("Rentang CGPA",
                                   min_value=float(round(min(0.0, cmin),2)),
                                   max_value=float(round(max(4.0, cmax),2)),
                                   value=(float(round(cmin,2)), float(round(cmax,2))))

    sub = df.copy()
    if gcol: sub = sub[sub[gcol].astype(str).isin(g_sel)]
    if ccol: sub = sub[sub[ccol].astype(str).isin(c_sel)]
    if ycol: sub = sub[sub[ycol].astype(str).isin(y_sel)]
    if cgcol: sub = sub[(sub[cgcol] >= rmin) & (sub[cgcol] <= rmax)]

    with right:
        c1, c2, c3 = st.columns(3)
        c1.metric("Observasi", f"{len(sub):,}")
        c2.metric("Kolom", f"{sub.shape[1]}")
        if cgcol:
            c3.metric("Rata-rata CGPA", f"{sub[cgcol].mean():.2f}")
        else:
            c3.metric("Rata-rata CGPA", "-")

        tabs = st.tabs(["Tabel", "Distribusi Umum", "Top Kategori"])
        with tabs[0]:
            st.dataframe(sub.head(1000), use_container_width=True)
            st.download_button("Download data terfilter (CSV)", sub.to_csv(index=False), "filtered_data.csv")

        with tabs[1]:
            agecol = safe_col(sub, ["Age","age"])
            if agecol:
                st.plotly_chart(px.histogram(sub, x=agecol, nbins=30, title="Distribusi Usia"),
                                use_container_width=True)
            if cgcol:
                st.plotly_chart(px.histogram(sub, x=cgcol, nbins=30, title="Distribusi CGPA"),
                                use_container_width=True)
            sleepcol = safe_col(sub, ["Sleep Duration","sleep","Sleep"])
            if sleepcol:
                st.plotly_chart(px.histogram(sub, x=sleepcol, nbins=30, title="Distribusi Durasi Tidur"),
                                use_container_width=True)

        with tabs[2]:
            if gcol:
                st.plotly_chart(px.pie(sub, names=gcol, title="Komposisi Gender"), use_container_width=True)
            if ccol:
                top_course = (sub[ccol].value_counts().reset_index()
                              .rename(columns={"index":ccol, ccol:"count"}).head(15))
                st.plotly_chart(px.bar(top_course, x="count", y=ccol, orientation="h",
                                       title="Top Course by Jumlah Sampel"), use_container_width=True)

# ========== DASHBOARD ==========
elif page == "📊 Dashboard":
    st.subheader("Visualisasi Interaktif 📈")

    # target kandidat
    bool_candidates = [c for c in df.columns if c in COMMON_BOOL]
    for c in df.columns:
        vals = pd.Series(df[c].dropna().astype(str).str.lower().unique())
        if 1 <= len(vals) <= 5 and set(vals).issubset({"yes","no","y","n","true","false","0","1"}):
            if c not in bool_candidates:
                bool_candidates.append(c)
    target = st.selectbox("Pilih target (Yes/No) untuk prevalensi",
                          options=bool_candidates or ["(tidak ada)"])

    gcol = safe_col(df, ["Gender","gender"])
    ycol = safe_col(df, ["Year of Study","Year","year"])
    cgcol = safe_col(df, ["CGPA","cgpa"])
    slcol = safe_col(df, ["Sleep Duration","sleep","Sleep"])

    tnorm = normalize_yesno(df[target]) if target != "(tidak ada)" else None

    colA, colB = st.columns(2, gap="large")
    if tnorm is not None and tnorm.notna().any():
        if gcol:
            by_gender = (df[[gcol]].assign(target=tnorm).groupby(gcol)["target"].mean().reset_index())
            by_gender["Prevalence (%)"] = by_gender["target"] * 100
            colA.plotly_chart(px.bar(by_gender, x=gcol, y="Prevalence (%)",
                                     title=f"Prevalensi {target} by Gender"),
                              use_container_width=True)
        if ycol:
            by_year = (df[[ycol]].assign(target=tnorm).groupby(ycol)["target"].mean().reset_index())
            by_year["Prevalence (%)"] = by_year["target"] * 100
            colB.plotly_chart(px.line(by_year, x=ycol, y="Prevalence (%)", markers=True,
                                      title=f"Prevalensi {target} by Tahun Studi"),
                              use_container_width=True)
    else:
        colA.info("Pilih target biner untuk melihat prevalensi.")
        colB.empty()

    st.markdown("---")
    tabs = st.tabs(["Heatmap Korelasi", "CGPA vs Sleep", "Boxplot CGPA", "Sankey (Jika Ada)"])

    with tabs[0]:
        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] >= 2:
            corr = num_df.corr(numeric_only=True)
            st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto",
                                      title="Heatmap Korelasi (Numerik)"),
                            use_container_width=True)
        else:
            st.info("Butuh ≥2 kolom numerik untuk korelasi.")

    with tabs[1]:
        if cgcol and slcol:
            color_col = target if (target != "(tidak ada)" and target in df.columns) else None
            st.plotly_chart(px.scatter(df, x=cgcol, y=slcol, color=color_col,
                                       title="CGPA vs Durasi Tidur"),
                            use_container_width=True)
        else:
            st.info("Kolom CGPA atau Durasi Tidur tidak ditemukan.")

    with tabs[2]:
        if cgcol and target in df.columns:
            st.plotly_chart(px.box(df, x=target, y=cgcol, points="outliers",
                                   title=f"Sebaran CGPA per status {target}"),
                            use_container_width=True)
        else:
            st.info("Butuh kolom CGPA dan target.")

    with tabs[3]:
        # Sankey: Depression -> Treatment (jika keduanya ada)
        dep = safe_col(df, ["Depression","depression"])
        trt = safe_col(df, ["Did you seek treatment?","Treatment","treatment"])
        if dep and trt:
            a = normalize_yesno(df[dep])
            b = normalize_yesno(df[trt])
            temp = pd.DataFrame({"Depression": a, "Treatment": b}).dropna()
            lab = ["Dep=0","Dep=1","Treat=0","Treat=1"]
            # Build counts
            c00 = int(((temp["Depression"]==0)&(temp["Treatment"]==0)).sum())
            c01 = int(((temp["Depression"]==0)&(temp["Treatment"]==1)).sum())
            c10 = int(((temp["Depression"]==1)&(temp["Treatment"]==0)).sum())
            c11 = int(((temp["Depression"]==1)&(temp["Treatment"]==1)).sum())
            fig = px.sankey(
                node = dict(label=lab, pad=15, thickness=14),
                link = dict(
                    source=[0,0,1,1],  # Dep 0-> Treat 0/1; Dep 1-> Treat 0/1
                    target=[2,3,2,3],
                    value=[c00,c01,c10,c11]
                ),
                title="Alur Depression → Treatment"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Butuh kolom Depression dan Treatment untuk membuat Sankey.")

# ========== MODELING ==========
elif page == "🧮 Modeling":
    st.subheader("Pemodelan Klasifikasi 🧮")

    # cari semua kandidat biner Yes/No
    all_bool = []
    for c in df.columns:
        vals = pd.Series(df[c].dropna().astype(str).str.lower().unique())
        if 1 <= len(vals) <= 5 and set(vals).issubset({"yes","no","y","n","true","false","0","1"}):
            all_bool.append(c)
    if not all_bool:
        st.warning("Tidak ada kolom biner (Yes/No) untuk target.")
        st.stop()

    target = st.selectbox("Pilih kolom target (Yes/No):", options=all_bool)
    y = normalize_yesno(df[target])
    X = df.drop(columns=[target])

    # identifikasi tipe fitur
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include=["object","category"]).columns.tolist()

    # drop kolom kategorikal yang terlalu unik/ID-like
    too_many = [c for c in cat_cols if X[c].nunique() > max(50, 0.5*len(X))]
    if too_many:
        X = X.drop(columns=too_many)
        cat_cols = [c for c in cat_cols if c not in too_many]

    # buang baris target NaN
    mask = y.notna()
    X = X.loc[mask].copy(); y = y.loc[mask].astype(int)

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

    algo = st.selectbox("Algoritma", ["Logistic Regression","Random Forest"])
    if algo == "Logistic Regression":
        clf = LogisticRegression(max_iter=250)
    else:
        clf = RandomForestClassifier(n_estimators=400, random_state=42)

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])

    test_size = st.slider("Test size", 0.1, 0.5, 0.2, step=0.05)
    if st.button("Latih Model"):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)

        c1, c2 = st.columns(2)
        c1.metric("Akurasi (hold-out)", f"{acc*100:.1f}%")

        cm = confusion_matrix(y_test, pipe.predict(X_test))
        cm_df = pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"])
        c2.dataframe(cm_df, use_container_width=True)

        st.markdown("**Classification Report**")
        cr = classification_report(y_test, pipe.predict(X_test), output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(cr).T, use_container_width=True)

        # Feature importance untuk Random Forest
        if algo == "Random Forest":
            try:
                # Ambil nama fitur setelah one-hot
                ohe = pipe.named_steps["pre"].transformers_[1][1]
                cat_names = ohe.get_feature_names_out(cat_cols) if hasattr(ohe, "get_feature_names_out") else []
                feature_names = list(num_cols) + list(cat_names)
                importances = pipe.named_steps["clf"].feature_importances_
                fimp = (pd.DataFrame({"feature": feature_names, "importance": importances})
                        .sort_values("importance", ascending=False).head(20))
                st.plotly_chart(px.bar(fimp, x="importance", y="feature", orientation="h",
                                       title="Top Feature Importance (RF)"),
                                use_container_width=True)
            except Exception:
                pass

        st.markdown("---")
        st.markdown("### Prediksi Kasus Baru")
        with st.expander("Masukkan nilai prediktor"):
            inputs = {}
            for c in num_cols:
                v = float(np.nanmedian(X[c])) if c in X.columns else 0.0
                inputs[c] = st.number_input(f"{c}", value=v)
            for c in cat_cols:
                cats = ["(kosong)"] + sorted([str(x) for x in X[c].dropna().unique()]) if c in X.columns else ["(kosong)"]
                inputs[c] = st.selectbox(f"{c}", options=cats)

        if st.button("Prediksi"):
            row = {}
            for c in num_cols: row[c] = [inputs[c]]
            for c in cat_cols: row[c] = [None if inputs[c]=="(kosong)" else inputs[c]]
            newX = pd.DataFrame(row)
            prob = float(pipe.predict_proba(newX)[0,1])
            st.success(f"Probabilitas {target} = {prob*100:.1f}%")

        st.markdown("---")
        if st.button("Skor seluruh dataset & unduh"):
            proba_all = pipe.predict_proba(X)[:,1]
            out = df.loc[mask].copy()
            out[f"proba_{target}"] = proba_all
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("Download skor (CSV)", out.to_csv(index=False), "scored_dataset.csv")

    else:
        st.info("Atur parameter lalu klik **Latih Model**.")

# ========== INSIGHT & REPORT ==========
elif page == "📝 Insight & Laporan":
    st.subheader("Insight Otomatis & Unduhan 📝")
    notes = []

    # Prevalensi cepat
    for t in ["Depression","Anxiety","Panic attack"]:
        if t in df.columns:
            p = normalize_yesno(df[t]).mean()
            if pd.notna(p):
                notes.append(f"Perkiraan prevalensi **{t}**: **{p*100:.1f}%** dari sampel.")

    # Perbandingan CGPA dep vs non-dep
    cg = safe_col(df, ["CGPA","cgpa"])
    dep = safe_col(df, ["Depression","depression"])
    if cg and dep:
        dx = normalize_yesno(df[dep])
        mean_dep = df.loc[dx==1, cg].mean()
        mean_nodep = df.loc[dx==0, cg].mean()
        if pd.notna(mean_dep) and pd.notna(mean_nodep):
            notes.append(f"Rata-rata **CGPA** kelompok depression=1: **{mean_dep:.2f}**, depression=0: **{mean_nodep:.2f}**.")

    if notes:
        st.markdown("### Ringkasan Temuan")
        for n in notes:
            st.markdown(f"- {n}")
    else:
        st.info("Belum ada insight otomatis untuk struktur data saat ini.")

    st.markdown("---")
    st.markdown("### Unduhan")
    st.download_button("Download Data (CSV)", df.to_csv(index=False), "dataset_mental_health.csv")
    md = "# Laporan Singkat – Mental Health of Students (Kelompok Bjorkanism)\n\n"
    if notes:
        md += "\n".join([f"- {n}" for n in notes])
    else:
        md += "(Tidak ada insight otomatis)."
    st.download_button("Download Ringkasan (Markdown)", md, "laporan_ringkas.md")

# ========== ABOUT ==========
else:
    st.subheader("Tentang Aplikasi ℹ️")
    st.write("""
Aplikasi satu-berkas ini meniru alur eksplorasi ala demo-flight namun untuk **kesehatan mental mahasiswa**:
- **Explorer** untuk filter cepat & unduh data terfilter  
- **Dashboard** untuk prevalensi target (Yes/No), korelasi numerik, hubungan CGPA–tidur, boxplot per status  
- **Modeling** dengan pra-proses otomatis (StandardScaler + OneHot) dan dua algoritma (LogReg & Random Forest)  
- **Insight & Laporan** untuk ringkasan temuan dan export

Silakan kustomisasi warna, logo institusi, atau tambah halaman kuesioner sesuai kebutuhan.
""")
