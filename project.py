import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, mean_squared_error, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram

# --- Styling tambahan ---
st.markdown(
    """
    <style>
    .main {
        background-color: #F8F9FA;
    }
    .title {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #333;
    }
    /* Ubah warna background sidebar */
    section[data-testid="stSidebar"] {
        background-color: #fce4ec;  /* Pink Soft */
    }
    /* Header pertama */
    .sidebar-header-1 {
        color: purple;
        font-size: 20px;
        margin-bottom: 15px;
    }
    /* Header kedua dengan garis atas */
    .sidebar-header-2 {
        color: purple;
        font-size: 20px;
        margin-top: 25px;
        padding-top: 10px;
        border-top: 2px solid #ba68c8;  /* Ungu pastel */
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Judul Utama ---
st.markdown('<div class="title">📊 Project Data Science : Classification, Regression, & Clustering</div>', unsafe_allow_html=True)
st.write("Selamat datang! Pilih dataset dan metode analisis pada sidebar untuk mendapatkan insight lebih mendalam.")

# --- Sidebar: Upload & Navigasi ---
st.sidebar.markdown('<div class="sidebar-header-1">📁 Upload & Pengaturan Dataset</div>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.sidebar.success("Dataset berhasil diupload!")
else:
    st.sidebar.warning("Silakan upload file CSV Anda.")

# --- Sidebar: Tipe Analisis ---
st.sidebar.markdown('<div class="sidebar-header-2">⚙️ Pilih Tipe Analisis</div>', unsafe_allow_html=True)
analysis_type = st.sidebar.selectbox("Tipe Analisis", ["Classification", "Regression", "Clustering"])

if uploaded_file:
    # Tampilkan preview dataset
    st.subheader("📄 Preview Data")
    st.dataframe(df.head())

    # EDA dasar
    with st.container():
        st.markdown("### Exploratory Data Analysis (EDA)")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Deskripsi Data**")
            st.write(df.describe())
        with col2:
            st.write("**Tipe Data**")
            st.write(df.dtypes)
        if st.checkbox("Tampilkan Missing Value"):
            st.write(df.isnull().sum())

    # Encoding data kategorikal
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in df_encoded.select_dtypes(include='object').columns:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    # --- Classification & Regression ---
    if analysis_type in ["Classification", "Regression"]:
        st.markdown("### Pilih Target dan Fitur")
        target = st.selectbox("Pilih Kolom Target", df_encoded.columns)
        features = st.multiselect("Pilih Fitur (X)", df_encoded.columns.drop(target))

        if features:
            X = df_encoded[features]
            y = df_encoded[target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Model selection
            if analysis_type == "Classification":
                model_choice = st.sidebar.selectbox("Pilih Model Classification", 
                                                    ["Logistic Regression", "SVM", "Decision Tree", "Random Forest", "KNN", "Naive Bayes"])
                if model_choice == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_choice == "SVM":
                    model = SVC()
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier()
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier()
                elif model_choice == "KNN":
                    model = KNeighborsClassifier()
                elif model_choice == "Naive Bayes":
                    model = GaussianNB()
            else:
                model_choice = st.sidebar.selectbox("Pilih Model Regression", ["Linear Regression", "SVR"])
                if model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "SVR":
                    model = SVR()

            if st.button("Latih Model"):
                model.fit(X_train, y_train)
                if analysis_type == "Classification":
                    y_pred = model.predict(X_test)
                    st.markdown("#### Classification Report")
                    st.text(classification_report(y_test, y_pred))
                else:
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test, y_pred)
                    st.markdown("#### Evaluasi Regresi")
                    st.write(f"Mean Squared Error: **{mse:.2f}**")

                # Input manual prediksi
                st.markdown("#### Prediksi Berdasarkan Input")
                input_data = {}
                for col in features:
                    input_val = st.number_input(f"Masukkan nilai untuk '{col}'", value=float(X[col].mean()))
                    input_data[col] = input_val
                input_df = pd.DataFrame([input_data])
                pred = model.predict(input_df)[0]
                st.success(f"Hasil Prediksi: **{pred}**")

    # --- Clustering ---
    elif analysis_type == "Clustering":
        st.markdown("### Pilih Fitur untuk Clustering")
        cluster_features = st.multiselect("Pilih fitur numerik untuk Clustering", 
                                          df_encoded.select_dtypes(include=['int64', 'float64']).columns)
        if len(cluster_features) >= 2:
            X_cluster = df_encoded[cluster_features]
            clustering_method = st.sidebar.selectbox("Pilih Metode Clustering", ["K-Means", "Hierarchical", "DBSCAN"])

            if clustering_method == "K-Means":
                k = st.sidebar.slider("Jumlah Klaster (K)", 2, 10, 3)
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(X_cluster)
                score = silhouette_score(X_cluster, labels)
                st.write(f"**Silhouette Score:** {score:.2f}")

                X_cluster["Cluster"] = labels
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=X_cluster[cluster_features[0]], y=X_cluster[cluster_features[1]], hue=labels, palette="Set2", ax=ax)
                ax.set_title("Hasil Clustering - KMeans")
                st.pyplot(fig)

            elif clustering_method == "Hierarchical":
                method = st.sidebar.selectbox("Metode Linkage", ["ward", "single", "complete", "average"])
                Z = linkage(X_cluster, method=method)
                fig, ax = plt.subplots(figsize=(10, 5))
                dendrogram(Z, ax=ax)
                ax.set_title("Dendrogram - Hierarchical Clustering")
                st.pyplot(fig)

            elif clustering_method == "DBSCAN":
                eps = st.sidebar.slider("Epsilon", 0.1, 10.0, 0.5)
                min_samples = st.sidebar.slider("Min Samples", 1, 10, 3)
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                labels = dbscan.fit_predict(X_cluster)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                st.write(f"**Jumlah Klaster:** {n_clusters}")

                fig, ax = plt.subplots(figsize=(8, 6))
                sns.scatterplot(x=X_cluster[cluster_features[0]], y=X_cluster[cluster_features[1]], hue=labels, palette="Set1", ax=ax)
                ax.set_title("Hasil Clustering - DBSCAN")
                st.pyplot(fig)

        else:
            st.info("Pilih minimal 2 fitur numerik untuk clustering.")
