# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from lifelines import KaplanMeierFitter
import io

st.set_page_config(page_title="Mortality / Diagnosis Analysis", layout="wide")
st.title("📈 Phân tích tỷ lệ tử vong / chẩn đoán (Upload + Nhập tay + ML)")

# ---------- Helpers ----------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def preprocess_df(df):
    df = df.copy()
    # lowercase column names
    df.columns = [c.strip() for c in df.columns]
    # If diagnosis present (M/B), map to target 'target'
    if 'diagnosis' in df.columns:
        df['target'] = df['diagnosis'].map({'M':1, 'B':0})
    # drop unnamed all-nan
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def get_numeric_features(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

# ---------- Sidebar: Upload or Sample ----------
st.sidebar.header("1. Dữ liệu")
uploaded = st.sidebar.file_uploader("Tải file CSV lên (hoặc dùng sample)", type=["csv","xlsx"])
use_sample = False
df = None

if uploaded:
    try:
        if uploaded.name.endswith('.csv'):
            df = load_csv(uploaded)
        else:
            df = pd.read_excel(uploaded)
        st.sidebar.success(f"Đã tải: {uploaded.name}")
    except Exception as e:
        st.sidebar.error(f"Lỗi đọc file: {e}")
else:
    # try to load local data.csv if exists
    try:
        df = load_csv("data.csv")
        st.sidebar.info("Đang dùng data.csv trong thư mục (mẫu).")
        use_sample = True
    except Exception:
        st.sidebar.warning("Không có file tải lên và không tìm thấy data.csv. Bạn có thể upload file.")

if df is None:
    st.info("Hãy tải file CSV vào sidebar hoặc đặt file 'data.csv' vào thư mục project.")
    st.stop()

df = preprocess_df(df)

# ---------- Sidebar: Manual entry ----------
st.sidebar.header("2. Nhập thủ công")
with st.sidebar.expander("➕ Thêm mẫu (nhập tay)"):
    # show some numeric columns to optionally fill
    numeric_columns = get_numeric_features(df)
    # choose a small subset for manual entry
    sample_inputs = {}
    col1, col2 = st.columns(2)
    sample_inputs['id'] = st.text_input("id (tùy chọn)", value="")
    sample_inputs['diagnosis'] = st.selectbox("diagnosis", options=['M','B'])
    # let user fill first 3 numeric features if present
    shown_numeric = numeric_columns[:3]
    for c in shown_numeric:
        sample_inputs[c] = st.number_input(c, value=float(df[c].dropna().iloc[0]) if not df[c].dropna().empty else 0.0)
    if st.button("Thêm bản ghi thủ công"):
        # build row
        row = {}
        for k,v in sample_inputs.items():
            row[k] = v
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df = preprocess_df(df)
        st.success("Đã thêm bản ghi vào DataFrame (tạm thời trên session).")

# ---------- Filtering and preview ----------
st.subheader("Dữ liệu (preview)")
st.dataframe(df.head(200))

st.markdown(f"**Số bản ghi:** {len(df)}  \n**Cột:** {', '.join(df.columns[:30])}")

# ---------- Target / Label detection ----------
if 'target' not in df.columns:
    # try to find common label names
    possible = [c for c in df.columns if c.lower() in ('survived','survival','outcome','death','died')]
    if possible:
        label = possible[0]
        st.warning(f"Không thấy cột 'diagnosis' — phát hiện cột nhãn tiềm năng: {label}. Bạn có muốn map cột này thành target?")
        if st.button(f"Map {label} -> target (1=adverse)"):
            df['target'] = df[label].astype(str).map({'1':1,'0':0,'Yes':1,'No':0,'Alive':0,'Dead':1,'Died':1})
    else:
        st.info("Nếu file của bạn không có cột 'diagnosis', bạn có thể chọn cột nhãn thủ công trong phần ML bên dưới.")

# ---------- Quick counts & charts ----------
st.subheader("Phân bố nhãn")
if 'target' in df.columns:
    counts = df['target'].value_counts(dropna=False)
    fig1, ax1 = plt.subplots()
    labels = ['Adverse (1)','Non-adverse (0)']
    vals = [counts.get(1,0), counts.get(0,0)]
    ax1.pie(vals, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.set_title("Tỷ lệ Adverse vs Non-adverse")
    st.pyplot(fig1)

    st.write("Bảng tần suất:")
    st.write(pd.DataFrame({'value':vals}, index=labels))
else:
    st.warning("Chưa xác định biến mục tiêu (target). Nếu bạn có cột 'diagnosis' (M/B) hoặc 'survived' hãy map nó thành target.")

# ---------- CFR-like analysis (using diagnosis as proxy) ----------
st.subheader("Biểu đồ theo diagnosis (M/B)")
if 'diagnosis' in df.columns:
    fig2, ax2 = plt.subplots(figsize=(8,4))
    sns.countplot(x='diagnosis', data=df, order=df['diagnosis'].value_counts().index, ax=ax2)
    ax2.set_title("Số lượng theo diagnosis (M/B)")
    st.pyplot(fig2)

# ---------- Kaplan-Meier if followup_days present ----------
if 'followup_days' in df.columns and 'target' in df.columns:
    st.subheader("Phân tích sống sót — Kaplan-Meier (nếu có followup_days)")
    try:
        T = df['followup_days'].fillna(0)
        E = df['target'].astype(int)
        kmf = KaplanMeierFitter()
        kmf.fit(T, event_observed=E, label='Survival')
        ax = kmf.plot_survival_function()
        st.pyplot(ax.get_figure())
    except Exception as e:
        st.warning("Lỗi khi chạy Kaplan-Meier: " + str(e))

# ---------- Machine Learning ----------
st.subheader("Machine Learning: Dự đoán target")

# choose features
numeric_feats = get_numeric_features(df)
st.markdown(f"**Cột số (numeric) phát hiện:** {', '.join(numeric_feats[:20])}")
feat_choice = st.multiselect("Chọn đặc trưng để huấn luyện (nên là numeric)", numeric_feats, default=numeric_feats[:8])

label_choice = None
if 'target' in df.columns:
    label_choice = 'target'
else:
    # allow manual selection of label
    col_for_label = st.selectbox("Chọn cột làm label nếu chưa có 'target'", options=['(none)'] + list(df.columns))
    if col_for_label != '(none)':
        # try to map common formats
        df['target'] = df[col_for_label].map({'M':1,'B':0,'Yes':1,'No':0,'1':1,'0':0}).astype(float)
        label_choice = 'target'

if label_choice is None:
    st.info("Cần một cột label để huấn luyện mô hình.")
else:
    X = df[feat_choice].copy()
    y = df[label_choice].astype(float).fillna(0)
    # drop NA rows
    mask = X.notna().all(axis=1) & y.notna()
    X = X[mask]
    y = y[mask]
    st.write(f"Chuẩn bị dữ liệu: {X.shape[0]} bản ghi, {X.shape[1]} đặc trưng")
    if X.shape[0] < 10:
        st.warning("Số lượng mẫu ít để huấn luyện đáng tin cậy.")
    else:
        test_size = st.slider("Tỉ lệ test set", 0.1, 0.5, 0.25)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        model_name = st.selectbox("Chọn mô hình", ["Random Forest", "Logistic Regression"])
        if st.button("Huấn luyện & Đánh giá"):
            if model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=200, random_state=42)
            else:
                model = LogisticRegression(max_iter=2000)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Accuracy: {acc:.3f}")
            if y_prob is not None and len(np.unique(y_test))>1:
                try:
                    roc = roc_auc_score(y_test, y_prob)
                    st.write(f"ROC AUC: {roc:.3f}")
                except Exception:
                    pass
            st.write("Classification report:")
            st.text(classification_report(y_test, y_pred))
            # show feature importances for RF
            if model_name == "Random Forest":
                importances = model.feature_importances_
                fi = pd.Series(importances, index=feat_choice).sort_values(ascending=False).head(20)
                st.bar_chart(fi)
            # confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            st.pyplot(fig_cm)

# ---------- Download processed dataset ----------
st.subheader("Tải dữ liệu đã xử lý")
buf = io.BytesIO()
df.to_csv(buf, index=False)
st.download_button("⬇️ Tải CSV đã xử lý", data=buf.getvalue(), file_name="processed_data.csv", mime="text/csv")
