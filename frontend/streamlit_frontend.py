
import streamlit as st
import pandas as pd, os, requests
from datetime import datetime
from lifelines import KaplanMeierFitter

API_URL = os.getenv("API_URL","http://localhost:8000")
API_TOKEN = os.getenv("API_TOKEN","")
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

st.set_page_config(page_title="Mortality Dashboard", layout="wide")
st.title("Mortality Dashboard")

uploaded = st.file_uploader("Upload CSV/XLSX", type=["csv","xlsx"])
if uploaded:
    if uploaded.name.endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
else:
    try:
        r = requests.get(f"{API_URL}/patients/", headers=HEADERS, timeout=5)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
    except Exception:
        df = pd.read_csv("../sample_data.csv")

st.dataframe(df.head(200))
deaths = int((df['outcome'].astype(str) == 'Dead').sum()) if 'outcome' in df else 0
cases = int((df['outcome'].astype(str).isin(['Dead','Survived'])).sum()) if 'outcome' in df else len(df)
st.metric("Cases", cases)
st.metric("Deaths", deaths)
st.metric("CFR (%)", f"{(deaths/cases*100 if cases>0 else 0):.2f}%")
if 'followup_days' in df.columns:
    T = df['followup_days'].fillna(0)
    E = (df['outcome'].astype(str) == 'Dead').astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=E, label='KM')
    ax = kmf.plot_survival_function()
    st.pyplot(ax.get_figure())
