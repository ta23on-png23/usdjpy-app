import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from scipy.stats import norm
import plotly.graph_objs as go
from datetime import timedelta, datetime

# ==========================================
#  設定：パスワード
# ==========================================
DEMO_PASSWORD = "demo" 

# --- ページ設定 ---
st.set_page_config(page_title="ドル円AI短期予測", layout="wide")

# --- UI非表示 & 黒背景デザイン (CSS) ---
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    div[data-testid="stToolbar"] {visibility: hidden;}
    .stDeployButton {display:none;}
    
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6, p, div, span, label, li, .stMarkdown {
        color: #ffffff !important;
        font-family: sans-serif;
    }
    .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: #333333;
        font-weight: bold;
    }
    .stRadio > div {
        background-color: #333333;
        padding: 10px;
        border-radius: 10px;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
        padding-left: 0.5rem;
        padding-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- パスワード認証 ---
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False
    if st.session_state.password_correct:
        return True
    
    st.markdown("### 🔒 ドル円予測ツール (デモ版)")
    password = st.text_input("パスワード", type="password")
    if password == DEMO_PASSWORD:
        st.session_state.password_correct = True
        st.rerun()
    elif password:
        st.error("パスワードが違います")
    return False

if not check_password():
    st.stop()

# --- 数値変換 ---
def to_float(x):
    try:
        if isinstance(x, float): return x
        if isinstance(x, (pd.Series, pd.DataFrame)): return float(x.iloc[0]) if not x.empty else 0.0
        if hasattr(x, 'item'): return float(x.item())
        if isinstance(x, list): return float(x[0])
        return float(x)
    except: return 0.0

# --- 強力データ取得関数 ---
def get_forex_data_robust(interval="1h", period="1mo"):
    tickers_to_try = ["USDJPY=X", "JPY=X"]
    for ticker in tickers_to_try:
        try:
            temp_df = yf.download(ticker, period=period, interval=interval, progress=False)
            if not temp_df.empty and len(temp_df) > 20:
                return temp_df
        except:
            pass
    return pd.DataFrame()

# --- 乖離判定付き確率計算（トレンドフィルター強化版） ---
def calculate_reversion_probability(current_price, predicted_price, lower_bound, upper_bound, min_width=0.10, trend_direction=0):
    c = to_float(current_price)
    p = to_float(predicted_price)
    l = to_float(lower_bound)
    u = to_float(upper_bound)
    
    # 予測幅の最低値を設定
    width = u - l
    adjusted_width = max(width, min_width)
    sigma = adjusted_width / 2.0 

    if sigma == 0:
        base_prob = 50.0
    else:
        z_score = (p - c) / sigma
        damped_z = z_score * 0.5
        base_prob = norm.cdf(damped_z) * 100

    correction = 0.0
    note = "順張り(トレンド追随)"
    
    box_width = u - l
    if box_width < 0.01: box_width = 0.01

    # --- バンド乖離補正 ---
    if c > u: 
        excess = c - u
        ratio = excess / box_width
        correction = - (ratio * 20.0)
        correction = max(correction, -15.0)
        note = f"⚠️上値重め (調整警戒 {correction:.1f}%)"
    elif c < l: 
        excess = l - c
        ratio = excess / box_width
        correction = + (ratio * 20.0)
        correction = min(correction, 15.0)
        note = f"⚠️底堅い (反発期待 +{correction:.1f}%)"
    else: 
        center = (u + l) / 2
        dist_from_center = (c - center) / (box_width / 2) if box_width > 0 else 0
        correction += dist_from_center * -5.0

    # --- ★長期トレンドフィルター（矛盾解消ロジック） ---
    # trend_direction: 1 (長期上昇中), -1 (長期下落中), 0 (なし)
    if p < c and trend_direction == 1:
        penalty = 20.0 
        base_prob += penalty 
        note = "長期上昇中のため下値限定的"
    
    elif p > c and trend_direction == -1:
        penalty = 20.0
        base_prob -= penalty 
        note = "長期下落中のため上値限定的"

    final_prob = base_prob + correction
    final_prob = max(15.0, min(85.0, final_prob)) 
    
    return final_prob, note

# --- メイン処理 ---
st.markdown("### **🇺🇸🇯🇵 ドル円AI短期予測 (マルチタイムフレーム・トレンド補正版)**")

# === 時間足選択 ===
timeframe = st.radio(
    "⏱
