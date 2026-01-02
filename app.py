import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
import numpy as np # ダミーデータ作成用に追加
from scipy.stats import norm
import plotly.graph_objs as go
from datetime import timedelta, datetime

# ==========================================
#  設定：パスワード
# ==========================================
# ★ここがパスワードの設定です。今は "demo" になっています。
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
    h1, h2, h3, h4, h5, h6, p, div, span, label, li {
        color: #ffffff !important;
        font-family: sans-serif;
    }
    .stTextInput > div > div > input {
        color: #ffffff !important;
        background-color: #333333;
        font-weight: bold;
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

# --- 確率計算 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    c, p, l, u = to_float(current_price), to_float(predicted_price), to_float(lower_bound), to_float(upper_bound)
    sigma = (u - l) / 2.56 
    if sigma == 0: return 50.0
    z_score = (p - c) / sigma
    return norm.cdf(z_score) * 100

# --- ダミーデータ生成関数 (エラー回避用) ---
def create_dummy_data():
    # 過去7日分(168時間)のデータを適当に作成
    dates = pd.date_range(end=datetime.now(), periods=168, freq='h')
    base_price = 150.00
    # ランダムウォークさせる
    np.random.seed(42)
    changes = np.random.randn(168) * 0.1
    prices = base_price + np.cumsum(changes)
    
    df = pd.DataFrame(index=dates)
    df['Close'] = prices
    df['Open'] = prices + np.random.randn(168) * 0.05
    df['High'] = prices + 0.1
    df['Low'] = prices - 0.1
    df.index.name = 'Date'
    return df

# --- メイン処理 ---
st.markdown("### **🇺🇸🇯🇵 ドル円AI短期予測 (1時間足)**")
st.markdown("""
<div style="margin-top: -10px; margin-bottom: 20px;">
    <span style="font-size: 0.7rem; opacity: 0.8;">※黄色い帯の中にローソク足があれば「予測通り」、飛び出していれば「予測外」の動きです。</span>
</div>
""", unsafe_allow_html=True)

ticker = "USDJPY=X"
is_dummy = False # ダミーデータかどうかのフラグ

try:
    with st.spinner('データ取得中...'):
        # 1. まずリアルデータを試す
        try:
            df = yf.download(ticker, period="7d", interval="1h", progress=False)
        except:
            df = pd.DataFrame() # 失敗したら空にする

    # 2. リアルデータが空っぽなら、ダミーデータを作る (安全装置)
    if df.empty:
        is_dummy = True
        df = create_dummy_data()
        st.warning("⚠️ リアルタイムデータの取得に失敗しました。サーバー見本用の**サンプルデータ**を表示しています。")

    # --- データ整形 ---
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    cols = {c.lower(): c for c in df.columns}
    date_c = next((c for k, c in cols.items() if 'date' in k or 'time' in k), df.columns[0])
    close_c = next((c for k, c in cols.items() if 'close' in k), df.columns[1])

    # タイムゾーン処理 (ダミーの場合はスキップ)
    if not is_dummy:
        try:
            df[date_c] = pd.to_datetime(df[date_c]).dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
        except:
            df[date_c] = pd.to_datetime(df[date_c])
    else:
        df[date_c] = pd.to_datetime(df[date_c])

    df_p = pd.DataFrame()
    df_p['ds'] = df[date_c]
    df_p['y'] = df[close_c]
    
    current_price = to_float(df_p['y'].iloc[-1])
    last_date = df_p['ds'].iloc[-1]

    st.write(f"**現在値: {current_price:,.2f} 円**")
    st.write(f"<span style='font-size:0.8rem; color:#aaa'>基準日時: {last_date.strftime('%m/%d %H:%M')}</span>", unsafe_allow_html=True)

    # --- Prophetによる予測 ---
    m = Prophet(changepoint_prior_scale=0.1, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(df_p)
    
    future = m.make_future_dataframe(periods=13, freq='h')
    forecast = m.predict(future)

    # --- 予測結果の抽出 ---
    st.markdown("#### **📈 短期予測 (上昇確率)**")
    
    targets = [1, 2, 4, 8, 12]
    probs = []
    labels = []
    prices = []

    for i, h in enumerate(targets):
        target_time = last_date + timedelta(hours=h)
        row = forecast.iloc[(forecast['ds'] - target_time).abs().argsort()[:1]].iloc[0]
        
        pred = to_float(row['yhat'])
        prob = calculate_probability(current_price, pred, to_float(row['yhat_lower']), to_float(row['yhat_upper']))
        
        probs.append(prob)
        labels.append(f"{h}H後")
        prices.append(pred)

    # --- 棒グラフ ---
    bar_colors = ['#ff4b4b' if p < 50 else '#00cc96' for p in probs]

    fig_bar = go.Figure(data=[go.Bar(
        x=labels,
        y=probs,
        text=[f"{p:.1f}%" for p in probs],
        textposition='auto',
        marker_color=bar_colors
    )])
    
    fig_bar.update_layout(
        template="plotly_dark",
        height=200,
        margin=dict(l=0, r=0, t=20, b=20),
        yaxis=dict(range=[0, 100], title="上昇確率 (%)"),
        showlegend=False
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'staticPlot': True})

    # 詳細数値
    st.markdown("#### **詳細数値**")
    detail_data = {
        "時間": labels,
        "予測レート": [f"{p:.2f} 円" for p in prices],
        "上昇確率": [f"{p:.1f} %" for p in probs]
    }
    st.dataframe(pd.DataFrame(detail_data), hide_index=True, use_container_width=True)

    # --- 過去1週間のチャート ---
    st.markdown("#### **過去1週間の推移とAIの軌道**")
    
    fig_chart = go.Figure()

    # 1. 実測ローソク足
    fig_chart.add_trace(go.Candlestick(
        x=df[date_c],
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='実測'
    ))
    
    # 2. 黄色い帯（予測範囲）
    fig_chart.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False
    ))
    fig_chart.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 255, 0, 0.15)',
        hoverinfo='skip', showlegend=False, name='予測範囲'
    ))

    # 3. 黄色い線（AIの中心予測）
    fig_chart.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='AI軌道', line=dict(color='yellow', width=2)
    ))

    # X軸範囲固定
    x_min = df[date_c].min()
    x_max = forecast['ds'].max()

    fig_chart.update_layout(
        template="plotly_dark",
        height=400,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            range=[x_min, x_max],
            type="date",
            fixedrange=True, 
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            fixedrange=True
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig_chart, use_container_width=True, config={'displayModeBar': False, 'staticPlot': False})

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
