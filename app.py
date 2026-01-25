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

# --- 強力データ取得関数 (時間足対応版) ---
def get_forex_data_robust(interval="1h", period="1mo"):
    tickers_to_try = ["USDJPY=X", "JPY=X"]
    for ticker in tickers_to_try:
        try:
            # yfinanceでデータ取得
            temp_df = yf.download(ticker, period=period, interval=interval, progress=False)
            if not temp_df.empty and len(temp_df) > 20:
                return temp_df
        except:
            pass
    return pd.DataFrame()

# --- 乖離判定付き確率計算（時間足別感度調整） ---
def calculate_reversion_probability(current_price, predicted_price, lower_bound, upper_bound, min_width=0.10):
    c = to_float(current_price)
    p = to_float(predicted_price)
    l = to_float(lower_bound)
    u = to_float(upper_bound)
    
    # 予測幅の最低値を設定（時間足によって感度を変えるため引数化）
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
        minor_correction = dist_from_center * -5.0
        correction += minor_correction

    final_prob = base_prob + correction
    final_prob = max(10.0, min(90.0, final_prob))
    
    return final_prob, note

# --- メイン処理 ---
st.markdown("### **🇺🇸🇯🇵 ドル円AI短期予測 (マルチタイムフレーム)**")

# === 時間足選択 ===
timeframe = st.radio(
    "⏱️ 時間足を選択してください",
    ["1時間足 (1H)", "15分足 (15m)", "5分足 (5m)"],
    horizontal=True
)

st.markdown("""
<div style="margin-top: 5px; margin-bottom: 20px;">
    <span style="font-size: 0.7rem; opacity: 0.8;">※黄色い帯（AI予測）と紫色の帯（ボリンジャーバンド）の重なりでトレンドを判断します。</span>
</div>
""", unsafe_allow_html=True)

# 設定値の決定
if timeframe == "1時間足 (1H)":
    api_interval = "1h"
    api_period = "1mo"
    min_width_setting = 0.10  # 1時間足はノイズが大きいので10銭幅を見る
    # 予測ターゲット: 時間単位
    target_configs = [
        (1, "1H後"), (2, "2H後"), (4, "4H後"), (8, "8H後"), (12, "12H後")
    ]
    time_unit = "hours"
    
elif timeframe == "15分足 (15m)":
    api_interval = "15m"
    api_period = "1mo" # 15分足は1ヶ月分取得可能
    min_width_setting = 0.05  # 15分足は少し敏感に（5銭幅）
    # 予測ターゲット: 分単位
    target_configs = [
        (15, "15分後"), (30, "30分後"), (60, "1H後"), (120, "2H後"), (240, "4H後")
    ]
    time_unit = "minutes"
    
else: # 5分足
    api_interval = "5m"
    api_period = "5d"  # 5分足で1ヶ月は重すぎるため直近5日
    min_width_setting = 0.03  # 5分足はかなり敏感に（3銭幅）
    # 予測ターゲット: 分単位
    target_configs = [
        (5, "5分後"), (15, "15分後"), (30, "30分後"), (60, "1H後"), (120, "2H後")
    ]
    time_unit = "minutes"


try:
    with st.spinner(f'{timeframe} データ取得＆AI解析中...'):
        df = get_forex_data_robust(interval=api_interval, period=api_period)

    if df.empty:
        st.error("⚠️ データが取得できませんでした。時間をおいて再接続してください。")
        st.stop()

    # --- データ整形 ---
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    cols = {c.lower(): c for c in df.columns}
    date_c = next((c for k, c in cols.items() if 'date' in k or 'time' in k), df.columns[0])
    close_c = next((c for k, c in cols.items() if 'close' in k), df.columns[1])

    try:
        df[date_c] = pd.to_datetime(df[date_c]).dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
    except:
        df[date_c] = pd.to_datetime(df[date_c])

    # テクニカル計算
    df['SMA20'] = df[close_c].rolling(window=20).mean()
    df['STD'] = df[close_c].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (df['STD'] * 2)
    df['BB_Lower'] = df['SMA20'] - (df['STD'] * 2)

    df_p = pd.DataFrame()
    df_p['ds'] = df[date_c]
    df_p['y'] = df[close_c]
    
    current_price = to_float(df_p['y'].iloc[-1])
    last_date = df_p['ds'].iloc[-1]

    st.write(f"**現在値 ({timeframe}): {current_price:,.2f} 円**")
    st.write(f"<span style='font-size:0.8rem; color:#aaa'>基準日時: {last_date.strftime('%m/%d %H:%M')}</span>", unsafe_allow_html=True)

    # --- Prophet予測 ---
    # 5分足などは周期性が日次と合わないことがあるため微調整
    m = Prophet(
        changepoint_prior_scale=0.15, 
        daily_seasonality=True if api_interval == "1h" else False, # 短期足は日次より細かいトレンド重視
        weekly_seasonality=True, 
        yearly_seasonality=False
    )
    if api_interval in ["5m", "15m"]:
        m.add_seasonality(name='hourly', period=1/24, fourier_order=5) # 短期足用の周期追加

    m.fit(df_p)
    
    # 将来枠の作成（ターゲットの最大時間までカバーするようにperiodsを設定）
    # 5分足で2時間後(120分)まで見るなら、120/5 = 24 periods必要
    freq_str = 'h' if api_interval == '1h' else ('15min' if api_interval == '15m' else '5min')
    periods_needed = 30 # 少し多めに確保
    
    future = m.make_future_dataframe(periods=periods_needed, freq=freq_str)
    forecast = m.predict(future)

    # --- 予測結果抽出 ---
    st.markdown("#### **📈 短期予測 (上昇 vs 下落)**")
    
    probs_up = []
    probs_down = []
    labels = []
    prices = []
    notes = []
    colors_up = []
    colors_down = []

    for val, label_text in target_configs:
        # ターゲット時間の計算
        if time_unit == "hours":
            target_time = last_date + timedelta(hours=val)
        else:
            target_time = last_date + timedelta(minutes=val)
            
        # 最も近い予測ポイントを探す
        row = forecast.iloc[(forecast['ds'] - target_time).abs().argsort()[:1]].iloc[0]
        
        pred = to_float(row['yhat'])
        
        prob_up, note = calculate_reversion_probability(
            current_price, pred, 
            to_float(row['yhat_lower']), 
            to_float(row['yhat_upper']),
            min_width=min_width_setting
        )
        prob_down = 100.0 - prob_up
        
        price_diff = abs(pred - current_price)
        # 誤差範囲の判定（足によって許容誤差を変える）
        threshold = 0.15 if api_interval == "1h" else (0.08 if api_interval == "15m" else 0.05)
        
        if price_diff < threshold:
            c_up = '#808080'
            c_down = '#808080'
            note = f"誤差範囲 (変動幅 {price_diff:.2f}円)"
        else:
            c_up = '#00cc96'
            c_down = '#ff4b4b'
        
        probs_up.append(prob_up)
        probs_down.append(prob_down)
        labels.append(label_text)
        prices.append(pred)
        notes.append(note)
        colors_up.append(c_up)
        colors_down.append(c_down)

    # --- 棒グラフ ---
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=labels, y=probs_up, name='上昇確率', text=[f"{p:.1f}%" for p in probs_up], textposition='auto', marker_color=colors_up))
    fig_bar.add_trace(go.Bar(x=labels, y=probs_down, name='下落確率', text=[f"{p:.1f}%" for p in probs_down], textposition='auto', marker_color=colors_down))
    
    fig_bar.update_layout(
        template="plotly_dark",
        height=250,
        margin=dict(l=0, r=0, t=30, b=20),
        yaxis=dict(range=[0, 100], title="確率 (%)"),
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_bar, use_container_width=True, config={'staticPlot': True})

    # 詳細数値
    st.markdown("#### **詳細数値 & AI判断**")
    detail_data = {
        "時間": labels,
        "予測レート": [f"{p:.2f} 円" for p in prices],
        "上昇確率": [f"{p:.1f} %" for p in probs_up],
        "下落確率": [f"{p:.1f} %" for p in probs_down],
        "判定/状況": notes
    }
    st.dataframe(pd.DataFrame(detail_data), hide_index=True, use_container_width=True)

    # --- チャート表示 (色改良版・8円幅) ---
    st.markdown("#### **推移・AI軌道・テクニカル指標**")
    
    fig_chart = go.Figure()

    # 1. ボリンジャーバンド
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['BB_Upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig_chart.add_trace(go.Scatter(
        x=df[date_c], y=df['BB_Lower'], mode='lines', line=dict(width=0),
        fill='tonexty', 
        fillcolor='rgba(138, 43, 226, 0.3)', 
        name='BB(±2σ)', hoverinfo='skip'
    ))

    # 2. 実測ローソク足
    fig_chart.add_trace(go.Candlestick(
        x=df[date_c],
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='実測',
        increasing=dict(line=dict(color='#00cc96', width=1), fillcolor='rgba(0,0,0,0)'),
        decreasing=dict(line=dict(color='#ff4b4b', width=1), fillcolor='rgba(0,0,0,0)')
    ))

    # 3. SMA
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['SMA20'], mode='lines', name='20SMA', line=dict(color='cyan', width=1.5)))
    
    # 4. AI予測範囲
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig_chart.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0),
        fill='tonexty', 
        fillcolor='rgba(255, 255, 0, 0.4)', 
        hoverinfo='skip', showlegend=False, name='AI予測範囲'
    ))

    # 5. AI予測線
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='AI軌道', line=dict(color='yellow', width=2)))

    # 表示範囲計算
    # 時間軸：データ開始〜予測終了まで
    x_max = forecast['ds'].max()
    x_min = df[date_c].min() # 取得データ全体を表示
    
    # 価格軸：上下8円幅（±4円）で固定
    y_range_min = current_price - 4.0
    y_range_max = current_price + 4.0

    fig_chart.update_layout(
        template="plotly_dark",
        height=600,
        plot_bgcolor='#000000',
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(
            range=[x_min, x_max],
            type="date",
            fixedrange=False,  # ズーム可能にする
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            range=[y_range_min, y_range_max],
            fixedrange=True
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig_chart, use_container_width=True, config={'displayModeBar': False, 'staticPlot': False})

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
