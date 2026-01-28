import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from scipy.stats import norm
import plotly.graph_objs as go
from datetime import timedelta, datetime
import pytz
import requests
import json
from streamlit_autorefresh import st_autorefresh

# ==========================================
#  設定：パスワード & LINE連携
# ==========================================
DEMO_PASSWORD = "demo" 

# ★ここにLINE Developersで取得した情報を入力してください★
LINE_CHANNEL_ACCESS_TOKEN = "ここにアクセストークンを貼り付け" 
LINE_USER_ID = "ここにあなたのユーザーIDを貼り付け" 

# --- ページ設定 ---
st.set_page_config(page_title="ドル円AI短期予測 (5分足固定版)", layout="wide")

# --- 自動更新設定 (5分 = 300,000ミリ秒) ---
count = st_autorefresh(interval=300000, limit=None, key="fizzbuzzcounter")

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
    .stSlider > div > div > div > div {
        color: #00cc96 !important;
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
    
    st.markdown("### USD/JPY 予測ツール")
    password = st.text_input("パスワード", type="password")
    if password == DEMO_PASSWORD:
        st.session_state.password_correct = True
        st.rerun()
    elif password:
        st.error("パスワードが違います")
    return False

if not check_password():
    st.stop()

# --- LINE通知関数 ---
def send_line_notification(message):
    if not LINE_CHANNEL_ACCESS_TOKEN or not LINE_USER_ID or "ここに" in LINE_CHANNEL_ACCESS_TOKEN:
        return False

    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_CHANNEL_ACCESS_TOKEN}"
    }
    data = {
        "to": LINE_USER_ID,
        "messages": [{"type": "text", "text": message}]
    }
    try:
        requests.post(url, headers=headers, data=json.dumps(data))
        return True
    except:
        return False

# --- 数値変換 ---
def to_float(x):
    try:
        if isinstance(x, float): return x
        if isinstance(x, (pd.Series, pd.DataFrame)): return float(x.iloc[0]) if not x.empty else 0.0
        if hasattr(x, 'item'): return float(x.item())
        if isinstance(x, list): return float(x[0])
        return float(x)
    except: return 0.0

# --- リアルタイム価格強制取得 ---
def get_realtime_data():
    try:
        ticker = yf.Ticker("USDJPY=X")
        df_now = ticker.history(period="5d", interval="1m")
        if not df_now.empty:
            df_now.index = df_now.index.tz_convert('Asia/Tokyo')
            latest_price = float(df_now['Close'].iloc[-1])
            latest_time = df_now.index[-1]
            return latest_price, latest_time, df_now
    except:
        pass
    return None, None, pd.DataFrame()

# --- 強力データ取得関数 ---
def get_forex_data_robust():
    tickers_to_try = ["USDJPY=X", "JPY=X"]
    for ticker in tickers_to_try:
        try:
            # 5分足を直近5日分取得
            temp_df = yf.download(ticker, period="5d", interval="5m", progress=False)
            if not temp_df.empty and len(temp_df) > 20:
                return temp_df
        except:
            pass
    return pd.DataFrame()

# --- 乖離判定付き確率計算 ---
def calculate_reversion_probability(current_price, predicted_price, lower_bound, upper_bound, min_width=0.03, trend_direction=0):
    c = to_float(current_price)
    p = to_float(predicted_price)
    l = to_float(lower_bound)
    u = to_float(upper_bound)
    
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
    note = "順張り"
    
    box_width = u - l
    if box_width < 0.01: box_width = 0.01

    if c > u: 
        excess = c - u
        ratio = excess / box_width
        correction = - (ratio * 20.0)
        correction = max(correction, -15.0)
        note = f"上値重め (調整警戒 {correction:.1f}%)"
    elif c < l: 
        excess = l - c
        ratio = excess / box_width
        correction = + (ratio * 20.0)
        correction = min(correction, 15.0)
        note = f"底堅い (反発期待 +{correction:.1f}%)"
    else: 
        center = (u + l) / 2
        dist_from_center = (c - center) / (box_width / 2) if box_width > 0 else 0
        correction += dist_from_center * -5.0

    # 長期トレンドフィルター
    if p < c and trend_direction == 1:
        penalty = 20.0 
        base_prob += penalty 
        note = "長期上昇中のため下値限定"
    elif p > c and trend_direction == -1:
        penalty = 20.0
        base_prob -= penalty 
        note = "長期下落中のため上値限定"

    final_prob = base_prob + correction
    final_prob = max(15.0, min(85.0, final_prob)) 
    
    return final_prob, note

# --- バックテスト機能 (時間フィルター付き・72時間版) ---
def perform_backtest_persistent(df, forecast_df, min_width_setting, trend_window, threshold):
    """
    過去72時間分のデータでテスト。
    """
    df_merged = pd.merge(df, forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], left_on=df.columns[0], right_on='ds', how='inner')
    
    # 最後の行（現在進行中の足）が含まれていると結果が揺れるため除外
    # ただし、リアルタイム判定には使うため、ここではバックテスト用にコピーして処理
    df_fixed = df_merged.copy()
    
    cutoff_date = df_fixed['ds'].max() - timedelta(hours=72)
    backtest_data = df_fixed[df_fixed['ds'] >= cutoff_date].copy().reset_index(drop=True)
    
    results = []
    active_trade = None 
    
    for i in range(len(backtest_data)):
        row = backtest_data.iloc[i]
        current_time = row['ds']
        current_hour = current_time.hour 
        
        o_price = to_float(row['Open'])
        h_price = to_float(row['High'])
        l_price = to_float(row['Low'])
        c_price = to_float(row['Close'])
        
        # --- 1. 決済判定 ---
        if active_trade is not None:
            outcome = None
            pnl = 0.0
            
            hit_tp = False
            hit_sl = False
            
            if active_trade['type'] == 'BUY':
                if h_price >= active_trade['tp']: hit_tp = True
                if l_price <= active_trade['sl']: hit_sl = True
            elif active_trade['type'] == 'SELL':
                if l_price <= active_trade['tp']: hit_tp = True
                if h_price >= active_trade['sl']: hit_sl = True
            
            if hit_sl and hit_tp:
                outcome = "LOSS"
                pnl = -15.0
            elif hit_sl:
                outcome = "LOSS"
                pnl = -15.0
            elif hit_tp:
                outcome = "WIN"
                pnl = 15.0
            
            if outcome:
                results.append({
                    "エントリー": active_trade['start_time'].strftime('%m/%d %H:%M'),
                    "決済日時": current_time.strftime('%m/%d %H:%M'),
                    "売買": active_trade['type'],
                    "価格": active_trade['entry_price'],
                    "結果": outcome,
                    "P/L(pips)": pnl
                })
                active_trade = None 
                continue 
        
        # --- 2. 新規エントリー判定 ---
        if active_trade is None:
            # 時間フィルター: 2時〜8時はエントリーしない
            if 2 <= current_hour < 9:
                continue

            pred = to_float(row['yhat'])
            
            current_trend_sma = to_float(row['Trend_SMA']) if 'Trend_SMA' in row else c_price
            trend_dir = 0
            if c_price > current_trend_sma: trend_dir = 1
            elif c_price < current_trend_sma: trend_dir = -1
            
            prob_up, _ = calculate_reversion_probability(
                c_price, pred, 
                to_float(row['yhat_lower']), to_float(row['yhat_upper']),
                min_width=min_width_setting,
                trend_direction=trend_dir
            )
            
            action = None
            if prob_up >= threshold:
                action = "BUY"
            elif prob_up <= (100.0 - threshold):
                action = "SELL"
                
            if action:
                entry_price = c_price
                tp_dist = 0.15 
                sl_dist = 0.15 
                
                if action == "BUY":
                    active_trade = {
                        'type': 'BUY',
                        'entry_price': entry_price,
                        'tp': entry_price + tp_dist,
                        'sl': entry_price - sl_dist,
                        'start_time': current_time
                    }
                else:
                    active_trade = {
                        'type': 'SELL',
                        'entry_price': entry_price,
                        'tp': entry_price - tp_dist,
                        'sl': entry_price + sl_dist,
                        'start_time': current_time
                    }
                    
    return pd.DataFrame(results)

# --- メイン処理 ---
st.markdown("### **ドル円AI短期予測 (5分足専用・固定検証版)**")

# === 時間足は5分固定 ===
timeframe = "5分足 (5m)"
api_interval = "5m"
api_period = "5d" # 5日分取得
min_width_setting = 0.03
future_configs = [(5, "5分後"), (15, "15分後"), (30, "30分後"), (60, "1H後")]
past_configs = [(5, "5分前"), (15, "15分前"), (30, "30分前"), (60, "1H前")]
time_unit = "minutes"
trend_window = 100 

# === 通知設定 ===
notify_threshold = st.slider(
    "🔔 LINE通知判定 / エントリー閾値 (%)",
    min_value=70, max_value=95, value=80, step=5,
    help="この確率を超えた場合、LINE通知を行い、バックテストのエントリー基準としても使用します。"
)

try:
    with st.spinner('5分足データ取得中...'):
        df = get_forex_data_robust()

    if df.empty:
        st.error("データが取得できませんでした。時間をおいて再接続してください。")
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
    df['Trend_SMA'] = df[close_c].rolling(window=trend_window).mean()

    df_p = pd.DataFrame()
    df_p['ds'] = df[date_c]
    df_p['y'] = df[close_c]
    
    # --- ★【重要】AI学習データの固定化 ---
    # 最後の行（現在進行中の足）を除外して学習させることで、次の足が確定するまで結果を固定する
    df_train = df_p.iloc[:-1].copy()
    
    # Prophet学習
    m = Prophet(
        changepoint_prior_scale=0.15, 
        daily_seasonality=False,
        weekly_seasonality=True, 
        yearly_seasonality=False
    )
    m.add_seasonality(name='hourly', period=1/24, fourier_order=5)
    m.fit(df_train) # 確定足のみで学習
    
    # 予測作成
    future = m.make_future_dataframe(periods=40, freq='5min')
    forecast = m.predict(future)

    # --- 現在値の表示 (ここはリアルタイム) ---
    realtime_price, realtime_time, df_recent_1m = get_realtime_data()
    
    # チャートの最後の確定足データ
    last_fixed_price = to_float(df_p['y'].iloc[-2]) # 確定足の終値
    last_fixed_date = df_p['ds'].iloc[-2]

    if realtime_price is not None:
        current_price = realtime_price
        display_time = realtime_time.strftime('%m/%d %H:%M')
        # 現在進行中の足のデータをリアルタイム値で更新（表示用）
        # ただしAI学習には使わない
    else:
        current_price = to_float(df_p['y'].iloc[-1])
        now_jst_fallback = datetime.now(pytz.timezone('Asia/Tokyo'))
        display_time = now_jst_fallback.strftime('%m/%d %H:%M')

    # トレンド判定 (確定足ベース)
    current_trend_sma = to_float(df['Trend_SMA'].iloc[-2])
    trend_dir = 0
    if not pd.isna(current_trend_sma):
        if last_fixed_price > current_trend_sma: trend_dir = 1 
        else: trend_dir = -1 

    st.write(f"**現在値 ({timeframe}): {current_price:,.2f} 円**")
    trend_text = "長期上昇トレンド中" if trend_dir == 1 else ("長期下落トレンド中" if trend_dir == -1 else "レンジ相場")
    st.write(f"<span style='font-size:0.9rem; color:#ddd'>{trend_text} (現在日時: {display_time})</span>", unsafe_allow_html=True)

    # =========================================
    #  過去データ分析
    # =========================================
    st.markdown("#### **📉 直近のAI判断 (過去の答え合わせ)**")
    
    past_data_list = []
    
    for val, label_text in past_configs:
        # 基準は「最後の確定足」の時間
        target_time = last_fixed_date - timedelta(minutes=val)
        
        # 1. その時点の「実際の価格」を探す
        past_actual_price = None
        try:
            row_past = df_p.iloc[(df_p['ds'] - target_time).abs().argsort()[:1]].iloc[0]
            if abs((row_past['ds'] - target_time).total_seconds()) < 600:
                past_actual_price = to_float(row_past['y
