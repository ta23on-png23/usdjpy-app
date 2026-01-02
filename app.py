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

# --- 強力データ取得関数 ---
def get_forex_data_robust():
    tickers_to_try = ["USDJPY=X", "JPY=X"]
    for ticker in tickers_to_try:
        try:
            temp_df = yf.download(ticker, period="1mo", interval="1h", progress=False)
            if not temp_df.empty and len(temp_df) > 24:
                return temp_df
        except:
            pass
        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=29)
            temp_df = yf.download(ticker, start=start_dt, end=end_dt, interval="1h", progress=False)
            if not temp_df.empty and len(temp_df) > 24:
                return temp_df
        except:
            pass
    return pd.DataFrame()

# --- ★最重要：乖離（平均回帰）判定付き確率計算 ---
def calculate_reversion_probability(current_price, predicted_price, lower_bound, upper_bound):
    """
    黄色い枠（AI予測範囲）からの乖離を見て、行き過ぎた相場の「戻り」を加味する
    """
    c = to_float(current_price)
    p = to_float(predicted_price)
    l = to_float(lower_bound)
    u = to_float(upper_bound)
    
    # 1. 基礎トレンド確率 (Zスコア)
    sigma = (u - l) / 2.56
    if sigma == 0: base_prob = 50.0
    else:
        z_score = (p - c) / sigma
        base_prob = norm.cdf(z_score) * 100

    # 2. 乖離補正 (Mean Reversion Logic)
    # 黄色い枠の幅
    box_width = u - l
    if box_width == 0: box_width = 0.01

    correction = 0.0
    note = "順張り(トレンド追随)"
    
    # 【ケースA】上に突き抜けている場合 (Overbought)
    if c > u:
        # どれくらい突き抜けたか (乖離率)
        excess = c - u
        ratio = excess / box_width # 枠の幅に対して何倍突き抜けたか
        
        # 突き抜けた分だけ、強力に「確率を下げる（下落調整を予測）」
        correction = - (ratio * 40.0) # 係数40: かなり強く戻そうとする力
        correction = max(correction, -40.0) # 最大でも40%ダウンまで
        
        base_prob += correction
        note = f"⚠️上振れ乖離 (調整警戒 -{abs(correction):.1f}%)"

    # 【ケースB】下に突き抜けている場合 (Oversold)
    elif c < l:
        # どれくらい突き抜けたか
        excess = l - c
        ratio = excess / box_width
        
        # 突き抜けた分だけ、強力に「確率を上げる（自律反発を予測）」
        correction = + (ratio * 40.0)
        correction = min(correction, 40.0)
        
        base_prob += correction
        note = f"⚠️下振れ乖離 (反発期待 +{abs(correction):.1f}%)"

    # 【ケースC】枠内の場合 (Normal)
    else:
        # 枠内だが、上限・下限ギリギリの場合の微調整
        # 中心からの距離を見る
        center = (u + l) / 2
        dist_from_center = (c - center) / (box_width / 2) # -1.0 ~ 1.0
        
        # 端っこにいるほど少しだけ逆張り圧力をかける（ゴム紐の原理）
        minor_correction = dist_from_center * -5.0 # 最大±5%程度の微調整
        base_prob += minor_correction

    # 0~100に収める
    final_prob = max(1.0, min(99.0, base_prob))
    
    return final_prob, note

# --- メイン処理 ---
st.markdown("### **🇺🇸🇯🇵 ドル円AI短期予測 (乖離修正ロジック搭載)**")
st.markdown("""
<div style="margin-top: -10px; margin-bottom: 20px;">
    <span style="font-size: 0.7rem; opacity: 0.8;">※黄色い枠（AI予測範囲）から価格が大きく外れた場合、「行き過ぎ」と判断して反発・調整の可能性を加味します。</span>
</div>
""", unsafe_allow_html=True)

try:
    with st.spinner('市場データ取得＆乖離計算中...'):
        df = get_forex_data_robust()

    if df.empty:
        st.error("⚠️ データが取得できませんでした。しばらく時間を置いて再接続してください。")
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

    # テクニカル計算 (表示用)
    df['SMA20'] = df[close_c].rolling(window=20).mean()
    df['STD'] = df[close_c].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (df['STD'] * 2)
    df['BB_Lower'] = df['SMA20'] - (df['STD'] * 2)

    # Prophetデータ作成
    df_p = pd.DataFrame()
    df_p['ds'] = df[date_c]
    df_p['y'] = df[close_c]
    
    current_price = to_float(df_p['y'].iloc[-1])
    last_date = df_p['ds'].iloc[-1]

    st.write(f"**現在値: {current_price:,.2f} 円**")
    st.write(f"<span style='font-size:0.8rem; color:#aaa'>基準日時: {last_date.strftime('%m/%d %H:%M')}</span>", unsafe_allow_html=True)

    # --- Prophet予測 ---
    m = Prophet(changepoint_prior_scale=0.15, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
    m.fit(df_p)
    future = m.make_future_dataframe(periods=13, freq='h')
    forecast = m.predict(future)

    # --- 予測結果の抽出 ---
    st.markdown("#### **📈 短期予測 (上昇 vs 下落)**")
    
    targets = [1, 2, 4, 8, 12]
    probs_up = []
    probs_down = []
    labels = []
    prices = []
    notes = []

    for i, h in enumerate(targets):
        target_time = last_date + timedelta(hours=h)
        row = forecast.iloc[(forecast['ds'] - target_time).abs().argsort()[:1]].iloc[0]
        
        pred = to_float(row['yhat'])
        
        # ★乖離判定ロジックを使用
        prob_up, note = calculate_reversion_probability(
            current_price, 
            pred, 
            to_float(row['yhat_lower']), 
            to_float(row['yhat_upper'])
        )
        prob_down = 100.0 - prob_up
        
        probs_up.append(prob_up)
        probs_down.append(prob_down)
        labels.append(f"{h}H後")
        prices.append(pred)
        notes.append(note)

    # --- 棒グラフ ---
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=labels, y=probs_up, name='上昇確率', text=[f"{p:.1f}%" for p in probs_up], textposition='auto', marker_color='#00cc96'))
    fig_bar.add_trace(go.Bar(x=labels, y=probs_down, name='下落確率', text=[f"{p:.1f}%" for p in probs_down], textposition='auto', marker_color='#ff4b4b'))
    
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
    
    # 乖離状況によって文字色を変えるなどの処理（データフレーム上ではテキストで表現）
    detail_data = {
        "時間": labels,
        "予測レート": [f"{p:.2f} 円" for p in prices],
        "上昇確率": [f"{p:.1f} %" for p in probs_up],
        "下落確率": [f"{p:.1f} %" for p in probs_down],
        "乖離判定": notes # ここに「上振れ乖離」などの理由が出る
    }
    st.dataframe(pd.DataFrame(detail_data), hide_index=True, use_container_width=True)

    # --- チャート表示 ---
    st.markdown("#### **直近1週間の推移・AI軌道・テクニカル指標**")
    
    fig_chart = go.Figure()

    # ボリンジャーバンド
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['BB_Upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['BB_Lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(0, 200, 255, 0.1)', name='BB(±2σ)', hoverinfo='skip'))

    # 実測
    fig_chart.add_trace(go.Candlestick(x=df[date_c], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='実測'))

    # SMA
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['SMA20'], mode='lines', name='20SMA', line=dict(color='cyan', width=1.5)))
    
    # AI予測範囲 (黄色)
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 255, 0, 0.15)', hoverinfo='skip', showlegend=False, name='AI予測範囲'))

    # AI予測線
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='AI軌道', line=dict(color='yellow', width=2)))

    # 表示範囲
    x_max = forecast['ds'].max()
    x_min = last_date - timedelta(days=7)

    fig_chart.update_layout(
        template="plotly_dark",
        height=450,
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
