import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from scipy.stats import norm
import plotly.graph_objs as go

# --- 安全な数値変換関数 ---
def to_float(x):
    try:
        if isinstance(x, float): return x
        if isinstance(x, (pd.Series, pd.DataFrame)):
            if x.empty: return 0.0
            return float(x.to_numpy()[0])
        if hasattr(x, 'item'): return float(x.item())
        if isinstance(x, list): return float(x[0])
        return float(x)
    except: return 0.0

# --- ページ設定 ---
st.set_page_config(page_title="USD/JPY AI確率予測", layout="wide")
st.title('📈 USD/JPY AI確率予測モニター')

# --- サイドバー ---
st.sidebar.header("操作盤")
if st.sidebar.button('🔄 データを最新に更新'):
    st.rerun()
st.sidebar.markdown("""
**表示の見方**
- **上昇確率**: 現在の価格より上がる確率
- **60%以上**: 買いのチャンス (緑)
- **40%以下**: 売りのチャンス (赤)
""")

# --- 確率計算関数 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    c, p, l, u = to_float(current_price), to_float(predicted_price), to_float(lower_bound), to_float(upper_bound)
    sigma = (u - l) / 2.56
    if sigma == 0: return 50.0
    z_score = (p - c) / sigma
    return norm.cdf(z_score) * 100

# --- メイン処理 ---
ticker = "USDJPY=X"

try:
    # 1. データ取得
    with st.spinner(f'{ticker} のデータを取得中...'):
        raw_data = yf.download(ticker, period="2y", interval="1h", progress=False)
    
    if raw_data.empty:
        st.error("データの取得に失敗しました。")
        st.stop()

    # --- データ整形 ---
    df = raw_data.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # カラム特定
    date_col, open_col, high_col, low_col, close_col = None, None, None, None, None
    for col in df.columns:
        c_str = str(col).lower()
        if 'date' in c_str or 'time' in c_str: date_col = col
        if 'open' in c_str: open_col = col
        if 'high' in c_str: high_col = col
        if 'low' in c_str: low_col = col
        if 'close' in c_str: close_col = col

    if date_col is None: date_col = df.columns[0]
    if close_col is None: close_col = df.columns[1]

    df_ohlc = pd.DataFrame()
    df_ohlc['ds'] = pd.to_datetime(df[date_col]).dt.tz_localize(None)
    df_ohlc['Open'] = df[open_col] if open_col else df[close_col]
    df_ohlc['High'] = df[high_col] if high_col else df[close_col]
    df_ohlc['Low'] = df[low_col] if low_col else df[close_col]
    df_ohlc['Close'] = df[close_col]

    df_clean = pd.DataFrame({'ds': df_ohlc['ds'], 'y': df_ohlc['Close']})
    latest_close = to_float(df_clean['y'].iloc[-1])
    latest_time = df_clean['ds'].iloc[-1]

    # --- 2. 画面トップ表示 ---
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(label="現在レート (直近終値)", value=f"{latest_close:.3f} 円", delta="最新更新")
    with col2:
        st.info(f"最終データ日時: {latest_time.strftime('%Y/%m/%d %H:%M')}")

    # --- 3. AI学習と予測 ---
    with st.spinner('AIが未来を計算中...'):
        m = Prophet(changepoint_prior_scale=0.05, daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(df_clean)
        future = m.make_future_dataframe(periods=24, freq='H')
        forecast = m.predict(future)

    # --- 4. 確率判定テーブル ---
    st.subheader('🎯 未来の上昇・下落確率')
    future_forecast = forecast[forecast['ds'] > latest_time].copy()
    targets = [1, 4, 8, 24]
    results = []
    for h in targets:
        if len(future_forecast) >= h:
            row = future_forecast.iloc[h-1]
            pred_val = to_float(row['yhat'])
            prob_up = calculate_probability(latest_close, pred_val, to_float(row['yhat_lower']), to_float(row['yhat_upper']))
            trend = "➡️ レンジ"
            if prob_up >= 60: trend = "↗️ 上昇優勢"
            elif 100-prob_up >= 60: trend = "↘️ 下落優勢"
            results.append({
                "対象": f"{h}時間後", "予測日時": row['ds'].strftime('%m/%d %H:%M'),
                "現在価格": f"{latest_close:.3f}", "予測価格": f"{pred_val:.3f}",
                "上昇確率": f"{prob_up:.1f} %", "下落確率": f"{100-prob_up:.1f} %", "判定": trend
            })
    st.table(pd.DataFrame(results).set_index("対象"))

    # --- 5. グラフ表示 (ローソク足) ---
    st.subheader('📊 予測推移チャート (ローソク足＆AI予測)')
    
    fig = go.Figure()

    # ローソク足
    fig.add_trace(go.Candlestick(
        x=df_ohlc['ds'],
        open=df_ohlc['Open'], high=df_ohlc['High'],
        low=df_ohlc['Low'], close=df_ohlc['Close'],
        name='実測値',
        increasing_line_color='#00CC96',
        decreasing_line_color='#EF553B'
    ))

    # AI予測ライン(黄色)
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat'],
        mode='lines', name='AI予測ライン',
        line=dict(color='yellow', width=2)
    ))

    # 予測範囲(薄い黄色)
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_upper'],
        mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'],
        mode='lines', line=dict(width=0),
        fill='tonexty', fillcolor='rgba(255, 255, 0, 0.2)',
        hoverinfo='skip', showlegend=False, name='予測範囲'
    ))

    fig.add_hline(y=latest_close, line_dash="dash", line_color="white", annotation_text="現在")

    fig.update_layout(
        title="実測ローソク足とAI予測ライン",
        yaxis_title="価格 (円)",
        template="plotly_dark",
        height=600,
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
