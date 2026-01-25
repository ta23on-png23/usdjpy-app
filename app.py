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

# --- 乖離判定付き確率計算 ---
def calculate_reversion_probability(current_price, predicted_price, lower_bound, upper_bound, min_width=0.10, trend_direction=0):
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

# --- バックテスト機能 (15pips版) ---
def perform_backtest_15pips(df, forecast_df, min_width_setting, trend_window):
    """
    過去48時間分のデータで「確率80%以上で順張りエントリー、15pips利確損切り」をテストする
    """
    df_merged = pd.merge(df, forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], left_on=df.columns[0], right_on='ds', how='inner')
    
    cutoff_date = df_merged['ds'].max() - timedelta(hours=48)
    backtest_data = df_merged[df_merged['ds'] >= cutoff_date].copy().reset_index(drop=True)
    
    results = []
    
    for i in range(len(backtest_data) - 1):
        row = backtest_data.iloc[i]
        next_row = backtest_data.iloc[i+1] 
        
        current_price = to_float(row['Close'])
        pred = to_float(row['yhat'])
        
        current_trend_sma = to_float(row['Trend_SMA']) if 'Trend_SMA' in row else current_price
        
        trend_dir = 0
        if current_price > current_trend_sma: trend_dir = 1
        elif current_price < current_trend_sma: trend_dir = -1
            
        prob_up, _ = calculate_reversion_probability(
            current_price, pred, 
            to_float(row['yhat_lower']), to_float(row['yhat_upper']),
            min_width=min_width_setting,
            trend_direction=trend_dir
        )
        
        action = None
        if prob_up >= 80.0:
            action = "BUY"
        elif prob_up <= 20.0: 
            action = "SELL"
            
        if action:
            entry_price = current_price
            tp_pips = 0.15 
            sl_pips = 0.15 
            
            outcome = "DRAW" 
            pnl = 0.0
            
            next_high = to_float(next_row['High'])
            next_low = to_float(next_row['Low'])
            next_close = to_float(next_row['Close'])
            
            if action == "BUY":
                tp_price = entry_price + tp_pips
                sl_price = entry_price - sl_pips
                
                hit_tp = next_high >= tp_price
                hit_sl = next_low <= sl_price
                
                if hit_sl:
                    outcome = "LOSS"
                    pnl = -15.0 
                elif hit_tp:
                    outcome = "WIN"
                    pnl = 15.0  
                else:
                    pnl = (next_close - entry_price) * 100
                    outcome = "TIME_EXIT"

            elif action == "SELL":
                tp_price = entry_price - tp_pips
                sl_price = entry_price + sl_pips
                
                hit_tp = next_low <= tp_price
                hit_sl = next_high >= sl_price
                
                if hit_sl:
                    outcome = "LOSS"
                    pnl = -15.0 
                elif hit_tp:
                    outcome = "WIN"
                    pnl = 15.0  
                else:
                    pnl = (entry_price - next_close) * 100
                    outcome = "TIME_EXIT"
            
            results.append({
                "時間": row['ds'].strftime('%m/%d %H:%M'),
                "売買": action,
                "Entry": entry_price,
                "確率": f"{prob_up:.1f}%" if action=="BUY" else f"{100-prob_up:.1f}%",
                "結果": outcome,
                "P/L(pips)": round(pnl, 1)
            })
            
    return pd.DataFrame(results)

# --- メイン処理 ---
st.markdown("### **ドル円AI短期予測 (マルチタイムフレーム・トレンド補正版)**")

# === 時間足選択 ===
timeframe = st.radio(
    "時間足を選択してください",
    ["1時間足 (1H)", "15分足 (15m)", "5分足 (5m)"],
    horizontal=True
)

# 設定値の決定
if timeframe == "1時間足 (1H)":
    api_interval = "1h"
    api_period = "1mo"
    min_width_setting = 0.10
    target_configs = [(1, "1H後"), (2, "2H後"), (4, "4H後"), (8, "8H後"), (12, "12H後")]
    time_unit = "hours"
    trend_window = 50 
    
elif timeframe == "15分足 (15m)":
    api_interval = "15m"
    api_period = "1mo"
    min_width_setting = 0.05
    target_configs = [(15, "15分後"), (30, "30分後"), (60, "1H後"), (120, "2H後"), (240, "4H後")]
    time_unit = "minutes"
    trend_window = 80 
    
else: # 5分足
    api_interval = "5m"
    api_period = "5d"
    min_width_setting = 0.03
    target_configs = [(5, "5分後"), (15, "15分後"), (30, "30分後"), (60, "1H後"), (120, "2H後")]
    time_unit = "minutes"
    trend_window = 100 


try:
    with st.spinner(f'{timeframe} データ取得中...'):
        df = get_forex_data_robust(interval=api_interval, period=api_period)

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
    
    current_price = to_float(df_p['y'].iloc[-1])
    current_trend_sma = to_float(df['Trend_SMA'].iloc[-1])
    last_date = df_p['ds'].iloc[-1]
    
    trend_dir = 0
    if not pd.isna(current_trend_sma):
        if current_price > current_trend_sma: trend_dir = 1 
        else: trend_dir = -1 

    st.write(f"**現在値 ({timeframe}): {current_price:,.2f} 円**")
    
    trend_text = "長期上昇トレンド中" if trend_dir == 1 else ("長期下落トレンド中" if trend_dir == -1 else "レンジ相場")
    st.write(f"<span style='font-size:0.9rem; color:#ddd'>{trend_text} (基準日時: {last_date.strftime('%m/%d %H:%M')})</span>", unsafe_allow_html=True)

    # --- Prophet予測 ---
    prior_scale = 0.05 if api_interval == "5m" else 0.15 
    
    m = Prophet(
        changepoint_prior_scale=prior_scale, 
        daily_seasonality=True if api_interval == "1h" else False,
        weekly_seasonality=True, 
        yearly_seasonality=False
    )
    if api_interval in ["5m", "15m"]:
        m.add_seasonality(name='hourly', period=1/24, fourier_order=5)

    m.fit(df_p)
    
    freq_str = 'h' if api_interval == '1h' else ('15min' if api_interval == '15m' else '5min')
    periods_needed = 30
    future = m.make_future_dataframe(periods=periods_needed, freq=freq_str)
    forecast = m.predict(future)

    # --- 予測結果抽出 ---
    st.markdown("#### **短期予測 (上昇 vs 下落)**")
    
    probs_up = []
    probs_down = []
    labels = []
    prices = []
    notes = []
    colors_up = []
    colors_down = []

    for val, label_text in target_configs:
        if time_unit == "hours":
            target_time = last_date + timedelta(hours=val)
        else:
            target_time = last_date + timedelta(minutes=val)
            
        row = forecast.iloc[(forecast['ds'] - target_time).abs().argsort()[:1]].iloc[0]
        pred = to_float(row['yhat'])
        
        prob_up, note = calculate_reversion_probability(
            current_price, pred, 
            to_float(row['yhat_lower']), 
            to_float(row['yhat_upper']),
            min_width=min_width_setting,
            trend_direction=trend_dir 
        )
        prob_down = 100.0 - prob_up
        
        price_diff = abs(pred - current_price)
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

    # --- チャート表示 ---
    st.markdown("#### **推移・AI軌道・テクニカル指標**")
    
    fig_chart = go.Figure()

    # ボリンジャーバンド
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['BB_Upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig_chart.add_trace(go.Scatter(
        x=df[date_c], y=df['BB_Lower'], mode='lines', line=dict(width=0),
        fill='tonexty', 
        fillcolor='rgba(138, 43, 226, 0.3)', 
        name='BB(±2σ)', hoverinfo='skip'
    ))

    # ローソク足
    fig_chart.add_trace(go.Candlestick(
        x=df[date_c],
        open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='実測',
        increasing=dict(line=dict(color='#00cc96', width=1), fillcolor='rgba(0,0,0,0)'),
        decreasing=dict(line=dict(color='#ff4b4b', width=1), fillcolor='rgba(0,0,0,0)')
    ))

    # SMA
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['SMA20'], mode='lines', name='20SMA (短期)', line=dict(color='cyan', width=1.5)))
    fig_chart.add_trace(go.Scatter(x=df[date_c], y=df['Trend_SMA'], mode='lines', name='長期トレンド線', line=dict(color='orange', width=2, dash='dash')))
    
    # AI予測
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', line=dict(width=0), hoverinfo='skip', showlegend=False))
    fig_chart.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', line=dict(width=0),
        fill='tonexty', 
        fillcolor='rgba(255, 255, 0, 0.4)', 
        hoverinfo='skip', showlegend=False, name='AI予測範囲'
    ))
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='AI軌道', line=dict(color='yellow', width=2)))

    # 表示範囲
    x_max = forecast['ds'].max()
    x_min = df[date_c].min() 
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
            fixedrange=False,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            range=[y_range_min, y_range_max],
            fixedrange=True
        ),
        showlegend=True
    )
    
    st.plotly_chart(fig_chart, use_container_width=True, config={'displayModeBar': False, 'staticPlot': False})

    # --- バックテスト結果表示 ---
    st.markdown("---")
    st.markdown("### 🔙 **過去48時間のバックテスト結果 (15pips 利確/損切り)**")
    st.markdown("""
    <div style="font-size:0.8rem; color:#aaa; margin-bottom:10px;">
    ルール: AIの方向確率が80%を超えた時点でエントリー。次の足の高値/安値が15pips(0.15円)に達したら決済。<br>
    ※同じ足で利確と損切りの両方に達した場合は「負け」としてカウントする厳しめの判定です。
    </div>
    """, unsafe_allow_html=True)
    
    bt_results = perform_backtest_15pips(df, forecast, min_width_setting, trend_window)
    
    if not bt_results.empty:
        total_trades = len(bt_results)
        wins = len(bt_results[bt_results['結果'] == "WIN"])
        losses = len(bt_results[bt_results['結果'] == "LOSS"])
        time_exits = len(bt_results[bt_results['結果'] == "TIME_EXIT"])
        
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pips = bt_results['P/L(pips)'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総取引回数", f"{total_trades} 回")
        col2.metric("勝率", f"{win_rate:.1f} %")
        col3.metric("合計獲得pips", f"{total_pips:+.1f} pips", delta_color="normal")
        col4.metric("内訳", f"勝{wins} / 負{losses} / 分{time_exits}")
        
        # --- 追加: 損益推移グラフ (改良版: 棒グラフ + 折れ線グラフ) ---
        st.markdown("### 📊 **損益推移 (単独 & 累積)**")
        
        bt_results['Cumulative_PL'] = bt_results['P/L(pips)'].cumsum()
        
        fig_pnl = go.Figure()
        
        # 1. 単独損益 (棒グラフ)
        # 勝ちは緑、負けは赤、引き分け(0)はグレー
        bar_colors = []
        for val in bt_results['P/L(pips)']:
            if val > 0: bar_colors.append('#00cc96') # 緑
            elif val < 0: bar_colors.append('#ff4b4b') # 赤
            else: bar_colors.append('#808080') # グレー

        fig_pnl.add_trace(go.Bar(
            x=bt_results['時間'],
            y=bt_results['P/L(pips)'],
            name='単独損益',
            marker_color=bar_colors,
            opacity=0.6 # 透けさせてラインを見やすく
        ))
        
        # 2. 累積損益 (折れ線グラフ)
        fig_pnl.add_trace(go.Scatter(
            x=bt_results['時間'], 
            y=bt_results['Cumulative_PL'], 
            mode='lines+markers', 
            name='累積損益',
            line=dict(color='yellow', width=3) # 黄色で目立たせる
        ))
        
        # 基準線 (0, ±100, ±200, ±300)
        lines_to_draw = [0, 100, -100, 200, -200, 300, -300]
        for val in lines_to_draw:
            color = 'white' if val == 0 else ('#333' if abs(val) < 300 else '#555')
            width = 1 if val == 0 else 1
            dash = 'solid' if val == 0 else 'dash'
            
            fig_pnl.add_hline(y=val, line_dash=dash, line_color=color, line_width=width, annotation_text=f"{val} pips" if val !=0 else "±0")

        # レイアウト調整 (見やすくするために±300以上の範囲も自動考慮)
        # Y軸の範囲計算 (単独損益のバーと累積ラインの両方が入るように)
        vals_to_check = pd.concat([bt_results['P/L(pips)'], bt_results['Cumulative_PL']])
        y_max = max(350, vals_to_check.max() + 50)
        y_min = min(-350, vals_to_check.min() - 50)
        
        fig_pnl.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=30, b=20),
            yaxis=dict(title="pips", range=[y_min, y_max]),
            xaxis=dict(title="日時", type='category'), 
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

        st.dataframe(bt_results, hide_index=True, use_container_width=True)
    else:
        st.info("過去48時間以内に条件(確率80%以上)を満たすエントリーポイントはありませんでした。")

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
