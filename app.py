import streamlit as st
import yfinance as yf
from prophet import Prophet
import pandas as pd
from scipy.stats import norm
import plotly.graph_objs as go
from datetime import timedelta, datetime
import pytz

# ==========================================
#  設定：パスワード
# ==========================================
DEMO_PASSWORD = "demo" 

# --- ページ設定 ---
st.set_page_config(page_title="ドル円AI短期予測 (5分足固定版)", layout="wide")

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

# --- 数値変換 ---
def to_float(x):
    try:
        if isinstance(x, float): return x
        if isinstance(x, (pd.Series, pd.DataFrame)): return float(x.iloc[0]) if not x.empty else 0.0
        if hasattr(x, 'item'): return float(x.item())
        if isinstance(x, list): return float(x[0])
        return float(x)
    except: return 0.0

# --- リアルタイム価格 & 履歴取得 ---
def get_realtime_data():
    try:
        ticker = yf.Ticker("USDJPY=X")
        # 直近のデータを取得して現在値を確認
        df_now = ticker.history(period="1d", interval="1m")
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

# --- バックテスト機能 ---
def perform_backtest_persistent(df_fixed, forecast_df, min_width_setting, trend_window, threshold):
    """
    過去72時間分のデータでテスト。
    """
    # df_fixedとforecastをマージ
    df_merged = pd.merge(df_fixed, forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='inner')
    
    cutoff_date = df_merged['ds'].max() - timedelta(hours=72)
    backtest_data = df_merged[df_merged['ds'] >= cutoff_date].copy().reset_index(drop=True)
    
    results = []
    active_trade = None 
    
    for i in range(len(backtest_data)):
        row = backtest_data.iloc[i]
        current_time = row['ds']
        current_hour = current_time.hour 
        
        # エラー修正: 確実に存在するカラム名を使用
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
            
            # Trend_SMAがあるか確認
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
st.markdown("### **ドル円AI短期予測 (5分足専用・完全固定版)**")

# === 設定 ===
timeframe = "5分足 (5m)"
api_interval = "5m"
api_period = "5d" 
min_width_setting = 0.03
future_configs = [(5, "5分後"), (15, "15分後"), (30, "30分後"), (60, "1H後")]
past_configs = [(5, "5分前"), (15, "15分前"), (30, "30分前"), (60, "1H前")]
time_unit = "minutes"
trend_window = 100 

# === 閾値スライダー ===
entry_threshold = st.slider(
    "エントリー判定閾値 (%)", 
    min_value=70, 
    max_value=95, 
    value=80, 
    step=5,
    help="AIの確信度がこの数値以上の場合のみエントリーします。"
)

try:
    with st.spinner('5分足データ取得中...'):
        df = get_forex_data_robust()

    if df.empty:
        st.error("データが取得できませんでした。時間をおいて再接続してください。")
        st.stop()

    # --- データ整形とカラム名統一 ---
    df = df.reset_index()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # カラム名を統一 (Open, High, Low, Close, ds)
    cols_map = {}
    for c in df.columns:
        cl = c.lower()
        if 'date' in cl or 'time' in cl: cols_map[c] = 'ds'
        elif 'open' in cl: cols_map[c] = 'Open'
        elif 'high' in cl: cols_map[c] = 'High'
        elif 'low' in cl: cols_map[c] = 'Low'
        elif 'close' in cl: cols_map[c] = 'Close'
    
    df = df.rename(columns=cols_map)
    
    # タイムゾーン処理
    try:
        df['ds'] = pd.to_datetime(df['ds']).dt.tz_convert('Asia/Tokyo').dt.tz_localize(None)
    except:
        df['ds'] = pd.to_datetime(df['ds'])

    # テクニカル計算
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['STD'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA20'] + (df['STD'] * 2)
    df['BB_Lower'] = df['SMA20'] - (df['STD'] * 2)
    df['Trend_SMA'] = df['Close'].rolling(window=trend_window).mean()

    # --- ★【重要】データ固定化 ---
    # Prophet学習用データ (OHLCすべて保持する)
    df_p = df.copy()
    df_p = df_p.rename(columns={'Close': 'y'}) # Prophetはターゲットをyとする
    
    # 最後の行（現在進行中の足）を除外して「固定データ」を作る
    df_fixed = df_p.iloc[:-1].copy()

    # --- Prophet学習 (固定データ使用) ---
    m = Prophet(
        changepoint_prior_scale=0.15, 
        daily_seasonality=False,
        weekly_seasonality=True, 
        yearly_seasonality=False
    )
    m.add_seasonality(name='hourly', period=1/24, fourier_order=5)
    m.fit(df_fixed) # 確定足のみで学習
    
    # 予測作成
    future = m.make_future_dataframe(periods=40, freq='5min')
    forecast = m.predict(future)

    # --- 現在値の表示 (ここはリアルタイム) ---
    realtime_price, realtime_time, df_recent_1m = get_realtime_data()
    
    # チャートの最後の確定足データ
    last_fixed_price = to_float(df_fixed['y'].iloc[-1])
    last_fixed_date = df_fixed['ds'].iloc[-1]

    if realtime_price is not None:
        current_price = realtime_price
        display_time = realtime_time.strftime('%m/%d %H:%M')
    else:
        current_price = to_float(df_p['y'].iloc[-1]) 
        now_jst_fallback = datetime.now(pytz.timezone('Asia/Tokyo'))
        display_time = now_jst_fallback.strftime('%m/%d %H:%M')

    # トレンド判定 (確定足ベース)
    current_trend_sma = to_float(df_fixed['Trend_SMA'].iloc[-1])
    trend_dir = 0
    if not pd.isna(current_trend_sma):
        if last_fixed_price > current_trend_sma: trend_dir = 1 
        else: trend_dir = -1 

    st.write(f"**現在値 (5分足): {current_price:,.2f} 円**")
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
            row_past = df_fixed.iloc[(df_fixed['ds'] - target_time).abs().argsort()[:1]].iloc[0]
            if abs((row_past['ds'] - target_time).total_seconds()) < 600:
                past_actual_price = to_float(row_past['y'])
        except:
            pass

        # 2. その時点の「AI予測値(yhat)」を探す
        row_fc = forecast.iloc[(forecast['ds'] - target_time).abs().argsort()[:1]].iloc[0]
        past_pred = to_float(row_fc['yhat'])
        
        if past_actual_price is not None:
            p_up, note = calculate_reversion_probability(
                past_actual_price, past_pred, 
                to_float(row_fc['yhat_lower']), 
                to_float(row_fc['yhat_upper']),
                min_width=min_width_setting,
                trend_direction=trend_dir 
            )
            p_down = 100.0 - p_up
            
            past_data_list.append({
                "時間": label_text,
                "当時のレート": f"{past_actual_price:.2f} 円",
                "AIトレンド判定": f"上 {p_up:.0f}% / 下 {p_down:.0f}%",
                "乖離状況": note
            })
        else:
             past_data_list.append({"時間": label_text, "当時のレート": "-", "AIトレンド判定": "-", "乖離状況": "-"})

    st.dataframe(pd.DataFrame(past_data_list), hide_index=True, use_container_width=True)

    # =========================================
    #  未来予測
    # =========================================
    st.markdown("#### **📈 短期予測 (上昇 vs 下落)**")
    
    probs_up = []
    probs_down = []
    labels = []
    
    # グラフ用データ作成
    for val, label_text in future_configs:
        # 起点は確定足の時間
        t_time = last_fixed_date + timedelta(minutes=val)
        
        r = forecast.iloc[(forecast['ds'] - t_time).abs().argsort()[:1]].iloc[0]
        p = to_float(r['yhat'])
        
        # 判定には「現在のリアルタイム価格」を使う
        p_up, note = calculate_reversion_probability(
            current_price, p, 
            to_float(r['yhat_lower']), to_float(r['yhat_upper']), 
            min_width=min_width_setting, trend_direction=trend_dir
        )
        probs_up.append(p_up)
        probs_down.append(100.0 - p_up)
        labels.append(label_text)

    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(x=labels, y=probs_up, name='上昇確率', marker_color='#00cc96'))
    fig_bar.add_trace(go.Bar(x=labels, y=probs_down, name='下落確率', marker_color='#ff4b4b'))
    fig_bar.update_layout(template="plotly_dark", height=250, margin=dict(l=0, r=0, t=30, b=20), barmode='group')
    st.plotly_chart(fig_bar, use_container_width=True)

    # 詳細数値
    st.markdown("#### **詳細数値 & AI判断**")
    detail_data = {
        "時間": labels,
        "上昇確率": [f"{p:.1f} %" for p in probs_up],
        "下落確率": [f"{p:.1f} %" for p in probs_down]
    }
    st.dataframe(pd.DataFrame(detail_data), hide_index=True, use_container_width=True)

    # --- チャート表示 ---
    st.markdown("#### **推移・AI軌道**")
    fig_chart = go.Figure()
    # チャートも固定データ(df_fixed)を表示
    fig_chart.add_trace(go.Candlestick(
        x=df_fixed['ds'], 
        open=df_fixed['Open'], high=df_fixed['High'], low=df_fixed['Low'], close=df_fixed['y'], 
        name='実測(確定足)'
    ))
    fig_chart.add_trace(go.Scatter(x=df_fixed['ds'], y=df_fixed['SMA20'], mode='lines', name='SMA20', line=dict(color='cyan', width=1)))
    fig_chart.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='AI軌道', line=dict(color='yellow', width=2)))
    
    x_max = forecast['ds'].max()
    x_min = df_fixed['ds'].min()
    
    fig_chart.update_layout(template="plotly_dark", height=500, xaxis=dict(range=[x_min, x_max]), yaxis=dict(fixedrange=False))
    st.plotly_chart(fig_chart, use_container_width=True)

    # --- バックテスト結果表示 ---
    st.markdown("---")
    st.markdown("### 🔙 **過去72時間のバックテスト (保有継続・時間フィルター版)**")
    
    st.markdown(f"""
    <div style="font-size:0.8rem; color:#aaa; margin-bottom:10px;">
    ルール: AIの方向確率が <b>{entry_threshold}%</b> を超えた時点でエントリー。ポジションは常に1つ。<br>
    ±15pips(0.15円)に到達するまで、時間をまたいでポジションを保有し続けます。<br>
    <span style="color:#ff4b4b;">※日本時間 02:00〜08:59 の間はエントリーしません。(決済は行われます)</span><br>
    ※データは確定足のみを使用しているため、リロードしても結果は固定されます。
    </div>
    """, unsafe_allow_html=True)
    
    # 固定データ(df_fixed)を渡す
    bt_results = perform_backtest_persistent(df_fixed, forecast, min_width_setting, trend_window, entry_threshold)
    
    if not bt_results.empty:
        total_trades = len(bt_results)
        wins = len(bt_results[bt_results['結果'] == "WIN"])
        losses = len(bt_results[bt_results['結果'] == "LOSS"])
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pips = bt_results['P/L(pips)'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("総取引回数", f"{total_trades} 回")
        col2.metric("勝率", f"{win_rate:.1f} %")
        col3.metric("合計獲得pips", f"{total_pips:+.1f} pips", delta_color="normal")
        col4.metric("内訳", f"勝{wins} / 負{losses}")
        
        bt_results['Cumulative_PL'] = bt_results['P/L(pips)'].cumsum()
        
        fig_pnl = go.Figure()
        bar_colors = ['#00cc96' if v > 0 else '#ff4b4b' for v in bt_results['P/L(pips)']]
        fig_pnl.add_trace(go.Bar(x=bt_results['決済日時'], y=bt_results['P/L(pips)'], name='単独損益', marker_color=bar_colors, opacity=0.6))
        fig_pnl.add_trace(go.Scatter(x=bt_results['決済日時'], y=bt_results['Cumulative_PL'], mode='lines+markers', name='累積損益', line=dict(color='yellow', width=3)))
        
        fig_pnl.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=30, b=20), xaxis=dict(title="決済日時", type='category'))
        st.plotly_chart(fig_pnl, use_container_width=True)
        st.dataframe(bt_results, hide_index=True, use_container_width=True)
    else:
        st.info(f"過去72時間以内に条件(確率{entry_threshold}%以上)を満たすエントリーポイントはありませんでした。")

except Exception as e:
    st.error(f"エラーが発生しました: {e}")
