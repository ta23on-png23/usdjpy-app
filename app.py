import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from scipy.stats import norm
import datetime

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

# --- 関数: 上昇確率の計算 ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    # 値を強制的に「数値(float)」に変換してエラーを防ぐ
    try:
        current_price = float(current_price)
        predicted_price = float(predicted_price)
        lower_bound = float(lower_bound)
        upper_bound = float(upper_bound)
    except:
        return 50.0

    sigma = (upper_bound - lower_bound) / 2.56
    
    if sigma == 0:
        return 50.0
        
    z_score = (predicted_price - current_price) / sigma
    prob_up = norm.cdf(z_score) * 100
    return prob_up

# --- メイン処理 ---
ticker = "USDJPY=X"

try:
    # 1. データ取得
    with st.spinner(f'{ticker} の最新データを取得中...'):
        raw_data = yf.download(ticker, period="2y", interval="1h")
    
    if raw_data.empty:
        st.error("データの取得に失敗しました。")
        st.stop()

    # --- 【重要】データ構造の強力なクリーニング ---
    # MultiIndexカラム（2段組みの列名）になっている場合、1段目に平坦化する
    if isinstance(raw_data.columns, pd.MultiIndex):
        raw_data.columns = raw_data.columns.get_level_values(0)

    # 'Close'列だけを取り出し、余計な列を削除
    if 'Close' in raw_data.columns:
        df = raw_data[['Close']].copy()
    else:
        # Closeが見つからない場合、2列目を強制的に採用（1列目はOpenの可能性があるため）
        df = raw_data.iloc[:, [0]].copy() # 安全策として1列目を採用

    # インデックス（日時）を列に戻す
    df = df.reset_index()

    # カラム名を強制的に ['ds', 'y'] に変更する（名前が何であれ）
    # dfの列は [Date/Datetime, Close] の順になっているはず
    df.columns = ['ds', 'y']
    
    # タイムゾーン情報の削除
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    # --- 【修正】値の取り出し方を変更（.item()を使用して確実に数値にする） ---
    latest_close_series = df['y'].iloc[-1]
    # Series型なら値を取り出す、そうでなければそのまま
    if hasattr(latest_close_series, 'item'):
        latest_close = latest_close_series.item()
    else:
        latest_close = float(latest_close_series)
        
    latest_time = df['ds'].iloc[-1]

    # --- 2. 画面トップ表示 ---
    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric(
            label="現在レート (直近終値)",
            value=f"{latest_close:.3f} 円",
            delta="最新更新"
        )
    with col2:
        st.info(f"最終データ日時: {latest_time.strftime('%Y/%m/%d %H:%M')}")

    # --- 3. AI学習と予測 ---
    with st.spinner('AIが未来を計算中...'):
        m = Prophet(
            changepoint_prior_scale=0.05,
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=False
        )
        m.fit(df)

        future = m.make_future_dataframe(periods=24, freq='H')
        forecast = m.predict(future)

    # --- 4. 確率判定テーブル ---
    st.subheader('🎯 未来の上昇・下落確率 (現在価格比)')

    future_forecast = forecast[forecast['ds'] > latest_time].copy()
    targets = [1, 4, 8, 24]
    results = []

    for h in targets:
        if len(future_forecast) >= h:
            row = future_forecast.iloc[h-1]
            
            # 確実に数値化
            pred_val = float(row['yhat'])
            lower = float(row['yhat_lower'])
            upper = float(row['yhat_upper'])
            target_time = row['ds']

            # 確率計算
            prob_up = calculate_probability(latest_close, pred_val, lower, upper)
            prob_down = 100 - prob_up

            # 判定
            trend = "➡️ レンジ"
            if prob_up >= 60:
                trend = "↗️ 上昇優勢"
            elif prob_down >= 60:
                trend = "↘️ 下落優勢"

            results.append({
                "対象": f"{h}時間後",
                "予測日時": target_time.strftime('%m/%d %H:%M'),
                "現在価格": f"{latest_close:.3f}",
                "予測価格": f"{pred_val:.3f}",
                "上昇確率": f"{prob_up:.1f} %",
                "下落確率": f"{prob_down:.1f} %",
                "判定": trend
            })

    st.table(pd.DataFrame(results).set_index("対象"))

    # --- 5. グラフ表示 ---
    st.subheader('📊 予測推移チャート')
    fig = plot_plotly(m, forecast)
    fig.add_hline(y=latest_close, line_dash="dash", line_color="white", annotation_text="現在価格")
    fig.update_layout(
        title="青線: AI予測 / 水色帯: 予測範囲 / 黒点: 実績",
        yaxis_title="価格 (円)",
        xaxis_title="日時",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"予期せぬエラーが発生しました: {e}")
