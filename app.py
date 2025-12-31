import streamlit as st
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
from scipy.stats import norm
import datetime
import pytz

# --- ページ設定 ---
st.set_page_config(page_title="USD/JPY AI確率予測", layout="wide")
st.title('📈 USD/JPY AI確率予測モニター')

# --- サイドバー：更新ボタン ---
st.sidebar.header("操作盤")
if st.sidebar.button('🔄 データを最新に更新'):
    st.rerun()

st.sidebar.markdown("""
**表示の見方**
- **上昇確率**: 現在の価格より上がる確率
- **60%以上**: 買いのチャンス (緑)
- **40%以下**: 売りのチャンス (赤)
""")

# --- 関数: 上昇確率の計算ロジック ---
def calculate_probability(current_price, predicted_price, lower_bound, upper_bound):
    # Prophetの80%信頼区間から標準偏差(sigma)を逆算
    # 信頼区間幅 = 2.56 * sigma (正規分布近似)
    sigma = (upper_bound - lower_bound) / 2.56
    
    if sigma == 0:
        return 50.0
        
    # Zスコア計算 (予測値が現在値からどれくらい離れているか)
    z_score = (predicted_price - current_price) / sigma
    
    # 累積分布関数で確率を算出(%)
    prob_up = norm.cdf(z_score) * 100
    return prob_up

# --- メイン処理 ---
ticker = "USDJPY=X"

try:
    # 1. データ取得 (過去2年分、1時間足)
    with st.spinner(f'{ticker} の最新データを取得中...'):
        data = yf.download(ticker, period="2y", interval="1h")
    
    if data.empty:
        st.error("データの取得に失敗しました。時間をおいて再試行してください。")
        st.stop()

    # データの整形
    df = data.reset_index()
    # カラム名のゆらぎ吸収
    if 'Date' in df.columns:
        df = df.rename(columns={'Date': 'ds', 'Close': 'y'})
    elif 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'ds', 'Close': 'y'})
    
    df = df[['ds', 'y']]
    # タイムゾーン削除（Prophet用）
    df['ds'] = pd.to_datetime(df['ds']).dt.tz_localize(None)

    # 最新価格の取得
    latest_close = df['y'].iloc[-1]
    latest_time = df['ds'].iloc[-1]

    # --- 2. 画面トップ：現在レート表示 ---
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
    with st.spinner('AIが未来を計算中... (確率算出)'):
        # モデル設定: ドル円の特性に合わせて調整
        m = Prophet(
            changepoint_prior_scale=0.05, # トレンド変化への感度
            daily_seasonality=True,       # 1日の時間帯による癖
            weekly_seasonality=True,      # 曜日の癖
            yearly_seasonality=False
        )
        m.fit(df)

        # 未来24時間分の枠を作成
        future = m.make_future_dataframe(periods=24, freq='H')
        forecast = m.predict(future)

    # --- 4. 確率判定テーブルの作成 ---
    st.subheader('🎯 未来の上昇・下落確率 (現在価格比)')

    # 現在時刻より未来の予測データだけを取り出す
    future_forecast = forecast[forecast['ds'] > latest_time].copy()
    
    # チェックしたい時間（1, 4, 8, 24時間後）
    targets = [1, 4, 8, 24]
    results = []

    for h in targets:
        # データが存在するか確認
        if len(future_forecast) >= h:
            row = future_forecast.iloc[h-1] # indexは0始まりなので-1
            
            pred_val = row['yhat']
            lower = row['yhat_lower']
            upper = row['yhat_upper']
            target_time = row['ds']

            # 確率計算
            prob_up = calculate_probability(latest_close, pred_val, lower, upper)
            prob_down = 100 - prob_up

            # 判定と色付け
            trend = "➡️ レンジ"
            
            if prob_up >= 60:
                trend = "↗️ 上昇優勢"
            elif prob_down >= 60:
                trend = "↘️ 下落優勢"

            # 結果リストに追加
            results.append({
                "対象": f"{h}時間後",
                "予測日時": target_time.strftime('%m/%d %H:%M'),
                "現在価格": f"{latest_close:.3f}",
                "予測価格": f"{pred_val:.3f}",
                "上昇確率": f"{prob_up:.1f} %",
                "下落確率": f"{prob_down:.1f} %",
                "判定": trend
            })

    # 表を表示
    st.table(pd.DataFrame(results).set_index("対象"))

    # --- 5. 予測チャートの表示 ---
    st.subheader('📊 予測推移チャート')
    
    fig = plot_plotly(m, forecast)
    
    # 現在価格のライン（白の点線）を追加
    fig.add_hline(
        y=latest_close, 
        line_dash="dash", 
        line_color="white", 
        annotation_text="現在価格", 
        annotation_position="bottom right"
    )

    fig.update_layout(
        title="青線: AI予測 / 水色帯: 予測のブレ幅 / 黒点: 実績",
        yaxis_title="価格 (円)",
        xaxis_title="日時",
        template="plotly_dark",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"予期せぬエラーが発生しました: {e}")