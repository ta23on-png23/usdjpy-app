# --- 乖離判定付き確率計算（修正版） ---
def calculate_reversion_probability(current_price, predicted_price, lower_bound, upper_bound):
    c = to_float(current_price)
    p = to_float(predicted_price)
    l = to_float(lower_bound)
    u = to_float(upper_bound)
    
    # 【修正1】予測の不確実性（幅）が小さすぎる場合の「最低幅」を設定
    # 為替で1時間先に0.05円未満の誤差しかないことは稀なため、ノイズフロアを設けます
    width = u - l
    min_width = 0.10  # 最低でも10銭程度の幅はあると仮定
    adjusted_width = max(width, min_width)
    
    # Sigmaの計算（幅を少し広めに解釈して過信を防ぐ）
    sigma = adjusted_width / 2.0 

    # Zスコアの計算と確率算出
    # 予測値と現在値の差が、想定される変動幅に対してどれくらい大きいか
    if sigma == 0:
        base_prob = 50.0
    else:
        z_score = (p - c) / sigma
        # 【修正2】感度を鈍らせる（係数 0.5 を乗算）
        # Prophetはトレンドを直線的に捉えがちなので、確信度を半分程度に割り引く
        damped_z = z_score * 0.5
        base_prob = norm.cdf(damped_z) * 100

    # 【修正3】乖離補正（Correction）のマイルド化
    # 旧: 最大40% → 新: 最大15% に抑制
    correction = 0.0
    note = "順張り(トレンド追随)"
    
    # バンド幅を再取得（計算用）
    box_width = u - l
    if box_width < 0.01: box_width = 0.01

    if c > u: 
        # 上振れしすぎ → 下がりやすい（確率は下がる）
        excess = c - u
        ratio = excess / box_width
        correction = - (ratio * 20.0) # 係数を40から20へ
        correction = max(correction, -15.0) # 最大補正を-15%でキャップ
        note = f"⚠️上値重め (調整警戒 {correction:.1f}%)"

    elif c < l: 
        # 下振れしすぎ → 上がりやすい（確率は上がる）
        excess = l - c
        ratio = excess / box_width
        correction = + (ratio * 20.0) # 係数を40から20へ
        correction = min(correction, 15.0) # 最大補正を+15%でキャップ
        note = f"⚠️底堅い (反発期待 +{correction:.1f}%)"

    else: 
        # バンド内での位置による微調整
        center = (u + l) / 2
        # 中心からの距離
        dist_from_center = (c - center) / (box_width / 2) if box_width > 0 else 0
        # 中心より上にいれば少し確率を下げる、下にいれば少し上げる（平均回帰）
        minor_correction = dist_from_center * -5.0
        correction += minor_correction

    final_prob = base_prob + correction
    
    # 【修正4】最終的な確率を 30%～70% の範囲に収まりやすくし、極端な値（1%や99%）を排除
    # 為替の短時間予測で90%越えは異常値のため、最大でも85-90%程度に留める
    final_prob = max(10.0, min(90.0, final_prob))
    
    return final_prob, note
