# -*- coding: utf-8 -*-
"""
Analysis tools — wraps StockTrendAnalyzer as an agent-callable tool.

Tools:
- analyze_trend: comprehensive technical trend analysis
"""

import logging
from typing import Optional

from src.agent.tools.registry import ToolParameter, ToolDefinition

logger = logging.getLogger(__name__)


def _fetch_trend_data(stock_code: str):
    """Fetch historical OHLCV (DataFrame) for trend analysis. DB first, then DataFetcher fallback."""
    from src.services.history_loader import load_history_df

    df, _ = load_history_df(stock_code, days=60)
    return df


def _handle_analyze_trend(stock_code: str) -> dict:
    """Run technical trend analysis on a stock."""
    from src.stock_analyzer import StockTrendAnalyzer

    if not (stock_code and str(stock_code).strip()):
        return {"error": "stock_code is required"}

    df = _fetch_trend_data(stock_code)
    if df is None or df.empty:
        return {"error": f"No historical data available for trend analysis on {stock_code}"}

    if len(df) < 20:
        return {"error": f"Insufficient data for trend analysis on {stock_code} (need >= 20 days)"}

    analyzer = StockTrendAnalyzer()
    try:
        result = analyzer.analyze(df, stock_code)
    except Exception:
        logger.warning("analyze_trend(%s): Trend analysis failed", stock_code, exc_info=True)
        return {"error": f"Trend analysis failed for {stock_code}"}

    return {
        "code": result.code,
        "trend_status": result.trend_status.value,
        "ma_alignment": result.ma_alignment,
        "trend_strength": result.trend_strength,
        "ma5": result.ma5,
        "ma10": result.ma10,
        "ma20": result.ma20,
        "ma60": result.ma60,
        "current_price": result.current_price,
        "bias_ma5": round(result.bias_ma5, 2),
        "bias_ma10": round(result.bias_ma10, 2),
        "bias_ma20": round(result.bias_ma20, 2),
        "volume_status": result.volume_status.value,
        "volume_ratio_5d": round(result.volume_ratio_5d, 2),
        "volume_trend": result.volume_trend,
        "support_ma5": result.support_ma5,
        "support_ma10": result.support_ma10,
        "resistance_levels": result.resistance_levels,
        "support_levels": result.support_levels,
        "macd_dif": round(result.macd_dif, 4),
        "macd_dea": round(result.macd_dea, 4),
        "macd_bar": round(result.macd_bar, 4),
        "macd_status": result.macd_status.value,
        "macd_signal": result.macd_signal,
        "rsi_6": round(result.rsi_6, 2),
        "rsi_12": round(result.rsi_12, 2),
        "rsi_24": round(result.rsi_24, 2),
        "rsi_status": result.rsi_status.value,
        "rsi_signal": result.rsi_signal,
        "buy_signal": result.buy_signal.value,
        "signal_score": result.signal_score,
        "signal_reasons": result.signal_reasons,
        "risk_factors": result.risk_factors,
    }


analyze_trend_tool = ToolDefinition(
    name="analyze_trend",
    description="Run comprehensive technical trend analysis on a stock. "
                "Fetches historical data from database or data source. "
                "Returns MA alignment, bias rates, MACD status, RSI levels, "
                "volume analysis, support/resistance levels, and a buy/sell signal "
                "with a score (0-100).",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code to analyze, e.g., '600519'",
        ),
    ],
    handler=_handle_analyze_trend,
    category="analysis",
)


# ============================================================
# calculate_ma — flexible moving average calculator
# ============================================================

def _handle_calculate_ma(stock_code: str, periods: Optional[str] = None, days: int = 120) -> dict:
    """Calculate moving averages for arbitrary periods from historical K-line data."""
    from src.services.history_loader import load_history_df

    df, source = load_history_df(stock_code, days=days)

    if df is None or df.empty:
        return {"error": f"No historical data for {stock_code}"}

    # Parse requested periods (default: 5,10,20,30,60,120,250)
    default_periods = [5, 10, 20, 30, 60, 120, 250]
    if periods:
        try:
            requested = [int(p.strip()) for p in periods.split(",") if p.strip().isdigit()]
            period_list = sorted(set(requested)) if requested else default_periods
        except Exception:
            period_list = default_periods
    else:
        period_list = default_periods

    close = df["close"]
    current_price = float(close.iloc[-1])
    result: dict = {
        "code": stock_code,
        "source": source,
        "current_price": round(current_price, 2),
        "data_points": len(df),
        "ma": {},
    }

    for period in period_list:
        if len(close) < period:
            result["ma"][f"ma{period}"] = None
            continue
        ma_val = float(close.rolling(window=period).mean().iloc[-1])
        bias = round((current_price - ma_val) / ma_val * 100, 2) if ma_val else None
        result["ma"][f"ma{period}"] = {
            "value": round(ma_val, 2),
            "bias_pct": bias,
            "price_above": current_price > ma_val,
        }

    # Summary: how many MAs is the price above?
    ma_values = [v for v in result["ma"].values() if v is not None]
    above_count = sum(1 for v in ma_values if v["price_above"])
    result["above_ma_count"] = above_count
    result["total_ma_count"] = len(ma_values)
    result["ma_alignment"] = (
        "多头排列" if above_count == len(ma_values)
        else "空头排列" if above_count == 0
        else f"混合({above_count}/{len(ma_values)}条均线上方)"
    )
    return result


calculate_ma_tool = ToolDefinition(
    name="calculate_ma",
    description="Calculate moving averages (MA5/10/20/30/60/120/250 or custom periods) "
                "for a stock. Returns each MA value, price bias %, and whether price "
                "is above each MA. Also returns overall MA alignment (多头/空头/混合).",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519'",
        ),
        ToolParameter(
            name="periods",
            type="string",
            description="Comma-separated MA periods to calculate (default: '5,10,20,30,60,120,250'). "
                        "E.g., '5,10,20,60'",
            required=False,
            default="5,10,20,30,60,120,250",
        ),
        ToolParameter(
            name="days",
            type="integer",
            description="Number of trading days to fetch history for (default: 120)",
            required=False,
            default=120,
        ),
    ],
    handler=_handle_calculate_ma,
    category="analysis",
)


# ============================================================
# get_volume_analysis — volume-price relationship analysis
# ============================================================

def _handle_get_volume_analysis(stock_code: str, days: int = 30) -> dict:
    """Analyse volume-price patterns over recent trading days."""
    from src.services.history_loader import load_history_df
    import pandas as pd

    df, source = load_history_df(stock_code, days=max(days + 20, 60))

    if df is None or df.empty:
        return {"error": f"No historical data for {stock_code}"}

    df = df.tail(days).copy()
    if len(df) < 5:
        return {"error": f"Insufficient data for volume analysis (got {len(df)} days, need >= 5)"}

    close = df["close"]
    volume = df["volume"]

    # Average volumes
    avg_vol_5 = float(volume.tail(5).mean())
    avg_vol_10 = float(volume.tail(10).mean())
    avg_vol_20 = float(volume.tail(20).mean()) if len(df) >= 20 else avg_vol_10
    latest_vol = float(volume.iloc[-1])
    vol_ratio_5d = round(latest_vol / avg_vol_5, 2) if avg_vol_5 > 0 else None
    vol_ratio_20d = round(latest_vol / avg_vol_20, 2) if avg_vol_20 > 0 else None

    # Price direction for each day
    price_up = close.diff() > 0  # True = up day

    # Volume-price correlation (last N days)
    try:
        import numpy as np
        vp_corr = float(pd.Series(volume.values, dtype=float).corr(pd.Series(close.values, dtype=float)))
        vp_corr = round(vp_corr, 3)
    except Exception:
        vp_corr = None

    # Detect shrinking volume on up days (bearish divergence) vs expanding on up days (healthy)
    up_days = df[price_up]
    down_days = df[~price_up]
    avg_up_vol = float(up_days["volume"].mean()) if len(up_days) > 0 else 0
    avg_down_vol = float(down_days["volume"].mean()) if len(down_days) > 0 else 0

    # Volume trend: compare last 5 days vs prior 5 days
    if len(volume) >= 10:
        recent_5_avg = float(volume.tail(5).mean())
        prior_5_avg = float(volume.iloc[-10:-5].mean())
        vol_trend_pct = round((recent_5_avg - prior_5_avg) / prior_5_avg * 100, 1) if prior_5_avg > 0 else 0
        vol_trend = "放量" if vol_trend_pct > 20 else "缩量" if vol_trend_pct < -20 else "量能平稳"
    else:
        vol_trend_pct = 0
        vol_trend = "数据不足"

    # High-volume days (> 2x 20d avg)
    high_vol_days = int((volume > avg_vol_20 * 2).sum()) if avg_vol_20 > 0 else 0

    # Volume-price pattern interpretation
    pattern = "未知"
    if avg_up_vol > avg_down_vol * 1.3:
        pattern = "量价配合良好（上涨放量、下跌缩量）"
    elif avg_down_vol > avg_up_vol * 1.3:
        pattern = "量价背离（下跌放量、上涨缩量，偏空）"
    elif vol_ratio_5d and vol_ratio_5d > 1.5:
        pattern = "近期明显放量"
    elif vol_ratio_5d and vol_ratio_5d < 0.6:
        pattern = "近期明显缩量"
    else:
        pattern = "量价关系中性"

    return {
        "code": stock_code,
        "source": source,
        "period_days": len(df),
        "latest_volume": latest_vol,
        "avg_volume_5d": round(avg_vol_5, 0),
        "avg_volume_20d": round(avg_vol_20, 0),
        "volume_ratio_vs_5d": vol_ratio_5d,
        "volume_ratio_vs_20d": vol_ratio_20d,
        "avg_up_day_volume": round(avg_up_vol, 0),
        "avg_down_day_volume": round(avg_down_vol, 0),
        "volume_trend": vol_trend,
        "volume_trend_pct": vol_trend_pct,
        "high_volume_days": high_vol_days,
        "volume_price_corr": vp_corr,
        "pattern": pattern,
    }


get_volume_analysis_tool = ToolDefinition(
    name="get_volume_analysis",
    description="Analyse volume-price relationship for a stock. Returns volume ratios, "
                "average volume on up vs down days, volume trend (expanding/shrinking), "
                "and pattern interpretation (量价配合/背离). Useful for confirming trend "
                "strength and detecting distribution or accumulation phases.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519'",
        ),
        ToolParameter(
            name="days",
            type="integer",
            description="Number of recent trading days to analyse (default: 30)",
            required=False,
            default=30,
        ),
    ],
    handler=_handle_get_volume_analysis,
    category="analysis",
)


# ============================================================
# analyze_pattern — candlestick / chart pattern recognition
# ============================================================

def _handle_analyze_pattern(stock_code: str, days: int = 60) -> dict:
    """Detect common candlestick and chart patterns in recent price history."""
    from src.services.history_loader import load_history_df

    df, source = load_history_df(stock_code, days=max(days, 120))

    if df is None or df.empty:
        return {"error": f"No historical data for {stock_code}"}

    df = df.tail(days).copy().reset_index(drop=True)
    if len(df) < 10:
        return {"error": f"Insufficient data for pattern analysis (got {len(df)} days, need >= 10)"}

    o = df["open"].values
    h = df["high"].values
    l = df["low"].values   # noqa: E741
    c = df["close"].values
    v = df["volume"].values if "volume" in df.columns else None

    patterns_detected = []
    n = len(c)

    # ---- Helpers ----
    def body(i):
        return abs(c[i] - o[i])

    def upper_shadow(i):
        return h[i] - max(c[i], o[i])

    def lower_shadow(i):
        return min(c[i], o[i]) - l[i]

    def is_bullish(i):
        return c[i] > o[i]

    def is_bearish(i):
        return c[i] < o[i]

    avg_body = sum(body(i) for i in range(n)) / n if n > 0 else 1

    # --- Single-candle patterns (last 3 days) ---
    for i in range(max(0, n - 3), n):
        bd = body(i)
        us = upper_shadow(i)
        ls = lower_shadow(i)

        # Doji
        if bd < avg_body * 0.1 and (us + ls) > bd * 3:
            patterns_detected.append({
                "pattern": "十字星 (Doji)", "type": "reversal_signal",
                "day_offset": -(n - 1 - i),
                "strength": "弱", "desc": "多空平衡，可能变盘信号"
            })

        # Hammer / Hanging Man
        if ls > body(i) * 2 and us < body(i) * 0.5:
            label = "锤子线 (Hammer)" if i == 0 or c[i] >= c[i - 1] else "上吊线 (Hanging Man)"
            patterns_detected.append({
                "pattern": label, "type": "reversal_signal",
                "day_offset": -(n - 1 - i),
                "strength": "中", "desc": "下影线长，潜在支撑/反转"
            })

        # Shooting Star / Inverted Hammer
        if us > body(i) * 2 and ls < body(i) * 0.5:
            label = "流星线 (Shooting Star)" if is_bearish(i) else "倒锤子"
            patterns_detected.append({
                "pattern": label, "type": "bearish_signal",
                "day_offset": -(n - 1 - i),
                "strength": "中", "desc": "上影线长，潜在压力/反转"
            })

        # Big bullish / bearish candle
        if bd > avg_body * 2.5:
            label = "大阳线" if is_bullish(i) else "大阴线"
            t = "bullish" if is_bullish(i) else "bearish"
            patterns_detected.append({
                "pattern": label, "type": t,
                "day_offset": -(n - 1 - i),
                "strength": "强", "desc": "实体大，方向明确"
            })

    # --- Multi-candle patterns (use last 10 days) ---
    if n >= 3:
        i = n - 1
        # Morning Star (早晨之星) — bottom reversal
        if (is_bearish(i - 2) and body(i - 2) > avg_body * 1.5
                and body(i - 1) < avg_body * 0.4
                and is_bullish(i) and body(i) > avg_body * 1.5
                and c[i] > (o[i - 2] + c[i - 2]) / 2):
            patterns_detected.append({
                "pattern": "早晨之星 (Morning Star)", "type": "bullish_reversal",
                "day_offset": -2, "strength": "强", "desc": "三根K线底部反转形态"
            })

        # Evening Star (黄昏之星) — top reversal
        if (is_bullish(i - 2) and body(i - 2) > avg_body * 1.5
                and body(i - 1) < avg_body * 0.4
                and is_bearish(i) and body(i) > avg_body * 1.5
                and c[i] < (o[i - 2] + c[i - 2]) / 2):
            patterns_detected.append({
                "pattern": "黄昏之星 (Evening Star)", "type": "bearish_reversal",
                "day_offset": -2, "strength": "强", "desc": "三根K线顶部反转形态"
            })

        # Engulfing (吞没形态)
        if (is_bullish(i) and is_bearish(i - 1)
                and o[i] < c[i - 1] and c[i] > o[i - 1]):
            patterns_detected.append({
                "pattern": "看涨吞没 (Bullish Engulfing)", "type": "bullish_reversal",
                "day_offset": -1, "strength": "强", "desc": "阳线完全覆盖前一阴线"
            })
        elif (is_bearish(i) and is_bullish(i - 1)
              and o[i] > c[i - 1] and c[i] < o[i - 1]):
            patterns_detected.append({
                "pattern": "看跌吞没 (Bearish Engulfing)", "type": "bearish_reversal",
                "day_offset": -1, "strength": "强", "desc": "阴线完全覆盖前一阳线"
            })

    # --- Chart patterns over the window ---
    # Double bottom detection (简化版: 两个相近低点 + 中间高点)
    recent_lows_idx = sorted(range(n), key=lambda i: l[i])[:5]
    if len(recent_lows_idx) >= 2:
        lo1, lo2 = sorted(recent_lows_idx[:2])
        if lo2 - lo1 >= 5 and abs(l[lo1] - l[lo2]) / max(l[lo1], l[lo2]) < 0.03:
            mid_high = max(h[lo1:lo2 + 1])
            if mid_high > l[lo1] * 1.03:
                patterns_detected.append({
                    "pattern": "双底 (Double Bottom)", "type": "bullish_reversal",
                    "day_offset": -(n - 1 - lo2),
                    "strength": "强", "desc": "两个相近低点，W型底部形态"
                })

    # Upward breakout: closes above 20d high (excluding last day itself)
    if n >= 21:
        high_20d = max(h[n - 21:n - 1])
        if c[-1] > high_20d and (v is None or v[-1] > sum(v[n - 6:n - 1]) / 5 * 1.5):
            patterns_detected.append({
                "pattern": "放量突破20日高点", "type": "bullish_breakout",
                "day_offset": 0, "strength": "强", "desc": "收盘突破近20日最高，量能配合"
            })

    # Price in consolidation box (box oscillation)
    if n >= 10:
        recent_high = max(h[n - 10:])
        recent_low = min(l[n - 10:])
        box_range_pct = (recent_high - recent_low) / recent_low * 100 if recent_low > 0 else 0
        if box_range_pct < 8:
            patterns_detected.append({
                "pattern": "箱体震荡", "type": "consolidation",
                "day_offset": 0, "strength": "中",
                "desc": f"近10日波幅 {box_range_pct:.1f}%，价格在区间内震荡"
            })

    # Deduplicate by pattern name, keep most recent
    seen = set()
    unique_patterns = []
    for p in reversed(patterns_detected):
        if p["pattern"] not in seen:
            seen.add(p["pattern"])
            unique_patterns.append(p)
    unique_patterns = list(reversed(unique_patterns))

    return {
        "code": stock_code,
        "source": source,
        "period_days": len(df),
        "current_price": round(float(c[-1]), 2),
        "patterns_count": len(unique_patterns),
        "patterns": unique_patterns,
        "summary": (
            "未发现明显形态" if not unique_patterns
            else "、".join(p["pattern"] for p in unique_patterns)
        ),
    }


analyze_pattern_tool = ToolDefinition(
    name="analyze_pattern",
    description="Detect candlestick and chart patterns in recent price history. "
                "Identifies: Doji, Hammer, Shooting Star, Morning/Evening Star, Engulfing, "
                "Double Bottom, upward breakout, box oscillation, and more. "
                "Returns pattern list with type (bullish/bearish/reversal) and strength.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519'",
        ),
        ToolParameter(
            name="days",
            type="integer",
            description="Number of recent trading days to scan (default: 60)",
            required=False,
            default=60,
        ),
    ],
    handler=_handle_analyze_pattern,
    category="analysis",
)


# ============================================================
# analyze_weekly_trend — weekly timeframe validation
# ============================================================

def _handle_analyze_weekly_trend(stock_code: str) -> dict:
    """Derive weekly K-line trend signals by resampling daily data."""
    from src.services.history_loader import load_history_df
    import pandas as pd

    df, source = load_history_df(stock_code, days=300)

    if df is None or df.empty:
        return {"error": f"No historical data for {stock_code}"}

    df = df.copy()

    # Ensure datetime index
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return {"error": f"Missing OHLC columns for {stock_code}"}

    vol_col = "volume" if "volume" in df.columns else None
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if vol_col:
        agg["volume"] = "sum"

    weekly = df.resample("W-FRI").agg(agg).dropna(subset=["close"])

    if len(weekly) < 10:
        return {"error": f"Insufficient weekly data for {stock_code} (only {len(weekly)} weeks)"}

    close = weekly["close"]
    current_price = float(close.iloc[-1])

    # Weekly MAs
    ma5w = float(close.rolling(5).mean().iloc[-1]) if len(close) >= 5 else None
    ma10w = float(close.rolling(10).mean().iloc[-1]) if len(close) >= 10 else None
    ma20w = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else None

    # MA alignment
    ma_values = [v for v in [ma5w, ma10w, ma20w] if v is not None]
    above_count = sum(1 for v in ma_values if current_price > v)
    total_mas = len(ma_values)

    if total_mas == 0:
        weekly_ma_alignment = "数据不足"
        is_weekly_bullish = None
    elif above_count == total_mas:
        weekly_ma_alignment = "多头排列"
        is_weekly_bullish = True
    elif above_count == 0:
        weekly_ma_alignment = "空头排列"
        is_weekly_bullish = False
    else:
        weekly_ma_alignment = f"混合({above_count}/{total_mas}均线上方)"
        is_weekly_bullish = above_count > total_mas / 2

    # Weekly RSI-14
    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi_series = 100 - (100 / (1 + rs))
    rsi_val = rsi_series.iloc[-1]
    weekly_rsi = float(rsi_val) if pd.notna(rsi_val) else None

    # Weekly MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    weekly_dif = float(dif.iloc[-1])
    weekly_dea = float(dea.iloc[-1])
    weekly_macd_golden = weekly_dif > weekly_dea

    # 5-week price change
    trend_pct = None
    if len(close) >= 6:
        trend_pct = round(
            (float(close.iloc[-1]) - float(close.iloc[-6])) / float(close.iloc[-6]) * 100, 1
        )

    # Plain-language summary
    rsi_desc = ""
    if weekly_rsi is not None:
        if weekly_rsi > 70:
            rsi_desc = "，周RSI超买区"
        elif weekly_rsi < 30:
            rsi_desc = "，周RSI超卖区（关注反弹）"

    macd_cross = "MACD金叉" if weekly_macd_golden else "MACD死叉"

    if is_weekly_bullish is True:
        summary = f"周线{weekly_ma_alignment}，{macd_cross}{rsi_desc}，中期趋势向上"
    elif is_weekly_bullish is False:
        summary = f"周线{weekly_ma_alignment}，{macd_cross}{rsi_desc}，中期趋势向下"
    else:
        summary = f"周线数据不足，{macd_cross}{rsi_desc}，暂无明确中期趋势判断"

    return {
        "code": stock_code,
        "source": source,
        "weekly_bars": len(weekly),
        "current_price": round(current_price, 2),
        "ma5_weekly": round(ma5w, 2) if ma5w is not None else None,
        "ma10_weekly": round(ma10w, 2) if ma10w is not None else None,
        "ma20_weekly": round(ma20w, 2) if ma20w is not None else None,
        "weekly_ma_alignment": weekly_ma_alignment,
        "is_weekly_bullish": is_weekly_bullish,
        "weekly_rsi": round(weekly_rsi, 1) if weekly_rsi is not None else None,
        "weekly_macd_golden_cross": weekly_macd_golden,
        "weekly_dif": round(weekly_dif, 4),
        "weekly_dea": round(weekly_dea, 4),
        "weekly_trend_pct_5w": trend_pct,
        "weekly_summary": summary,
    }


analyze_weekly_trend_tool = ToolDefinition(
    name="analyze_weekly_trend",
    description="Derive weekly K-line trend signals by resampling daily data into weekly bars. "
                "Returns weekly MA alignment (multi-head/bearish/mixed), RSI, MACD golden/dead "
                "cross, and a plain-Chinese summary. Use to validate whether the daily signal "
                "aligns with the medium-term weekly trend. Always call this alongside analyze_trend.",
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519'",
        ),
    ],
    handler=_handle_analyze_weekly_trend,
    category="analysis",
)


# ============================================================
# analyze_monthly_trend — monthly timeframe validation
# ============================================================

def _load_df_for_monthly(stock_code: str):
    """Load ≥3 years of daily history for monthly resampling."""
    from src.services.history_loader import load_history_df
    return load_history_df(stock_code, days=1500)


def _handle_analyze_monthly_trend(stock_code: str) -> dict:
    """Derive monthly K-line trend signals by resampling daily data.

    Degradation policy:
    - >= 36 monthly bars: full analysis with degraded=False
    - 6-35 bars: analysis with degraded=True (data note appended to summary)
    - < 6 bars: minimal summary only, sets degraded=True
    - No data / missing columns: returns {"error": ...}
    """
    import pandas as pd

    df, source = _load_df_for_monthly(stock_code)

    if df is None or (hasattr(df, "empty") and df.empty):
        return {"error": f"No historical data for {stock_code}"}

    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
    else:
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

    required = {"open", "high", "low", "close"}
    if not required.issubset(set(df.columns)):
        return {"error": f"Missing OHLC columns for {stock_code}"}

    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"

    monthly = df.resample("ME").agg(agg).dropna(subset=["close"])
    n_bars = len(monthly)
    degraded = n_bars < 36

    if n_bars < 6:
        return {
            "code": stock_code,
            "monthly_bars": n_bars,
            "degraded": True,
            "monthly_summary": "月线数据不足6根，无法判断长期趋势",
        }

    close = monthly["close"]
    current_price = float(close.iloc[-1])

    # Monthly MAs (3/6/12 months)
    ma3m = float(close.rolling(3).mean().iloc[-1]) if n_bars >= 3 else None
    ma6m = float(close.rolling(6).mean().iloc[-1]) if n_bars >= 6 else None
    ma12m = float(close.rolling(12).mean().iloc[-1]) if n_bars >= 12 else None

    ma_values = [v for v in [ma3m, ma6m, ma12m] if v is not None]
    above_count = sum(1 for v in ma_values if current_price > v)
    total_mas = len(ma_values)

    if total_mas == 0:
        monthly_ma_alignment = "数据不足"
        is_monthly_bullish = None
    elif above_count == total_mas:
        monthly_ma_alignment = "多头排列"
        is_monthly_bullish = True
    elif above_count == 0:
        monthly_ma_alignment = "空头排列"
        is_monthly_bullish = False
    else:
        monthly_ma_alignment = f"混合({above_count}/{total_mas}均线上方)"
        is_monthly_bullish = above_count > total_mas / 2

    # Monthly RSI-14
    delta = close.diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, float("inf"))
    rsi_series = 100 - (100 / (1 + rs))
    rsi_val = rsi_series.iloc[-1]
    monthly_rsi = float(rsi_val) if pd.notna(rsi_val) else None

    # Monthly MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    monthly_dif = float(dif.iloc[-1])
    monthly_dea = float(dea.iloc[-1])
    monthly_macd_golden = monthly_dif > monthly_dea

    # 6-month price change
    trend_pct_6m = None
    if n_bars >= 7:
        trend_pct_6m = round(
            (float(close.iloc[-1]) - float(close.iloc[-7])) / float(close.iloc[-7]) * 100, 1
        )

    # Plain-language summary
    rsi_desc = ""
    if monthly_rsi is not None:
        if monthly_rsi > 70:
            rsi_desc = "，月RSI超买区（长期高位）"
        elif monthly_rsi < 30:
            rsi_desc = "，月RSI超卖区（长期低位，关注价值回归）"

    macd_cross = "MACD金叉" if monthly_macd_golden else "MACD死叉"
    data_note = "（数据有限，仅供参考）" if degraded else ""

    if is_monthly_bullish is True:
        summary = f"月线{monthly_ma_alignment}，{macd_cross}{rsi_desc}，长期趋势向上{data_note}"
    elif is_monthly_bullish is False:
        summary = f"月线{monthly_ma_alignment}，{macd_cross}{rsi_desc}，长期趋势向下{data_note}"
    else:
        summary = f"月线数据有限，{macd_cross}{rsi_desc}，暂无明确长期趋势判断{data_note}"

    return {
        "code": stock_code,
        "source": source,
        "monthly_bars": n_bars,
        "degraded": degraded,
        "current_price": round(current_price, 2),
        "ma3_monthly": round(ma3m, 2) if ma3m is not None else None,
        "ma6_monthly": round(ma6m, 2) if ma6m is not None else None,
        "ma12_monthly": round(ma12m, 2) if ma12m is not None else None,
        "monthly_ma_alignment": monthly_ma_alignment,
        "is_monthly_bullish": is_monthly_bullish,
        "monthly_rsi": round(monthly_rsi, 1) if monthly_rsi is not None else None,
        "monthly_macd_golden_cross": monthly_macd_golden,
        "monthly_dif": round(monthly_dif, 4),
        "monthly_dea": round(monthly_dea, 4),
        "monthly_trend_pct_6m": trend_pct_6m,
        "monthly_summary": summary,
    }


analyze_monthly_trend_tool = ToolDefinition(
    name="analyze_monthly_trend",
    description=(
        "Derive monthly K-line trend signals by resampling daily data into monthly bars. "
        "Returns monthly MA alignment (MA3/6/12), RSI, MACD golden/dead cross, and a "
        "plain-Chinese summary. Use to validate the long-term trend direction. "
        "Complements analyze_trend (daily) and analyze_weekly_trend (weekly) for "
        "three-timeframe resonance analysis. Degrades gracefully when data < 36 months."
    ),
    parameters=[
        ToolParameter(
            name="stock_code",
            type="string",
            description="Stock code, e.g., '600519'",
        ),
    ],
    handler=_handle_analyze_monthly_trend,
    category="analysis",
)


ALL_ANALYSIS_TOOLS = [
    analyze_trend_tool,
    analyze_weekly_trend_tool,
    analyze_monthly_trend_tool,
    calculate_ma_tool,
    get_volume_analysis_tool,
    analyze_pattern_tool,
]
