# -*- coding: utf-8 -*-
"""Unit tests for analyze_monthly_trend tool handler."""
import pytest
import pandas as pd
from unittest.mock import patch


def _make_daily_df(n_days: int) -> pd.DataFrame:
    """Build synthetic daily OHLCV DataFrame."""
    import numpy as np
    dates = pd.date_range(end="2026-05-01", periods=n_days, freq="B")
    close = 10 + np.cumsum(np.random.randn(n_days) * 0.1)
    close = close.clip(min=1)
    df = pd.DataFrame({
        "date": dates,
        "open": close * 0.99,
        "high": close * 1.01,
        "low": close * 0.98,
        "close": close,
        "volume": np.random.randint(1_000_000, 5_000_000, n_days),
    })
    return df


class TestAnalyzeMonthlyTrend:
    def test_sufficient_data_returns_summary(self):
        """36+ monthly bars → full result with monthly_summary."""
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        df = _make_daily_df(1200)  # ~5 years
        with patch("src.agent.tools.analysis_tools._load_df_for_monthly", return_value=(df, "mock")):
            result = _handle_analyze_monthly_trend("600519")
        assert "monthly_summary" in result
        assert "monthly_ma_alignment" in result
        assert "is_monthly_bullish" in result
        assert "monthly_bars" in result
        assert result["monthly_bars"] >= 36

    def test_insufficient_data_degrades_gracefully(self):
        """6-35 monthly bars → returns degraded=True, still has monthly_summary."""
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        df = _make_daily_df(300)  # ~14 months
        with patch("src.agent.tools.analysis_tools._load_df_for_monthly", return_value=(df, "mock")):
            result = _handle_analyze_monthly_trend("600519")
        assert result.get("degraded") is True
        assert "monthly_summary" in result

    def test_no_data_returns_error(self):
        """None DataFrame → error key present."""
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        with patch("src.agent.tools.analysis_tools._load_df_for_monthly", return_value=(None, "mock")):
            result = _handle_analyze_monthly_trend("600519")
        assert "error" in result

    def test_summary_contains_chinese_direction(self):
        """monthly_summary must contain a Chinese keyword."""
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        df = _make_daily_df(1500)
        with patch("src.agent.tools.analysis_tools._load_df_for_monthly", return_value=(df, "mock")):
            result = _handle_analyze_monthly_trend("600519")
        summary = result.get("monthly_summary", "")
        assert any(kw in summary for kw in ["月线", "月K", "趋势", "数据"])
