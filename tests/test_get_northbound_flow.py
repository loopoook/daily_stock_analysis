# -*- coding: utf-8 -*-
"""Unit tests for get_northbound_flow tool handler."""
import pytest
import pandas as pd
from unittest.mock import patch


def _make_flow_df(net_buy_sh: float, net_buy_sz: float) -> pd.DataFrame:
    """Synthetic northbound flow summary DataFrame using real API column names."""
    return pd.DataFrame([
        {"交易日": "2026-05-09", "板块": "沪股通(港)", "成交净买额": net_buy_sh},
        {"交易日": "2026-05-09", "板块": "深股通(港)", "成交净买额": net_buy_sz},
    ])


class TestGetNorthboundFlow:
    def test_strong_inflow_returns_positive_signal(self):
        """Net inflow > 30 亿 → signal=strong_inflow."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        df = _make_flow_df(25.0, 20.0)  # total 45 亿
        with patch("src.agent.tools.data_tools._fetch_northbound_df", return_value=df):
            result = _handle_get_northbound_flow()
        assert result["signal"] == "strong_inflow"
        assert result["net_total_billion"] > 0

    def test_outflow_returns_negative_signal(self):
        """Net outflow total < -10 亿 → signal=outflow or strong_outflow."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        df = _make_flow_df(-15.0, -10.0)
        with patch("src.agent.tools.data_tools._fetch_northbound_df", return_value=df):
            result = _handle_get_northbound_flow()
        assert result["signal"] == "outflow"
        assert result["net_total_billion"] < 0

    def test_score_delta_within_range(self):
        """score_delta must be between -5 and +5."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        df = _make_flow_df(50.0, 30.0)  # very strong inflow
        with patch("src.agent.tools.data_tools._fetch_northbound_df", return_value=df):
            result = _handle_get_northbound_flow()
        assert result["score_delta"] == 5

    def test_api_failure_returns_not_available(self):
        """akshare failure → status=not_available, no exception raised."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        with patch("src.agent.tools.data_tools._fetch_northbound_df", side_effect=Exception("network")):
            result = _handle_get_northbound_flow()
        assert result["status"] == "not_available"
        assert "score_delta" not in result or result["score_delta"] == 0
