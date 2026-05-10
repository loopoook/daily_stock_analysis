# 北向资金日频流入 + 月线三级趋势 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增两个分析维度：① 市场整体日频北向净买入（作为情绪评分加权）；② 月线趋势分析（与日线/周线形成三级共振验证，数据不足时动态降级）

**Architecture:** 遵循现有 `analyze_weekly_trend` / `get_capital_flow` 的工具模式，分别在 `analysis_tools.py` 和 `data_tools.py` 新增工具函数，挂载到对应 Agent；Orchestrator 在 `_normalize_dashboard_payload` 增加两处确定性兜底——月线摘要注入 `data_perspective.trend_status.monthly_trend`，北向净买入对 `sentiment_score` 做 ±5 分加权。

**Tech Stack:** Python 3.10+, akshare, pandas, Jinja2, pytest

---

## 文件变动一览

| 操作 | 文件 | 职责 |
|------|------|------|
| 修改 | `src/agent/tools/analysis_tools.py` | 新增 `_handle_analyze_monthly_trend` 函数及 `analyze_monthly_trend_tool`，注册到 `ALL_ANALYSIS_TOOLS` |
| 修改 | `src/agent/tools/data_tools.py` | 新增 `_handle_get_northbound_flow` 函数及 `get_northbound_flow_tool`，注册到 `ALL_DATA_TOOLS` |
| 修改 | `src/agent/agents/technical_agent.py` | `tool_names` 加入 `analyze_monthly_trend`；system_prompt 新增三级时间框架规则；JSON 输出格式加 `monthly_trend` 字段 |
| 修改 | `src/agent/agents/intel_agent.py` | `tool_names` 加入 `get_northbound_flow`；system_prompt 新增北向解读规则；JSON 输出格式加 `northbound_signal` 字段 |
| 修改 | `src/agent/orchestrator.py` | `_normalize_dashboard_payload` 新增月线兜底注入 + 北向 sentiment_score 加权 |
| 修改 | `src/schemas/report_schema.py` | `TrendStatus` 新增 `monthly_trend: Optional[str]` |
| 修改 | `src/report_language.py` | 新增 `monthly_trend_label`、`northbound_label` 中英双语标签 |
| 修改 | `templates/report_markdown.j2` | 数据透视表新增月线趋势行 |
| 修改 | `templates/report_wechat.j2` | 新增月线趋势 + 北向净买入展示行 |
| 新增 | `tests/test_analyze_monthly_trend.py` | 月线工具单元测试 |
| 新增 | `tests/test_get_northbound_flow.py` | 北向工具单元测试 |

---

## Task 1：`analyze_monthly_trend` 工具函数

**Files:**
- Modify: `src/agent/tools/analysis_tools.py`（在 `analyze_weekly_trend_tool` 定义之后，`ALL_ANALYSIS_TOOLS` 列表之前插入）
- Create: `tests/test_analyze_monthly_trend.py`

- [ ] **Step 1.1：写失败测试**

新建 `tests/test_analyze_monthly_trend.py`：

```python
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
        """< 24 monthly bars → returns degraded flag, no error key."""
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        df = _make_daily_df(300)  # ~14 months
        with patch("src.agent.tools.analysis_tools._load_df_for_monthly", return_value=(df, "mock")):
            result = _handle_analyze_monthly_trend("600519")
        assert result.get("degraded") is True
        assert "monthly_summary" in result  # still returns a summary string

    def test_no_data_returns_error(self):
        """Empty DataFrame → error key present."""
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        with patch("src.agent.tools.analysis_tools._load_df_for_monthly", return_value=(None, "mock")):
            result = _handle_analyze_monthly_trend("600519")
        assert "error" in result

    def test_summary_contains_chinese_direction(self):
        """monthly_summary must contain direction word in Chinese."""
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        df = _make_daily_df(1500)
        with patch("src.agent.tools.analysis_tools._load_df_for_monthly", return_value=(df, "mock")):
            result = _handle_analyze_monthly_trend("600519")
        summary = result.get("monthly_summary", "")
        assert any(kw in summary for kw in ["月线", "月K", "趋势", "数据"])
```

- [ ] **Step 1.2：运行测试确认失败**

```bash
cd F:\ai-workspace\daily_stock_analysis
python -m pytest tests/test_analyze_monthly_trend.py -v 2>&1 | head -30
```

期望：`ImportError` 或 `AttributeError`（函数尚未存在）

- [ ] **Step 1.3：实现 `_handle_analyze_monthly_trend`**

在 `src/agent/tools/analysis_tools.py` 中，在 `analyze_weekly_trend_tool` 定义结束后、`ALL_ANALYSIS_TOOLS` 列表之前，插入：

```python
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
    - >= 36 monthly bars: full analysis (MA3/6/12, RSI, MACD)
    - 24-35 bars: partial analysis, sets degraded=True
    - < 24 bars: minimal summary, sets degraded=True
    - No data: returns {"error": ...}
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
```

然后将 `analyze_monthly_trend_tool` 加入 `ALL_ANALYSIS_TOOLS` 列表：

```python
ALL_ANALYSIS_TOOLS = [
    analyze_trend_tool,
    analyze_weekly_trend_tool,
    analyze_monthly_trend_tool,   # 新增
    calculate_ma_tool,
    get_volume_analysis_tool,
    analyze_pattern_tool,
]
```

- [ ] **Step 1.4：运行测试确认通过**

```bash
python -m pytest tests/test_analyze_monthly_trend.py -v
```

期望：4 个测试全部 PASS

- [ ] **Step 1.5：提交**

```bash
git add src/agent/tools/analysis_tools.py tests/test_analyze_monthly_trend.py
git commit -m "功能: 新增 analyze_monthly_trend 工具，月线三级趋势分析"
```

---

## Task 2：`get_northbound_flow` 工具函数

**Files:**
- Modify: `src/agent/tools/data_tools.py`（在 `get_capital_flow_tool` 之后追加）
- Create: `tests/test_get_northbound_flow.py`

- [ ] **Step 2.1：写失败测试**

新建 `tests/test_get_northbound_flow.py`：

```python
# -*- coding: utf-8 -*-
"""Unit tests for get_northbound_flow tool handler."""
import pytest
import pandas as pd
from unittest.mock import patch


def _make_flow_df(net_buy_sh: float, net_buy_sz: float) -> pd.DataFrame:
    """Synthetic northbound flow summary DataFrame."""
    return pd.DataFrame([
        {"日期": "2026-05-09", "市场": "沪股通", "资金净买": net_buy_sh, "资金净流入": net_buy_sh * 0.8},
        {"日期": "2026-05-09", "市场": "深股通", "资金净买": net_buy_sz, "资金净流入": net_buy_sz * 0.8},
    ])


class TestGetNorthboundFlow:
    def test_strong_inflow_returns_positive_signal(self):
        """Net inflow > 30 亿 → signal=strong_inflow."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        df = _make_flow_df(25.0, 20.0)  # total 45 亿
        with patch("src.agent.tools.data_tools._fetch_northbound_df", return_value=df):
            result = _handle_get_northbound_flow()
        assert result["signal"] in ("strong_inflow", "inflow")
        assert result["net_total_billion"] > 0

    def test_outflow_returns_negative_signal(self):
        """Net outflow total < -10 亿 → signal=outflow or strong_outflow."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        df = _make_flow_df(-15.0, -10.0)
        with patch("src.agent.tools.data_tools._fetch_northbound_df", return_value=df):
            result = _handle_get_northbound_flow()
        assert result["signal"] in ("outflow", "strong_outflow")
        assert result["net_total_billion"] < 0

    def test_score_delta_within_range(self):
        """score_delta must be between -5 and +5."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        df = _make_flow_df(50.0, 30.0)  # very strong inflow
        with patch("src.agent.tools.data_tools._fetch_northbound_df", return_value=df):
            result = _handle_get_northbound_flow()
        assert -5 <= result["score_delta"] <= 5

    def test_api_failure_returns_not_available(self):
        """akshare failure → status=not_available, no exception raised."""
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        with patch("src.agent.tools.data_tools._fetch_northbound_df", side_effect=Exception("network")):
            result = _handle_get_northbound_flow()
        assert result["status"] == "not_available"
        assert "score_delta" not in result or result["score_delta"] == 0
```

- [ ] **Step 2.2：运行测试确认失败**

```bash
python -m pytest tests/test_get_northbound_flow.py -v 2>&1 | head -20
```

期望：ImportError（函数尚未存在）

- [ ] **Step 2.3：实现 `_handle_get_northbound_flow`**

在 `src/agent/tools/data_tools.py` 中，`get_capital_flow_tool` 注册之后、文件末尾之前，插入：

```python
# ============================================================
# get_northbound_flow — daily market-level northbound net buy
# ============================================================

def _fetch_northbound_df():
    """Fetch northbound fund flow summary via akshare."""
    import akshare as ak
    return ak.stock_hsgt_fund_flow_summary_em()


def _handle_get_northbound_flow() -> dict:
    """Get today's market-level northbound (沪深港通) net buy data.

    Returns net inflow in 亿 CNY, a directional signal, and a score_delta
    (-5 to +5) for sentiment_score weighting in the orchestrator.
    """
    try:
        df = _fetch_northbound_df()
    except Exception as exc:
        logger.warning("get_northbound_flow: akshare fetch failed: %s", exc)
        return {"status": "not_available", "score_delta": 0,
                "note": f"北向资金数据获取失败: {exc}"}

    if df is None or df.empty:
        return {"status": "not_available", "score_delta": 0,
                "note": "北向资金数据暂不可用"}

    # Column name normalisation — akshare returns Chinese column names
    # Identify net-buy column (资金净买 / 资金净流入 etc.)
    net_col = None
    for candidate in ["资金净买", "净买额", "资金净流入", "净流入"]:
        if candidate in df.columns:
            net_col = candidate
            break

    if net_col is None:
        return {"status": "not_available", "score_delta": 0,
                "note": "北向资金数据列名无法识别"}

    try:
        net_total = float(df[net_col].sum())  # 单位：亿元
    except Exception:
        return {"status": "not_available", "score_delta": 0,
                "note": "北向资金数值解析失败"}

    # Direction signal
    if net_total >= 30:
        signal = "strong_inflow"
        score_delta = 5
    elif net_total >= 10:
        signal = "inflow"
        score_delta = 3
    elif net_total >= -10:
        signal = "neutral"
        score_delta = 0
    elif net_total >= -30:
        signal = "outflow"
        score_delta = -3
    else:
        signal = "strong_outflow"
        score_delta = -5

    # Plain-language summary
    direction_word = "净流入" if net_total >= 0 else "净流出"
    summary = (
        f"今日北向资金{direction_word} {abs(net_total):.1f} 亿元，"
        f"市场情绪{'偏多' if net_total > 0 else '偏空'}"
    )

    date_col = next((c for c in df.columns if "日期" in c or "date" in c.lower()), None)
    trade_date = str(df[date_col].iloc[0]) if date_col and len(df) > 0 else "unknown"

    return {
        "status": "available",
        "trade_date": trade_date,
        "net_total_billion": round(net_total, 2),
        "signal": signal,
        "score_delta": score_delta,
        "summary": summary,
    }


get_northbound_flow_tool = ToolDefinition(
    name="get_northbound_flow",
    description=(
        "Get today's market-level northbound fund (北向资金/沪深港通) net buy data. "
        "Returns net inflow in 亿 CNY, directional signal (strong_inflow/inflow/neutral/"
        "outflow/strong_outflow), and a score_delta (-5 to +5) for sentiment weighting. "
        "Use as a market-sentiment background indicator. Always available for A-share analysis."
    ),
    parameters=[],  # no parameters — market-level data
    handler=_handle_get_northbound_flow,
    category="data",
)

ALL_DATA_TOOLS.append(get_northbound_flow_tool)
```

- [ ] **Step 2.4：运行测试确认通过**

```bash
python -m pytest tests/test_get_northbound_flow.py -v
```

期望：4 个测试全部 PASS

- [ ] **Step 2.5：提交**

```bash
git add src/agent/tools/data_tools.py tests/test_get_northbound_flow.py
git commit -m "功能: 新增 get_northbound_flow 工具，日频北向净买入及评分加权"
```

---

## Task 3：TechnicalAgent — 接入月线工具

**Files:**
- Modify: `src/agent/agents/technical_agent.py`

- [ ] **Step 3.1：更新 `tool_names`**

将 `"analyze_monthly_trend"` 加入 `tool_names` 列表（紧跟 `"analyze_weekly_trend"` 之后）：

```python
tool_names = [
    "get_realtime_quote",
    "get_daily_history",
    "analyze_trend",
    "analyze_weekly_trend",
    "analyze_monthly_trend",   # 新增
    "calculate_ma",
    "get_volume_analysis",
    "analyze_pattern",
    "get_chip_distribution",
    "get_analysis_context",
]
```

- [ ] **Step 3.2：更新 `system_prompt`**

在 `system_prompt` 中修改 Workflow 第 2 步和 Multi-Timeframe Rule 段落，以及 Output Format 的 JSON schema：

**Workflow 第 2 步（替换）：**
```
2. Run `analyze_trend` (daily) AND `analyze_weekly_trend` (weekly) AND `analyze_monthly_trend` (monthly)
```

**Multi-Timeframe Rule（替换）：**
```
## Three-Timeframe Rule
- Monthly trend (from `analyze_monthly_trend`) is the **long-term direction**: skip if data < 24 months.
- Weekly trend (from `analyze_weekly_trend`) is the **medium-term filter**.
- Three-timeframe resonance (日线买信号 + 周线多头 + 月线多头) = highest confidence buy.
- Divergence (日线买信号 + 月线空头) = downgrade confidence; label signal as "hold" at most unless weekly also bullish.
- Record `monthly_summary` field from `analyze_monthly_trend` as `monthly_trend` in output; omit if tool returned degraded/error.
```

**Output Format JSON schema — 新增 `monthly_trend` 字段：**
```json
{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence summary",
  "key_levels": {"support": float, "resistance": float, "stop_loss": float},
  "trend_score": 0-100,
  "ma_alignment": "bullish|neutral|bearish",
  "volume_status": "heavy|normal|light",
  "pattern": "<detected pattern or none>",
  "weekly_trend": "<weekly_summary from analyze_weekly_trend>",
  "monthly_trend": "<monthly_summary from analyze_monthly_trend, omit if unavailable>"
}
```

- [ ] **Step 3.3：验证 TechnicalAgent 工具列表可加载**

```bash
python -c "
from src.agent.agents.technical_agent import TechnicalAgent
print('tool_names:', TechnicalAgent.tool_names)
assert 'analyze_monthly_trend' in TechnicalAgent.tool_names
print('OK')
" 2>&1
```

期望：打印工具列表，包含 `analyze_monthly_trend`，无报错

- [ ] **Step 3.4：提交**

```bash
git add src/agent/agents/technical_agent.py
git commit -m "功能: TechnicalAgent 接入月线工具，更新三级时间框架规则"
```

---

## Task 4：IntelAgent — 接入北向工具

**Files:**
- Modify: `src/agent/agents/intel_agent.py`

- [ ] **Step 4.1：更新 `tool_names`**

```python
tool_names = [
    "search_stock_news",
    "search_comprehensive_intel",
    "get_stock_info",
    "get_capital_flow",
    "get_northbound_flow",   # 新增
]
```

- [ ] **Step 4.2：更新 `system_prompt`**

在 Workflow 中新增第 4 步（现有第 4、5 步顺延），并新增北向解读规则段落，并在 JSON 输出格式中新增 `northbound_signal` 字段。

**Workflow 新增（在 `get_capital_flow` 步骤之后）：**
```
4. Call get_northbound_flow to get today's market-level northbound net buy (北向净买入).
   Note the score_delta it returns — this feeds directly into the final sentiment score weighting.
```

**新增段落 Northbound Flow Interpretation：**
```
## Northbound Flow Interpretation
- signal=strong_inflow (net > +30亿): strong positive market background, upgrade confidence
- signal=inflow (net +10~30亿): mildly positive background
- signal=neutral: no market-level tailwind or headwind
- signal=outflow / strong_outflow: foreign capital retreating, downgrade confidence on buy signals
- Always report the summary field verbatim in your reasoning.
```

**JSON 输出格式新增字段：**
```json
{
  ...现有字段...,
  "northbound_signal": "strong_inflow|inflow|neutral|outflow|strong_outflow|not_available",
  "northbound_score_delta": -5  // integer, from get_northbound_flow result
}
```

- [ ] **Step 4.3：验证 IntelAgent 工具列表**

```bash
python -c "
from src.agent.agents.intel_agent import IntelAgent
print('tool_names:', IntelAgent.tool_names)
assert 'get_northbound_flow' in IntelAgent.tool_names
print('OK')
" 2>&1
```

- [ ] **Step 4.4：提交**

```bash
git add src/agent/agents/intel_agent.py
git commit -m "功能: IntelAgent 接入北向工具，新增北向解读规则"
```

---

## Task 5：Orchestrator — 月线兜底 + 北向评分注入

**Files:**
- Modify: `src/agent/orchestrator.py`（`_normalize_dashboard_payload` 方法，在现有 weekly_trend 兜底之后）

- [ ] **Step 5.1：实现月线兜底注入**

在 `_normalize_dashboard_payload` 方法的 weekly_trend 注入代码块之后，加入月线兜底：

```python
# --- Monthly trend injection (after weekly_trend block) ---
_monthly_trend = _tech_raw_wt.get("monthly_trend")
if not _monthly_trend and ctx.stock_code:
    try:
        from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
        _mt_result = _handle_analyze_monthly_trend(ctx.stock_code)
        if isinstance(_mt_result, dict) and not _mt_result.get("error"):
            _monthly_trend = _mt_result.get("monthly_summary", "")
    except Exception as _mt_err:
        logger.debug("[Orchestrator] monthly_trend fallback failed: %s", _mt_err)
if _monthly_trend:
    _dp_mt = dashboard_block.get("data_perspective")
    if not isinstance(_dp_mt, dict):
        _dp_mt = {}
        dashboard_block["data_perspective"] = _dp_mt
    _ts_mt = _dp_mt.get("trend_status")
    if not isinstance(_ts_mt, dict):
        _ts_mt = {}
        _dp_mt["trend_status"] = _ts_mt
    _ts_mt["monthly_trend"] = _monthly_trend
```

- [ ] **Step 5.2：实现北向 sentiment_score 加权**

在月线注入代码之后，加入北向评分加权（在 `payload["sentiment_score"] = sentiment_score` 赋值之前）：

```python
# --- Northbound score weighting ---
_intel_op = self._latest_opinion(ctx, {"intel"})
_intel_raw = _intel_op.raw_data if _intel_op and isinstance(_intel_op.raw_data, dict) else {}
_nb_delta = _intel_raw.get("northbound_score_delta")
if _nb_delta is None:
    # Fallback: call deterministically if IntelAgent skipped it
    try:
        from src.agent.tools.data_tools import _handle_get_northbound_flow
        _nb_result = _handle_get_northbound_flow()
        _nb_delta = _nb_result.get("score_delta", 0) if _nb_result.get("status") == "available" else 0
    except Exception as _nb_err:
        logger.debug("[Orchestrator] northbound fallback failed: %s", _nb_err)
        _nb_delta = 0
if _nb_delta:
    sentiment_score = max(0, min(100, sentiment_score + int(_nb_delta)))
    logger.debug("[Orchestrator] northbound score_delta=%s → adjusted sentiment_score=%s", _nb_delta, sentiment_score)
```

注意：此代码需插入在 `payload["sentiment_score"] = sentiment_score` **之前**，找到该行并将加权逻辑放在紧接其上方。

- [ ] **Step 5.3：语法验证**

```bash
python -c "
import ast
with open('src/agent/orchestrator.py', encoding='utf-8') as f:
    ast.parse(f.read())
print('syntax OK')
"
```

- [ ] **Step 5.4：提交**

```bash
git add src/agent/orchestrator.py
git commit -m "功能: Orchestrator 新增月线兜底注入与北向 sentiment_score 加权"
```

---

## Task 6：Schema + 语言标签 + 模板

**Files:**
- Modify: `src/schemas/report_schema.py`
- Modify: `src/report_language.py`
- Modify: `templates/report_markdown.j2`
- Modify: `templates/report_wechat.j2`

- [ ] **Step 6.1：`report_schema.py` — 新增 `monthly_trend` 字段**

在 `TrendStatus` 类中，`weekly_trend` 字段之后加入：

```python
monthly_trend: Optional[str] = None
```

- [ ] **Step 6.2：`report_language.py` — 新增标签**

在中文标签字典 `weekly_trend_label` 之后加入：

```python
"monthly_trend_label": "月线趋势",
"northbound_label": "北向资金",
```

在英文标签字典对应位置加入：

```python
"monthly_trend_label": "Monthly Trend",
"northbound_label": "Northbound Flow",
```

- [ ] **Step 6.3：`report_markdown.j2` — 新增月线趋势行**

在现有 `weekly_trend` 行之后，加入月线行（格式保持一致）：

```jinja2
{% if trend_data.get('monthly_trend') %}
| {{ labels.monthly_trend_label }} | {{ trend_data.monthly_trend }} | 月线方向，长期趋势验证 |
{% endif %}
```

- [ ] **Step 6.4：`report_wechat.j2` — 新增月线 + 北向展示**

在现有 `weekly_trend` 展示行之后加入：

```jinja2
{% set core_mt = (dashboard.get('data_perspective') or {}).get('trend_status', {}).get('monthly_trend') %}
{% if core_mt %}
📆 **{{ labels.monthly_trend_label }}**: {{ core_mt }}
{% endif %}
```

北向净买入行已在上一版本新增（`volume_ratio` 修改时加入），无需重复添加；若未加则在此处补充。

- [ ] **Step 6.5：模板语法验证**

```bash
python -c "
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader('templates'))
for t in ['report_markdown.j2', 'report_wechat.j2']:
    env.get_template(t)
    print(t, 'OK')
"
```

- [ ] **Step 6.6：提交**

```bash
git add src/schemas/report_schema.py src/report_language.py templates/report_markdown.j2 templates/report_wechat.j2
git commit -m "功能: 新增月线趋势展示与北向标签，更新报告模板"
```

---

## Task 7：端到端冒烟测试

**Files:**
- 无新文件；运行现有测试套件

- [ ] **Step 7.1：运行全套测试**

```bash
python -m pytest tests/ -v --tb=short -q 2>&1 | tail -30
```

期望：所有现有测试通过，新增 8 个测试通过

- [ ] **Step 7.2：月线工具手动冒烟（需网络/数据库）**

```bash
python -c "
from src.agent.tools.analysis_tools import _handle_analyze_monthly_trend
r = _handle_analyze_monthly_trend('600519')
print('monthly_bars:', r.get('monthly_bars'))
print('degraded:', r.get('degraded'))
print('monthly_summary:', r.get('monthly_summary'))
" 2>&1
```

期望：打印月线摘要，无异常

- [ ] **Step 7.3：北向工具手动冒烟（需网络）**

```bash
python -c "
from src.agent.tools.data_tools import _handle_get_northbound_flow
r = _handle_get_northbound_flow()
print('status:', r.get('status'))
print('net_total_billion:', r.get('net_total_billion'))
print('signal:', r.get('signal'))
print('score_delta:', r.get('score_delta'))
print('summary:', r.get('summary'))
" 2>&1
```

期望：返回 available 状态和数值，或在非交易日返回 not_available

- [ ] **Step 7.4：最终提交**

```bash
git add .
git commit -m "功能: 完成北向资金+月线趋势功能，端到端测试通过"
```
