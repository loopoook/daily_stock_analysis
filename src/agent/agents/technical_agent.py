# -*- coding: utf-8 -*-
"""
TechnicalAgent — technical & price analysis specialist.

Responsible for:
- Fetching realtime quotes and historical K-line data
- Running technical indicators (trend, MA, volume, pattern)
- Producing a structured opinion on trend/momentum/support-resistance
"""

from __future__ import annotations

import logging
from typing import Optional

from src.agent.agents.base_agent import BaseAgent
from src.agent.protocols import AgentContext, AgentOpinion
from src.agent.runner import try_parse_json

logger = logging.getLogger(__name__)


class TechnicalAgent(BaseAgent):
    agent_name = "technical"
    max_steps = 8
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

    def system_prompt(self, ctx: AgentContext) -> str:
        skills = ""
        if self.skill_instructions:
            skills = f"\n## Active Trading Skills\n\n{self.skill_instructions}\n"
        baseline = ""
        if self.technical_skill_policy:
            baseline = f"\n{self.technical_skill_policy}\n"

        return f"""\
You are a **Technical Analysis Agent** specialising in Chinese A-shares, \
Hong Kong stocks, and US equities.

Your task: perform a thorough technical analysis of the given stock and \
output a structured JSON opinion.

## Workflow (execute stages in order)
1. Fetch realtime quote + daily history (if not already provided)
2. Run `analyze_trend` (daily) AND `analyze_weekly_trend` (weekly) AND `analyze_monthly_trend` (monthly)
3. Analyse volume-price relationship (`get_volume_analysis`) and chip distribution
4. Identify chart patterns

## Three-Timeframe Rule
- Monthly trend (from `analyze_monthly_trend`) is the **long-term direction**: call unconditionally; if the response has `degraded=true`, treat `monthly_summary` as low-confidence but still record it.
- Weekly trend (from `analyze_weekly_trend`) is the **medium-term filter**.
- Three-timeframe resonance (日线买信号 + 周线多头 + 月线多头) = highest confidence buy.
- Divergence (日线买信号 + 月线空头) = downgrade confidence; label signal as "hold" at most unless weekly also bullish.
- Record `monthly_summary` from `analyze_monthly_trend` as `monthly_trend` in output. On `error` key → omit the field. On `degraded=true` with a valid `monthly_summary` → include it, suffix with "（数据有限）".

{baseline}
{skills}
## Output Format
Return **only** a JSON object (no markdown fences):
{{
  "signal": "strong_buy|buy|hold|sell|strong_sell",
  "confidence": 0.0-1.0,
  "reasoning": "2-3 sentence summary",
  "key_levels": {{
    "support": <float>,
    "resistance": <float>,
    "stop_loss": <float>
  }},
  "trend_score": 0-100,
  "ma_alignment": "bullish|neutral|bearish",
  "volume_status": "heavy|normal|light",
  "pattern": "<detected pattern or none>",
  "weekly_trend": "<weekly_summary from analyze_weekly_trend>",
  "monthly_trend": "<monthly_summary string, or omit key>"
}}
"""

    def build_user_message(self, ctx: AgentContext) -> str:
        parts = [f"Perform technical analysis on stock **{ctx.stock_code}**"]
        if ctx.stock_name:
            parts[0] += f" ({ctx.stock_name})"
        parts.append("Use your tools to fetch any missing data, then output the JSON opinion.")
        return "\n".join(parts)

    def post_process(self, ctx: AgentContext, raw_text: str) -> Optional[AgentOpinion]:
        """Parse the JSON opinion from the LLM response."""
        parsed = try_parse_json(raw_text)
        if parsed is None:
            logger.warning("[TechnicalAgent] failed to parse opinion JSON")
            return None

        return AgentOpinion(
            agent_name=self.agent_name,
            signal=parsed.get("signal", "hold"),
            confidence=float(parsed.get("confidence", 0.5)),
            reasoning=parsed.get("reasoning", ""),
            key_levels={
                k: float(v) for k, v in parsed.get("key_levels", {}).items()
                if isinstance(v, (int, float))
            },
            raw_data=parsed,
        )

