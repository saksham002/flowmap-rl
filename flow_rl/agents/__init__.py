"""Agent objective classes + factory.

  fql       -> FQLAgent          (flow / regression BC + one-step Q-max; also BC-only)
  cql       -> CQLAgent          (FQL + conservative CQL penalty + dual ascent)
  meanflow  -> MeanFlowAgent     (staged MeanFlow-F2D2 BC: L_mf -> + L_div-mf, one run)
  shortcut  -> ShortcutAgent     (staged Shortcut-F2D2 BC: FM -> distill+semigroup -> + div, one run)
  meanflow_cql -> MeanFlowCQLAgent (meanflow policy + CQL + divergence-head log q)
  shortcut_cql -> ShortcutCQLAgent (shortcut policy + CQL + divergence-head log q)
"""

from flow_rl.agents.base import BaseAgent, TrainState, count_params, make_lr_schedule
from flow_rl.agents.fql import FQLAgent
from flow_rl.agents.cql import CQLAgent
from flow_rl.agents.shortcut import ShortcutAgent
from flow_rl.agents.shortcut_cql import ShortcutCQLAgent
from flow_rl.agents.meanflow import MeanFlowAgent
from flow_rl.agents.meanflow_cql import MeanFlowCQLAgent

_AGENTS = {"fql": FQLAgent, "cql": CQLAgent, "meanflow": MeanFlowAgent, "meanflow_cql": MeanFlowCQLAgent,
           "shortcut": ShortcutAgent, "shortcut_cql": ShortcutCQLAgent}


def make_agent(cfg) -> BaseAgent:
    name = cfg.agent.name
    if name not in _AGENTS:
        raise KeyError(f"unknown agent '{name}'. available: {list(_AGENTS)}")
    return _AGENTS[name](cfg)


__all__ = ["make_agent", "BaseAgent", "FQLAgent", "CQLAgent", "MeanFlowAgent", "MeanFlowCQLAgent", "ShortcutAgent",
           "ShortcutCQLAgent", "TrainState", "count_params", "make_lr_schedule"]
