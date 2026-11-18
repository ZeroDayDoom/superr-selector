"""Feature registry: every feature is documented at the point of creation.

Each module in ``superr.features`` exposes a ``FEATURE_DOCS`` dict mapping
column name -> metadata. The registry aggregates them, validates that every
feature column produced by the pipeline is documented, and generates
``reports/feature_dictionary.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class FeatureSpec:
    name: str
    definition: str          # what it is
    formula: str             # how it is computed
    source: str              # data source / producing module
    available_before_delivery: bool
    leakage_risk: str        # description of the risk and its control
    missing_handling: str
    relevance: str           # why it matters for pressure analytics


@dataclass
class Registry:
    specs: Dict[str, FeatureSpec] = field(default_factory=dict)
    duplicates: List[str] = field(default_factory=list)

    def register(self, docs: Dict[str, dict], replace: bool = False) -> None:
        for name, meta in docs.items():
            if name in self.specs and not replace:
                # same feature re-exposed from another perspective: keep the
                # first documentation and note the duplicate
                self.duplicates.append(name)
                continue
            if name in self.specs:
                raise ValueError(f"feature registered twice with replace: {name}")
            self.specs[name] = FeatureSpec(name=name, **meta)

    def validate(self, columns: List[str], ignore: Optional[set] = None) -> List[str]:
        """Return feature columns that are not documented."""
        ignore = ignore or set()
        return [c for c in columns if c not in self.specs and c not in ignore]

    def write_dictionary_md(self, path: Path) -> None:
        lines: List[str] = []
        a = lines.append
        a("# Feature Dictionary — SupeRR Selector Season 2")
        a("")
        a(f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}. "
          f"{len(self.specs)} documented features._")
        a("")
        a("All features are computed from strictly pre-delivery information")
        a("unless explicitly marked otherwise. Leakage risk and its control are")
        a("documented per feature; the dynamic leakage audit is described in")
        a("`scripts/run_features.py` and reported alongside the outputs.")
        a("")
        for name in sorted(self.specs):
            s = self.specs[name]
            a(f"## `{name}`")
            a(f"- **Definition:** {s.definition}")
            a(f"- **Formula:** {s.formula}")
            a(f"- **Source:** {s.source}")
            a(f"- **Available before delivery:** "
              f"{'YES' if s.available_before_delivery else 'NO (post-delivery / evidence only)'}")
            a(f"- **Leakage risk:** {s.leakage_risk}")
            a(f"- **Missing-data handling:** {s.missing_handling}")
            a(f"- **Relevance:** {s.relevance}")
            a("")
        path.write_text("\n".join(lines), encoding="utf-8")


def build_registry() -> Registry:
    from superr.features import (
        batting_pressure_features,
        bowling_pressure_features,
        match_pressure_features,
        matchup_features,
        role_features,
        scenario_features,
    )
    reg = Registry()
    for mod in (match_pressure_features, batting_pressure_features,
                bowling_pressure_features, scenario_features, role_features,
                matchup_features):
        reg.register(mod.FEATURE_DOCS)
    return reg
