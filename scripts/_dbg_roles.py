import sys
sys.path.insert(0, "src")
import pandas as pd
from pathlib import Path
sys.path.insert(0, "scripts")
from run_ratings import build_value_frame

df = build_value_frame(Path("data/processed"))
print("has batter_role:", "batter_role" in df.columns)
print("has bowler_role:", "bowler_role" in df.columns)
print([c for c in df.columns if "role" in c.lower()])
