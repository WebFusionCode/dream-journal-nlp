# src/symbols_ext.py
import re, yaml, os
import pandas as pd

def load_symbol_lexicon(path="config/symbols.yaml"):
    if not os.path.exists(path):
        return {}
    data = yaml.safe_load(open(path,"r", encoding="utf-8"))
    groups = data.get("groups", {})
    # Precompile regex lists
    compiled = {}
    for g, info in groups.items():
        words = info.get("words", [])
        patterns = [re.compile(rf"\b{re.escape(w)}\b", re.I) for w in words]
        compiled[g] = {
            "patterns": patterns,
            "meaning": info.get("meaning", "")
        }
    return compiled

def count_symbols_in_text(text, compiled):
    text = str(text)
    out = {}
    for g, info in compiled.items():
        c = 0
        for p in info["patterns"]:
            c += len(p.findall(text))
        out[g] = c
    return out

def symbol_summary_for_df(df, lexicon=None):
    # returns per-entry counts, totals and a per-group meaning table
    if lexicon is None:
        lexicon = load_symbol_lexicon()
    counts_df = df["text"].apply(lambda t: count_symbols_in_text(t, lexicon)).apply(pd.Series).fillna(0).astype(int)
    totals = counts_df.sum().sort_values(ascending=False).rename_axis("symbol_group").reset_index(name="total_count")
    # attach meaning
    meanings = []
    for g in totals["symbol_group"]:
        m = lexicon.get(g, {}).get("meaning", "")
        meanings.append(m)
    totals["meaning"] = meanings
    return counts_df, totals
