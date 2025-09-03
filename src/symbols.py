import argparse
import os
import re
import yaml
import pandas as pd

def ensure_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")

def load_lexicon(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    groups = data.get("groups", {})
    compiled = {g: [re.compile(rf"\b{re.escape(term)}\b", re.I) for term in terms]
                for g, terms in groups.items()}
    return compiled

def count_symbols(text: str, lex):
    text = str(text)
    out = {}
    for group, patterns in lex.items():
        c = 0
        for p in patterns:
            c += len(p.findall(text))
        out[group] = c
    return out

def main():
    ap = argparse.ArgumentParser(description="Symbol/archetype counter")
    ap.add_argument("--input", required=True, help="CSV with columns: date,text")
    ap.add_argument("--lex", required=True, help="YAML lexicon")
    ap.add_argument("--outdir", default="reports")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)
    if not {"date","text"}.issubset(df.columns):
        raise ValueError("Input CSV must include date,text")

    df["date"] = ensure_datetime(df["date"])
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    lex = load_lexicon(args.lex)
    symbol_rows = df["text"].apply(lambda t: count_symbols(t, lex)).apply(pd.Series)
    symbol_rows = symbol_rows.fillna(0).astype(int)
    per_entry = pd.concat([df[["date","text"]], symbol_rows], axis=1)
    per_entry.to_csv(os.path.join(args.outdir, "symbols_per_entry.csv"), index=False)

    timeline = per_entry.groupby("date").sum(numeric_only=True).reset_index()
    timeline.to_csv(os.path.join(args.outdir, "symbols_timeline.csv"), index=False)

    totals = symbol_rows.sum().reset_index()
    totals.columns = ["symbol_group","total_count"]
    totals = totals.sort_values("total_count", ascending=False)
    totals.to_csv(os.path.join(args.outdir, "symbols_totals.csv"), index=False)

    print("Saved:")
    print(f"- {os.path.join(args.outdir, 'symbols_per_entry.csv')}")
    print(f"- {os.path.join(args.outdir, 'symbols_timeline.csv')}")
    print(f"- {os.path.join(args.outdir, 'symbols_totals.csv')}")

if __name__ == "__main__":
    main()
