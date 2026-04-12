"""
aggregate_grid_results.py
-------------------------
Aggregate k-fold JSON reports from the sampler × augmentation grid,
produce a CSV summary and comparison plot, and select a recommended
training configuration (sampler-scale + aug-target).

Outputs:
- reports/grid_summary.csv
- reports/grid_summary.json
- reports/grid_summary.png

Run:
    python -u exohabitnet/scripts/aggregate_grid_results.py
"""
import json
import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REPORTS = Path("reports")
REPORTS.mkdir(exist_ok=True)


def parse_filename(fname: str):
    # Extract sampler-scale (e.g., scale1.0 or scale_1.0 or scale-1.0)
    s = None
    a = None
    m = re.search(r'scale[_-]?(\d+(?:\.\d+)?)', fname)
    if m:
        s = float(m.group(1))
    m2 = re.search(r'aug[_-]?(\d+)', fname)
    if m2:
        a = int(m2.group(1))
    # fallback patterns
    if s is None:
        m = re.search(r'scale(\d+(?:\.\d+)?)', fname)
        if m:
            s = float(m.group(1))
    return s, a


def main():
    rows = []
    files = sorted(REPORTS.glob('kfold_finetune_report*.json'))
    for f in files:
        try:
            j = json.loads(f.read_text())
        except Exception:
            continue
        sampler_scale, aug_target = parse_filename(f.name)
        rows.append({
            'file': f.name,
            'sampler_scale': sampler_scale,
            'aug_target': aug_target,
            'n_folds': j.get('n_folds'),
            'macro_f1_mean': j.get('macro_f1_mean'),
            'macro_f1_std': j.get('macro_f1_std'),
            'hab_recall_mean': j.get('hab_recall_mean'),
            'hab_recall_std': j.get('hab_recall_std')
        })

    if not rows:
        print('No grid JSON reports found in reports/.')
        return

    df = pd.DataFrame(rows)
    df = df.sort_values(by=['sampler_scale', 'aug_target'], na_position='last')
    csv_out = REPORTS / 'grid_summary.csv'
    json_out = REPORTS / 'grid_summary.json'
    df.to_csv(csv_out, index=False)
    df.to_json(json_out, orient='records', indent=2)
    print(f'Wrote {csv_out} and {json_out}')

    # Plot: macro_f1_mean vs sampler_scale, colored by aug_target
    plt.figure(figsize=(8, 5))
    markers = {None: 'o'}
    for at, group in df.groupby('aug_target'):
        plt.plot(group['sampler_scale'], group['macro_f1_mean'], marker='o', linestyle='-', label=f'aug={at}')
    plt.xlabel('sampler_scale')
    plt.ylabel('macro_f1_mean')
    plt.title('Sampler-scale vs Macro-F1 (by aug target)')
    plt.legend()
    plt.grid(True)
    png_out = REPORTS / 'grid_summary.png'
    plt.tight_layout()
    plt.savefig(png_out, dpi=150)
    plt.close()
    print(f'Wrote plot {png_out}')

    # Recommend best combo: prefer hab_recall_mean >= 0.8 then max macro_f1
    df_valid = df.copy()
    candidates = df_valid[df_valid['hab_recall_mean'] >= 0.8]
    if len(candidates) > 0:
        best = candidates.sort_values(by=['macro_f1_mean', 'hab_recall_mean'], ascending=False).iloc[0]
    else:
        best = df_valid.sort_values(by=['hab_recall_mean', 'macro_f1_mean'], ascending=False).iloc[0]

    rec = {
        'recommended_file': best['file'],
        'sampler_scale': float(best['sampler_scale']) if pd.notna(best['sampler_scale']) else None,
        'aug_target': int(best['aug_target']) if pd.notna(best['aug_target']) else None,
        'macro_f1_mean': float(best['macro_f1_mean']),
        'hab_recall_mean': float(best['hab_recall_mean'])
    }

    rec_out = REPORTS / 'grid_recommendation.json'
    rec_out.write_text(json.dumps(rec, indent=2))
    print('Recommendation saved to', rec_out)
    print('Recommended combo:', rec)


if __name__ == '__main__':
    main()
