#!/usr/bin/env python3
"""Read-only checks for retyped aggregate result tables included with this repository.
Not a reproduction of the underlying model training/evaluation.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def check(path, key, required, expected=5):
    with (ROOT / path).open(newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == expected, (path, len(rows))
    assert len(set(r[key] for r in rows)) == expected, path
    for row in rows:
        for col in required:
            val = float(row[col])
            assert 0 <= val <= 1, (path, row[key], col, val)
    print('OK:', path, '- models:', ', '.join(r[key] for r in rows))
    return rows

if __name__ == '__main__':
    for p in ('results/task2/patient_level_model_summary.csv','results/task3/patch_validation_summary.csv'):
        with (ROOT/p).open() as f:
            print('Columns:',p, next(csv.reader(f)))
    # Columns are checked dynamically against source-exported CSV schema.
    import csv
    a='results/task2/patient_level_model_summary.csv'
    b='results/task3/patch_validation_summary.csv'
    with (ROOT/a).open() as f: ha=next(csv.reader(f))
    with (ROOT/b).open() as f: hb=next(csv.reader(f))
    k1= 'Model' if 'Model' in ha else 'model'
    k2= 'Model' if 'Model' in hb else 'model'
    c1= [c for c in ('Pooled_AUC','Pooled_F1','AUC_mean','Pooled patient AUC','Pooled F1','pooled_patient_auc','pooled_f1','fold_auc_mean') if c in ha]
    c2= [c for c in ('balanced_accuracy','macro_f1','Balanced accuracy','Macro-F1') if c in hb]
    assert c1 and c2, 'Required metric columns missing'
    check(a,k1,c1)
    check(b,k2,c2,expected=6)
