#!/usr/bin/env python3
"""Export model-level sufficient summaries, never candidate identities or draws.

This command needs controlled local decision tables. Published CSVs contain only
screen/budget/engine counts and means. They support rate and clustered bootstrap
recalculation, not raw-record or detector replay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


def digest(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def summarize(d: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    d = d.copy()
    d['n_go'] = d.decision.eq('GO').astype(int)
    d['n_nogo'] = d.decision.eq('NO_GO').astype(int)
    d['n_abstain'] = d.decision.eq('ABSTAIN').astype(int)
    d['false_go'] = (d.decision.eq('GO') & d.true_pass.eq(0)).astype(int)
    d['false_nogo'] = (d.decision.eq('NO_GO') & d.true_pass.eq(1)).astype(int)
    specs = dict(n=('decision', 'size'), n_go=('n_go', 'sum'),
                 n_nogo=('n_nogo', 'sum'), n_abstain=('n_abstain', 'sum'),
                 false_go=('false_go', 'sum'), false_nogo=('false_nogo', 'sum'))
    if 'labels' in d:
        specs['mean_labels'] = ('labels', 'mean')
    out = d.groupby(keys, dropna=False, sort=True).agg(**specs).reset_index()
    # Multiple screens on the same trained seed share this cluster, not an ID per screen.
    def cluster(row):
        found = re.search(r'seed(\d+)', row.victim_id)
        seed = found.group(1) if found else row.victim_id
        domain = 'cifar10' if 'cifar' in row.victim_id else 'gtsrb'
        return domain + '_seed' + seed
    out['physical_victim_cluster'] = out.apply(cluster, axis=1)
    assert (out.n_go + out.n_nogo + out.n_abstain).eq(out.n).all()
    assert not {'sample_id', 'draw', 'is_poison', 'candidate_id'} & set(out.columns)
    return out


def export(cert: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    records = []
    sources = [
        ('audit_baselines', 'results_audit_baselines_v1/decisions.csv', ['block','victim_id','k','design','true_pass']),
        ('stratified_ui_corrected', 'results_stratified_ui_audit_v1_1/decisions.csv', ['block','victim_id','k','design','share','true_pass']),
        ('block16_corrected', 'results_gtsrb_srac_ui_confirmation_v1_1_erratum/decisions.csv', ['victim_id','k','engine','true_pass']),
    ]
    for name, rel, keys in sources:
        p = cert / rel
        d = pd.read_csv(p, dtype={'block':str, 'share':str})
        out = summarize(d, keys)
        dest = output / (name + '_by_screen.csv')
        out.to_csv(dest, index=False)
        records.append(dict(path=dest.name, rows=len(out), controlled_source_rows=len(d),
                            controlled_source_sha256=digest(p), sha256=digest(dest)))
    p = cert / 'results_ui_mechanism_ablation_v1/decisions.csv'
    d = pd.read_csv(p, dtype={'block':str})
    engines = ['interval_frozen','interval_used','interval_split','ui_split','ui_nobb','ui_full','suite']
    frames = []
    for engine in engines:
        for subset, mask in [('all_available',d[engine].isin(['GO','NO_GO','ABSTAIN'])),
                             ('block16_first20',d.block.eq('16') & d.draw.lt(20) & d[engine].isin(['GO','NO_GO','ABSTAIN']))]:
            t = d.loc[mask, ['block','victim_id','k','true_pass',engine]].rename(columns={engine:'decision'})
            if t.empty:
                continue
            t['engine'], t['subset'] = engine, subset
            frames.append(summarize(t, ['block','victim_id','k','true_pass','engine','subset']))
    out = pd.concat(frames, ignore_index=True)
    dest = output / 'mechanism_by_screen.csv'
    out.to_csv(dest,index=False)
    records.append(dict(path=dest.name,rows=len(out),controlled_source_rows=len(d),
                        controlled_source_sha256=digest(p),sha256=digest(dest)))
    receipt = dict(schema='srac-model-level-aggregates-20261019',
        scope='Model-level sufficient summaries only; no candidate identities, private labels, or per-draw tables.',
        grouping='physical_victim_cluster groups reused CIFAR checkpoint seeds across screens; point estimates may remain screen weighted.',
        mechanism_subset='block16_first20 permits matched no-Berger-Boos/Fisher comparison without releasing draw-level records.',
        files=records)
    (output / 'AGGREGATE_PROVENANCE.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--certificate-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(export(a.certificate_root,a.output),indent=2))


if __name__ == '__main__':
    main()
