"""
Shared setup for the notebooks: which cohorts, which run, and loaders for a run's tables.

    from nb_setup import *

Notebooks READ tables that the `sc-*` commands wrote; they do not recompute. Every notebook opens a run
with `open_run(report, cohort)` ('latest' unless `SC_NB_RUN` is set) and draws from its CSVs. The one
exception is a single-animal cell per notebook that walks the library pipeline so the numbers are
traceable — it uses `load_experiment()`.

Environment switches:
    SC_NB_SYNTHETIC=1   read the synthetic runs (`sc-make-synthetic-run`) and the in-memory synthetic
                        experiment instead of the real cohorts — what CI uses
    SC_NB_RUN=<id>      a run id instead of 'latest'
    SC_RESULTS_ROOT / SC_DATA_ROOT   as everywhere else (see sound_categorisation/data/paths.py)
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behav_utils.plotting.styles import COLOURS, PALETTE, apply_style
from sound_categorisation.data.paths import REPO_ROOT, data_root, resolve_run, results_root
from sound_categorisation.reports.tables import normalise_site

SYNTHETIC = os.environ.get('SC_NB_SYNTHETIC', '') not in ('', '0')
RUN = os.environ.get('SC_NB_RUN', 'latest')

if SYNTHETIC:
    OPTO_COHORT, SWITCH_COHORT, MODEL_COHORT = 'synthetic', 'synthetic', 'synthetic_uniform'
else:
    OPTO_COHORT, SWITCH_COHORT, MODEL_COHORT = 'opto1-cohort', 'behaviour1-cohort', 'opto1-cohort'

apply_style()
pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 40)
pd.set_option('display.precision', 3)

GENOTYPE_COLOUR = {'wt': PALETTE[0], 'het': PALETTE[1], 'unknown': '0.5'}


def banner() -> None:
    mode = 'SYNTHETIC (SC_NB_SYNTHETIC=1)' if SYNTHETIC else 'real cohorts'
    print(f'mode: {mode} | results root: {results_root()} | data root: {data_root()} | run: {RUN}')


# ── runs ─────────────────────────────────────────────────────────────────────

def open_run(report: str, cohort: str, run: str = RUN) -> Path:
    """The run directory for (report, cohort); a readable error naming the command that creates it."""
    try:
        path = resolve_run(report, cohort, run)
    except FileNotFoundError as e:
        hint = {'opto_contrasts': f'sc-reports opto-contrasts --all --cohort {cohort}   (or sc-reports battery)',
                'switch_adaptation': f'sc-reports switch-adaptation --cohort {cohort}',
                'light_artefact': f'sc-reports light-artefact --cohort {cohort}',
                'model_identification': 'slurm/submit.sh grid-search / condition, then sc-consensus (see docs/runs.md)'}[report]
        raise FileNotFoundError(f'{e}\nNo {report} run for cohort {cohort!r}. Create one with:\n    {hint}'
                                + ('\n(or set SC_NB_SYNTHETIC=1 after sc-make-synthetic-run)' if not SYNTHETIC else ''))
    print(f'{report}/{cohort}/{path.name}')
    return path


def _read_all(run: Path, name: str, exclude_group: bool = True) -> pd.DataFrame:
    files = sorted(glob.glob(str(run / '**' / f'{name}.csv'), recursive=True))
    if exclude_group:
        files = [f for f in files if '/group/' not in f]
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return normalise_site(df) if name in ('contrasts', 'levels', 'group_rows', 'group_tests', 'trajectory') else df


def load_contrasts(run: Path) -> pd.DataFrame:
    """Every per-animal contrasts.csv of an opto-contrasts run, one frame (columns include distribution,
    site (ppc | alm_uni | alm_bi), trial_class, kind, unit, stat, diff, ci_lo, ci_hi, boot_p, perm_p)."""
    return _read_all(run, 'contrasts')


def load_levels(run: Path) -> pd.DataFrame:
    """Every per-animal levels.csv: the observed value of each stat in each condition the contrasts were
    built from — (kind, phase) names the condition, e.g. within/non_opto = laser-off trials of opto sessions."""
    return _read_all(run, 'levels')


def load_group(run: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(group_rows, group_tests) across every condition of an opto run."""
    return _read_all(run, 'group_rows', exclude_group=False), _read_all(run, 'group_tests', exclude_group=False)


def load_trajectory(run: Path) -> pd.DataFrame:
    """Per-session trajectory rows of an opto run (one copy per session)."""
    t = _read_all(run, 'trajectory')
    if len(t):
        t = t.drop_duplicates(subset=['animal', 'session_idx', 'trial_class'])
    return t


def condition_dir(run: Path, distribution: str, site: str = 'ppc', trial_class: str = 'opto') -> Path:
    """<run>/<distribution>/<site>_<trial_class>, site in ppc | alm_uni | alm_bi."""
    return run / distribution / f'{site}_{trial_class}'


def load_readouts(run: Path, animal: str, distribution: str, site: str = 'ppc', trial_class: str = 'opto') -> dict:
    """The readouts.npz of one animal/condition as a dict of arrays (keys like
    'opto__non_opto__curve_x', 'masking__opto__um'); empty if the run was --fast."""
    f = condition_dir(run, distribution, site, trial_class) / animal / 'readouts.npz'
    if not f.exists():
        return {}
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}


def load_switches(run: Path) -> dict:
    """The switch-adaptation tables: switches, pre_post, convergence, sessions, overnight,
    psychometrics, psychometric_curves."""
    d = run / 'switches'
    return {f.stem: pd.read_csv(f) for f in sorted(d.glob('*.csv'))}


def load_light_artefact(run: Path, distribution: str = 'Uniform') -> dict:
    """The light-artefact tables for one distribution: contrasts (light-on − off per set and stat, plus
    kind='site_dod'), levels, bins (P(B) per stimulus bin on vs off), group_rows, group_tests."""
    d = run / distribution
    out = {'contrasts': _read_all(d, 'light_contrasts'), 'levels': _read_all(d, 'levels'),
           'bins': _read_all(d, 'choice_by_bin')}
    out['group_rows'] = pd.read_csv(d / 'group' / 'group_rows.csv') if (d / 'group' / 'group_rows.csv').exists() else pd.DataFrame()
    out['group_tests'] = pd.read_csv(d / 'group' / 'group_tests.csv') if (d / 'group' / 'group_tests.csv').exists() else pd.DataFrame()
    return out


def load_consensus(run: Path, distribution: str = 'uniform') -> tuple[pd.DataFrame, str]:
    """(assignments, summary text) of a model-identification run for one distribution."""
    d = run / 'consensus' / distribution
    return pd.read_csv(d / 'assignments.csv'), (d / 'summary.txt').read_text()


# ── the experiment (for the single-animal pipeline cells) ───────────────────

def load_experiment():
    """The real snapshot, or the synthetic experiment under SC_NB_SYNTHETIC=1."""
    if SYNTHETIC:
        from sound_categorisation.data.cohort import ensure_presets
        from sound_categorisation.data.synthetic import synthetic_experiment
        ensure_presets()
        return synthetic_experiment()
    from sound_categorisation.data.cohort import load_experiment_any
    return load_experiment_any()


def first_animal(df: pd.DataFrame, genotype: str | None = None) -> str:
    sub = df if genotype is None else df[df.genotype == genotype]
    return sorted(sub.animal.unique())[0]


__all__ = ['SYNTHETIC', 'RUN', 'OPTO_COHORT', 'SWITCH_COHORT', 'MODEL_COHORT', 'REPO_ROOT',
           'GENOTYPE_COLOUR', 'COLOURS', 'PALETTE', 'np', 'pd', 'plt', 'Path',
           'banner', 'open_run', 'load_contrasts', 'load_levels', 'load_group', 'load_trajectory', 'condition_dir',
           'load_readouts', 'load_switches', 'load_light_artefact', 'load_consensus', 'load_experiment', 'first_animal']
