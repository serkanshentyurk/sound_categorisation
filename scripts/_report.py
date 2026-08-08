"""Shared assembly for the PPC / ALM opto PDF reports.

Torch-free: loads the experiment via ``behav_utils`` (snapshot or CSV) and never
imports ``scripts.config`` (which pulls in the fitting stack). Holds the loaders,
the genotype lookup, the contrast computations, and a ``PdfPages`` page builder
used by the four report scripts. The scripts are thin CLI wrappers around this.

Design notes
------------
* Per-animal page order is psychometric first (model-free behaviour), then the
  contrast grids: within-phase -> between-phase -> delta-of-deltas (-> ALM-vs-PPC
  for the ALM reports).
* Unit convention (matches the notebooks): within-phase panels draw the trial CI
  + permutation p (laser randomised per trial); between-phase and delta-of-deltas
  draw the trial+session dual CI with paired p and the "excludes 0" mark
  (session type was assigned per session, so the session bootstrap is the honest
  interval).
* PPC reports carry the base stat set; ALM reports additionally carry
  ``reaction_time`` and ``reaction_time_jitter``.
"""
from __future__ import annotations

import math
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# The project layer (plotting/) and scripts/ need the repo root on the path,
# independent of the caller's cwd, before the project-level imports below.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from behav_utils.data.loading import load_experiment
from behav_utils.data.ops.selection import select_sessions
from behav_utils.data.ops.filtering import filter_trials
from behav_utils.analysis import (
    compute_delta_stat, compute_interaction, collect_rows, compare_groups,
    compute_psychometric, compute_um, average_um,
)
from behav_utils.analysis.adaptation import compute_adaptation
from behav_utils.plotting import plot_stat_comparison_single, plot_interaction_single
from behav_utils.plotting.psychometric import plot_psychometric
from behav_utils.plotting.update_matrix import plot_um
from behav_utils.config.schema import load_cohorts
from plotting.opto import plot_delta_swarm  # project-level swarm (per-animal points + rank p)

# ── stat sets (mirror the notebooks) ─────────────────────────────────────────
STATS = ['accuracy', 'hard_accuracy', 'easy_accuracy', 'recency', 'side_bias', 'psychometric', 'lose_shift', 'win_stay']
STATS_RT = STATS + ['reaction_time', 'reaction_time_jitter']
SENSITIVITY = ['accuracy', 'hard_accuracy', 'easy_accuracy', 'sigma', 'recency', 'lapse_low', 'lapse_high', 'lose_shift', 'win_stay']
BIAS = ['mu', 'side_bias']
DISPLAY = SENSITIVITY + BIAS
DISPLAY_RT = DISPLAY + ['reaction_time', 'reaction_time_jitter']

DU = ('trials', 'sessions')      # dual CI on between-phase / delta-of-deltas
N_PERM, N_BOOT = 100, 100
PSYCH = True                     # draw the psychometric page (slow: curve bootstrap)
ADAPT = True                     # draw the adaptation page for non-uniform phases
                                 # (slow: a PSE fit per rolling window per session)
ADAPT_FIXED_SIGMA = 0.175        # provisional fixed σ for the pse normative line —
                                 # one σ for every animal, so a single normative line
                                 # is comparable across the cohort. Set to the
                                 # cohort-typical / SBI-mean value when known; P3
                                 # replaces this with the per-animal SBI posterior.

SITE_TYPE = {'uni': 'alm_control_uni', 'bi': 'alm_control_bi'}
OFF = '#7f7f7f'
ON = '#1f77b4'
ALL = '#2ca02c'
ADAPT_COL = {'opto': ON, 'masking': OFF}   # adaptation trajectory colours
GENO_COL = {'het': '#d62728', 'wt': '#2ca02c'}  # genotype colours (group adaptation)


# ── loading / genotype ───────────────────────────────────────────────────────
def load_any(config_path=None, snapshot_path=None):
    """Load the experiment (snapshot if available, else CSV via config).

    Torch-free. When ``snapshot_path`` is not given it defaults to the SAME
    location the notebooks use — ``snapshot_dir(repo)/sound_cat_snapshot.pkl``
    (cluster path on Linux, ``<repo>/../../data/behaviour/snapshots/`` locally) —
    so a normal run needs no ``--snapshot``. Falls back to the CSV loader only if
    no snapshot is found, and raises a clear error (naming the snapshot path it
    tried) rather than the cryptic raw-data-dir error if that also fails.
    """
    config_path = Path(config_path) if config_path else _ROOT / 'config.yaml'
    # Resolve the default snapshot path the notebook way (repo-derived / cluster).
    if snapshot_path is None:
        try:
            from scripts.snapshot import snapshot_dir, SNAPSHOT_FILENAME
            snapshot_path = snapshot_dir(_ROOT) / SNAPSHOT_FILENAME
        except Exception:
            snapshot_path = None
    snapshot_path = Path(snapshot_path) if snapshot_path else None

    if snapshot_path and snapshot_path.exists():
        from scripts.snapshot import load_snapshot
        experiment, _ = load_snapshot(
            snapshot_path, config_path=config_path if config_path.exists() else None)
        print(f'loaded snapshot: {snapshot_path}')
        return experiment

    if snapshot_path is not None:
        warnings.warn(f'no snapshot at {snapshot_path}; trying CSV (needs raw data mounted)')
    if config_path.exists():
        try:
            return load_experiment(config_path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f'{exc}\n\nNo snapshot found either (looked for {snapshot_path}). '
                f'Pass --snapshot /path/to/sound_cat_snapshot.pkl, or mount the raw '
                f'data dir the config points to.') from None
    raise FileNotFoundError(
        f'no snapshot at {snapshot_path!r} and no config at {config_path}')


def gather_genotypes(experiment):
    """{animal_id: genotype} and {genotype: [animal_id, ...]} from loaded animals.

    Genotype is read from each animal's ``.genotype`` (loaded from
    animal_metadata.json) — the single source of truth. Missing -> 'unknown'.
    """
    by_animal = {aid: str(getattr(a, 'genotype', 'unknown') or 'unknown').lower()
                 for aid, a in experiment.animals.items()}
    unknown = sorted(a for a, g in by_animal.items() if g in ('unknown', 'none', ''))
    if unknown:
        warnings.warn(f"{len(unknown)} animal(s) without genotype: {', '.join(unknown)}")
    groups: dict = {}
    for aid, g in by_animal.items():
        groups.setdefault(g, []).append(aid)
    for g in groups:
        groups[g].sort()
    return by_animal, groups


# ── session collection (real select_sessions; scripts call these) ────────────
def collect_sessions_ppc(animal, distribution):
    return {
        'opto': select_sessions(animal, distribution=distribution, session_type='opto'),
        'masking': select_sessions(animal, distribution=distribution, session_type='masking'),
    }


def collect_sessions_alm(animal, distribution, site):
    stype = SITE_TYPE[site]
    return {
        'alm': select_sessions(animal, distribution=distribution, session_type=stype),
        'masking': select_sessions(animal, distribution=distribution, session_type='masking'),
        'opto': select_sessions(animal, distribution=distribution, session_type='opto'),
    }


# ── contrast computations ────────────────────────────────────────────────────
def ppc_contrasts(opto, masking, toi, stats, n_boot=None, n_perm=None):
    """PPC: within-phase ({toi} vs non_opto), between-phase (opto vs masking,
    all trials), and their delta-of-deltas. ``n_boot=0`` gives point diffs only
    (for the group fold) and skips the interaction object.
    """
    n_boot = N_BOOT if n_boot is None else n_boot
    n_perm = N_PERM if n_perm is None else n_perm
    key = f'{toi}_vs_non_opto'
    o_toi, o_non = filter_trials(opto, trial_type=toi), filter_trials(opto, trial_type='non_opto')
    m_toi, m_non = filter_trials(masking, trial_type=toi), filter_trials(masking, trial_type='non_opto')
    o_all, m_all = filter_trials(opto, trial_type='all'), filter_trials(masking, trial_type='all')
    r = {}
    if o_toi and o_non:
        r['within'] = compute_delta_stat({toi: o_toi, 'non_opto': o_non}, stats=stats,
                                          reference='non_opto', n_permutations=n_perm,
                                          n_bootstrap=n_boot, resample_units=DU)
    if m_toi and m_non:
        r['within_masking'] = compute_delta_stat({toi: m_toi, 'non_opto': m_non}, stats=stats,
                                                  reference='non_opto', n_permutations=n_perm,
                                                  n_bootstrap=n_boot, resample_units=DU)
    if o_all and m_all:
        r['between'] = compute_delta_stat({'opto': o_all, 'masking': m_all}, stats=stats,
                                          reference='masking', n_permutations=0,
                                          n_bootstrap=n_boot, resample_units=DU)
    if n_boot > 0 and 'within' in r and 'within_masking' in r:
        r['dod'] = compute_interaction(r['within'], r['within_masking'], contrast=key,
                                       label_a='opto', label_b='masking')
    return r, key


def alm_contrasts(alm, masking, opto, toi, stats, n_boot=None, n_perm=None):
    """ALM: the four contrasts — within-phase ({toi} vs non_opto), between-phase
    (ALM vs masking, all trials), delta-of-deltas, and ALM vs PPC-opto (all
    trials). ``n_boot=0`` -> point diffs only, interaction skipped.
    """
    n_boot = N_BOOT if n_boot is None else n_boot
    n_perm = N_PERM if n_perm is None else n_perm
    key = f'{toi}_vs_non_opto'
    a_toi, a_non = filter_trials(alm, trial_type=toi), filter_trials(alm, trial_type='non_opto')
    m_toi, m_non = filter_trials(masking, trial_type=toi), filter_trials(masking, trial_type='non_opto')
    a_all, m_all, o_all = (filter_trials(alm, trial_type='all'),
                           filter_trials(masking, trial_type='all'),
                           filter_trials(opto, trial_type='all'))
    r = {}
    if a_toi and a_non:
        r['within'] = compute_delta_stat({toi: a_toi, 'non_opto': a_non}, stats=stats,
                                          reference='non_opto', n_permutations=n_perm,
                                          n_bootstrap=n_boot, resample_units=DU)
    if m_toi and m_non:
        r['within_masking'] = compute_delta_stat({toi: m_toi, 'non_opto': m_non}, stats=stats,
                                                  reference='non_opto', n_permutations=n_perm,
                                                  n_bootstrap=n_boot, resample_units=DU)
    if a_all and m_all:
        r['between'] = compute_delta_stat({'alm': a_all, 'masking': m_all}, stats=stats,
                                          reference='masking', n_permutations=0,
                                          n_bootstrap=n_boot, resample_units=DU)
    if n_boot > 0 and 'within' in r and 'within_masking' in r:
        r['dod'] = compute_interaction(r['within'], r['within_masking'], contrast=key,
                                       label_a='alm', label_b='masking')
    if a_all and o_all:
        r['vs_ppc'] = compute_delta_stat({'alm': a_all, 'opto': o_all}, stats=stats,
                                         reference='opto', n_permutations=0,
                                         n_bootstrap=n_boot, resample_units=DU)
    return r, key


def _dod_point(r, key):
    """Per-animal delta-of-deltas point = within delta - within_masking delta."""
    wa = r['within']['contrasts'][key]['diffs']
    wm = r['within_masking']['contrasts'][key]['diffs']
    return {s: wa[s] - wm.get(s, np.nan) for s in wa}


# ── figure builders ──────────────────────────────────────────────────────────
def _grid(result, stats, kind, units, title, ncols=4):
    nrows = math.ceil(len(stats) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.0 * nrows), squeeze=False)
    axf = axes.ravel()
    for ax, stat in zip(axf, stats):
        if kind == 'cmp':
            plot_stat_comparison_single(result, stat, ax=ax, units=units)
        else:
            plot_interaction_single(result, stat, ax=ax, units=units)
    for ax in axf[len(stats):]:
        fig.delaxes(ax)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def _psychometric(phase_sessions, toi, suptitle):
    """One panel per phase, each overlaying non_opto (grey), {toi} (blue), and
    all-trials (green). phase_sessions: list of (panel_title, sessions)."""
    n = max(1, len(phase_sessions))
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.0), squeeze=False)
    for ax, (title, sess) in zip(axes[0], phase_sessions):
        non = filter_trials(sess, trial_type='non_opto')
        toi_c = filter_trials(sess, trial_type=toi)
        allc = filter_trials(sess, trial_type='all')
        if allc:
            plot_psychometric(compute_psychometric(allc), ax=ax, color=ALL, label='all trials')
        if non:
            plot_psychometric(compute_psychometric(non), ax=ax, color=OFF, label='non_opto')
        if toi_c:
            plot_psychometric(compute_psychometric(toi_c), ax=ax, color=ON, label=toi)
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def _um_animal(phase_sessions, toi, suptitle):
    """Per-condition update matrices: rows = phases, cols = [non_opto, {toi}]."""
    n = max(1, len(phase_sessions))
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n), squeeze=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # empty UM cells -> NaN (blank)
        for r, (title, sess) in enumerate(phase_sessions):
            non = filter_trials(sess, trial_type='non_opto')
            toi_c = filter_trials(sess, trial_type=toi)
            if non:
                plot_um(compute_um(non), ax=axes[r, 0])
            axes[r, 0].set_title(f'{title} \u00b7 non_opto', fontsize=9)
            if toi_c:
                plot_um(compute_um(toi_c), ax=axes[r, 1])
            axes[r, 1].set_title(f'{title} \u00b7 {toi}', fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def _group_psychometric(sba, by_animal, phase_specs, suptitle):
    """WT-vs-HET pooled all-trials psychometric, one panel per phase.
    phase_specs: list of (panel_title, phase_key)."""
    n = max(1, len(phase_specs))
    fig, axes = plt.subplots(1, n, figsize=(4.7 * n, 4.0), squeeze=False)
    for ax, (title, pk) in zip(axes[0], phase_specs):
        for g, col in [('wt', OFF), ('het', ON)]:
            ids = [a for a in sba if by_animal.get(a) == g]
            pooled = []
            for a in ids:
                pooled += filter_trials(sba[a].get(pk, []), trial_type='all')
            if pooled:
                plot_psychometric(compute_psychometric(pooled), ax=ax, color=col,
                                  label=f'{g} (n={len(ids)})')
        ax.set_title(f'{title} \u00b7 all trials', fontsize=9)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def _group_um(sba, by_animal, phase_key, toi, suptitle):
    """Genotype-mean UM (average of per-animal UMs): rows = [het, wt],
    cols = [non_opto, {toi}], for one phase."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 8), squeeze=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # empty UM cells -> NaN (blank)
        for row, g in enumerate(['het', 'wt']):
            ids = [a for a in sba if by_animal.get(a) == g]
            for col, tt in enumerate(['non_opto', toi]):
                ums = []
                for a in ids:
                    c = filter_trials(sba[a].get(phase_key, []), trial_type=tt)
                    if c:
                        ums.append(compute_um(c))
                if ums:
                    plot_um(average_um(ums, min_sources=2), ax=axes[row, col])
                axes[row, col].set_title(f"{g} \u00b7 {tt} (n={len(ids)})", fontsize=9)
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def _swarm(df, res, display, title, ncols=4):
    nrows = math.ceil(len(display) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.3 * ncols, 3.3 * nrows), squeeze=False)
    axf = axes.ravel()
    for ax, stat in zip(axf, display):
        plot_delta_swarm(df, stat, ax=ax, p_value=res.get(stat, {}).get('p'),
                         group_col='group', value_col='value')
    for ax in axf[len(display):]:
        fig.delaxes(ax)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def _rows(diffs, aid, by_animal):
    return collect_rows([{'stat': k, 'value': float(v)} for k, v in diffs.items()],
                        animal=aid, group=by_animal[aid])


def _mean_curve(segs):
    """Mean over a list of ``{'trials','values'}`` at each rolling-window centre
    (centres align across sessions when window & step match)."""
    acc = defaultdict(list)
    for e in segs:
        for x, y in zip(e['trials'], e['values']):
            if np.isfinite(y):
                acc[float(x)].append(y)
    if not acc:
        return None, None
    xs = np.array(sorted(acc))
    return xs, np.array([np.mean(acc[x]) for x in xs])


def _adapt_curves(animal, distribution, window=50, step=10):
    """Per-session rolling-PSE trajectories for one animal at ``distribution``,
    standardised to the expert-uniform baseline. The pse normative offset uses a
    FIXED provisional σ (``ADAPT_FIXED_SIGMA``) — one σ for every animal, so a
    single normative line is comparable across the cohort; σ affects only that
    line, not the trajectories. Never raises. Returns
    ``{'sessions': [{'session_type','trials','values'}], 'normative': float}``.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            r = compute_adaptation(animal, distribution, stat_names=['pse'],
                                   mode='per_session', window=window, step=step,
                                   sigma_source='fixed', sigma_value=ADAPT_FIXED_SIGMA,
                                   trials='all')
        segs = [{'session_type': e.get('session_type', ''),
                 'trials': np.asarray(e['trials'], float),
                 'values': np.asarray(e['values']['pse'], float)}
                for e in r['sessions']]
        return {'sessions': segs,
                'normative': float(r.get('normative', {}).get('pse', np.nan))}
    except Exception as exc:
        warnings.warn(f"{getattr(animal, 'animal_id', '?')}: adaptation "
                      f"({distribution}) failed: {exc}")
        return {'sessions': [], 'normative': np.nan}


def _animal_means(curves_by_animal, aids, phase):
    """Each animal's mean curve for ``phase`` -> list of ``{'trials','values'}``
    (the mean over these is the mean-of-animals — equal weight per animal, the
    correct unit of replication, not session-pooled)."""
    out = []
    for a in aids:
        segs = [e for e in curves_by_animal[a]['sessions'] if e['session_type'] == phase]
        mx, my = _mean_curve(segs)
        if mx is not None:
            out.append({'trials': mx, 'values': my})
    return out


def _adapt_finish(ax, has_data, norm, empty_msg):
    """Shared axes dressing: expert-uniform 0-line + provisional fixed-σ normative."""
    ax.axhline(0.0, color='0.6', lw=1, zorder=1)               # expert-uniform baseline
    if np.isfinite(norm):
        ax.axhline(norm, color='crimson', ls='--', lw=1.2, zorder=1,
                   label='normative (fixed σ)')
    if not has_data and empty_msg:
        ax.text(0.5, 0.5, empty_msg, transform=ax.transAxes,
                ha='center', va='center', color='0.5')
    ax.set_xlabel('trial in session')
    ax.set_ylabel('rolling PSE (− expert baseline)')
    if ax.get_legend_handles_labels()[0]:
        ax.legend(frameon=False, fontsize=8)


def _adaptation(animal, distribution, suptitle):
    """Per-animal adaptation into ``distribution``: thin line per session + thick
    per-type mean, opto vs masking, standardised to the expert-uniform 0-line.
    Normative line uses a fixed provisional σ (``ADAPT_FIXED_SIGMA``); P3 swaps
    it for the SBI posterior. Never raises."""
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    cur = _adapt_curves(animal, distribution)
    drew = False
    for phase, col in ADAPT_COL.items():
        segs = [e for e in cur['sessions'] if e['session_type'] == phase]
        for e in segs:                                         # thin: one per session
            ax.plot(e['trials'], e['values'], color=col, lw=0.8, alpha=0.35, zorder=2)
        mx, my = _mean_curve(segs)                             # thick: per-type mean
        if mx is not None:
            ax.plot(mx, my, color=col, lw=2.4, zorder=3,
                    label=f'{phase} (mean of {len(segs)})')
            drew = True
    _adapt_finish(ax, drew, cur['normative'], 'no adaptation data at this phase')
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    return fig


def _group_adaptation(animals, by_animal, distribution, suptitle):
    """Group (WT vs HET) adaptation, 1x4: HET opto-vs-masking | WT opto-vs-masking
    | masking HET-vs-WT | opto HET-vs-WT. Thin line = each animal's mean curve;
    thick line = mean across animals (equal weight per animal). Single fixed-σ
    normative line, comparable across the cohort. Never raises per animal."""
    cba = {aid: _adapt_curves(an, distribution) for aid, an in animals.items()}
    aids = list(cba)
    het = [a for a in aids if by_animal.get(a) == 'het']
    wt = [a for a in aids if by_animal.get(a) == 'wt']
    norm = next((cba[a]['normative'] for a in aids
                 if np.isfinite(cba[a].get('normative', np.nan))), np.nan)

    def panel(ax, series, empty_msg):
        has = False
        for label, gaids, phase, col in series:
            ams = _animal_means(cba, gaids, phase)
            for am in ams:                                     # thin: one per animal (its mean)
                ax.plot(am['trials'], am['values'], color=col, lw=0.7, alpha=0.35, zorder=2)
            gx, gy = _mean_curve(ams)                          # thick: mean of animals
            if gx is not None:
                ax.plot(gx, gy, color=col, lw=2.6, zorder=4, label=f'{label} (n={len(ams)})')
                has = True
        _adapt_finish(ax, has, norm, empty_msg)

    fig, ax = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    panel(ax[0], [(ph, het, ph, ADAPT_COL[ph]) for ph in ADAPT_COL], 'no HET sessions')
    ax[0].set_title(f'HET (n={len(het)}) · opto vs masking', fontsize=10)
    panel(ax[1], [(ph, wt, ph, ADAPT_COL[ph]) for ph in ADAPT_COL], 'no WT sessions')
    ax[1].set_title(f'WT (n={len(wt)}) · opto vs masking', fontsize=10)
    panel(ax[2], [('het', het, 'masking', GENO_COL['het']),
                  ('wt', wt, 'masking', GENO_COL['wt'])], 'no masking sessions')
    ax[2].set_title('masking · HET vs WT', fontsize=10)
    panel(ax[3], [('het', het, 'opto', GENO_COL['het']),
                  ('wt', wt, 'opto', GENO_COL['wt'])], 'no opto sessions')
    ax[3].set_title('opto · HET vs WT', fontsize=10)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    return fig


# ── per-animal PDFs ──────────────────────────────────────────────────────────
def build_ppc_per_animal(animal, sbt, aid, genotype, distribution, toi, out_path):
    opto, masking = sbt.get('opto', []), sbt.get('masking', [])
    r, key = ppc_contrasts(opto, masking, toi, STATS)
    tag = f"{aid} \u00b7 {genotype} \u00b7 {distribution}"
    with PdfPages(out_path) as pdf:
        if PSYCH:
            phases_ps = [('opto', opto), ('masking', masking)]
            pdf.savefig(_psychometric(phases_ps, toi, f"{tag} \u00b7 psychometric")); plt.close('all')
            pdf.savefig(_um_animal(phases_ps, toi, f"{tag} \u00b7 update matrices")); plt.close('all')
        if 'within' in r:
            pdf.savefig(_grid(r['within'], DISPLAY, 'cmp', ('trials',),
                              f"{tag} \u00b7 within-phase \u00b7 OPTO sessions ({toi} vs non_opto)")); plt.close('all')
        if 'within_masking' in r:
            pdf.savefig(_grid(r['within_masking'], DISPLAY, 'cmp', ('trials',),
                              f"{tag} \u00b7 within-phase \u00b7 MASKING sessions ({toi} vs non_opto)")); plt.close('all')
        if 'between' in r:
            pdf.savefig(_grid(r['between'], DISPLAY, 'cmp', DU,
                              f"{tag} \u00b7 between-phase (opto vs masking, all trials)")); plt.close('all')
        if 'dod' in r:
            pdf.savefig(_grid(r['dod'], DISPLAY, 'int', DU,
                              f"{tag} \u00b7 delta-of-deltas (silencing beyond artifact)")); plt.close('all')
        if ADAPT and animal is not None and distribution.lower() != 'uniform':
            pdf.savefig(_adaptation(animal, distribution,
                        f"{tag} \u00b7 adaptation \u00b7 rolling PSE (opto vs masking)")); plt.close('all')
    return out_path


def build_alm_per_animal(sbt, aid, genotype, distribution, site_label, toi, out_path):
    alm, masking, opto = sbt.get('alm', []), sbt.get('masking', []), sbt.get('opto', [])
    r, key = alm_contrasts(alm, masking, opto, toi, STATS_RT)
    tag = f"{aid} \u00b7 {genotype} \u00b7 {distribution} \u00b7 ALM-{site_label}"
    with PdfPages(out_path) as pdf:
        if PSYCH:
            phases_ps = [('ALM', alm), ('masking', masking), ('opto', opto)]
            pdf.savefig(_psychometric(phases_ps, toi, f"{tag} \u00b7 psychometric")); plt.close('all')
            pdf.savefig(_um_animal(phases_ps, toi, f"{tag} \u00b7 update matrices")); plt.close('all')
        if 'within' in r:
            pdf.savefig(_grid(r['within'], DISPLAY_RT, 'cmp', ('trials',),
                              f"{tag} \u00b7 within-phase \u00b7 ALM sessions ({toi} vs non_opto)")); plt.close('all')
        if 'within_masking' in r:
            pdf.savefig(_grid(r['within_masking'], DISPLAY_RT, 'cmp', ('trials',),
                              f"{tag} \u00b7 within-phase \u00b7 MASKING sessions ({toi} vs non_opto)")); plt.close('all')
        if 'between' in r:
            pdf.savefig(_grid(r['between'], DISPLAY_RT, 'cmp', DU,
                              f"{tag} \u00b7 between-phase (ALM vs masking, all trials)")); plt.close('all')
        if 'dod' in r:
            pdf.savefig(_grid(r['dod'], DISPLAY_RT, 'int', DU,
                              f"{tag} \u00b7 delta-of-deltas (silencing beyond artifact)")); plt.close('all')
        if 'vs_ppc' in r:
            pdf.savefig(_grid(r['vs_ppc'], DISPLAY_RT, 'cmp', DU,
                              f"{tag} \u00b7 ALM vs PPC-opto (all trials)")); plt.close('all')
    return out_path


# ── group (WT vs HET) PDFs ───────────────────────────────────────────────────
def build_ppc_group(animals, sba, by_animal, distribution, toi, out_path):
    rows = {'within': [], 'within_masking': [], 'between': [], 'dod': []}
    for aid, sbt in sba.items():
        r, key = ppc_contrasts(sbt.get('opto', []), sbt.get('masking', []), toi, STATS,
                               n_boot=0, n_perm=0)
        if 'within' in r:
            rows['within'] += _rows(r['within']['contrasts'][key]['diffs'], aid, by_animal)
        if 'within_masking' in r:
            rows['within_masking'] += _rows(r['within_masking']['contrasts'][key]['diffs'], aid, by_animal)
        if 'between' in r:
            rows['between'] += _rows(r['between']['contrasts']['opto_vs_masking']['diffs'], aid, by_animal)
        if 'within' in r and 'within_masking' in r:
            rows['dod'] += _rows(_dod_point(r, key), aid, by_animal)
    titles = {'within': f'within-phase \u00b7 OPTO sessions ({toi} vs non_opto)',
              'within_masking': f'within-phase \u00b7 MASKING sessions ({toi} vs non_opto)',
              'between': 'between-phase (opto vs masking, all trials)',
              'dod': 'delta-of-deltas (silencing beyond artifact)'}
    prelude = []
    if PSYCH:
        prelude.append(_group_psychometric(
            sba, by_animal, [('opto', 'opto'), ('masking', 'masking')],
            f'{distribution} \u00b7 group psychometric (WT vs HET)'))
        prelude.append(_group_um(sba, by_animal, 'opto', toi,
                                 f'{distribution} \u00b7 opto \u00b7 genotype-mean UM'))
        prelude.append(_group_um(sba, by_animal, 'masking', toi,
                                 f'{distribution} \u00b7 masking \u00b7 genotype-mean UM'))
    if ADAPT and distribution.lower() != 'uniform':
        prelude.append(_group_adaptation(
            animals, by_animal, distribution,
            f'{distribution} \u00b7 group adaptation \u00b7 rolling PSE (WT vs HET)'))
    _write_group_pdf(rows, titles, DISPLAY, distribution, out_path, prelude=prelude)
    return out_path


def build_alm_group(sba, by_animal, distribution, site_label, toi, out_path):
    rows = {'within': [], 'within_masking': [], 'between': [], 'dod': [], 'vs_ppc': []}
    for aid, sbt in sba.items():
        r, key = alm_contrasts(sbt.get('alm', []), sbt.get('masking', []), sbt.get('opto', []),
                               toi, STATS_RT, n_boot=0, n_perm=0)
        if 'within' in r:
            rows['within'] += _rows(r['within']['contrasts'][key]['diffs'], aid, by_animal)
        if 'within_masking' in r:
            rows['within_masking'] += _rows(r['within_masking']['contrasts'][key]['diffs'], aid, by_animal)
        if 'between' in r:
            rows['between'] += _rows(r['between']['contrasts']['alm_vs_masking']['diffs'], aid, by_animal)
        if 'vs_ppc' in r:
            rows['vs_ppc'] += _rows(r['vs_ppc']['contrasts']['alm_vs_opto']['diffs'], aid, by_animal)
        if 'within' in r and 'within_masking' in r:
            rows['dod'] += _rows(_dod_point(r, key), aid, by_animal)
    titles = {'within': f'within-phase \u00b7 ALM sessions ({toi} vs non_opto)',
              'within_masking': f'within-phase \u00b7 MASKING sessions ({toi} vs non_opto)',
              'between': 'between-phase (ALM vs masking, all trials)',
              'dod': 'delta-of-deltas (silencing beyond artifact)',
              'vs_ppc': 'ALM vs PPC-opto (all trials)'}
    prefix = f'{distribution} \u00b7 ALM-{site_label}'
    prelude = []
    if PSYCH:
        prelude.append(_group_psychometric(
            sba, by_animal, [('ALM', 'alm'), ('masking', 'masking'), ('opto', 'opto')],
            f'{prefix} \u00b7 group psychometric (WT vs HET)'))
        for pk, lab in [('alm', 'ALM'), ('masking', 'masking'), ('opto', 'opto')]:
            prelude.append(_group_um(sba, by_animal, pk, toi,
                                     f'{prefix} \u00b7 {lab} \u00b7 genotype-mean UM'))
    _write_group_pdf(rows, titles, DISPLAY_RT, prefix, out_path, prelude=prelude)
    return out_path


def _write_group_pdf(rows, titles, display, prefix, out_path, prelude=()):
    with PdfPages(out_path) as pdf:
        for fig in prelude:
            pdf.savefig(fig); plt.close(fig)
        for key in titles:
            df = pd.DataFrame(rows[key])
            if not len(df):
                print(f'  {key}: no rows, skipped')
                continue
            res = compare_groups(df, group_col='group')
            n = df['animal'].nunique()
            title = f"{prefix} \u00b7 {titles[key]} \u2014 WT vs HET (n={n})"
            pdf.savefig(_swarm(df, res, display, title)); plt.close('all')


# ── quick knobs / selftest ───────────────────────────────────────────────────
def configure(fast=False):
    """--fast: scalar-only stats, few draws, no psychometric page — a seconds-long
    structure check rather than the full overnight report."""
    global STATS, STATS_RT, DISPLAY, DISPLAY_RT, N_BOOT, N_PERM, PSYCH, ADAPT
    if fast:
        STATS = ['accuracy', 'side_bias']
        STATS_RT = ['accuracy', 'side_bias', 'reaction_time', 'reaction_time_jitter']
        DISPLAY = list(STATS)
        DISPLAY_RT = list(STATS_RT)
        N_BOOT, N_PERM, PSYCH, ADAPT = 30, 30, False, False


def _synthetic_sessions(out_types, seed=0):
    """Tiny tagged synthetic sessions per type, for --selftest (no data load)."""
    from datetime import date as _date
    from behav_utils.data.structures import TrialData, SessionData, SessionMetadata

    def _tr(n, rng, b, rtm):
        s = rng.uniform(-1, 1, n); c = (s > 0).astype(float); ch = (s > b).astype(float)
        f = rng.random(n) < 0.15; ch[f] = 1 - ch[f]
        return TrialData(trial_number=np.arange(n), stimulus=s, category=c, choice=ch,
                         outcome=(ch == c).astype(float), correct=(ch == c),
                         abort=np.zeros(n, bool), opto_on=rng.random(n) < 0.3,
                         reaction_time=np.clip(rng.normal(rtm, 40, n), 80, None))
    out = {}
    for j, (k, (stype, rtm, b0)) in enumerate(out_types.items()):
        rng = np.random.default_rng(seed + j + 1)
        out[k] = [SessionData(session_id=f'{stype}{i}', session_idx=i, date=_date(2024, 1, 1 + i),
                              metadata=SessionMetadata(fields={'stage': 'x', 'distribution': 'Uniform'}),
                              trials=_tr(180, rng, rng.normal(b0, .2), rtm), session_type=stype)
                  for i in range(3)]
    return out


def run_selftest(out_dir):
    """Build one PPC and one ALM per-animal PDF from synthetic data — confirms the
    assembly runs in this environment without loading any experiment."""
    configure(fast=True)
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ppc = _synthetic_sessions({'opto': ('opto', 250, 0.35), 'masking': ('masking', 270, 0.05)})
    p1 = build_ppc_per_animal(None, ppc, 'SELFTEST', 'het', 'Uniform', 'opto', out_dir / 'selftest_ppc.pdf')
    alm = _synthetic_sessions({'alm': ('alm_control_uni', 300, 0.25),
                               'masking': ('masking', 270, 0.05), 'opto': ('opto', 250, 0.35)})
    p2 = build_alm_per_animal(alm, 'SELFTEST', 'het', 'Uniform', 'uni', 'opto', out_dir / 'selftest_alm_uni.pdf')
    print(f'selftest OK -> {p1}\n            {p2}')
    return p1, p2
