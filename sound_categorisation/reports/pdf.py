"""PDF assembly: per-animal and group documents from computed results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from sound_categorisation.reports import figures as F
from sound_categorisation.reports.compute import BETWEEN_KEY, PHASES, AnimalResult, GroupResult, Settings

__all__ = ['animal_pdf', 'group_pdf']

_TITLES = {
    'within': 'within-phase · {a} sessions ({toi} vs non_opto)',
    'within_masking': 'within-phase · MASKING sessions ({toi} vs non_opto)',
    'between': 'between-phase ({between}, all trials)',
    'dod': 'delta-of-deltas (silencing beyond artefact)',
    'vs_ppc': 'ALM vs PPC-opto (all trials)',
}


def _title(kind: str, r_or_g) -> str:
    design = r_or_g.design
    return _TITLES[kind].format(a='OPTO' if design == 'ppc' else 'ALM', toi=r_or_g.toi,
                                between=BETWEEN_KEY[design].replace('_', ' '))


def _save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def animal_pdf(r: AnimalResult, out_path, settings: Settings) -> Path:
    """Psychometric + UM pages, then within → within_masking → between → dod (→ vs_ppc), then the session trajectory."""
    display = settings.display_for(r.design)
    tag = r.tag
    with PdfPages(out_path) as pdf:
        if r.readouts:
            _save(pdf, F.psychometric_page(r, f'{tag} · psychometric'))
            _save(pdf, F.update_matrix_page(r, f'{tag} · update matrices'))
        c = r.contrasts
        if 'within' in c:
            _save(pdf, F.contrast_grid(c['within'], display, F.TRIAL_UNITS, f'{tag} · {_title("within", r)}'))
        if 'within_masking' in c:
            _save(pdf, F.contrast_grid(c['within_masking'], display, F.TRIAL_UNITS,
                                       f'{tag} · {_title("within_masking", r)}'))
        if 'between' in c:
            _save(pdf, F.contrast_grid(c['between'], display, settings.units, f'{tag} · {_title("between", r)}'))
        if 'dod' in c:
            _save(pdf, F.interaction_grid(c['dod'], display, settings.units, f'{tag} · {_title("dod", r)}'))
        if 'vs_ppc' in c:
            _save(pdf, F.contrast_grid(c['vs_ppc'], display, settings.units, f'{tag} · {_title("vs_ppc", r)}'))
        if r.trajectory is not None:
            _save(pdf, F.trajectory_page(r, f'{tag} · session trajectory'))
    return Path(out_path)


def group_pdf(g: GroupResult, per_animal: Dict[str, AnimalResult], out_path, settings: Settings) -> Path:
    """Group psychometric + genotype-mean UMs, adaptation, then one swarm page per contrast kind."""
    display = settings.display_for(g.design)
    prefix = f'{g.distribution}' + (f' · ALM-{g.site}' if g.site else '')
    with PdfPages(out_path) as pdf:
        if per_animal and any(r.readouts for r in per_animal.values()):
            _save(pdf, F.group_psychometric_page(per_animal, f'{prefix} · group psychometric (WT vs HET)'))
            for phase in PHASES[g.design]:
                _save(pdf, F.group_update_matrix_page(per_animal, phase, g.toi,
                                                      f'{prefix} · {phase} · genotype-mean UM'))
        if g.trajectories:
            _save(pdf, F.group_trajectory_page(g, f'{prefix} · session trajectories (WT vs HET)'))
        kinds = [k for k in ('within', 'within_masking', 'between', 'dod', 'vs_ppc') if k in set(g.rows['kind'])]
        for kind in kinds:
            n = g.rows.loc[g.rows['kind'] == kind, 'animal'].nunique()
            _save(pdf, F.swarm_page(g, kind, display, f'{prefix} · {_title(kind, g)} — WT vs HET (n={n})'))
    return Path(out_path)
