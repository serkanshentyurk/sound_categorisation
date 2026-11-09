"""
The daily tracker's report: one self-contained HTML file, figures embedded as PNG, no server.

    from sound_categorisation.reports.daily_tracker import write_daily_report
    write_daily_report(tracker, path, source='user@<login node>:/ceph/.../Raw')

The page opens on the overview: expected animals with no session on ceph first, then the animals that need a
look (a flag on or since their latest session), then the rest, each with a 14-day trend. Clicking an animal
shows only that animal; the address then ends in #<animal>, so it can be bookmarked or sent. An animal's
view has the days newest first (with the session number), values against date with preset changes marked,
the latest psychometric curve, and per condition the pooled and mean summaries next to the individual
sessions. ← and → step through the animals, Esc returns to the overview, and ↑ top is always there.
Without JavaScript, and when printed, every section shows one after the other (a page per animal on paper).

Layout is decided here; the panels come from plotting.daily_tracker. Figures are matplotlib Figure objects
(no pyplot), so building the report never touches a GUI backend.
"""

from __future__ import annotations

import base64
import html
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from behav_utils.plotting.readouts import plot_psychometric_curve
from matplotlib.figure import Figure

from sound_categorisation.plotting.daily_tracker import LIGHT_COLOURS, plot_condition_curves, plot_daily_values

# column: (header, how to show it, what it means, shown on hover)
COLUMNS = {
    'session_no': ('session', 'count', "the animal's session number on ceph (first = 1; test runs and "
                                        "sessions_to_ignore not counted)"),
    'preset_no': ('on preset', 'count', 'sessions on this preset so far, this one included'),
    'preset': ('preset', 'text', 'the rig preset that ran (Session_Type in Trial_Summary)'),
    'stage': ('stage', 'text', 'the training stage the rig ran'),
    'distribution': ('distribution', 'text', 'stimulus distribution'),
    'rig': ('rig', 'text', 'Rig_ID from Session_Parameters'),
    'trials': ('trials', 'count', 'trials in the session (early steps: water deliveries)'),
    'responded': ('responded', 'count', 'trials with a lick in the response window'),
    'correct': ('correct', 'count', 'correct trials, light trials included: what the mouse earned'),
    'no_response': ('no response', 'count', "Trial_Outcome 'Abort': no lick in the response window"),
    'early': ('early lick', 'count', "Trial_Outcome 'Early': aborted by an early lick"),
    'light_trials': ('light trials', 'count', 'trials with light (laser, or the masking LED)'),
    'accuracy': ('accuracy', 'prop', 'proportion correct, trials without light'),
    'hard_accuracy': ('hard', 'prop', 'accuracy on |stimulus| < 0.3, trials without light'),
    'easy_accuracy': ('easy', 'prop', 'accuracy on |stimulus| ≥ 0.3, trials without light'),
    'right_choices_minus_50': ('right choices − 0.5', 'signed', 'P(choose right) − 0.5, trials without light'),
    'accuracy_right_minus_left': ('accuracy right − left', 'signed',
                                  'accuracy on right-rewarded minus left-rewarded trials, trials without light'),
    'pse': ('PSE', 'signed', 'point of subjective equality, free 4-parameter fit (category B positive)'),
    'sigma': ('sigma', 'prop', 'psychometric width'),
    'lapse_low': ('lapse low', 'prop', 'lapse rate at the low end'),
    'lapse_high': ('lapse high', 'prop', 'lapse rate at the high end'),
}
DAY_SHORT = ['session_no', 'preset_no', 'preset', 'trials', 'correct', 'accuracy', 'hard_accuracy',
             'easy_accuracy', 'right_choices_minus_50', 'accuracy_right_minus_left']
DAY_FULL = ['session_no', 'preset_no', 'preset', 'stage', 'distribution', 'rig', 'trials', 'responded', 'correct',
            'no_response', 'early', 'light_trials', 'accuracy', 'hard_accuracy', 'easy_accuracy',
            'right_choices_minus_50', 'accuracy_right_minus_left', 'pse', 'lapse_low', 'lapse_high']
EARLY_STEP_COLUMNS = {'session_no', 'preset_no', 'preset', 'stage', 'distribution', 'rig', 'trials', 'light_trials'}
SUMMARY_STATS = ['accuracy', 'hard_accuracy', 'easy_accuracy', 'right_choices_minus_50',
                 'accuracy_right_minus_left', 'pse', 'sigma', 'lapse_low', 'lapse_high']
STATUS = {0: ('flag', 'b-red', 'a flag on or since its latest session'),
          1: ('earlier', 'b-amber', 'flags earlier in the window only'),
          2: ('ok', 'b-green', 'no flags in the window')}

CSS = """
:root { --fg: #1d1d1f; --bg: #ffffff; --muted: #6e6e73; --line: #d2d2d7; --head: #f5f5f7; --link: #0b63ce;
        --red: #b42318; --redbg: #fee4e2; --amber: #b45309; --amberbg: #fef3c7; --green: #15803d; --greenbg: #dcfce7; }
@media (prefers-color-scheme: dark) {
  :root { --fg: #f5f5f7; --bg: #1c1c1e; --muted: #a1a1a6; --line: #3a3a3c; --head: #2c2c2e; --link: #6cb4ff;
          --red: #ff8a80; --redbg: #3d1c1a; --amber: #fbbf24; --amberbg: #3b2f12; --green: #4ade80;
          --greenbg: #12301d; }
  img { background: #ffffff; border-radius: 4px; }
}
* { box-sizing: border-box; }
body { font: 14px/1.45 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--fg);
       background: var(--bg); margin: 0; padding: 18px 24px 70px; }
a { color: var(--link); }
h1 { font-size: 20px; margin: 0 0 4px; } h2 { font-size: 17px; margin: 18px 0 6px; }
h3 { font-size: 14px; margin: 18px 0 4px; }
.muted { color: var(--muted); }
.nav { display: flex; gap: 16px; align-items: baseline; flex-wrap: wrap; margin: 2px 0 8px; font-size: 13px; }
.nav .here { font-weight: 600; font-size: 17px; }
table { border-collapse: collapse; margin: 6px 0 12px; font-size: 12px; }
th, td { border-bottom: 1px solid var(--line); padding: 3px 7px; text-align: right; white-space: nowrap;
         vertical-align: middle; }
th { background: var(--head); font-weight: 600; }
th[title] { cursor: help; text-decoration: underline dotted var(--muted); text-underline-offset: 3px; }
td.t, th.t { text-align: left; } td.wrap { white-space: normal; min-width: 160px; max-width: 340px; }
tr.empty td { color: var(--muted); }
.flag { color: var(--amber); background: var(--amberbg); }
.badge { display: inline-block; padding: 1px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }
.b-red { color: var(--red); background: var(--redbg); } .b-amber { color: var(--amber); background: var(--amberbg); }
.b-green { color: var(--green); background: var(--greenbg); } .b-grey { color: var(--muted); background: var(--head); }
.ready { color: var(--green); font-weight: 600; }
.scroll { overflow-x: auto; max-width: 100%; }
img { max-width: 100%; }
svg.spark { color: var(--link); vertical-align: middle; }
details { margin: 2px 0 12px; } summary { cursor: pointer; color: var(--muted); font-size: 12px; }
.totop { position: fixed; right: 18px; bottom: 18px; background: var(--head); color: var(--fg);
         border: 1px solid var(--line); border-radius: 16px; padding: 4px 12px; font-size: 12px;
         text-decoration: none; }
body.views section.view { display: none; } body.views section.view.current { display: block; }
@media print { body.views section.view { display: block; } .nav, .totop { display: none; }
               section.animal { page-break-before: always; } }
"""

JS = """
(function () {
  var views = Array.prototype.slice.call(document.querySelectorAll('section.view'));
  var ids = views.map(function (v) { return v.id; });
  var animals = views.filter(function (v) { return v.classList.contains('animal'); })
                     .map(function (v) { return v.id; });
  document.body.classList.add('views');
  function current() { return decodeURIComponent(location.hash.slice(1)); }
  function show() {
    var id = ids.indexOf(current()) >= 0 ? current() : 'home';
    views.forEach(function (v) { v.classList.toggle('current', v.id === id); });
    window.scrollTo(0, 0);
  }
  window.addEventListener('hashchange', show);
  document.addEventListener('keydown', function (e) {
    if (e.metaKey || e.ctrlKey || e.altKey) { return; }
    var i = animals.indexOf(current());
    if (e.key === 'Escape') { location.hash = 'home'; }
    else if (e.key === 'ArrowRight' && animals.length) {
      location.hash = animals[i < 0 ? 0 : Math.min(i + 1, animals.length - 1)];
    } else if (e.key === 'ArrowLeft' && i > 0) { location.hash = animals[i - 1]; }
  });
  show();
})();
"""


# ── small helpers ────────────────────────────────────────────────────────────

def _fmt(v, kind: str) -> str:
    if v is None or (kind != 'text' and pd.isna(v)):
        return ''
    if kind == 'count':
        return str(int(v))
    if kind == 'prop':
        return f'{float(v):.2f}'
    if kind == 'signed':
        return f'{float(v):+.2f}'
    return html.escape(str(v))


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=85, bbox_inches='tight')
    return '<img alt="" src="data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode() + '">'


def _head(col: str) -> tuple:
    label, kind, tip = COLUMNS[col]
    return label, 't' if kind == 'text' else '', tip


def _table(header: list, rows: list, row_classes: list | None = None) -> str:
    """header: (label, css class[, hover text]); rows: lists of (html, css class[, hover text])."""
    def cell(tag, item):
        text, cls = item[0], item[1]
        title = f' title="{html.escape(item[2])}"' if len(item) > 2 and item[2] else ''
        return f'<{tag} class="{cls}"{title}>{text}</{tag}>'
    th = ''.join(cell('th', h) for h in header)
    body = []
    for k, cells in enumerate(rows):
        cls = f' class="{row_classes[k]}"' if row_classes and row_classes[k] else ''
        body.append(f'<tr{cls}>' + ''.join(cell('td', c) for c in cells) + '</tr>')
    return f'<div class="scroll"><table><tr>{th}</tr>{"".join(body)}</table></div>'


def _sparkline(values, lo: float, hi: float, title: str, w: int = 112, h: int = 24) -> str:
    """Inline SVG trend: one point per day, gaps where there is no value."""
    vals = [v if v is not None and np.isfinite(v) else None for v in values]
    pts = [(i, v) for i, v in enumerate(vals) if v is not None]
    if not pts:
        return ''
    n = max(len(vals) - 1, 1)

    def xy(i, v):
        return 2 + (w - 4) * i / n, 2 + (h - 4) * (1 - (min(max(v, lo), hi) - lo) / ((hi - lo) or 1))
    segs, cur = [], []
    for i, v in enumerate(vals):
        if v is None:
            segs, cur = (segs + [cur] if len(cur) > 1 else segs), []
        else:
            cur.append('%.1f,%.1f' % xy(i, v))
    if len(cur) > 1:
        segs.append(cur)
    lines = ''.join(f'<polyline points="{" ".join(s)}" fill="none" stroke="currentColor" stroke-width="1.2"/>'
                    for s in segs)
    dots = ''.join('<circle cx="%.1f" cy="%.1f" r="1.4" fill="currentColor"/>' % xy(i, v) for i, v in pts)
    lx, ly = xy(*pts[-1])
    return (f'<svg class="spark" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>{html.escape(title)}</title>'
            f'{lines}{dots}<circle cx="{lx:.1f}" cy="{ly:.1f}" r="2.6" fill="currentColor"/></svg>')


def _real_sessions(tracker, animal: str) -> pd.DataFrame:
    s = tracker.sessions
    s = s[(s['animal'] == animal) & s['folder'].notna()]
    real = s[~s['test_run'].fillna(False).astype(bool)]
    return (real if len(real) else s).sort_values('date')


def _flags_text(f: pd.DataFrame) -> tuple:
    """('side bias (09 Nov); missing day (07 Nov, 08 Nov)', hover text with the details)."""
    if f.empty:
        return '', ''
    parts = []
    for flag, g in f.groupby('flag', sort=False):
        parts.append(f"{flag} ({', '.join(d.strftime('%d %b') for d in sorted(g['date']))})")
    tip = '\n'.join(f"{r.date:%d %b}: {r.flag} — {r.detail}" for r in f.sort_values('date').itertuples())
    return html.escape('; '.join(parts)), tip


def _status(tracker, animal: str) -> tuple:
    last = _real_sessions(tracker, animal).iloc[-1]
    f = tracker.flags[tracker.flags['animal'] == animal]
    recent, earlier = f[f['date'] >= last['date']], f[f['date'] < last['date']]
    return last, recent, earlier, (0 if len(recent) else 1 if len(earlier) else 2)


def _move_on(tracker, animal: str) -> tuple:
    m = tracker.move_on[tracker.move_on['animal'] == animal]
    if m.empty:
        return '', 't', ''
    m = m.iloc[0]
    if m['ready'] is True or m['ready'] == True:  # noqa: E712
        return '<span class="ready">ready</span>', 't', m['detail']
    if m['ready'] is None or pd.isna(m['ready']):
        return '—', 't muted', m['detail']
    return 'not yet', 't', m['detail']


def _session_label(row) -> str:
    n = row.get('session_no')
    return f"{row['date']:%d %b}" + ('' if pd.isna(n) else f' · #{int(n)}')


# ── the overview ─────────────────────────────────────────────────────────────

def _missing_table(tracker) -> str:
    miss = tracker.missing
    if miss is None or miss.empty:
        return ''
    rows = []
    for r in miss.itertuples():
        last = '' if r.last_session is None or pd.isna(r.last_session) else r.last_session.strftime('%a %d %b')
        since = 'never on ceph' if not last else f'{int(r.days_since)} days ago'
        rows.append([('<span class="badge b-grey">no session</span>', 't'), (html.escape(r.animal), 't'),
                     (last, 't'), (since, 't')])
    return ('<h2>Expected, but no session on ceph in the window</h2>'
            '<p class="muted">Not run, or not pushed. From daily_tracker.expected_cohorts in config.yaml.</p>'
            + _table([('', 't'), ('animal', 't'), ('last session on ceph', 't'), ('', 't')], rows))


def _overview(tracker) -> str:
    window = [tracker.today - pd.Timedelta(days=i) for i in range(tracker.days - 1, -1, -1)]
    order = sorted(tracker.animals, key=lambda a: (_status(tracker, a)[3], a))
    header = [('', 't'), ('animal', 't'), ('latest', 't', 'latest session: date · session number'),
              _head('preset'), ('trend', 't', 'accuracy over the window (trials, on early steps)'),
              _head('trials'), _head('correct'), _head('accuracy'), _head('hard_accuracy'),
              _head('right_choices_minus_50'), _head('accuracy_right_minus_left'),
              ('flags on or since the latest session', 't'), ('earlier flags', 't'),
              ('move on', 't', "TRAINING.md's criterion for the current step; hover for details")]
    rows = []
    for animal in order:
        last, recent, earlier, rank = _status(tracker, animal)
        label, cls, tip = STATUS[rank]
        d = tracker.sessions[tracker.sessions['animal'] == animal].set_index('date')
        early = last['group'] == 'early step'
        if early:
            trend = _sparkline([d['trials'].get(day) for day in window], 0,
                               max(300.0, float(np.nanmax(d['trials'].to_numpy(dtype=float)))), 'trials per day')
        else:
            trend = _sparkline([d['accuracy'].get(day) for day in window], 0.4, 1.0,
                               'accuracy per day (0.4 to 1.0)')

        def show(col, last=last, early=early):
            return '' if early and col not in EARLY_STEP_COLUMNS else _fmt(last[col], COLUMNS[col][1])
        recent_text, recent_tip = _flags_text(recent)
        earlier_text, earlier_tip = _flags_text(earlier)
        mv = _move_on(tracker, animal)
        rows.append([
            (f'<span class="badge {cls}">{label}</span>', 't', tip),
            (f'<a href="#{html.escape(animal)}">{html.escape(animal)}</a>', 't'),
            (_session_label(last), 't'), (_fmt(last['preset'] or last['stage'], 'text'), 't'), (trend, 't'),
            (show('trials'), ''), (show('correct'), ''), (show('accuracy'), ''), (show('hard_accuracy'), ''),
            (show('right_choices_minus_50'), ''), (show('accuracy_right_minus_left'), ''),
            (recent_text, 't wrap flag' if recent_text else 't', recent_tip),
            (earlier_text, 't wrap muted', earlier_tip), mv])
    return _table(header, rows)


# ── one animal ───────────────────────────────────────────────────────────────

def _nav(tracker, animal: str, prev_a: str | None, next_a: str | None) -> str:
    prev_link = f'<a href="#{html.escape(prev_a)}">‹ {html.escape(prev_a)}</a>' if prev_a else ''
    next_link = f'<a href="#{html.escape(next_a)}">{html.escape(next_a)} ›</a>' if next_a else ''
    return (f'<div class="nav"><a href="#home">← overview</a>{prev_link}<span class="here">{html.escape(animal)}</span>'
            f'{next_link}<span class="muted">{tracker.today:%a %d %b %Y} · ← → next animal, Esc overview</span></div>')


def _day_tables(tracker, animal: str) -> str:
    d = tracker.sessions[tracker.sessions['animal'] == animal]
    first = d.loc[d['folder'].notna(), 'date'].min()
    d = d[d['date'] >= first].sort_values('date', ascending=False)
    f = tracker.flags[tracker.flags['animal'] == animal]

    def build(cols):
        header = [('date', 't')] + [_head(c) for c in cols] + [('flags', 't')]
        rows, classes = [], []
        for _, r in d.iterrows():
            mine = f[f['date'] == r['date']]
            flag_text = '; '.join(f"{x.flag} ({x.detail})" for x in mine.itertuples())
            empty = pd.isna(r.get('folder'))
            test = (not empty) and bool(r.get('test_run'))
            early = (not empty) and r.get('group') == 'early step'
            cells = [(f"{r['date']:%a %d %b}" + (' (test run)' if test else ''), 't')]
            for c in cols:
                hide = empty or (early and c not in EARLY_STEP_COLUMNS)
                cells.append(('' if hide else _fmt(r.get(c), COLUMNS[c][1]), 't' if COLUMNS[c][1] == 'text' else ''))
            cells.append((html.escape(flag_text), 't wrap flag' if flag_text else 't'))
            rows.append(cells)
            classes.append('empty' if empty else '')
        return _table(header, rows, classes)
    return build(DAY_SHORT) + f'<details><summary>all columns</summary>{build(DAY_FULL)}</details>'


def _changes(sessions: pd.DataFrame) -> list:
    """(date, label) where the preset changes. Hard-A and Hard-B of the same kind count as one preset, so
    the daily alternation is not marked: hard-A / hard-B -> 'hard', opto-hard-A / opto-hard-B -> 'opto-hard'."""
    out, prev = [], None
    for _, r in sessions.sort_values('date').iterrows():
        key = re.sub(r'-[AB]$', '', r['preset'] or r['stage'] or '')
        if prev is not None and key != prev:
            out.append((r['date'], key))
        prev = key
    return out


def _day_figure(tracker, animal: str) -> str:
    s = tracker.settings
    real = _real_sessions(tracker, animal)
    changes = _changes(real)
    if (real['group'] == 'early step').all():
        fig = Figure(figsize=(5, 3.2))
        ax = fig.subplots()
        plot_daily_values(tracker.sessions, animal, ['trials'], ax, labels=['trials'], ylabel='trials',
                          hline=s.early_step_min_trials if s else None, changes=changes)
        fig.tight_layout()
        return _png(fig)
    fig = Figure(figsize=(16, 3.3))
    axes = fig.subplots(1, 4)
    plot_daily_values(tracker.sessions, animal, ['accuracy', 'hard_accuracy', 'easy_accuracy'], axes[0],
                      labels=['accuracy', 'hard', 'easy'], ylabel='proportion correct', hline=0.5, ylim=(0.3, 1.02),
                      changes=changes)
    plot_daily_values(tracker.sessions, animal, ['right_choices_minus_50', 'accuracy_right_minus_left'], axes[1],
                      labels=['right choices − 0.5', 'accuracy right − left'], ylabel='toward right', hline=0.0,
                      changes=changes)
    plot_daily_values(tracker.sessions, animal, ['correct', 'trials'], axes[2], labels=['correct', 'trials'],
                      ylabel='trials', hline=s.min_correct_trials if s else None, changes=changes)
    latest = tracker.latest.get(animal)
    if latest is not None:
        plot_psychometric_curve(latest[2], axes[3], color=LIGHT_COLOURS['no light'], show_band=False)
        axes[3].set_title(f"latest full-task session: {latest[0]:%d %b} ({latest[1]})", fontsize=9)
        axes[3].set_xlabel('stimulus')
        axes[3].set_ylabel('P(choose B)')
    else:
        axes[3].set_axis_off()
    fig.tight_layout()
    return _png(fig)


def _in_order(conditions: pd.DataFrame) -> list:
    """Presets in the order the animal first ran them (within the summarised sessions)."""
    if conditions.empty:
        return []
    return list(conditions.groupby('preset')['first_date'].min().sort_values().index)


def _condition_tables(tracker, animal: str) -> str:
    c = tracker.conditions[tracker.conditions['animal'] == animal]
    if c.empty:
        return '<p class="muted">No full-task conditions yet.</p>'
    cs = tracker.condition_sessions[tracker.condition_sessions['animal'] == animal]
    parts = []
    for preset in _in_order(c):
        for label in ('no light', 'light'):
            sub = c[(c['preset'] == preset) & (c['trials'] == label)]
            if sub.empty:
                continue
            first = sub.iloc[0]
            mine = cs[(cs['preset'] == preset) & (cs['trials'] == label)]
            dates = mine[['date', 'session_no']].drop_duplicates().sort_values('date')
            header = [('', 't'), ('pooled', '', 'one fit on all the trials of these sessions'),
                      ('mean ± SD', '', 'mean and SD of the per-session values')] + \
                     [(_session_label(r), '') for _, r in dates.iterrows()]
            rows = []
            for stat in SUMMARY_STATS:
                row = sub[sub['stat'] == stat]
                if row.empty:
                    continue
                row = row.iloc[0]
                label_, kind, tip = COLUMNS[stat]
                sd = '' if pd.isna(row['sd']) else f" ± {row['sd']:.2f}"
                cells = [(label_, 't', tip), (_fmt(row['pooled'], kind), ''),
                         ('' if pd.isna(row['mean']) else _fmt(row['mean'], kind) + sd, '')]
                per = mine[mine['stat'] == stat]
                per = dict(zip(per['date'], per['value']))
                cells += [(_fmt(per.get(d, np.nan), kind), '') for d in dates['date']]
                rows.append(cells)
            parts.append(f'<h3>{html.escape(preset)}, trials {label} '
                         f'<span class="muted">(last {int(first["n_sessions"])} sessions, '
                         f'{first["first_date"]:%d %b} to {first["last_date"]:%d %b})</span></h3>'
                         + _table(header, rows))
    return ''.join(parts)


def _condition_figure(tracker, animal: str) -> str:
    presets = _in_order(tracker.conditions[tracker.conditions['animal'] == animal])
    if not presets:
        return ''
    ncol = min(4, len(presets))
    nrow = int(np.ceil(len(presets) / ncol))
    fig = Figure(figsize=(4 * ncol, 3.2 * nrow))
    axes = fig.subplots(nrow, ncol, squeeze=False)
    for ax, preset in zip(axes.flat, presets):
        for label in ('no light', 'light'):
            cur = tracker.curves.get((animal, preset, label))
            if cur is not None:
                plot_condition_curves(cur, ax, color=LIGHT_COLOURS[label], label=f'{label}, pooled')
        ax.set_title(preset, fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc='upper left')
    for ax in list(axes.flat)[len(presets):]:
        ax.set_axis_off()
    fig.tight_layout()
    return _png(fig)


def _animal_section(tracker, animal: str, prev_a, next_a) -> str:
    mv, _, detail = _move_on(tracker, animal)
    move = f'<p>Move on: {mv} <span class="muted">{html.escape(detail)}</span></p>' if mv else ''
    return (f'<section class="view animal" id="{html.escape(animal)}">' + _nav(tracker, animal, prev_a, next_a) + move
            + '<h3>Each day, newest first</h3>' + _day_tables(tracker, animal) + _day_figure(tracker, animal)
            + '<h3>Conditions: the last sessions of each preset</h3>'
            + '<p class="muted">Pooled: one fit on all their trials. Mean ± SD: over the per-session values, '
              'shown to the right. Curves: each session faint, their mean dashed, the pooled fit solid.</p>'
            + _condition_tables(tracker, animal) + _condition_figure(tracker, animal)
            + _nav(tracker, animal, prev_a, next_a) + '</section>')


def _settings_section(tracker, source: str) -> str:
    t = tracker.settings.table(tracker.rule) if tracker.settings else pd.DataFrame()
    rows = [[(html.escape(str(r['setting'])), 't'), (html.escape(str(r['value'])), 't'),
             (html.escape(str(r['set in'])), 't')] for _, r in t.iterrows()]
    notes = ''.join(f'<li>{html.escape(n)}</li>' for n in tracker.notes)
    return ('<section class="view" id="settings"><div class="nav"><a href="#home">← overview</a></div>'
            '<h2>Numbers used, and where they are set</h2>'
            '<p>To change one: edit <code>config.yaml</code> (block <code>daily_tracker</code>) in the repository, '
            'commit, then run <code>sc-update</code> on the PC that runs the tracker. The bias thresholds are '
            "<code>BiasRule</code>'s, shared with the switch report.</p>"
            + _table([('setting', 't'), ('value', 't'), ('set in', 't')], rows)
            + f'<p class="muted">Data: {html.escape(source)}. Statistics use trials without light; side bias '
              "uses the free-fit PSE. Right and left follow each session's sound contingency.</p>"
            + (f'<h3>Loader notes</h3><ul class="muted">{notes}</ul>' if notes else '') + '</section>')


def write_daily_report(tracker, path: Path, *, source: str = '', notice: str = '') -> Path:
    """Write the report to ``path`` (one HTML file) and return it. ``notice`` is shown under the title
    (e.g. that the page was not refreshed from ceph)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start = tracker.today - pd.Timedelta(days=tracker.days - 1)
    order = sorted(tracker.animals, key=lambda a: (_status(tracker, a)[3], a))
    newest = (f' Newest session file written {tracker.newest_file:%a %d %b %H:%M}.'
              if tracker.newest_file is not None else '')
    home = [f'<section class="view" id="home"><h1>Daily behaviour tracker, {tracker.today:%A %d %B %Y}</h1>',
            f'<p class="muted">{len(tracker.animals)} animals with a session from {start:%d %b} to '
            f'{tracker.today:%d %b}.{newest} Click an animal for its page. '
            '<a href="#settings">Numbers used</a>.</p>',
            f'<p class="flag" style="padding: 4px 8px">{html.escape(notice)}</p>' if notice else '',
            _missing_table(tracker), '<h2>Overview</h2>',
            _overview(tracker) if tracker.animals else '<p>No animal has a session in the window.</p>', '</section>']
    sections = [_animal_section(tracker, a, order[k - 1] if k else None, order[k + 1] if k + 1 < len(order) else None)
                for k, a in enumerate(order)]
    page = ('<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>Daily tracker {tracker.today.isoformat()}</title><style>{CSS}</style></head><body>'
            + '\n'.join(home + sections + [_settings_section(tracker, source)])
            + '<a class="totop" href="#home" onclick="window.scrollTo(0, 0); return false;">↑ top</a>'
            + f'<script>{JS}</script></body></html>')
    path.write_text(page, encoding='utf-8')
    return path
