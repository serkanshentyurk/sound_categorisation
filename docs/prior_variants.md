# Prior variants

`sound_categorisation/models/priors.py` names two priors for BE and SC, over the same four parameters
per model.

| variant | learning ranges |
|---|---|
| `manuscript` | as the networks before mid-November 2026 were trained (`BEParams.get_bounds`, `SCParams.get_bounds`) |
| `learning` | wider: BE `eta_learning` 0.005 to 0.9 and `eta_relax` 0.001 to 0.4, drawn uniformly in the logarithm; SC `gamma` 0.30 to 0.999 |

`settings.SBI_PRIOR` (now `learning`) is the one every model fit uses: the SBI networks, the grid search
grid (`inference/grid_search.py`: 8,000 points per model spanning the prior, log-spaced where the prior is
log-uniform), synthetic cohorts and calibration draws. A saved network records its variant and is
conditioned, calibrated and checked with it, so the earlier networks keep working on their own ranges.
`sc-sbi-range --prior` simulates either variant and places the animals in it.

A log-uniform parameter is handled in *flat space*, its natural logarithm, where the prior is a uniform
box: the density estimator is trained there, and recovery (error relative to the prior width, contraction)
is measured there (`scale` column of the recovery tables: `log` or `linear`). Everything else, posterior
samples and estimates included, is in the parameters' own units.

## Why these wider learning ranges

A scan of the simulator on Uniform (4 sessions of 400 trials, burn-in 1,000, two seeds per setting;
`sigma_percep` 0.05 / 0.15 / 0.3, `A_repulsion` 0 / 0.2), pooled statistics.

**BE** (`eta_learning` 0.001 to 0.9, `eta_relax` 0 to 0.4; 420 runs):

- Accuracy is highest with slow learning and little relaxation: 0.79 to 0.82 at `sigma_percep` 0.05,
  `eta_learning` 0.01 to 0.05, `eta_relax` at most 0.01. The manuscript range reaches this only in its
  corner (`eta_learning` near 0.05, `eta_relax` near 0.01), which few uniform draws hit.
- No setting exceeded an accuracy of about 0.83 (0.82 in the scan; 0.83 with `sigma_percep` at 0.005, a
  tenth of the prior's floor), and at the accurate settings the fitted psychometric sigma stayed above
  about 0.4. This is BE itself on Uniform, not its prior: the boundary belief stays broad.
- Recency follows the learning rate: about 0 below `eta_learning` 0.05, 0.07 to 0.25 at 0.2, 0.28 to
  0.56 at 0.9.
- At `eta_learning` 0.001 accuracy falls again: the belief has not settled within the burn-in.

Hence `eta_learning` 0.005 to 0.9 and `eta_relax` 0.001 to 0.4, both drawn uniformly in the logarithm,
so that the slow end is actually sampled (a uniform draw over 0.005 to 0.9 would put 5 % of draws below
0.05).

**SC** (`gamma` 0.3 to 0.999, `sigma_update` 0.02 to 2.0; 504 runs):

- Within the manuscript range (`gamma` at least 0.8) recency stays between -0.08 and 0.10.
- Lower retention is the one direction that changes this: with `gamma` at most 0.5 and `sigma_update`
  0.5 to 1.0, recency averages 0.10 and reaches 0.28, at accuracies of 0.65 to 0.82. With a narrow
  update (`sigma_update` at most 0.05) low `gamma` gives negative win-stay instead.
- Widening `sigma_update` does not help: at 0.02 accuracy is 0.50 to 0.70 and psychometric fits fail;
  at 2.0 accuracy is 0.60 to 0.69. Both are below the session filter's 0.70.

Hence `gamma` 0.30 to 0.999 (uniform) and `sigma_update` unchanged.

## What the wider ranges gain and cost

Gain, on the 19 expert mice (Uniform, pooled statistics; an animal is covered by a model when no
statistic is outside the central 99 % of the model's simulations and a simulation lies nearby):
BE covered 4 animals under `manuscript` and 13 under `learning`; SC 6 and 15; neither model 9 and 3.
(`manuscript` from 2,000 draws per model, `learning` from 1,000.)

Cost: positive recency is BE's signature under the manuscript prior. A BE that learns slowly has
none, and an SC with `gamma` below about 0.5 and a broad update has some, so the wider ranges make the
two models produce the same statistics more often. `sc-sbi-range` measures this next to the coverage
(`recognised`): the share of a model's draws that a classifier trained on other draws of both models
calls that model, over every draw and over the draws that pass the session accuracy rule (where the
animals are). The classifier is gradient boosting on the same statistics, a stand-in for what a
network could tell apart; 0.5 is chance.

Pooled statistics, 1,000 draws per model, Uniform, training configuration:

| variant | BE recognised | BE, passing draws | SC recognised | SC, passing draws |
|---|---|---|---|---|
| `manuscript` | 0.99 | 0.92 (36 draws) | 0.99 | 1.00 (777) |
| `learning` | 0.86 | 0.59 (114) | 0.89 | 0.96 (571) |

Under `learning` a BE draw that passes the accuracy rule is called BE little more often than a coin
would. The confusion sits where BE learns slowly: among BE draws with a pooled accuracy of 0.70 to
0.83, those with `eta_learning` below 0.05 were called BE 34 to 46 % of the time, those above 0.15,
86 %. An accurate BE needs a stable boundary, and a stable boundary leaves no trial-to-trial trace: on
a stationary distribution it is then the same observer as a slowly updating SC. This is a limit of
the Uniform expert data, not of the prior, and it concerns these summary statistics; whether the
update matrix separates the two there is a separate question.

## Tried and removed: a side bias and a lapse

Two choice parameters shared by both models were tried in November 2026,
`P(B) = (1 - lapse) * P_model(B at s_hat - bias) + lapse / 2`, with bias within ±0.4 on the stimulus
axis and lapse 0 to 0.3, alone and together with the wider learning ranges. They did not bring the
animals inside the models, and the lapse cost BE accuracy it does not have to spare:

| prior | BE covers | SC covers | neither | BE draws passing the accuracy rule |
|---|---|---|---|---|
| `manuscript` | 4 | 6 | 9 | 3.6 % |
| `manuscript` + bias and lapse | 1 | 5 | 14 | 0.8 % |
| `learning` | 13 | 15 | 3 | 11.4 % |
| `learning` + bias and lapse | 4 | 14 | 5 | 3.3 % |

(19 expert mice, Uniform, pooled, 1,000 draws per model; the `manuscript` coverage from 2,000.) The
positive psychometric midpoints that motivated the bias (16 of 19 animals, median 0.19) are inside
what the models produce under `learning` without it. The parameters and the two variants that used them were
removed from the code.

## Running it

    sc-sbi-range --cohort all-mice --distribution uniform --prior all

writes `<run>/range/uniform/<variant>/` for each variant, and `comparison.csv` and
`comparison_animals.csv` beside them. A variant can also be run alone; the comparison is rebuilt from
every variant simulated so far in that run.
