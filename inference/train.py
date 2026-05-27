"""
Per-animal SBI training (script-style entry point).

Direct training function bypassing the SBIFitter class. Used by CLI
scripts that want full control over the training loop. For most code,
prefer SBIFitter (in inference.fitter) which provides a cleaner API.

Public API:
    train_per_animal_snpe — Train SNPE on one animal's stimulus sequence
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any


def train_per_animal_snpe(
    model_type: str,
    fitting_data: 'FittingData',
    stat_names: List[str],
    n_simulations: int = 10_000,
    burn_in: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train SNPE for one animal using its real stimulus sequence.

    This is the per-animal training path (as opposed to amortised training
    which trains once on generic data). Produces a posterior that is
    conditioned on this animal's specific stimulus sequence.

    Args:
        model_type: 'be' or 'sc'.
        fitting_data: FittingData for one animal.
        stat_names: Summary stat names (CAN include update_matrix).
        n_simulations: Number of training simulations.
        burn_in: Burn-in trials for model initialisation.
        seed: Random seed.

    Returns:
        Dict with 'posterior', 'prior', 'simulator', 'sbi_sim',
        'param_names', 'model_type', 'stat_names', 'burn_in',
        'training_time', 'n_valid'.
    """
    import torch
    from sbi.inference import SNPE
    from inference.simulator import (
        create_be_simulator, create_sc_simulator,
        get_sbi_prior, wrap_for_sbi,
    )

    name = model_type.upper()
    aid = fitting_data.animal_id
    pooled = fitting_data.pool()
    stim, cat = pooled['stimuli'], pooled['categories']

    print(f"  Training per-animal SNPE [{name}] for {aid} "
          f"({n_simulations:,} sims, {len(stim)} trials)...")

    creator = create_be_simulator if model_type == 'be' else create_sc_simulator
    sim = creator(stim, cat, stat_names=stat_names, burn_in=burn_in)
    prior = get_sbi_prior(sim)
    sbi_sim = wrap_for_sbi(sim)

    t0 = time.time()
    theta = prior.sample((n_simulations,))
    x = torch.stack([sbi_sim(t) for t in theta])

    valid = ~torch.any(torch.isnan(x), dim=1)
    n_valid = valid.sum().item()
    print(f"    {n_valid}/{n_simulations} valid "
          f"({100 * n_valid / n_simulations:.0f}%)")

    inference_engine = SNPE(prior=prior)
    inference_engine.append_simulations(theta[valid], x[valid])
    posterior = inference_engine.build_posterior(inference_engine.train())

    dt = time.time() - t0
    print(f"    Done in {dt / 60:.1f} min")

    return {
        'posterior': posterior, 'prior': prior,
        'simulator': sim, 'sbi_sim': sbi_sim,
        'param_names': sim.get_param_names(),
        'model_type': model_type, 'stat_names': stat_names,
        'burn_in': burn_in, 'training_time': dt, 'n_valid': n_valid,
    }
    
