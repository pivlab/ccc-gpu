"""Named benchmark grids ("presets"), data instead of commented-out parameter lists.

``smoke`` grids finish in seconds (sanity / CI-adjacent). ``paper`` grids
reproduce the poster/README sweeps and can take a long time (the CPU reference
for the biggest cases is minutes per point) -- run them on the reference GPU box.
"""

# coef / scaling grids use (features, samples, n_jobs); ari uses
# (n_features, n_parts, n_objs, k).

COEF_PRESETS = {
    "smoke": {"features": [50, 100], "samples": [100], "n_jobs": [1]},
    # Poster/README-style sweep: end-to-end GPU-vs-CPU over a feature grid.
    "paper": {
        "features": [500, 1000, 2000, 5000, 10000],
        "samples": [1000],
        "n_jobs": [24],
    },
}

ARI_PRESETS = {
    "smoke": {"n_features": [10, 50], "n_parts": [10], "n_objs": [300], "k": [10]},
    "paper": {
        "n_features": [100, 200, 1000],
        "n_parts": [20],
        "n_objs": [300],
        "k": [10],
    },
}

SCALING_PRESETS = {
    "smoke": {"features": [100], "samples": [500], "n_jobs": [1, 2]},
    "paper": {"features": [1000], "samples": [1000], "n_jobs": [1, 6, 12, 24]},
}

PRESETS_BY_MODE = {
    "coef": COEF_PRESETS,
    "ari": ARI_PRESETS,
    "scaling": SCALING_PRESETS,
}


def get_preset(mode: str, name: str) -> dict:
    """Return the grid dict for ``mode``/``name`` or raise a clear error."""
    presets = PRESETS_BY_MODE.get(mode)
    if presets is None:
        msg = f"No presets for mode {mode!r}"
        raise KeyError(msg)
    if name not in presets:
        available = ", ".join(sorted(presets))
        msg = f"Unknown preset {name!r} for mode {mode!r} (available: {available})"
        raise KeyError(msg)
    return dict(presets[name])
