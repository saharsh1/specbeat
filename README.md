# specbeat

A faint spectroscopic companion leaves a beat pattern in the Fourier transform
of a blended line: the log-amplitude picks up a cosine whose height is the
companion-to-primary line-depth ratio and whose frequency is the velocity
separation. `specbeat` implements that measurement and the statistics that go
with it, on simulated spectra.

It accompanies the paper *Faint Companions to Hot Stars* (Shahaf et al.). The
paper explains the method; this package lets you run it, break it, and see
where it stops working. Everything here is simulation — there is no data
ingestion layer, and reproducing the paper's stellar results is not the point.

The three notebooks in `notebooks/` are the intended way in.

## Install

```sh
git clone https://github.com/saharsh1/specbeat
cd specbeat
pip install -e .
```

Python 3.10 or newer. The dependencies are numpy, scipy and matplotlib; add
`.[notebooks]` if you also want Jupyter. Nothing else is required, and a test
in the suite checks that nothing else creeps in.

If you would rather not install anything, the notebooks also run straight from
a source checkout: their first cell puts `src/` on the path when the package
is not importable.

## A first measurement

```python
import numpy as np
from specbeat import amplitude, detect, mc, nulls, simulate

dv, n_pix = 1.0, 256           # velocity step (km/s) and segment length
sigma_pix = 0.02               # per-pixel noise
zeta_max = 0.03                # fit band (s/km)
v_grid = np.linspace(20.0, 120.0, 11)

# A primary line with a companion 30% as deep, 60 km/s away.
wave = simulate.generate_wavelength_grid(dv=dv, N=n_pix)
spec, _, _ = simulate.generate_simulated_spectrum(
    wave, dv, N=n_pix, line_depth=0.20, v=0.0, vG=12.0, gamma=3.0, vsini=0.0,
    second_component=True, v2=60.0, vG2=12.0, gamma2=3.0, vsini2=0.0,
    alpha=0.30)
spec = simulate.add_noise(spec, sigma_pix, seed=42)

# The same sigma_ft goes into the fit and into the Monte Carlo below.
sigma_ft = amplitude.amp_noise_sigma(sigma_pix, n=n_pix)

hit = detect.detection_statistic(spec, dv, v_grid, sigma_ft=sigma_ft,
                                 zeta_max=zeta_max)
print(f"T = {hit['T']:.1f} at v = {hit['v_best']:.0f} km/s, "
      f"alpha = {hit['alpha_hat']:.3f}")

# Scanning over v breaks the fixed-v null, so calibrate it by simulation.
template = mc.make_template(dv, n_pix, line_depth=0.20, vG=12.0, gamma=3.0)
null = mc.run_null_mc(sigma_pix, n_pix, dv, v_grid, n_draws=1000, seed=11,
                      sigma_ft=sigma_ft, template=template, zeta_max=zeta_max)
fit = nulls.fit_gamma_null(null["T"])

print(f"threshold = {nulls.gamma_null_threshold(0.05, fit['a'], fit['theta'], fit['w0']):.1f}")
print(f"p         = {nulls.gamma_null_pvalue(hit['T'], fit['a'], fit['theta'], fit['w0']):.1e}")
```

```text
T = 21.2 at v = 60 km/s, alpha = 0.268
threshold = 5.6
p         = 7.2e-07
```

The threshold is 5.6 rather than the textbook 2.71 because the velocity was
scanned rather than known in advance. Notebook 03 is about that difference.

## What is in the package

| module | what it does |
| --- | --- |
| `kernels` | Fourier transforms of the rotational, Gaussian and Lorentzian broadening kernels |
| `simulate` | synthetic single- and double-lined spectra, plus noise |
| `amplitude` | the log-\|FT\| observable, its design matrix, and bounded weighted least squares |
| `phase` | the phase observable and its fit, for when the primary velocity is known |
| `detect` | the velocity scan and the detection statistic |
| `nulls` | thresholds, p-values and upper limits under the `alpha >= 0` boundary |
| `power` | Fisher information, non-centrality, detection power |
| `mc` | Monte-Carlo nulls and injection-recovery |
| `plotting` | the matplotlib settings used for the paper figures |

Velocities are in km/s, the Fourier conjugate variable `zeta` is in s/km, and
kernel widths (`vG`, `gamma`) are FWHM in km/s. The Fourier grid is always
`np.fft.rfftfreq(N, d=dv)[1:]`, with the DC bin dropped.

## Scope

Two restrictions are built in, and both are places the paper goes further.

The fits are single-epoch. Functions that would take a stack of epochs accept
the shape but raise `NotImplementedError`.

The companion and the primary are assumed to share one broadening kernel, so
their ratio `R(zeta)` is 1. This is the LSF-limited case of eq. 12, and it is a
good approximation when the instrumental profile dominates. When the two stars
have visibly different broadening it fails, and the failure is visible in the
fit residual — notebook 02 shows what it looks like. The design matrix has the
columns for a polynomial correction, but the default fits do not use them.

## Tests

```sh
pytest -q
```

377 tests, about seven seconds. They cover the algebra of each fit, the boundary
statistics against their closed forms, Monte-Carlo recovery of the null
parameters, and a set of regression tests for mistakes that were made at least
once while writing this (`tests/test_traps.py`).

## Citing

See `CITATION.cff`. Please cite the paper; the code is MIT licensed and you are
free to do what you like with it.
