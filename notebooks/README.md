# Notebooks

Three notebooks, meant to be read in order. Each one assumes the *ideas* of the
ones before it, never their variables — every notebook re-simulates its own
data and runs on its own.

1. **`01_single_star_in_fourier_space.ipynb`** — shape and position separate
   under a Fourier transform, and broadening kernels multiply so their logs add.
   Then $v\sin i$ from a single line, and the lesson that only what a smooth
   baseline cannot absorb is measurable. §2.1, eqs. 1–8, Figs. 1–2, Apps. A and
   B.1. About 3 s.

2. **`02_the_companion_signature.ipynb`** — the companion as a cosine beat in
   the log-amplitude, its two failure modes, and the same signal seen in the
   Fourier phase once the primary velocity is known. §2.2, §3.2, §4.2,
   eqs. 9–14, Fig. 3, App. C. About 2 s.

3. **`03_detection_and_calibration.ipynb`** — how precisely $\alpha$ can be
   measured in each channel; the null distribution of the detection statistic
   when $\alpha\ge0$; and what scanning over $v$ does to it. §3, eqs. 15–19,
   Figs. 4–5, App. B.2. About 7 s.

Run them from this directory. Monte-Carlo draw counts are sized for a fast
run-through; production values are noted in the text where it matters.

## Getting the package importable

Either `pip install -e .` from the repository root, or nothing at all — the
first cell of each notebook falls back to putting `../src` on the path when
`specbeat` is not installed.

## The fiducial star

Declared in notebook 1 and reused by default: line depth 0.5, $v_G = 15$,
$v_\gamma = 5$, $v\sin i = 20$ km/s ($\epsilon = 0.5$), $dv = 1$ km/s,
$N = 2048$, $\sigma_{\rm pix} = 0.01$. Where a cell departs from it — to match a
paper figure, to keep a Monte Carlo fast, or to isolate an LSF-limited case —
it says so and says why.

## Conventions

Velocities in km/s, $\zeta$ in s/km, kernel widths as FWHM in km/s. The Fourier
grid is `np.fft.rfftfreq(N, d=dv)[1:]`, with the DC bin always dropped. Where
there is noise, the fit band comes from the SNR rule
(`truncate_to_snr_cutoff`) rather than a bare $\zeta_{\max}$. Seeds are fixed
throughout. Most quantities are built by hand first and with the library
second, and the agreement is printed.

Plots use the paper's style, via `specbeat.plotting.set_paper_style`. LaTeX
rendering is off, so no TeX installation is needed.
