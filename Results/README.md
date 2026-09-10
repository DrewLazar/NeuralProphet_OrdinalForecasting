# Results

Per-run output from the simulation notebook. One row per (config, seed), not
summaries: `noise_sweep.csv` has 100 rows for 5 noise levels × 20 seeds. The
tables in the paper are means and standard deviations across seeds, printed by
`pretty_summary` in the notebook.

| File | Backs |
|---|---|
| `multiseed_reference_40.csv` | Table 4, and the latent-recovery correlations in Section 4.1.4. Reference configuration, 40 seeds (22–61) |
| `multiseed_trend1.csv` | Table 5, one trend reversal at t = 1750 |
| `multiseed_trend2.csv` | Table 5, two reversals at t = 1200, 2400 |
| `multiseed_trend3.csv` | Table 5, three reversals at t = 900, 1800, 2700 |
| `noise_sweep.csv` | Table 6 and Figure 5. σ ∈ {0.15, 0.25, 0.40, 0.60, 0.80}, 20 seeds per level (42–61) |
| `noise_100.csv` | The σ = 1.0 result quoted in Section 4.1.5 |

Trend conditions use segment slopes alternating at ±4 × 10⁻⁴.

Figure 5 is drawn from `noise_sweep.csv`, so it can be regenerated without
re-running the sweep.
