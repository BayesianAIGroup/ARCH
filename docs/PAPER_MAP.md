# Paper-to-code map

Reference: the supplied 32-page *Arbitrarily Conditioned Hierarchical Flows for Spatiotemporal Events*, NeurIPS 2026. This maps existing implementations, not regenerated numerical agreement.

| Paper | Experiment | Launcher task | Coverage |
|---|---|---|---|
| §5.1, Table 1 | Relative intensity L2, 8×8 grid | `synthetic --action intensity-error` | ARCH and supplied baseline prediction arrays |
| §5.1, Fig. 1, appendix Fig. 4 | Event-time intensity plots | `synthetic --action plot` | Existing plotting routines and checkpoints |
| §5.2, Table 2 | Next event | `real --task next_event` | ARCH only |
| §5.2, Table 3 | Inverse inference | `real --task inverse_inference` | ARCH only |
| §5.2, Table 4 | Internal gaps 5/10 | `real --task trajectory_recovery` | ARCH only |
| §5.2, Table 5 | Future suffix 5/10/15/20/30 | `real --task forecast --suffix-length K` | ARCH joint forecast only |
| §5.3, Table 6 | Held-out NLL | `real --task likelihood` | Seahorse ARCH implementation and separate data |
| §§5.4–5.6, Tables 7–12 | Ablations and architecture comparisons | Excluded | Requested scope excludes these |
| Appendix D, Table 14 | Random recovery 10/20/30% | `real --task random_imputation --missing-ratio R` | ARCH random recovery (v2) |
| Appendix D, Table 15 | Partial attributes 5/10% | `real --task partial_attributes` | ARCH only |
| Appendix E | Inter-event midpoint spatial surfaces | `synthetic --action midpoints` | Source, queries and saved baseline predictions |
| Appendix E.1 | Temporal intensity, 10 queries/interval | `synthetic --action temporal` | Source, dense queries and temporal checkpoints |

Joint Flow rows in Tables 3/4/15 are also excluded with the architecture comparisons. The supplied files do not reproduce every baseline row from scratch. See [the audit](CURATION.md) for source/paper masking and training differences.

## Metric ordering

The launcher preserves function return values in `metrics.json`:

| Task | Return structure |
|---|---|
| next_event | `[ARCH time RMSE, ARCH mean spatial distance]` |
| forecast | `[ARCH joint time RMSE, ARCH joint mean spatial distance]` |
| random_imputation | `[ARCH time RMSE, ARCH mean spatial distance]` |
| trajectory_recovery | map keyed by `5`, `10`; `model` pair `[time RMSE, mean spatial distance]` |
| partial_attributes | map keyed by `0.05`, `0.1`; same ARCH-only format |
| inverse_inference | map keyed by cases `0`, `1`, `2`; same ARCH-only format |
| likelihood | `[time prediction error, spatial prediction error, LL dictionary]` |

Inverse case 0 is the first event, 1 the second event, 2 the first two events. The paper uses first and first-two settings. Summaries flatten these into keys such as `5/model/0`. LL summaries include numeric metadata too; the Table 6 quantity is `2/avg_nll_joint_raw_lon_lat`. It is the negative of average joint LL in raw coordinates, not normalized-coordinate density.

`std_population` uses NumPy ddof=0. `standard_error_population` divides it by the square root of the run count. These are distinct, and neither establishes statistical significance. Every CLI invocation trains the chosen source task separately; it does not assert one checkpoint was reused across tasks.
