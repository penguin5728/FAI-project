# b13902057 — 6 Nimmt! agents

## Final submission

| File | Class | Use |
|------|-------|-----|
| `best_player1.py` | `BestPlayer1` | Gap-trap-aware IS-MCTS, **symmetric** opponent model |
| `best_player2.py` | `BestPlayer2` | Same search, **asymmetric** opponent model (diversity) |

Both are **self-contained** (no imports of other player modules or of any
baseline) and need only `player_idx` to construct; every tuning knob is a
keyword argument with a default.

### Core idea

A determinized Information-Set MCTS. Opponents' hands are hidden and the
tournament scores by *rank*, so each of ~10k iterations per move:

1. **determinizes** the hidden hands by dealing the unseen card pool (which
   also contains the never-dealt cards — correct epistemic uncertainty);
2. descends a **UCB1 tree keyed by our own card sequence**, opponents playing a
   fast greedy policy, each round resolved exactly like the engine;
3. **greedy-rolls** the remaining cards to game end;
4. backpropagates the **normalized seat rank**.

The move played is the most-visited root child.

The key ingredient over a plain greedy/MCTS is the **gap-trap** term in the base
policy `_eval`. A round resolves low→high, so any *unseen* card in the gap
between a row's end and the card we place can be played by an opponent **before**
our card resolves; if enough land there the row fills to five and our card is the
trapped sixth (we take the row). `_eval` adds the exact binomial probability of
that event × the row's bullheads, on top of the strong `×10` take-avoidance.
This was the decisive improvement: it moved the agent from ~2.65 to ~2.5 average
rank against the strongest released baseline (B10), i.e. up to its level.

`BestPlayer1` applies the gap term to **both** our moves and the simulated
opponents (`opp_gap_w == gap_w`). `BestPlayer2` applies it only to **our** moves
and models opponents with plain greedy (`opp_gap_w = 0`) — statistically equal
in strength, deliberately different in style for robustness across opponent
pools.

## Development files (not the final agents)

| File | What it is |
|------|------------|
| `player1.py` | Depth-2 determinization Monte-Carlo (rank objective) — early approach |
| `player2.py` | Snapshot of the symmetric gap-aware MCTS (same as BestPlayer1) |
| `player3.py` | Plain determinized IS-MCTS (no gap term) — the baseline the gap idea improved on |
| `player4.py` | Working copy of the gap-aware MCTS with the `gap_w` / `opp_gap_w` knobs |
| `bench.py` | Offline evaluation harness (see below) |

### Methods tried (and what the data showed)

- **Hyperparameter tuning of the plain MCTS** (`c`, `root_k`, reward shaping,
  `final` rule): no movement — all settings sat at ~2.65 vs B10. The agent runs
  ~15k sims/move, so it was never sim-starved; the ceiling was *policy* quality.
- **A pure analytic gap-trap heuristic** (no search): much worse (~3.0) — it
  under-penalized taking rows and took ~18 bullheads/game.
- **Gap-trap term inside the MCTS policy**: the win. Best at a *moderate*
  weight (`gap_w≈1.0`); large weights (≥2) overshoot and regress.
- **Asymmetric opponent model** (`opp_gap_w=0`): statistically tied with
  symmetric — kept only for diversity.

## Reproducing the evaluation

`bench.py` is an offline harness used during development. It pits a "hero"
agent against opponents over many shared deals and reports average rank /
bullheads. It can import the TA baselines **for local measurement only** — the
submitted agents never import them.

> Note: the TA baseline `.so` files segfault when imported in-process on this
> setup, so head-to-head vs. baselines was measured through `run_tournament.py`
> (subprocess-isolated), not `bench.py`. `bench.py` was used for baseline-free
> sparring (e.g. vs. `player3`).

Run the final agents against the released baseline:

```bash
python run_tournament.py --config configs/tournament/best_vs_b10.json
```

(Requires the TA baseline modules to be present; they are not part of this
submission.)
