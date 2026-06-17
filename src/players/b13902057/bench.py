"""In-process benchmark / sweep harness for the b13902057 agents.

Runs a `hero` agent against three `opponent` agents over many deals and reports
the tournament-relevant metric: average rank (lower is better), plus average
penalty and strict-win rate. Deals use per-game seeds shared across configs
(common random numbers), so two configs are compared on the *same* hands and
the difference in their scores is signal, not deal luck.

This is an evaluation/tuning script (not a submitted player). It is allowed to
import the TA baselines for measurement; the final players never import them.

Usage:
    python -m src.players.b13902057.bench eval \
        --hero src.players.b13902057.player3:player3 \
        --opp  src.players.TA.public_baselines2:Baseline10 \
        --games 64
    python -m src.players.b13902057.bench sweep   # runs the player3 grid
"""

import argparse
import importlib
import time

from src.engine import Engine


def _resolve(spec):
    """'module.path:ClassName' -> class object."""
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def evaluate(hero_cls, hero_kwargs, opp_cls, opp_kwargs=None,
             n_games=64, base_seed=1000, n_players=4, n_rounds=10,
             timeout=None, rotate=True):
    """Play `n_games` of hero vs (n_players-1) opponents; return metrics dict.

    Seat is rotated across games so any positional bias averages out. The
    engine seed is `base_seed + g`, deterministic and identical across configs.
    """
    opp_kwargs = opp_kwargs or {}
    ranks = []
    pens = []
    wins = 0.0
    worst_move = 0.0
    t_start = time.perf_counter()

    for g in range(n_games):
        hero_seat = (g % n_players) if rotate else 0
        players = []
        for s in range(n_players):
            if s == hero_seat:
                inst = hero_cls(player_idx=s, **hero_kwargs)
                players.append(_TimedWrapper(inst) if timeout is None else inst)
            else:
                players.append(opp_cls(player_idx=s, **opp_kwargs))

        cfg = {"n_players": n_players, "n_rounds": n_rounds,
               "timeout": timeout, "seed": base_seed + g}
        scores, _ = Engine(cfg, players).play_game()

        mine = scores[hero_seat]
        lower = sum(1 for i, sc in enumerate(scores) if i != hero_seat and sc < mine)
        equal = sum(1 for i, sc in enumerate(scores) if i != hero_seat and sc == mine)
        rank = 1.0 + lower + 0.5 * equal
        ranks.append(rank)
        pens.append(mine)
        if lower == 0 and equal == 0:
            wins += 1.0

        hp = players[hero_seat]
        if isinstance(hp, _TimedWrapper) and hp.worst > worst_move:
            worst_move = hp.worst

    n = len(ranks)
    return {
        "avg_rank": sum(ranks) / n,
        "avg_pen": sum(pens) / n,
        "win_pct": 100.0 * wins / n,
        "worst_move_s": worst_move,
        "wall_s": time.perf_counter() - t_start,
        "games": n,
    }


class _TimedWrapper:
    """Wraps an agent to record its slowest action() (only used when the engine
    timeout is disabled, so the agent's own wall-budget governs speed)."""

    def __init__(self, inst):
        self.inst = inst
        self.player_idx = inst.player_idx
        self.worst = 0.0

    def action(self, hand, history):
        t = time.perf_counter()
        r = self.inst.action(hand, history)
        dt = time.perf_counter() - t
        if dt > self.worst:
            self.worst = dt
        return r


def _fmt(tag, m):
    return (f"{tag:<34} rank={m['avg_rank']:.4f}  pen={m['avg_pen']:5.1f}  "
            f"win={m['win_pct']:5.1f}%  worst={m['worst_move_s']*1000:6.1f}ms  "
            f"({m['wall_s']:.1f}s, n={m['games']})")


def cmd_eval(args):
    hero_cls = _resolve(args.hero)
    opp_cls = _resolve(args.opp)
    hero_kwargs = {}
    for kv in args.hero_args or []:
        k, v = kv.split("=")
        hero_kwargs[k] = _coerce(v)
    m = evaluate(hero_cls, hero_kwargs, opp_cls, n_games=args.games,
                 base_seed=args.seed, timeout=args.timeout)
    print(_fmt(args.hero.split(":")[-1] + " " + str(hero_kwargs), m))


def _coerce(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def cmd_sweep(args):
    """Grid sweep of player3 (MCTS) hyperparameters vs the chosen opponent.

    Uses a fixed iteration count (iters) so configs are compared on equal
    compute, deterministically and fast. The winning structural config is then
    deployed with the wall-clock budget for the real 1.0s-per-move setting.
    """
    hero_cls = _resolve(args.hero)
    opp_cls = _resolve(args.opp)
    base = dict(iters=args.iters, n_det=64, budget=9.9, hard_cap=9.9)

    grid = []
    for c in (0.3, 0.6, 1.0, 1.4):
        grid.append(("c", dict(base, c=c, root_k=8, reward="rank", final="visit")))
    for rk in (6, 8, 10, 12):
        grid.append(("root_k", dict(base, c=0.6, root_k=rk, reward="rank", final="visit")))
    for rw in ("rank", "pen"):
        grid.append(("reward", dict(base, c=0.6, root_k=8, reward=rw, final="visit")))
    for fn in ("visit", "mean"):
        grid.append(("final", dict(base, c=0.6, root_k=8, reward="rank", final=fn)))

    print(f"# sweep: {args.hero} vs {args.opp}  games={args.games} iters={args.iters}")
    results = []
    for tag, kw in grid:
        m = evaluate(hero_cls, kw, opp_cls, n_games=args.games,
                     base_seed=args.seed, timeout=None)
        knob = {k: kw[k] for k in ("c", "root_k", "reward", "final")}
        print(_fmt(f"[{tag}] {knob}", m))
        results.append((m["avg_rank"], tag, knob, m))
    results.sort(key=lambda x: x[0])
    print("\n# best by avg_rank:")
    for rank, tag, knob, m in results[:5]:
        print(_fmt(f"[{tag}] {knob}", m))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pe = sub.add_parser("eval")
    pe.add_argument("--hero", required=True)
    pe.add_argument("--opp", default="src.players.TA.public_baselines2:Baseline10")
    pe.add_argument("--hero-args", nargs="*", default=[])
    pe.add_argument("--games", type=int, default=64)
    pe.add_argument("--seed", type=int, default=1000)
    pe.add_argument("--timeout", type=float, default=None)
    pe.set_defaults(func=cmd_eval)

    ps = sub.add_parser("sweep")
    ps.add_argument("--hero", default="src.players.b13902057.player3:player3")
    ps.add_argument("--opp", default="src.players.TA.public_baselines2:Baseline10")
    ps.add_argument("--games", type=int, default=48)
    ps.add_argument("--iters", type=int, default=4000)
    ps.add_argument("--seed", type=int, default=1000)
    ps.set_defaults(func=cmd_sweep)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
