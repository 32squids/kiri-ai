# Kiri AI: The Duel

A self-play reinforcement learning agent for **Kiri Ai: The Duel**, a
2-player simultaneous-selection dueling game played on a 5-space board.
Both players secretly pick two cards per turn (movement, stance changes,
attacks) which are revealed and resolved together — closer to *rock-paper-scissors*
with positioning than a turn-based game. First to land two hits wins.

The full rules are in [kiri-ai-ruleset.txt](kiri-ai-ruleset.txt).

This project implements the game from scratch (state, resolution engine,
Gymnasium environment) and trains a [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
agent against itself to play it.

## Why this is interesting

- **Simultaneous, imperfect-information resolution.** Both players commit
  moves blind, one of three special attacks is dealt secretly per player,
  and hit resolution has a mutual-cancellation rule (two attacks that would
  each hit the other player instead cancel out). [resolution.py](resolution.py)
  is a from-scratch rules engine for this, built on the data model in
  [state.py](state.py).
- **A fixed, side-agnostic action/observation space.** The action space
  is a stable superset over every structurally valid card pair (not just
  what's legal in the current state — legality is exposed separately via
  action masking), and observations are encoded in a canonical frame so
  the same policy network can play either starting side without knowing
  which one it physically is. See the design notes in [env.py](env.py).
- **Historical self-play.** [train.py](train.py) trains one [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
  policy against a pool of its own past snapshots (mostly recent, occasionally
  older) rather than a single always-moving mirror of itself, which is
  more stable than pure self-mirroring. Progress is tracked separately
  against a fixed uniform-random opponent so the eval metric doesn't
  drift as the training opponent gets stronger.

## Results

Evaluated against a fixed uniform-random opponent (never the moving
self-play opponent), win rate climbs from ~50% to **80%+** within the
first 200k training steps under the default sparse reward:

| Training steps | Win rate vs. random |
|---:|---:|
| 0 | ~50% (untrained) |
| 200,000 | ~80% |

(`mean_reward` in `runs/*/eval/evaluations.npz` is `2*win_rate - 1` under
sparse reward, since draws aren't reachable — see `GameState.winner`.)

## Project layout

| File | Purpose |
|---|---|
| [state.py](state.py) | Game data model: board, cards, stances, hands, cooldowns. |
| [resolution.py](resolution.py) | Pure turn-resolution engine: given a state and both players' chosen cards, resolves movement, attacks, and hits. |
| [env.py](env.py) | Gymnasium `KiriDuelEnv`: wraps state/resolution as a single-agent env with action masking, a pluggable opponent policy, and self-play support. |
| [train.py](train.py) | Self-play training script (MaskablePPO via sb3-contrib). |
| [play_cli.py](play_cli.py) | Interactive terminal CLI to play against a random, cycling, or trained-model opponent. |
| [kiri-ai-ruleset.txt](kiri-ai-ruleset.txt) | The full game ruleset. |

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.10+, PyTorch, Gymnasium, Stable-Baselines3, and sb3-contrib.

## Training

```bash
python train.py --opponent selfplay --total-timesteps 200000
```

Key options (see `python train.py --help` for the full list):

- `--opponent selfplay|random` — historical self-play (default) or a
  fixed random opponent for the whole run (useful as a sanity check that
  the env/reward/masking plumbing actually produces a learning signal).
- `--reward-mode sparse|dense` — win/loss only at match end, or a reward
  per hit landed/taken.
- `--output-dir DIR` — where snapshots, the best/final model, and
  TensorBoard logs are written (default `runs/latest`).

Monitor training with TensorBoard:

```bash
tensorboard --logdir runs/latest/tb
```

## Playing against the agent

```bash
python play_cli.py --opponent model --model runs/latest/best/best_model.zip
```

Or try `--opponent random` / `--opponent cycle` to play against the
environment's baseline opponents without needing a trained model at all.
