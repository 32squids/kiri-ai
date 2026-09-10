"""
Self-play training script for Kiri Ai: The Duel.

Trains a single MaskablePPO (sb3-contrib) policy against KiriDuelEnv.
Since observations are already encoded in a canonical, side-agnostic
frame (see env.py), one policy plays both physical starting sides
without needing separate logic.

Self-play opponent (--opponent selfplay, the default): the env's
non-learner side is driven by `SelfPlayOpponent`, an indirection object
whose `.model` attribute can be swapped without reconstructing the env.
`SelfPlaySnapshotCallback` periodically saves the live model to a
bounded-size pool of on-disk snapshots and re-rolls which one (or the
live model itself) `.model` points to -- this is a lightweight
"historical self-play" approach: mostly playing a recent version of
itself, occasionally an older frozen snapshot, which is materially more
stable than always playing an exact mirror of the live, still-changing
policy (the latter tends to chase its own tail).

The opponent identity is re-rolled every `--snapshot-freq` environment
steps, not at exact episode boundaries -- a mid-episode swap would be
strange for a human opponent but is harmless here (the opponent is
queried fresh each of its own turns regardless), and hooking exact
episode boundaries would need a custom VecEnv wrapper for one env this
size isn't worth adding yet.

--opponent random is a debug/sanity mode: skip self-play entirely and
train against the env's default uniform-random legal-move policy for
the whole run. Useful for confirming the plumbing (env, action masking,
reward) actually produces a learning signal before layering self-play
on top.

Either way, progress is tracked against a SEPARATE, fixed evaluation
env whose opponent is always uniform-random (never self-play) -- a
moving self-play opponent would make the eval metric incomparable
across time, since the "difficulty" would drift with the policy itself.
Under the default sparse reward, eval's mean_reward is directly
2*win_rate - 1 against that fixed random baseline (draws are not
possible -- see GameState.winner), so it should trend from ~0 toward 1
as training progresses.

Usage:
    python train.py [--opponent selfplay|random] [--total-timesteps N]
                     [--reward-mode sparse|dense] [--max-turns N]
                     [--snapshot-freq N] [--pool-size N] [--latest-prob P]
                     [--eval-freq N] [--eval-episodes N]
                     [--output-dir DIR] [--seed N]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Optional, cast

import gymnasium as gym
import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from env import KiriDuelEnv


def _mask_fn(env: gym.Env) -> np.ndarray:
    # ActionMasker always calls this with the KiriDuelEnv it wraps; typed
    # as the general gym.Env here only so it satisfies ActionMasker's own
    # (contravariant) callable parameter type.
    return cast(KiriDuelEnv, env).action_masks()


class SelfPlayOpponent:
    """
    A KiriDuelEnv `opponent_policy` backed by a swappable model
    reference. `.model is None` means "play uniformly at random" (the
    starting state, before any model exists yet); otherwise queries
    that model's masked policy, sampled (not deterministic) so the
    opponent doesn't collapse to one predictable line of play.
    """

    def __init__(self, model: Optional[MaskablePPO] = None) -> None:
        self.model = model

    def __call__(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        if self.model is None:
            legal_actions = np.flatnonzero(action_mask)
            return int(np.random.choice(legal_actions))
        action, _ = self.model.predict(observation, action_masks=action_mask, deterministic=False)
        return int(action)


class SelfPlaySnapshotCallback(BaseCallback):
    """
    Every `snapshot_freq` steps: save the live model as a new pool
    snapshot (evicting the oldest once the pool exceeds `pool_size`),
    then re-roll `opponent.model` -- with probability `latest_prob` the
    live model itself, otherwise a uniformly random snapshot from the
    pool (frozen at whatever point it was saved).
    """

    def __init__(
        self,
        opponent: SelfPlayOpponent,
        snapshot_dir: Path,
        snapshot_freq: int,
        pool_size: int,
        latest_prob: float,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.opponent = opponent
        self.snapshot_dir = snapshot_dir
        self.snapshot_freq = snapshot_freq
        self.pool_size = pool_size
        self.latest_prob = latest_prob
        self._pool: List[Path] = []
        self._next_id = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.snapshot_freq == 0:
            path = self.snapshot_dir / f"snapshot_{self._next_id}.zip"
            self.model.save(path)
            self._pool.append(path)
            self._next_id += 1

            if len(self._pool) > self.pool_size:
                self._pool.pop(0).unlink(missing_ok=True)

            if self._pool and random.random() > self.latest_prob:
                chosen = random.choice(self._pool)
                self.opponent.model = MaskablePPO.load(chosen)
                if self.verbose:
                    print(f"[selfplay] step {self.num_timesteps}: opponent -> {chosen.name}")
            else:
                # self.model is BaseCallback's handle on whatever was passed to
                # .learn() -- always a MaskablePPO here, just typed generically.
                self.opponent.model = cast(MaskablePPO, self.model)
                if self.verbose:
                    print(f"[selfplay] step {self.num_timesteps}: opponent -> live model")
        return True


def make_train_env(opponent: SelfPlayOpponent, reward_mode: str, max_turns: int) -> Monitor:
    env = KiriDuelEnv(opponent_policy=opponent, reward_mode=reward_mode, max_turns=max_turns)
    env = ActionMasker(env, _mask_fn)
    return Monitor(env)


def make_eval_env(reward_mode: str, max_turns: int) -> Monitor:
    # opponent_policy=None -> KiriDuelEnv's default uniform-random legal
    # policy, kept fixed for the whole run (see module docstring).
    env = KiriDuelEnv(opponent_policy=None, reward_mode=reward_mode, max_turns=max_turns)
    env = ActionMasker(env, _mask_fn)
    return Monitor(env)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-play MaskablePPO training for Kiri Ai: The Duel.")
    parser.add_argument("--opponent", choices=["selfplay", "random"], default="selfplay",
                         help="selfplay (default): historical self-play via a snapshot pool. "
                              "random: train against a fixed uniform-random opponent the whole run (debug/sanity mode).")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--reward-mode", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--max-turns", type=int, default=100,
                         help="Per-episode turn cap (training convenience, not a rule -- see env.py).")
    parser.add_argument("--snapshot-freq", type=int, default=10_000,
                         help="Environment steps between self-play opponent snapshots/re-rolls.")
    parser.add_argument("--pool-size", type=int, default=10, help="Max snapshots kept on disk at once.")
    parser.add_argument("--latest-prob", type=float, default=0.5,
                         help="Probability each re-roll picks the live model over a pool snapshot.")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Environment steps between evaluation passes.")
    parser.add_argument("--eval-episodes", type=int, default=50, help="Episodes per evaluation pass.")
    parser.add_argument("--output-dir", type=str, default="runs/latest",
                         help="Where snapshots, the best/final model, and tensorboard logs are written.")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    opponent = SelfPlayOpponent(model=None)
    train_env = make_train_env(opponent, args.reward_mode, args.max_turns)
    eval_env = make_eval_env(args.reward_mode, args.max_turns)

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        seed=args.seed,
        tensorboard_log=str(output_dir / "tb"),
    )

    callbacks: List[BaseCallback] = [
        MaskableEvalCallback(
            eval_env,
            n_eval_episodes=args.eval_episodes,
            eval_freq=args.eval_freq,
            best_model_save_path=str(output_dir / "best"),
            log_path=str(output_dir / "eval"),
            verbose=1,
        ),
    ]

    if args.opponent == "selfplay":
        opponent.model = model  # from now on, the training env's opponent is the live model itself
        snapshot_dir = output_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            SelfPlaySnapshotCallback(
                opponent, snapshot_dir, args.snapshot_freq, args.pool_size, args.latest_prob, verbose=1,
            )
        )
    # opponent.model stays None for --opponent random: KiriDuelEnv's own
    # default uniform-random policy is used for the entire training run.

    model.learn(total_timesteps=args.total_timesteps, callback=CallbackList(callbacks))
    model.save(str(output_dir / "final_model"))
    print(f"\nDone. Final model saved to {output_dir / 'final_model'}.zip")


if __name__ == "__main__":
    main()
