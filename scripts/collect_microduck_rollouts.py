"""Duck-camera frames from the Microduck simulator, each labelled with the action the teacher
would take: a training set for Laya Vision on the same task and action space
`eval_microduck.py` scores.

The teacher is `eval_microduck.teacher_action`: the ground-truth oracle, except that it turns
left whenever the ball is out of the camera's view, because a picture of empty floor cannot say
which way is shorter. The duck does not always do what the teacher says. With probability
`--epsilon` it takes a random move instead, and the frame it lands on is labelled all the same,
so the set holds the off-course states a learner will wander into and the way back from them
(DAgger's point, without the learner in the loop yet). Frames with no ball in view are
all bare floor, so only `--keep-blind` of them are kept. A fraction of episodes (`--near-frac`)
start with the duck placed close to the ball, because a kick is one decision an episode and
would otherwise be rare in the data.

Seeds never overlap the eval's 0..9: train from `--seed-base` (1000), val from
`--seed-base + 100000`. The output is Laya Vision's JSONL layout, the same as its game frames:

    <out>/train.jsonl, val.jsonl   {"id", "episode", "step", "image", "question", "label"}
    <out>/images/<id>.png          the 256 x 256 duck camera
    <out>/meta.json, _READY

    uv run python scripts/collect_microduck_rollouts.py --out runs/data/microduck_kick
    modal volume put laya-datasets runs/data/microduck_kick vqa/microduck_kick
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_microduck import (  # a sibling script, not a package
    ACTIONS,
    KICK_SETTLE_S,
    SUCCESS_M,
    move,
    question,
    read_truth,
    teacher_action,
)

from quackd.sim2d.render import render_duckcam
from quackd.sim2d.world import ARENA_HALF, DUCK_R, World
from quackd.transport.base import Intent
from quackd.transport.sim2d import Sim2DTransport

LABELS = list(ACTIONS)
VAL_SEED_OFFSET = 100_000


def place_near_ball(world: World, rng: random.Random) -> None:
    """Put the duck 0.15 to 0.6 m from the ball, facing anywhere."""
    duck = world.ducks[0]
    lim = ARENA_HALF - DUCK_R
    for _ in range(100):
        r = rng.uniform(0.15, 0.6)
        a = rng.uniform(-math.pi, math.pi)
        x, y = world.ball.x + r * math.cos(a), world.ball.y + r * math.sin(a)
        if abs(x) < lim and abs(y) < lim:
            duck.x, duck.y = x, y
            duck.theta = rng.uniform(-math.pi, math.pi)
            return


async def episode(
    seed: int, split: str, out: Path, args: argparse.Namespace
) -> tuple[list[dict[str, Any]], bool]:
    rng = random.Random(seed)
    world = World(seed=seed)
    if rng.random() < args.near_frac:
        place_near_ball(world, rng)
    transport = Sim2DTransport(seed=seed, world=world)
    await transport.connect()
    q = question("cam")["action"]
    records: list[dict[str, Any]] = []
    try:
        for step in range(args.max_steps):
            label = teacher_action(world)
            rid = f"{split}-{seed:06d}-{step:03d}"
            # the duck acts on every step, but a frame with no ball in it is the same bare
            # floor over and over: keeping all of them would make "turn left" most of the set
            if read_truth(world).in_view or rng.random() < args.keep_blind:
                render_duckcam(world, 256).save(out / "images" / f"{rid}.png")
                records.append(
                    {
                        "id": rid,
                        "episode": f"{split}-{seed:06d}",
                        "step": step,
                        "image": f"images/{rid}.png",
                        "state_text": None,
                        "question": q,
                        "label": LABELS.index(label),
                    }
                )
            action = label
            if rng.random() < args.epsilon:
                action = rng.choice(["FORWARD", "LEFT", "RIGHT"])
            if action == "KICK":
                await transport.send_intent(Intent.do("kick_right"))
                await transport.sleep(KICK_SETTLE_S)
            else:
                await move(transport, action)
            if world.kicks_connected and world.ball_displacement_m >= SUCCESS_M:
                return records, True
    finally:
        await transport.close()
        await transport.clock.stop()
    return records, False


async def split(name: str, seeds: range, out: Path, args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    wins = 0
    for seed in seeds:
        recs, won = await episode(seed, name, out, args)
        records += recs
        wins += won
    # finetune_long holds out the LAST n_calib train records for temperature fitting, so the
    # file is shuffled: otherwise that holdout would be the final few episodes, not a sample
    random.Random(0).shuffle(records)
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    counts = Counter(LABELS[r["label"]] for r in records)
    return {
        "episodes": len(seeds),
        "frames": len(records),
        "teacher_success_rate": round(wins / len(seeds), 3),
        "labels": dict(counts),
    }


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("runs/data/microduck_kick"))
    ap.add_argument("--train-episodes", type=int, default=800)
    ap.add_argument("--val-episodes", type=int, default=80)
    ap.add_argument("--seed-base", type=int, default=1000)
    ap.add_argument("--epsilon", type=float, default=0.3, help="chance of a random move")
    ap.add_argument("--near-frac", type=float, default=0.3, help="episodes started near the ball")
    ap.add_argument("--max-steps", type=int, default=60)
    ap.add_argument(
        "--keep-blind", type=float, default=0.25, help="share of ball-out-of-view frames kept"
    )
    args = ap.parse_args()
    if args.seed_base < 10:
        sys.exit("--seed-base must stay clear of the eval's seeds 0..9")

    (args.out / "images").mkdir(parents=True, exist_ok=True)
    base = args.seed_base
    meta = {
        "source": "quackd scripts/collect_microduck_rollouts.py",
        "task": "find-and-kick, microduck:sim2d, duck camera",
        "teacher": "eval_microduck.teacher_action",
        "epsilon": args.epsilon,
        "near_frac": args.near_frac,
        "keep_blind": args.keep_blind,
        "max_steps": args.max_steps,
        "options": LABELS,
        "train": await split("train", range(base, base + args.train_episodes), args.out, args),
        "val": await split(
            "val",
            range(base + VAL_SEED_OFFSET, base + VAL_SEED_OFFSET + args.val_episodes),
            args.out,
            args,
        ),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1))
    (args.out / "_READY").touch()
    print(json.dumps({k: meta[k] for k in ("train", "val")}, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
