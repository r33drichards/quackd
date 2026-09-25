"""Find-and-kick on the Microduck simulator, scored against the world rather than the pilot.

A policy sees a picture each step and picks one of four low-level actions, the way Laya Vision
plays its games: the screen is the image and the buttons are the options. The verifier never
asks the policy how it went. It reads the simulator's own ground truth every step (where the
duck is, where the ball is, whether a kick connected, how far the ball ended up from where it
started) and scores the episode from that.

Policies:

    laya     Laya Vision (`laya.load_vlm`), greedy, one forward pass per step. `--view` picks
             what it is shown: `cam` (the duck's own camera, what a robot has), `top` (the
             top-down arena, what a game screen would be) or `both`.
    random   uniform over the four actions: the floor.
    blob     a hand-written controller on the duck's camera alone (quackd's colour-blob
             detector): turn to the ball, walk up, kick. What a policy can do with no
             privileged state.
    oracle   the same controller on ground truth: the ceiling for this action space.
    teacher  the oracle, except that it turns left whenever the ball is out of the camera's
             view: the labels `collect_microduck_rollouts.py` trains Laya Vision on.
    pilot    quackd's own scripted pilot (`FakeProvider`) driving the composite verbs
             (`search_scan`, `walk_to`, `kick`) through the real agent loop. A different
             action space, so it has no per-step trace, only the end-of-episode verdict.

Success is the bar `ducks/find-and-kick.duck` sets: a kick that connected and a ball that ended
at least 0.3 m from where it started. Pushing the ball with the body can move it too, which is
reported separately (`displaced`) and does not count.

    uv run python scripts/eval_microduck.py --policy laya random blob oracle pilot --seeds 0-9
    uv run python scripts/eval_microduck.py --policy laya --seeds 0 --video runs/eval/laya-seed0.mp4

Laya Vision is not a quackd dependency: install it next to quackd (`uv pip install -e
../laya-vision torchvision "transformers>=5.3"`). `--video` needs `imageio` and
`imageio-ffmpeg`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from PIL import Image, ImageDraw

from quackd.perception.color_blob import ColorBlobDetector
from quackd.sim2d.render import render_duckcam, render_topdown
from quackd.sim2d.world import KICK_CONE_DEG, KICK_RANGE_M, World
from quackd.transport.base import Intent
from quackd.transport.sim2d import Sim2DTransport

REPO = Path(__file__).resolve().parent.parent

# ── the action space ──────────────────────────────────────────────────────────────────

ACTIONS = {
    "FORWARD": "walk forward",
    "LEFT": "turn left",
    "RIGHT": "turn right",
    "KICK": "kick the ball in front of you",
}
TWIST = {"FORWARD": (0.25, 0.0, 0.0), "LEFT": (0.0, 0.0, 0.6), "RIGHT": (0.0, 0.0, -0.6)}
ACTION_S = 0.5
"""Sim seconds one move lasts: 0.125 m forward, or about 17 degrees of turn."""
RESEND_S = 0.2
"""The world stops a duck whose last move is older than 0.3 s (upstream's deadman)."""
KICK_SETTLE_S = 1.5
"""After a kick, time for the ball to roll before the next picture."""
SUCCESS_M = 0.3
"""`ducks/find-and-kick.duck`: "Ball displaced more than 0.3 m"."""

QUESTION_TEXT = {
    "cam": (
        "You are a small duck robot looking through your own camera. Somewhere in the arena is "
        "an orange ball. Turn to find it, walk up to it until it is close and in front of you, "
        "then kick it. Which action should you take now?"
    ),
    "top": (
        "You are the yellow triangle seen from above, pointing the way you face. Walk to the "
        "orange ball until it is right in front of your point, then kick it. The blue circle is "
        "a person; leave them alone. Which action should you take now?"
    ),
    "both": (
        "You are a small duck robot. The first picture is your own camera; the second shows the "
        "arena from above, where you are the yellow triangle pointing the way you face. Walk to "
        "the orange ball until it is right in front of you, then kick it. Which action should "
        "you take now?"
    ),
}


def question(view: str) -> dict[str, Any]:
    return {
        "action": {"type": "choice", "instructions": QUESTION_TEXT[view], "criteria": dict(ACTIONS)}
    }


# ── ground truth ──────────────────────────────────────────────────────────────────────


@dataclass
class Truth:
    """What the world says, read straight off it. The policy is never shown this."""

    t: float
    duck: tuple[float, float, float]
    ball: tuple[float, float]
    dist_m: float
    """Duck centre to ball centre."""
    bearing_deg: float
    """Ball relative to the duck's heading; positive is to its left."""
    in_view: bool
    """Inside the camera's 90 degree field of view."""
    kickable: bool
    """A kick now would connect: within `KICK_RANGE_M` and `KICK_CONE_DEG`."""
    displacement_m: float
    kicks: int
    kicks_connected: int


def read_truth(world: World) -> Truth:
    d = world.ducks[0]
    dist, bearing = world.relative(world.ball.x, world.ball.y)
    cam_dist, cam_bearing = world.relative(world.ball.x, world.ball.y, camera=True)
    bearing_deg = math.degrees(bearing)
    return Truth(
        t=round(world.t, 3),
        duck=(round(d.x, 3), round(d.y, 3), round(d.theta, 3)),
        ball=(round(world.ball.x, 3), round(world.ball.y, 3)),
        dist_m=round(dist, 3),
        bearing_deg=round(bearing_deg, 1),
        in_view=abs(math.degrees(cam_bearing)) <= 45.0 and cam_dist > 0,
        kickable=dist <= KICK_RANGE_M and abs(bearing_deg) <= KICK_CONE_DEG,
        displacement_m=round(world.ball_displacement_m, 3),
        kicks=world.kicks,
        kicks_connected=world.kicks_connected,
    )


# ── policies ──────────────────────────────────────────────────────────────────────────


class Policy(Protocol):
    name: str

    def reset(self, seed: int) -> None: ...

    def act(self, world: World, cam: Image.Image, top: Image.Image) -> tuple[str, dict[str, float]]:
        """The action, and the probabilities behind it (empty for a policy that has none)."""
        ...


class RandomPolicy:
    name = "random"

    def reset(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def act(self, world, cam, top):
        return self.rng.choice(list(ACTIONS)), {}


def steer(bearing_deg: float, dist_m: float) -> str:
    """Turn to the ball, walk up, kick: the controller both scripted policies share."""
    # the deadband is wider than one turn (17 degrees), or the controller rocks across it
    # forever, and narrower than the kick cone (35), so a ball it walks up to is kickable
    if abs(bearing_deg) > 20.0:
        return "LEFT" if bearing_deg > 0 else "RIGHT"
    if dist_m > 0.22:
        return "FORWARD"
    return "KICK"


class OraclePolicy:
    """`steer` on the world's own numbers."""

    name = "oracle"

    def reset(self, seed: int) -> None:
        pass

    def act(self, world, cam, top):
        dist, bearing = world.relative(world.ball.x, world.ball.y)
        return steer(math.degrees(bearing), dist), {}


def teacher_action(world: World) -> str:
    """`steer` on ground truth while the ball is in the camera's view, and a turn to the left
    while it is not: the oracle, made consistent with what a camera can show. Out of view,
    the picture cannot say which way is shorter, so a label that sometimes says left and
    sometimes right for the same empty floor would teach nothing. These are the labels
    `collect_microduck_rollouts.py` writes."""
    if not read_truth(world).in_view:
        return "LEFT"
    dist, bearing = world.relative(world.ball.x, world.ball.y)
    return steer(math.degrees(bearing), dist)


class TeacherPolicy:
    """`teacher_action`: the ceiling for a policy imitating those labels."""

    name = "teacher"

    def reset(self, seed: int) -> None:
        pass

    def act(self, world, cam, top):
        return teacher_action(world), {}


class BlobPolicy:
    """`steer` on what the camera sees, and a turn to the left while it sees nothing."""

    name = "blob"

    def __init__(self) -> None:
        self.detector = ColorBlobDetector()

    def reset(self, seed: int) -> None:
        pass

    def act(self, world, cam, top):
        balls = [d for d in self.detector.detect(cam) if d.label == "ball"]
        if not balls:
            return "LEFT", {}
        ball = max(balls, key=lambda d: d.area)
        dist = ball.est_distance_m if ball.est_distance_m is not None else 1.0
        # the detector measures from the camera; the kick needs the body's distance
        return steer(ball.bearing_deg or 0.0, dist), {}


class LayaPolicy:
    def __init__(
        self, model: str, view: str, revision: str | None = None, label: str = "laya"
    ) -> None:
        import laya  # laya-vision, not the text-only `laya` on PyPI: it has `load_vlm`

        if not hasattr(laya, "load_vlm"):
            sys.exit(
                "`import laya` found the text-only Laya. "
                "Install laya-vision: uv pip install -e ../laya-vision"
            )
        self.model_id = model
        self.agent = laya.load_vlm(model, revision=revision)
        self.view = view
        self.q = question(view)
        self.name = f"{label}[{view}]"

    def reset(self, seed: int) -> None:
        pass

    def act(self, world, cam, top):
        state = {"cam": {"image": cam}, "top": {"image": top}, "both": {"images": [cam, top]}}[
            self.view
        ]
        answer = self.agent.predict(state, self.q)["answers"]["action"]
        return answer["choice"], {k: round(float(v), 4) for k, v in answer["probabilities"].items()}


# ── an episode ────────────────────────────────────────────────────────────────────────


@dataclass
class Episode:
    policy: str
    seed: int
    success: bool
    """A kick connected and the ball ended at least `SUCCESS_M` from its start."""
    kicked: bool
    """At least one kick connected."""
    displaced: bool
    """The ball ended at least `SUCCESS_M` from its start, by any means (pushing included)."""
    steps: int
    sim_s: float
    start_dist_m: float
    min_dist_m: float
    final_dist_m: float
    progress: float
    """How much of the starting distance the duck closed at its closest: (start - min) / start."""
    first_kickable_step: int | None
    kicks: int
    kicks_connected: int
    ball_displacement_m: float
    ball_in_view_frac: float | None
    kickable_frac: float | None
    actions: dict[str, int] = field(default_factory=dict)
    decision_ms: float | None = None
    trace: list[dict[str, Any]] = field(default_factory=list)


class Video:
    """World | duck cam | the decision, one frame every `every_s` of sim time."""

    def __init__(self, transport: Sim2DTransport, size: int = 320, every_s: float = 0.1) -> None:
        self.transport = transport
        self.size = size
        self.every_s = every_s
        self.frames: list[Image.Image] = []
        self.caption = ""
        self.probs: dict[str, float] = {}
        self._last = -1e9
        transport.add_tick_hook(self._tick)

    def _tick(self, world: World) -> None:
        if world.t - self._last >= self.every_s - 1e-9:
            self._last = world.t
            self.frames.append(self.render(world))

    def render(self, world: World) -> Image.Image:
        s = self.size
        panel_w = 200
        frame = Image.new("RGB", (2 * s + panel_w + 8, s + 28), (24, 24, 28))
        frame.paste(render_topdown(world, s), (0, 28))
        frame.paste(render_duckcam(world, s), (s + 4, 28))
        draw = ImageDraw.Draw(frame)
        draw.text((6, 8), f"t={world.t:5.1f}s  {self.caption}", fill=(235, 235, 235))
        draw.text((s + 10, 8), "duck cam", fill=(170, 170, 170))
        x0, y = 2 * s + 12, 36
        truth = read_truth(world)
        for line in (
            f"ball dist  {truth.dist_m:.2f} m",
            f"bearing    {truth.bearing_deg:+.0f} deg",
            f"kickable   {'yes' if truth.kickable else 'no'}",
            f"kicks      {truth.kicks_connected}/{truth.kicks}",
            f"ball moved {truth.displacement_m:.2f} m",
        ):
            draw.text((x0, y), line, fill=(220, 220, 220))
            y += 16
        if self.probs:
            y += 10
            draw.text((x0, y), "P(action)", fill=(170, 170, 170))
            y += 16
            top = max(self.probs, key=lambda k: self.probs[k])
            for name, p in self.probs.items():
                colour = (255, 170, 60) if name == top else (120, 150, 230)
                draw.rectangle((x0 + 62, y + 3, x0 + 62 + int(p * 110), y + 11), fill=colour)
                draw.text((x0, y), name, fill=(220, 220, 220))
                y += 16
        return frame

    def save(self, path: Path, fps: int = 10) -> Path:
        import imageio.v2 as imageio
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        hold = [self.frames[-1]] * (fps * 2)  # linger on the ending
        with imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=8) as w:
            for f in self.frames + hold:
                w.append_data(np.asarray(f))  # type: ignore[attr-defined]
        return path


async def move(transport: Sim2DTransport, action: str) -> None:
    vx, vy, wz = TWIST[action]
    left = ACTION_S
    while left > 1e-9:
        await transport.send_intent(Intent.move(vx, vy, wz))
        chunk = min(RESEND_S, left)
        await transport.sleep(chunk)
        left -= chunk
    await transport.send_intent(Intent.stop())


async def run_episode(
    policy: Policy,
    seed: int,
    max_steps: int,
    video: Video | None = None,
    transport: Sim2DTransport | None = None,
) -> Episode:
    transport = transport or Sim2DTransport(seed=seed)
    await transport.connect()
    world = transport.world
    policy.reset(seed)
    start = read_truth(world)
    min_dist = start.dist_m
    first_kickable: int | None = None
    counts: Counter[str] = Counter()
    trace: list[dict[str, Any]] = []
    ms: list[float] = []
    seen = kickable = 0
    steps = 0
    try:
        for step in range(max_steps):
            before = read_truth(world)
            seen += before.in_view
            kickable += before.kickable
            if before.kickable and first_kickable is None:
                first_kickable = step
            cam = render_duckcam(world, 256)
            top = render_topdown(world, 256)
            t0 = time.perf_counter()
            action, probs = policy.act(world, cam, top)
            ms.append((time.perf_counter() - t0) * 1000)
            counts[action] += 1
            steps = step + 1
            if video is not None:
                video.caption = f"{policy.name}  seed {seed}  step {steps}: {action}"
                video.probs = probs
            if action == "KICK":
                await transport.send_intent(Intent.do("kick_right"))
                await transport.sleep(KICK_SETTLE_S)
            else:
                await move(transport, action)
            after = read_truth(world)
            min_dist = min(min_dist, after.dist_m)
            trace.append(
                {
                    "step": steps,
                    "action": action,
                    "probs": probs,
                    "before": asdict(before),
                    "connected": after.kicks_connected > before.kicks_connected,
                }
            )
            if after.kicks_connected and after.displacement_m >= SUCCESS_M:
                break
    finally:
        await transport.close()
        await transport.clock.stop()
    end = read_truth(world)
    return Episode(
        policy=policy.name,
        seed=seed,
        success=end.kicks_connected > 0 and end.displacement_m >= SUCCESS_M,
        kicked=end.kicks_connected > 0,
        displaced=end.displacement_m >= SUCCESS_M,
        steps=steps,
        sim_s=end.t,
        start_dist_m=start.dist_m,
        min_dist_m=round(min_dist, 3),
        final_dist_m=end.dist_m,
        progress=round(max(0.0, (start.dist_m - min_dist) / start.dist_m), 3),
        first_kickable_step=first_kickable,
        kicks=end.kicks,
        kicks_connected=end.kicks_connected,
        ball_displacement_m=end.displacement_m,
        ball_in_view_frac=round(seen / steps, 3) if steps else None,
        kickable_frac=round(kickable / steps, 3) if steps else None,
        actions=dict(counts),
        decision_ms=round(statistics.median(ms), 1) if ms else None,
        trace=trace,
    )


async def run_pilot(seed: int, max_steps: int, runs_dir: Path) -> Episode:
    """quackd's scripted pilot through the real agent loop, graded by the same world."""
    from quackd.agent.loop import RunConfig, run_duck
    from quackd.agent.providers.fake import FakeProvider
    from quackd.duckfile.parser import load_duck

    transport = Sim2DTransport(seed=seed)
    start = read_truth(transport.world)
    # the loop connects the transport itself; the minimum distance is read off every tick
    closest = [start.dist_m]
    transport.add_tick_hook(lambda w: closest.__setitem__(0, min(closest[0], read_truth(w).dist_m)))
    result = await run_duck(
        RunConfig(
            duck=load_duck(str(REPO / "ducks" / "find-and-kick.duck")),
            provider=FakeProvider.for_duck("find-and-kick"),
            transport=transport,
            detector=ColorBlobDetector(),
            runs_dir=runs_dir,
            max_steps=max_steps,
        )
    )
    await transport.clock.stop()
    end = read_truth(transport.world)
    return Episode(
        policy="pilot",
        seed=seed,
        success=end.kicks_connected > 0 and end.displacement_m >= SUCCESS_M,
        kicked=end.kicks_connected > 0,
        displaced=end.displacement_m >= SUCCESS_M,
        steps=result.steps,
        sim_s=end.t,
        start_dist_m=start.dist_m,
        min_dist_m=round(closest[0], 3),
        final_dist_m=end.dist_m,
        progress=round(max(0.0, (start.dist_m - closest[0]) / start.dist_m), 3),
        first_kickable_step=None,
        kicks=end.kicks,
        kicks_connected=end.kicks_connected,
        ball_displacement_m=end.displacement_m,
        ball_in_view_frac=None,
        kickable_frac=None,
        actions={"verbs": result.steps},
        trace=[
            {
                "pilot_outcome": result.outcome,
                "pilot_reason": result.reason,
                "run_dir": str(result.run_dir),
            }
        ],
    )


# ── the sweep ─────────────────────────────────────────────────────────────────────────


def summarise(episodes: list[Episode]) -> dict[str, Any]:
    n = len(episodes)

    def mean(key: str) -> float | None:
        vals = [getattr(e, key) for e in episodes if getattr(e, key) is not None]
        return round(statistics.fmean(vals), 3) if vals else None

    return {
        "episodes": n,
        "success_rate": round(sum(e.success for e in episodes) / n, 3),
        "kicked_rate": round(sum(e.kicked for e in episodes) / n, 3),
        "displaced_rate": round(sum(e.displaced for e in episodes) / n, 3),
        "mean_progress": mean("progress"),
        "mean_min_dist_m": mean("min_dist_m"),
        "mean_final_dist_m": mean("final_dist_m"),
        "mean_ball_displacement_m": mean("ball_displacement_m"),
        "mean_steps": mean("steps"),
        "mean_kicks": mean("kicks"),
        "mean_ball_in_view_frac": mean("ball_in_view_frac"),
        "mean_kickable_frac": mean("kickable_frac"),
        "median_decision_ms": mean("decision_ms"),
        "actions": dict(sum((Counter(e.actions) for e in episodes), Counter())),
    }


def normalise(summary: dict[str, dict[str, Any]]) -> None:
    """The games benchmark's convention: 0 is random play, 1 is the oracle."""
    lo, hi = summary.get("random"), summary.get("oracle")
    if lo is None or hi is None:
        return
    for s in summary.values():
        for key in ("success_rate", "mean_progress"):
            span = hi[key] - lo[key]
            s[f"normalised_{key}"] = round((s[key] - lo[key]) / span, 3) if span else None


def parse_seeds(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b) + 1) if b else [int(a)])
    return out


def table(summary: dict[str, dict[str, Any]]) -> str:
    cols = [
        ("success_rate", "success"),
        ("kicked_rate", "kicked"),
        ("displaced_rate", "displaced"),
        ("mean_progress", "progress"),
        ("mean_min_dist_m", "min dist m"),
        ("mean_ball_displacement_m", "ball moved m"),
        ("mean_steps", "steps"),
        ("mean_ball_in_view_frac", "ball in view"),
        ("normalised_success_rate", "norm. success"),
        ("normalised_mean_progress", "norm. progress"),
        ("median_decision_ms", "ms/step"),
    ]
    head = "| policy | " + " | ".join(c[1] for c in cols) + " |"
    rule = "|---|" + "---:|" * len(cols)
    rows = []
    for name, s in summary.items():
        cells = [
            "-" if s.get(k) is None else f"{s[k]:.2f}" if isinstance(s[k], float) else str(s[k])
            for k, _ in cols
        ]
        rows.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join([head, rule, *rows])


async def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--policy",
        nargs="+",
        default=["laya", "random", "blob", "oracle", "pilot"],
        choices=["laya", "random", "blob", "oracle", "teacher", "pilot"],
    )
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 0,3,7")
    ap.add_argument(
        "--max-steps", type=int, default=60, help="decisions per episode (pilot: verbs)"
    )
    ap.add_argument("--model", default="thaitea/laya-vision")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--label", default="laya", help="the name laya's rows are reported under")
    ap.add_argument(
        "--view",
        nargs="+",
        default=["cam"],
        choices=["cam", "top", "both"],
        help="what laya is shown; several run one after another",
    )
    ap.add_argument(
        "--out", type=Path, default=REPO / "runs" / "eval" / "microduck-find-and-kick.json"
    )
    ap.add_argument(
        "--video",
        type=Path,
        default=None,
        help="record the first seed of the first policy to this .mp4 (not the pilot)",
    )
    ap.add_argument("--no-trace", action="store_true", help="leave per-step traces out of the JSON")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    policies: list[Policy | str] = []
    for name in args.policy:
        if name == "laya":
            policies.extend(
                LayaPolicy(args.model, view, args.revision, args.label) for view in args.view
            )
        elif name == "pilot":
            policies.append("pilot")
        else:
            simple: dict[str, type[Policy]] = {
                "random": RandomPolicy,
                "blob": BlobPolicy,
                "oracle": OraclePolicy,
                "teacher": TeacherPolicy,
            }
            policies.append(simple[name]())

    results: dict[str, list[Episode]] = {}
    video_path: Path | None = None
    for policy in policies:
        name = policy if isinstance(policy, str) else policy.name
        eps = results.setdefault(name, [])
        for seed in seeds:
            t0 = time.perf_counter()
            if isinstance(policy, str):
                ep = await run_pilot(seed, args.max_steps, args.out.parent / "pilot-runs")
            else:
                video = None
                transport = Sim2DTransport(seed=seed)
                if args.video is not None and video_path is None:
                    video = Video(transport)
                ep = await run_episode(policy, seed, args.max_steps, video, transport)
                if video is not None:
                    video_path = video.save(args.video)
            eps.append(ep)
            print(
                f"{name:12s} seed {seed}: {'SUCCESS' if ep.success else 'fail   '} "
                f"kicked={ep.kicked} ball_moved={ep.ball_displacement_m:.2f}m "
                f"dist {ep.start_dist_m:.2f}->{ep.min_dist_m:.2f}m steps={ep.steps} "
                f"({time.perf_counter() - t0:.0f}s)",
                flush=True,
            )

    summary = {name: summarise(eps) for name, eps in results.items()}
    normalise(summary)
    report = {
        "task": "find-and-kick",
        "sim": "microduck:sim2d",
        "seeds": seeds,
        "max_steps": args.max_steps,
        "action_s": ACTION_S,
        "success_m": SUCCESS_M,
        "model": args.model if "laya" in args.policy else None,
        "revision": args.revision,
        "summary": summary,
        "episodes": {
            name: [{**asdict(e), **({"trace": []} if args.no_trace else {})} for e in eps]
            for name, eps in results.items()
        },
        "video": str(video_path) if video_path else None,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1))
    print()
    print(table(summary))
    print(f"\nwrote {args.out}" + (f"\nvideo {video_path}" if video_path else ""))


if __name__ == "__main__":
    asyncio.run(main())
