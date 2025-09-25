# manim -pqh main.py GaussianPhysicsScene (quick)
# manim -p -r 1080,1920 main.py GaussianPhysicsScene (production vertical)

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from manim import *

# -----------------------
# Global configuration
# -----------------------
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30
DURATION = 30

BG_COLOR = "#0F1115"
BALL_COLOR = YELLOW
BALL_RADIUS = 0.036
BALL_COUNT = 1200
SPAWN_RATE = 40.0  # balls per second
GRAVITY = -9.8 * 1.8
RESTITUTION = 0.42
FRICTION_COEFF = 0.45
LINEAR_DAMPING = 0.35
WALL_RESTITUTION = 0.4
FLOOR_FRICTION = 0.18
PAIR_DAMPING = 0.02

SIGMA = 1.0
X_CLIP = 3.0
WALL_X_LIMITS = (-3.3, 3.3)
SPAWN_Y = 6.5
FLOOR_Y = -6.2

DT = 1.0 / 120.0
SUBSTEPS = 2

SEED = 7

# Configure Manim
config.background_color = BG_COLOR
config.pixel_width = VIDEO_WIDTH
config.pixel_height = VIDEO_HEIGHT
config.frame_rate = FPS
config.frame_height = 14.0
config.frame_width = config.frame_height * (9.0 / 16.0)
config.output_file = "gaussian_physics"

random.seed(SEED)
np.random.seed(SEED)


@dataclass
class BallState:
    position: np.ndarray
    velocity: np.ndarray
    radius: float
    id: int


class PhysicsWorld:
    def __init__(
        self,
        gravity: float,
        restitution: float,
        friction: float,
        radius: float,
        spawn_rate: float,
        max_balls: int,
        spawn_y: float,
        floor_y: float,
        wall_limits: Tuple[float, float],
        linear_damping: float,
        wall_restitution: float,
        floor_friction: float,
        pair_damping: float,
    ) -> None:
        self.gravity = gravity
        self.restitution = restitution
        self.friction = friction
        self.radius = radius
        self.spawn_rate = spawn_rate
        self.max_balls = max_balls
        self.spawn_y = spawn_y
        self.floor_y = floor_y
        self.left_wall, self.right_wall = wall_limits
        self.linear_damping = linear_damping
        self.wall_restitution = wall_restitution
        self.floor_friction = floor_friction
        self.pair_damping = pair_damping
        self.balls: List[BallState] = []
        self.spawn_accumulator = 0.0
        self.time = 0.0
        self.cell_size = radius * 2.6
        self.grid: Dict[Tuple[int, int], List[int]] = {}
        self.next_id = 0
        self.new_ball_indices: List[int] = []

        self.neighbor_offsets = [
            (dx, dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]

    def reset_new_indices(self) -> None:
        self.new_ball_indices.clear()

    def spawn_ball(self) -> None:
        x_pos = float(np.random.normal(0.0, SIGMA))
        x_pos = float(np.clip(x_pos, -X_CLIP, X_CLIP))
        x_pos = np.clip(
            x_pos,
            self.left_wall + self.radius + 1e-3,
            self.right_wall - self.radius - 1e-3,
        )
        ball = BallState(
            position=np.array([x_pos, self.spawn_y], dtype=np.float64),
            velocity=np.zeros(2, dtype=np.float64),
            radius=self.radius,
            id=self.next_id,
        )
        self.next_id += 1
        self.new_ball_indices.append(len(self.balls))
        self.balls.append(ball)

    def rebuild_spatial_hash(self) -> None:
        self.grid.clear()
        inv_cell = 1.0 / self.cell_size
        for idx, ball in enumerate(self.balls):
            cell = (
                int(math.floor(ball.position[0] * inv_cell)),
                int(math.floor(ball.position[1] * inv_cell)),
            )
            self.grid.setdefault(cell, []).append(idx)

    def resolve_floor_and_walls(self, ball: BallState) -> None:
        r = ball.radius
        if ball.position[0] - r < self.left_wall:
            ball.position[0] = self.left_wall + r
            if ball.velocity[0] < 0.0:
                ball.velocity[0] = -ball.velocity[0] * self.wall_restitution
        elif ball.position[0] + r > self.right_wall:
            ball.position[0] = self.right_wall - r
            if ball.velocity[0] > 0.0:
                ball.velocity[0] = -ball.velocity[0] * self.wall_restitution

        if ball.position[1] - r < self.floor_y:
            ball.position[1] = self.floor_y + r
            if ball.velocity[1] < 0.0:
                ball.velocity[1] = -ball.velocity[1] * self.restitution
                ball.velocity[0] *= (1.0 - self.floor_friction)
                if abs(ball.velocity[1]) < 0.08:
                    ball.velocity[1] = 0.0

    def resolve_pair(self, i: int, j: int) -> None:
        ball_a = self.balls[i]
        ball_b = self.balls[j]
        delta = ball_b.position - ball_a.position
        dist_sq = float(delta[0] * delta[0] + delta[1] * delta[1])
        min_dist = ball_a.radius + ball_b.radius
        min_dist_sq = min_dist * min_dist
        if dist_sq >= min_dist_sq:
            return
        if dist_sq < 1e-10:
            delta = np.array([self.radius * 0.1, 0.0])
            dist_sq = float(delta[0] * delta[0] + delta[1] * delta[1])
        dist = math.sqrt(dist_sq)
        normal = delta / dist
        penetration = min_dist - dist
        penetration = min(penetration, min_dist * 0.9)
        correction = 0.5 * penetration * normal
        ball_a.position -= correction
        ball_b.position += correction

        relative_velocity = ball_a.velocity - ball_b.velocity
        vel_along_normal = float(np.dot(relative_velocity, normal))
        if vel_along_normal > 0.0:
            vel_along_normal = 0.0
        impulse_mag = -(1.0 + self.restitution) * vel_along_normal * 0.5
        impulse_vec = impulse_mag * normal
        ball_a.velocity += -impulse_vec
        ball_b.velocity += impulse_vec

        tangent = relative_velocity - vel_along_normal * normal
        tan_norm = np.linalg.norm(tangent)
        if tan_norm > 1e-9:
            tangent /= tan_norm
            rel_tangent = float(np.dot(relative_velocity, tangent))
            max_friction = self.friction * abs(impulse_mag)
            friction_impulse = -np.clip(rel_tangent * 0.5, -max_friction, max_friction)
            friction_vec = friction_impulse * tangent
            ball_a.velocity += -friction_vec
            ball_b.velocity += friction_vec

        ball_a.velocity *= (1.0 - self.pair_damping)
        ball_b.velocity *= (1.0 - self.pair_damping)

    def step(self, dt: float) -> List[int]:
        self.reset_new_indices()
        self.time += dt
        self.spawn_accumulator += dt * self.spawn_rate
        while self.spawn_accumulator >= 1.0 and len(self.balls) < self.max_balls:
            self.spawn_ball()
            self.spawn_accumulator -= 1.0

        sub_dt = dt / SUBSTEPS
        gravity_vec = np.array([0.0, self.gravity], dtype=np.float64)
        for _ in range(SUBSTEPS):
            for ball in self.balls:
                ball.velocity += gravity_vec * sub_dt
                ball.velocity *= max(0.0, 1.0 - self.linear_damping * sub_dt)
                ball.position += ball.velocity * sub_dt
                self.resolve_floor_and_walls(ball)

            self.rebuild_spatial_hash()
            for i, ball in enumerate(self.balls):
                cell_x = int(math.floor(ball.position[0] / self.cell_size))
                cell_y = int(math.floor(ball.position[1] / self.cell_size))
                for dx, dy in self.neighbor_offsets:
                    key = (cell_x + dx, cell_y + dy)
                    for j in self.grid.get(key, []):
                        if j <= i:
                            continue
                        self.resolve_pair(i, j)
                self.resolve_floor_and_walls(ball)
        return list(self.new_ball_indices)


class GaussianPhysicsScene(Scene):
    def construct(self) -> None:
        self.camera.background_color = BG_COLOR
        world = PhysicsWorld(
            gravity=GRAVITY,
            restitution=RESTITUTION,
            friction=FRICTION_COEFF,
            radius=BALL_RADIUS,
            spawn_rate=SPAWN_RATE,
            max_balls=BALL_COUNT,
            spawn_y=SPAWN_Y,
            floor_y=FLOOR_Y,
            wall_limits=WALL_X_LIMITS,
            linear_damping=LINEAR_DAMPING,
            wall_restitution=WALL_RESTITUTION,
            floor_friction=FLOOR_FRICTION,
            pair_damping=PAIR_DAMPING,
        )

        balls_group = VGroup()
        self.add(balls_group)

        def updater(group: VGroup, dt: float) -> None:
            if dt <= 0.0:
                return
            new_indices = world.step(dt)
            for idx in new_indices:
                ball = world.balls[idx]
                circle = Circle(radius=ball.radius, color=BALL_COLOR, fill_color=BALL_COLOR, fill_opacity=1.0)
                circle.set_stroke(width=0.0)
                circle.move_to([ball.position[0], ball.position[1], 0.0])
                group.add(circle)
            for mob, ball in zip(group, world.balls):
                mob.move_to([ball.position[0], ball.position[1], 0.0])

        balls_group.add_updater(updater)
        self.wait(DURATION)
        balls_group.remove_updater(updater)
