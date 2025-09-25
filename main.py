from manim import *
import numpy as np
from collections import defaultdict

BALL_COUNT = 1100
SPAWN_RATE = 38.0
BALL_RADIUS = 0.035
GRAVITY = np.array([0.0, -6.2])
RESTITUTION = 0.45
FRICTION = 0.22
LINEAR_DAMPING = 0.08
DT = 1.0 / 190.0
SUBSTEPS = 1
N_BINS = 21
SIGMA = 1.0
X_CLIP = 2.8
BIN_WALL_HEIGHT_RATIO = 0.55
GHOST_ALPHA = 0.22
SEED = 7
VIDEO_WIDTH = 1080
VIDEO_HEIGHT = 1920
FPS = 30
DURATION = 30
BG_COLOR = "#0F1115"
BALL_COLOR = YELLOW

config.pixel_width = VIDEO_WIDTH
config.pixel_height = VIDEO_HEIGHT
config.frame_rate = FPS
config.output_file = "gaussian_physics"
config.background_color = BG_COLOR
config.frame_height = 14.0
config.frame_width = config.frame_height * (VIDEO_WIDTH / VIDEO_HEIGHT)

FRAME_WIDTH = config.frame_width
FRAME_HEIGHT = config.frame_height

SLEEP_SPEED_THRESHOLD = 0.12
SLEEP_TIME = 0.5
GRID_CELL_SIZE = BALL_RADIUS * 3.0
PACKING_HEIGHT = 2.05 * BALL_RADIUS
GHOST_OMEGA = 6.5
JITTER_SCALE = 0.15 * 2 * BALL_RADIUS


class Ball:
    def __init__(self, position, velocity, radius, circle):
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.radius = radius
        self.circle = circle
        self.sleep_timer = 0.0
        self.sleeping = False

    def sync_mobject(self):
        self.circle.move_to([self.position[0], self.position[1], 0])

    def wake(self):
        self.sleep_timer = 0.0
        self.sleeping = False

    def update_sleep_state(self, dt):
        speed = np.linalg.norm(self.velocity)
        if speed < SLEEP_SPEED_THRESHOLD:
            self.sleep_timer += dt
            if self.sleep_timer >= SLEEP_TIME:
                self.sleeping = True
        else:
            self.wake()


class Bins:
    def __init__(self, floor_y):
        self.n_bins = N_BINS
        self.floor_y = floor_y
        self.left = -FRAME_WIDTH * 0.48
        self.right = FRAME_WIDTH * 0.48
        self.width = self.right - self.left
        self.bin_width = self.width / self.n_bins
        self.edges = np.linspace(self.left, self.right, self.n_bins + 1)
        self.centers = self.edges[:-1] + self.bin_width * 0.5
        self.separator_height = BIN_WALL_HEIGHT_RATIO * (FRAME_HEIGHT * 0.5)
        self.counts = np.zeros(self.n_bins, dtype=int)
        self.ghost_heights = np.zeros(self.n_bins, dtype=float)
        self.ghost_velocities = np.zeros(self.n_bins, dtype=float)
        self.ghost_bars = self._create_ghost_bars()
        self.separators = self._create_separators()
        self.floor = self._create_floor()

    def _create_floor(self):
        floor_thickness = BALL_RADIUS * 0.6
        floor = Rectangle(width=self.width, height=floor_thickness, fill_opacity=0.3, color=WHITE, stroke_width=0)
        floor.move_to([0, self.floor_y - floor_thickness * 0.5, 0])
        floor.set_fill(color=WHITE, opacity=0.15)
        floor.set_z_index(-2)
        return floor

    def _create_separators(self):
        group = VGroup()
        thickness = self.bin_width * 0.05
        for edge in self.edges[1:-1]:
            rect = Rectangle(width=thickness, height=self.separator_height, stroke_width=0)
            rect.set_fill(color=WHITE, opacity=0.28)
            rect.move_to([edge, self.floor_y + self.separator_height * 0.5, 0])
            rect.set_z_index(-1)
            group.add(rect)
        return group

    def _create_ghost_bars(self):
        bars = VGroup()
        for center in self.centers:
            rect = Rectangle(width=self.bin_width * 0.82, height=PACKING_HEIGHT * 0.5, stroke_width=0)
            rect.set_fill(color=Color("#5aa382"), opacity=GHOST_ALPHA)
            rect.move_to([center, self.floor_y + PACKING_HEIGHT * 0.25, 0])
            rect.set_z_index(-3)
            bars.add(rect)
        return bars

    def bin_index(self, x):
        rel = (x - self.left) / self.bin_width
        idx = int(np.clip(np.floor(rel), 0, self.n_bins - 1))
        return idx

    def update_counts(self, balls):
        self.counts[:] = 0
        for ball in balls:
            idx = self.bin_index(ball.position[0])
            self.counts[idx] += 1

    def update_ghosts(self, dt):
        targets = self.counts.astype(float) * PACKING_HEIGHT
        for i in range(self.n_bins):
            height = self.ghost_heights[i]
            velocity = self.ghost_velocities[i]
            target = targets[i]
            accel = -2 * GHOST_OMEGA * velocity - (GHOST_OMEGA**2) * (height - target)
            velocity += accel * dt
            height += velocity * dt
            if height < 0.0:
                height = 0.0
                velocity = 0.0
            self.ghost_velocities[i] = velocity
            self.ghost_heights[i] = height
            bar = self.ghost_bars[i]
            bar.stretch_to_fit_height(max(height, 1e-3))
            bar.move_to([self.centers[i], self.floor_y + height * 0.5, 0])


class PhysicsWorld:
    def __init__(self, bins, ball_group):
        self.bins = bins
        self.ball_group = ball_group
        self.balls = []
        self.time_accumulator = 0.0
        self.spawn_accumulator = 0.0
        self.fixed_dt = DT
        self.sub_dt = DT / SUBSTEPS
        self.floor_y = bins.floor_y
        self.wall_top_y = self.floor_y + bins.separator_height
        self.left_wall = bins.left
        self.right_wall = bins.right
        self.rng = np.random.default_rng(SEED)
        self.total_spawned = 0

    def step(self, dt):
        self.time_accumulator += dt
        while self.time_accumulator >= self.fixed_dt:
            self.time_accumulator -= self.fixed_dt
            self.fixed_update(self.fixed_dt)

    def fixed_update(self, dt):
        self.spawn_accumulator += SPAWN_RATE * dt
        while self.spawn_accumulator >= 1.0 and self.total_spawned < BALL_COUNT:
            self.spawn_ball()
            self.spawn_accumulator -= 1.0

        for _ in range(SUBSTEPS):
            self.integrate(self.sub_dt)
            self.resolve_collisions()
            self.update_sleep_states(self.sub_dt)
        self.bins.update_counts(self.balls)
        self.bins.update_ghosts(dt)
        for ball in self.balls:
            ball.sync_mobject()

    def integrate(self, dt):
        for ball in self.balls:
            if not ball.sleeping:
                ball.velocity += GRAVITY * dt
                ball.velocity *= (1.0 - LINEAR_DAMPING * dt)
                ball.position += ball.velocity * dt
            self.handle_boundaries(ball)

    def handle_boundaries(self, ball):
        radius = ball.radius
        if ball.position[1] - radius < self.floor_y:
            ball.position[1] = self.floor_y + radius
            if ball.velocity[1] < 0:
                ball.velocity[1] = -ball.velocity[1] * RESTITUTION
            ball.velocity[0] *= (1.0 - FRICTION)
            if abs(ball.velocity[1]) < SLEEP_SPEED_THRESHOLD * 0.5:
                ball.velocity[1] = 0.0
        if ball.position[0] - radius < self.left_wall:
            ball.position[0] = self.left_wall + radius
            if ball.velocity[0] < 0:
                ball.velocity[0] = -ball.velocity[0] * RESTITUTION
        if ball.position[0] + radius > self.right_wall:
            ball.position[0] = self.right_wall - radius
            if ball.velocity[0] > 0:
                ball.velocity[0] = -ball.velocity[0] * RESTITUTION
        if ball.position[1] - radius < self.wall_top_y:
            for edge in self.bins.edges[1:-1]:
                if ball.position[0] <= edge and ball.position[0] + radius > edge:
                    penetration = ball.position[0] + radius - edge
                    ball.position[0] -= penetration
                    if ball.velocity[0] > 0:
                        ball.velocity[0] = -ball.velocity[0] * RESTITUTION
                        ball.velocity[1] *= (1.0 - FRICTION * 0.5)
                    ball.wake()
                elif ball.position[0] >= edge and ball.position[0] - radius < edge:
                    penetration = edge - (ball.position[0] - radius)
                    ball.position[0] += penetration
                    if ball.velocity[0] < 0:
                        ball.velocity[0] = -ball.velocity[0] * RESTITUTION
                        ball.velocity[1] *= (1.0 - FRICTION * 0.5)
                    ball.wake()

    def resolve_collisions(self):
        grid = defaultdict(list)
        origin_x = self.left_wall - GRID_CELL_SIZE * 2
        origin_y = self.floor_y - GRID_CELL_SIZE * 2
        for idx, ball in enumerate(self.balls):
            cell_x = int((ball.position[0] - origin_x) / GRID_CELL_SIZE)
            cell_y = int((ball.position[1] - origin_y) / GRID_CELL_SIZE)
            grid[(cell_x, cell_y)].append(idx)
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 0), (0, 1),
            (1, -1), (1, 0), (1, 1),
        ]
        visited_pairs = set()
        for cell, indices in grid.items():
            for dx, dy in neighbor_offsets:
                neighbor = (cell[0] + dx, cell[1] + dy)
                if neighbor not in grid:
                    continue
                for i in indices:
                    for j in grid[neighbor]:
                        if j <= i:
                            continue
                        pair_key = (min(i, j), max(i, j))
                        if pair_key in visited_pairs:
                            continue
                        visited_pairs.add(pair_key)
                        self.narrow_phase(self.balls[i], self.balls[j])

    def narrow_phase(self, a, b):
        delta = b.position - a.position
        dist_sq = np.dot(delta, delta)
        min_dist = a.radius + b.radius
        if dist_sq >= min_dist * min_dist:
            return
        dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 0.0
        if dist == 0.0:
            normal = np.array([1.0, 0.0])
        else:
            normal = delta / dist
        penetration = min_dist - dist
        correction = normal * (penetration * 0.5)
        if not a.sleeping:
            a.position -= correction
        if not b.sleeping:
            b.position += correction
        a.wake()
        b.wake()
        relative = b.velocity - a.velocity
        vel_along = np.dot(relative, normal)
        if vel_along > 0:
            return
        impulse = -(1.0 + RESTITUTION) * vel_along
        impulse *= 0.5
        impulse_vec = impulse * normal
        a.velocity -= impulse_vec
        b.velocity += impulse_vec
        tangent = np.array([-normal[1], normal[0]])
        rel_tan = np.dot(relative, tangent)
        friction_impulse = rel_tan * 0.5 * FRICTION
        a.velocity += friction_impulse * tangent
        b.velocity -= friction_impulse * tangent

    def update_sleep_states(self, dt):
        for ball in self.balls:
            ball.update_sleep_state(dt)

    def spawn_ball(self):
        raw_x = self.rng.normal(0.0, SIGMA)
        clipped = np.clip(raw_x, -X_CLIP, X_CLIP)
        normalized = (clipped + X_CLIP) / (2 * X_CLIP)
        bin_index = int(np.clip(round(normalized * (N_BINS - 1)), 0, N_BINS - 1))
        center = self.bins.centers[bin_index]
        jitter = self.rng.uniform(-1.0, 1.0) * JITTER_SCALE
        x = np.clip(
            center + jitter,
            self.bins.edges[bin_index] + BALL_RADIUS * 1.05,
            self.bins.edges[bin_index + 1] - BALL_RADIUS * 1.05,
        )
        spawn_y = self.wall_top_y + BALL_RADIUS * 4.0
        circle = Circle(radius=BALL_RADIUS, stroke_width=0)
        circle.set_fill(color=BALL_COLOR, opacity=1.0)
        circle.move_to([x, spawn_y, 0])
        circle.set_z_index(5)
        ball = Ball([x, spawn_y], [0.0, 0.0], BALL_RADIUS, circle)
        self.balls.append(ball)
        self.ball_group.add(circle)
        self.total_spawned += 1


class GaussianPhysicsScene(Scene):
    def construct(self):
        floor_y = -FRAME_HEIGHT * 0.5 + 0.8
        bins = Bins(floor_y)
        ball_group = VGroup()
        ball_group.set_z_index(5)
        self.add(bins.ghost_bars)
        self.add(bins.floor)
        self.add(bins.separators)
        self.add(ball_group)
        world = PhysicsWorld(bins, ball_group)
        driver = VectorizedPoint()

        def updater(_mob, dt):
            world.step(dt)

        driver.add_updater(updater)
        self.add(driver)
        self.wait(DURATION)
        driver.clear_updaters()


if __name__ == "__main__":
    scene = GaussianPhysicsScene()
    scene.render()
