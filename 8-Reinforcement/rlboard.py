# Maze simulation environment for Reinforcement Learning tutorial
# by Dmitry Soshnikov, http://soshnikov.com
# pygame port: Board stores the state, BoardRenderer draws it and keeps none.

import os
import random

import pygame

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

BACKGROUND_COLOR = (255, 255, 255)
GRID_COLOR = (77, 77, 77)
WATER_COLOR = (0, 0, 255)
TREE_COLOR = (0, 255, 0)
ARROW_COLOR = (40, 40, 40)

__all__ = ["Board", "BoardRenderer", "open_window", "clip", "probs", "imload"]


def clip(min, max, x):
    """restric the value between [min, max]"""
    if x < min:
        return min
    if x > max:
        return max
    return x


def probs(v, eps=1e-3):
    v = v - v.min() + eps
    v = v / v.sum()
    return v


# def probs(v):
#     """Normalise values to probabilities (plain Python, no numpy needed)."""
#     v = [float(x) - min(v) for x in v]
#     total = sum(v)
#     return [x / total for x in v] if total > 0 else v


def imload(fname, size):
    """Load an image from images/ and fit it into a size x size box."""
    path = fname if os.path.isabs(fname) else os.path.join(IMAGES_DIR, fname)
    img = pygame.image.load(path)
    if img.get_bitsize() not in (24, 32):  # palette surfaces cannot be scaled
        rgba = pygame.Surface(img.get_size(), pygame.SRCALPHA)
        rgba.blit(img, (0, 0))
        img = rgba
    w, h = img.get_size()
    scale = size / max(w, h)
    return pygame.transform.smoothscale(
        img, (max(1, round(w * scale)), max(1, round(h * scale)))
    )


class Board:
    class Cell:
        empty = 0
        water = 1
        wolf = 2
        tree = 3
        apple = 4

    def __init__(self, width, height, size=50):
        self.width = width
        self.height = height
        self.size = size + 2  # pixel pitch of one cell
        self.matrix = [[Board.Cell.empty] * height for _ in range(width)]
        self.human = (0, 0)
        self.frame_no = 0

    def randomize(
        self,
        water_size=5,
        num_water=3,
        num_wolves=1,
        num_trees=5,
        num_apples=3,
        seed=None,
    ):
        if seed:
            random.seed(seed)
        for _ in range(num_water):
            x = random.randint(0, self.width - 1)
            y = random.randint(0, self.height - 1)
            for _ in range(water_size):
                self.matrix[x][y] = Board.Cell.water
                x = clip(0, self.width - 1, x + random.randint(-1, 1))
                y = clip(0, self.height - 1, y + random.randint(-1, 1))
        self._place(num_trees, Board.Cell.tree)
        self._place(num_wolves, Board.Cell.wolf)
        self._place(num_apples, Board.Cell.apple)

    def _place(self, count, value):
        for _ in range(count):
            while True:
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if self.matrix[x][y] == Board.Cell.empty:
                    self.matrix[x][y] = value
                    break

    def at(self, pos=None):
        """
        return specific pos Cell

        such as Cell.wolf, Cell.apple etc
        """
        return self.matrix[(pos or self.human)[0]][(pos or self.human)[1]]

    def is_valid(self, pos):
        return 0 <= pos[0] < self.width and 0 <= pos[1] < self.height

    def move_pos(self, pos, dpos):
        return (pos[0] + dpos[0], pos[1] + dpos[1])

    def move(self, dpos, check_correctness=True):
        new_pos = self.move_pos(self.human, dpos)
        if self.is_valid(new_pos) or not check_correctness:
            self.human = new_pos

    def random_pos(self):
        return (random.randint(0, self.width - 1), random.randint(0, self.height - 1))

    def random_start(self):
        while True:
            pos = self.random_pos()
            if self.at(pos) == Board.Cell.empty:
                self.human = pos
                break

    def image(self, Q=None):
        """Return a Surface holding the current state (no window needed)."""
        return BoardRenderer().image(self, Q)

    def plot(self, Q=None):
        """Alias for image(); the board never owns a window."""
        return self.image(Q)

    def saveimage(self, filename, Q=None):
        pygame.image.save(self.image(Q), filename)

    def walk(self, policy, save_to=None, start=None):
        """Run one episode without rendering. Returns steps, or -1 on failure."""
        n = 0
        if start:
            self.human = start
        else:
            self.random_start()

        while True:
            if save_to:
                self.saveimage(save_to.format(self.frame_no))
                self.frame_no += 1
            if self.at() == Board.Cell.apple:
                return n  # success!
            if self.at() in [Board.Cell.wolf, Board.Cell.water]:
                return -1  # eaten by wolf or drowned
            while True:
                a = policy(self)
                new_pos = self.move_pos(self.human, a)
                if self.is_valid(new_pos) and self.at(new_pos) != Board.Cell.water:
                    self.move(a)  # do the actual move
                    break
            n += 1


# sprites only depend on the cell pitch, so they are memoised;
# a BoardRenderer still holds no state of its own.
_SPRITES = {}


def _sprite(name, pitch):
    if (name, pitch) not in _SPRITES:
        _SPRITES[name, pitch] = imload(name + ".png", pitch - 6)
    return _SPRITES[name, pitch]


class BoardRenderer:
    """Draws a Board and holds no state: every call repaints from scratch."""

    # Q values are read as L, R, U, D, as in the original
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))

    @staticmethod
    def size(board):
        return (board.width * board.size + 1, board.height * board.size + 1)

    @staticmethod
    def _blit(surface, name, rect, pitch):
        pic = _sprite(name, pitch)
        surface.blit(pic, pic.get_rect(center=rect.center))

    def _arrow(self, surface, rect, values):
        """Average policy direction of a cell, from its Q values."""
        p = probs(values)
        dx = sum(p[i] * d[0] for i, d in enumerate(self.directions))
        dy = sum(p[i] * d[1] for i, d in enumerate(self.directions))
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return
        cx, cy = rect.center
        reach = rect.width / 2 - 4
        end = (cx + dx * reach, cy + dy * reach)
        pygame.draw.line(
            surface, ARROW_COLOR, (cx - dx * reach, cy - dy * reach), end, 2
        )
        pygame.draw.circle(surface, ARROW_COLOR, (round(end[0]), round(end[1])), 3)

    def draw(self, surface, board, Q=None):
        """Repaint the board into a surface whose top-left corner is (0, 0)."""
        cell = board.size
        surface.fill(BACKGROUND_COLOR)
        for x in range(board.width):
            for y in range(board.height):
                rect = pygame.Rect(x * cell + 1, y * cell + 1, cell - 1, cell - 1)
                value = board.matrix[x][y]
                if (x, y) == board.human:
                    self._blit(surface, "human", rect, cell)
                elif value == Board.Cell.water:
                    surface.fill(WATER_COLOR, rect)
                elif value == Board.Cell.tree:
                    surface.fill(TREE_COLOR, rect)
                elif value == Board.Cell.wolf:
                    self._blit(surface, "wolf", rect, cell)
                elif value == Board.Cell.apple:
                    self._blit(surface, "apple", rect, cell)
                elif Q is not None:
                    self._arrow(surface, rect, Q[x][y])

        for i in range(board.width + 1):
            pygame.draw.line(
                surface, GRID_COLOR, (i * cell, 0), (i * cell, board.height * cell)
            )
        for j in range(board.height + 1):
            pygame.draw.line(
                surface, GRID_COLOR, (0, j * cell), (board.width * cell, j * cell)
            )
        return surface

    def image(self, board, Q=None):
        """A fresh Surface holding the current board state."""
        return self.draw(pygame.Surface(self.size(board)), board, Q)

    def present(self, screen, board, Q=None):
        """Repaint into an existing window surface and flip it."""
        self.draw(screen, board, Q)
        pygame.display.flip()


def open_window(board, caption="RL maze"):
    """Create the window; SCALED keeps the surface at board size on HiDPI."""
    pygame.init()
    pygame.display.set_caption(caption)
    return pygame.display.set_mode(BoardRenderer.size(board), pygame.SCALED)


class WalkResult:
    def __init__(self, steps: int, success: bool, cause: str, path):
        """
        WalkResult store the result for a single episode
        """
        self.steps = steps
        self.success = success
        self.cause = cause
        self.path = path

    def __str__(self):
        if self.success:
            return f"success with {self.steps} steps"
        return f"failed because {self.cause} after {self.steps} steps"


class Summary:
    def __init__(self, results: list[WalkResult]):
        """
        Calculate
        success / total
        average steps
        average success steps
        failures and causes
        """
        total_n = len(results)
        total_steps = 0

        success_n = 0
        success_steps = 0

        causes = {}

        for r in results:
            total_steps += r.steps

            if r.success:
                success_n += 1
                success_steps += r.steps
            else:
                causes[r.cause] = causes.get(r.cause, 0) + 1

        self.total_n = total_n
        self.total_steps = total_steps
        self.success_n = success_n
        self.success_steps = success_steps
        self.cause = causes

    def total_avg_steps(self):
        """
        return total steps / total_n
        means the average steps each episode cost
        """
        return self.total_steps / self.total_n

    def success_rate(self):
        """
        return success_n / total_n
        means success rate for this results
        """
        return self.success_n / self.total_n

    def success_avg_steps(self):
        """
        return success_steps / success_n

        means average steps for success game
        """
        return self.success_steps / self.success_n
