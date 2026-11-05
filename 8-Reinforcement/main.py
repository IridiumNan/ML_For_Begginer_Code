import random
import numpy as np
import time
import pygame
from rlboard import *
from rlboard import WalkResult

width, height = 8, 8

m = Board(width, height)
m.randomize(seed=13)

# create actions map
actions = {"U": (0, -1), "D": (0, 1), "L": (-1, 0), "R": (1, 0)}
action_idx = {a: i for i, a in enumerate(actions.keys())}

repeat = 3  # how many episodes to play
interval = 0.1  # seconds between two moves
max_steps = 500  # hard cap on one episode: a walk must always terminate


def play_episode(
    m: Board, policy, renderer: BoardRenderer | None, screen: pygame.Surface | None
) -> WalkResult:
    """Walk until apple / water / wolf, drawing after every move.

    To create a screen, use open_window function

    if renderer and screen is None, it will not draw a picture
    just process game without sleep

    Returns the details of the episode."""
    m.random_start()
    path = [m.human]

    while True:
        # if any(event.type == pygame.QUIT for event in pygame.event.get()):
        #     return None
        if renderer and screen:
            renderer.present(screen, m)
            time.sleep(interval)

        if m.at() == Board.Cell.apple:
            return WalkResult(len(path) - 1, True, "apple", path)
        if m.at() in (Board.Cell.wolf, Board.Cell.water):
            cause = "wolf" if m.at() == Board.Cell.wolf else "water"
            return WalkResult(len(path) - 1, False, cause, path)

        # the policy may be deterministic (qpolicy_strict): retrying the same
        # blocked action forever would hang, so fall back to a legal move
        dpos = actions[policy(m)]
        new_pos = m.move_pos(m.human, dpos)
        if not (m.is_valid(new_pos) and m.at(new_pos) != Board.Cell.water):
            choices = [
                d
                for d in actions.values()
                if m.is_valid(m.move_pos(m.human, d))
                and m.at(m.move_pos(m.human, d)) != Board.Cell.water
            ]
            if not choices:  # boxed in: nowhere legal to go
                return WalkResult(len(path) - 1, False, "stuck", path)
            dpos = random.choice(choices)
        m.move(dpos)
        path.append(m.human)
        if len(path) - 1 >= max_steps:  # safety net, a walk can never loop forever
            return WalkResult(len(path) - 1, False, "timeout", path)


def random_policy(m: Board):
    return random.choice(list(actions))


def print_statics(policy=random_policy, loop=100, header=None):
    """
    This function play episode with loop counts and policy
    """

    # TODO: Rewrite this function

    # s success steps (cumulate)
    # n success game counts
    # total total steps (cumulate)
    s, n, total = 0, 0, 0
    causes = {}
    for _ in range(loop):
        r = play_episode(m, policy, None, None)
        total += r.steps
        if r.success:
            s += r.steps
            n += 1
        else:
            causes[r.cause] = causes.get(r.cause, 0) + 1
    avg = s / n if n else float("inf")

    print(f"""
        {header}
        success {n} / {loop}
        avg path = {avg:.2f}
        total steps = {total} 
        failures = {causes or 0}\n
        """)


def play_with_render():
    """
    It just a demo function for drawing walk animation by pygame
    """
    renderer = BoardRenderer()
    screen = open_window(m)

    results = []
    for _ in range(repeat):
        r = play_episode(m, random_policy, renderer, screen)

        results.append(r)
        print(f"episode {len(results) + 1}: {r.cause} in {r.steps} steps")

    pygame.quit()
    return results


# constants for reward
move_reward = -0.1
goal_reward = 10
end_reward = -10


# reward function
def reward(m: Board, pos=None):
    """
    Get the reward based on current pos

    invalid | wolf | water pos  -> end_reward
    apple                       -> goal_reward
    other                       -> move_reward
    """
    pos = pos or m.human
    if not m.is_valid(pos):
        return end_reward
    match m.at(pos):
        case Board.Cell.wolf | Board.Cell.water:
            return end_reward
        case Board.Cell.apple:
            return goal_reward
    return move_reward


default_init_Q = (
    np.ones((width, height, len(actions)), dtype=np.float32) * 1.0 / len(actions)
)


def wrap_Q(Q):
    def inner_Q():
        return Q

    return inner_Q


def train_Q_table(init_Q=default_init_Q, epochs=5000, verbose=False):
    """
    init_Q: the initialize Q (a tensor)

    epochs: the training epochs

    verbose: control inner print function
    if True, print more details when training

    """
    Q = init_Q.copy()

    print_interval = 1000
    cut_offs = 0
    window_steps = 0  # steps in the current reporting window

    terminal_conditions = [end_reward, goal_reward]
    for epoch in range(epochs):
        m.random_start()

        n = 0
        cum_reward = 0

        lpath = []

        while True:
            x, y = m.human
            v = probs(Q[x, y])
            a = random.choices(list(actions), weights=v)[0]
            dpos = actions[a]
            m.move(dpos, check_correctness=False)
            r = reward(m)
            cum_reward += r

            alpha = np.exp(-n / 1e5)
            gamma = 0.5
            ai = action_idx[a]
            # apple / wolf / water / off the board all end the episode, so
            # there is no future value to bootstrap from those transitions
            terminal = r in terminal_conditions
            target = r if terminal else r + gamma * Q[x + dpos[0], y + dpos[1]].max()
            Q[x, y, ai] = (1 - alpha) * Q[x, y, ai] + alpha * target

            if terminal or cum_reward < -1000:
                window_steps += n
                lpath.append(n)
                break

            n += 1
            if n >= max_steps:  # safety net, an episode can never run forever
                window_steps += n
                lpath.append(n)
                cut_offs += 1
                break

        if verbose and (epoch + 1) % print_interval == 0:
            # avg episode length is the health signal: it must stay bounded
            print(
                f"epoch: {epoch + 1}  avg steps = "
                f"{window_steps / print_interval:.1f}  capped = {cut_offs}"
            )
            window_steps = 0

    if cut_offs:
        print(f"{cut_offs} episodes hit the {max_steps} step cap")
    return Q


def qpolicy_strict(Q):
    """
    Create a new policy function based the Q-Table
    """

    def qpolicy_inner(m):
        x, y = m.human
        v = probs(Q[x, y])
        # if all the same, select randomly
        best = np.flatnonzero(v == v.max())
        a = list(actions)[random.choice(list(best))]
        return a

    return qpolicy_inner


def evaluate_Q():
    pass
    # TODO:


def train_evaluate():
    total_loop = 1000
    batch_loop = 100
    curr_loop = 0

    curr_Q = default_init_Q
    while curr_loop < total_loop:
        curr_Q = train_Q_table(curr_Q, batch_loop)


def main():
    # random policy
    print_statics(header="Random policy")

    Q = np.ones((width, height, len(actions)), dtype=np.float32) * 1.0 / len(actions)
    print("before training")
    policy = qpolicy_strict(Q)
    print_statics(policy, 100, header="Q before traning")

    Q = train_Q_table(epochs=1000)
    print("after training")

    policy = qpolicy_strict(Q)
    print_statics(policy, 100, header="Q after training")


if __name__ == "__main__":
    main()
