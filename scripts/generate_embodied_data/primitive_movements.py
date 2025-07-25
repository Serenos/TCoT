import numpy as np


def describe_move(move_vec):
    names = [
        {-1: "backward", 0: None, 1: "forward"},
        {-1: "right", 0: None, 1: "left"},
        {-1: "down", 0: None, 1: "up"},
        {-1: "tilt down", 0: None, 1: "tilt up"},
        {},
        {-1: "rotate clockwise", 0: None, 1: "rotate counterclockwise"},
        {-1: "close gripper", 0: None, 1: "open gripper"},
    ]

    xyz_move = [names[i][move_vec[i]] for i in range(0, 3)]
    xyz_move = [m for m in xyz_move if m is not None]

    if len(xyz_move) != 0:
        description = "move " + " ".join(xyz_move)
    else:
        description = ""

    if move_vec[3] == 0:
        move_vec[3] = move_vec[4]  # identify rolling and pitching

    if move_vec[3] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[3][move_vec[3]]

    if move_vec[5] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[5][move_vec[5]]

    if move_vec[6] != 0:
        if len(description) > 0:
            description = description + ", "

        description = description + names[6][move_vec[6]]

    if len(description) == 0:
        description = "stop"

    return description


def classify_movement(move, threshold=0.03):
    diff = move[-1] - move[0]

    if np.sum(np.abs(diff[:3])) > 3 * threshold:
        diff[:3] *= 3 * threshold / np.sum(np.abs(diff[:3]))

    diff[3:6] /= 10

    move_vec = 1 * (diff > threshold) - 1 * (diff < -threshold)

    return describe_move(move_vec), move_vec


move_actions = dict()


def get_move_primitives_episode(episode):
    steps = list(episode["steps"])

    states = np.array([step["observation"]["state"] for step in steps])
    actions = [step["action"][:3].numpy() for step in steps]

    move_trajs = [states[i : i + 4] for i in range(len(states) - 1)]
    primitives = [classify_movement(move) for move in move_trajs]
    primitives.append(primitives[-1])

    for (move, _), action in zip(primitives, actions):
        if move in move_actions.keys():
            move_actions[move].append(action)
        else:
            move_actions[move] = [action]

    return primitives


def get_move_primitives(episode_id, builder):
    ds = builder.as_dataset(split=f"train[{episode_id}:{episode_id + 1}]")
    episode = next(iter(ds))

    return get_move_primitives_episode(episode)
