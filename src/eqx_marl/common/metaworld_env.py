"""
Continual World single-task environments on metaworld 3.x (v3 tasks), mirroring the
wrappers of continualworld/envs.py: horizon 200, 'random_init_all' goal randomization,
per-episode success flag. The benchmark's pinned mujoco-py Meta-World does not run on
Python 3.13, so the v3 tasks (39-dim obs, 4-dim action) stand in for the v1 tasks.

    env = make_env("push-v3", seed=0)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)   # info["success"] per step
    vec = VecEnv("push-v3", num_envs=8, seed=0)                    # numpy batch, auto-reset
    obs = vec.reset(); obs, rew, done, success = vec.step(actions)
"""

import numpy as np
import gymnasium as gym
import metaworld

HORIZON = 200  # META_WORLD_TIME_HORIZON in the benchmark (v3 default is 500)
CW10 = [
    "hammer-v3", "push-wall-v3", "faucet-close-v3", "push-back-v3", "stick-pull-v3",
    "handle-press-side-v3", "push-v3", "shelf-place-v3", "window-close-v3", "peg-unplug-side-v3",
]
OBS_DIM, ACT_DIM = 39, 4


class SuccessCounter(gym.Wrapper):
    """info['success'] is per step; the benchmark counts an episode as a success if any step succeeded."""

    def __init__(self, env):
        super().__init__(env)
        self.current_success = False

    def reset(self, **kwargs):
        self.current_success = False
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if info.get("success", False):
            self.current_success = True
        info["episode_success"] = self.current_success
        return obs, reward, terminated, truncated, info


def make_env(task: str, seed: int = 0, randomization: str = "random_init_all", render_mode=None, camera_name="corner2"):
    assert randomization in ("deterministic", "random_init_all"), randomization
    np.random.seed(seed)  # metaworld draws reset positions from the global numpy RNG
    mt1 = metaworld.MT1(task, seed=seed)
    env = mt1.train_classes[task](render_mode=render_mode, camera_name=camera_name)
    env.set_task(mt1.train_tasks[0])
    env._freeze_rand_vec = randomization != "random_init_all"  # as in RandomizationWrapper
    env.max_path_length = HORIZON  # metaworld truncates itself at max_path_length
    env = SuccessCounter(env)
    env.name = task
    return env


class VecEnv:
    """A list of independent copies stepped in a Python loop, with auto-reset on truncation."""

    def __init__(self, task: str, num_envs: int, seed: int = 0, randomization: str = "random_init_all"):
        self.envs = [make_env(task, seed=seed * 1000 + i, randomization=randomization) for i in range(num_envs)]
        self.num_envs = num_envs
        self.obs_dim, self.act_dim = OBS_DIM, ACT_DIM

    def reset(self):
        return np.stack([e.reset()[0] for e in self.envs]).astype(np.float32)

    def step(self, actions):
        """actions: (num_envs, 4) in [-1, 1]. Returns obs, reward, done, success (all (num_envs,))."""
        obs, rew, done, succ = [], [], [], []
        for e, a in zip(self.envs, np.asarray(actions)):
            o, r, term, trunc, info = e.step(a)
            d = term or trunc
            succ.append(info["episode_success"])
            if d:
                o, _ = e.reset()
            obs.append(o); rew.append(r); done.append(d)
        return (np.stack(obs).astype(np.float32), np.array(rew, np.float32),
                np.array(done, np.float32), np.array(succ, bool))


if __name__ == "__main__":
    import time
    env = make_env("push-v3", seed=0)
    o0, _ = env.reset(); o1, _ = env.reset()
    assert o0.shape == (OBS_DIM,)
    assert not np.allclose(o0[-3:], o1[-3:]) or not np.allclose(o0[4:7], o1[4:7]), "random_init_all should move goal/object"
    n = 0
    while True:
        o, r, term, trunc, info = env.step(env.action_space.sample()); n += 1
        if term or trunc:
            break
    assert n == HORIZON and trunc, (n, trunc)
    det = make_env("push-v3", seed=0, randomization="deterministic")
    a0, _ = det.reset(); a1, _ = det.reset()
    assert np.allclose(a0[-3:], a1[-3:]), "deterministic should keep the goal"
    vec = VecEnv("push-v3", num_envs=4, seed=0)
    obs = vec.reset(); t = time.time()
    for _ in range(HORIZON):
        obs, rew, done, succ = vec.step(np.random.uniform(-1, 1, (4, ACT_DIM)))
    assert done.all() and obs.shape == (4, OBS_DIM)
    print(f"metaworld env ok: {4 * HORIZON / (time.time() - t):.0f} steps/s over 4 envs")
