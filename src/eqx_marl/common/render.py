"""
Roll out a policy in a single brax env and render it to an mp4 (MuJoCo offscreen EGL
renderer) and to brax's interactive HTML viewer.

    os.environ["MUJOCO_GL"] = "egl"   # before importing mujoco
    states, returns = rollout(env, policy, rng)          # policy: obs (obs_dim,) -> action (act_dim,)
    render_mp4(env, states, returns, "ant.mp4")
    render_html(env, states, "ant.html")
"""

import os
import numpy as np
import jax
import jax.random as jr
import imageio
from PIL import Image, ImageDraw
from brax.io import html


def rollout(env, policy, rng, num_steps: int = 1000):
    """One episode (stops at the first termination). Returns pipeline states and the running return per step."""
    obs, state = env.reset(rng)
    step = jax.jit(env.step)
    policy = jax.jit(policy)
    states, returns, ret = [], [], 0.0
    for _ in range(num_steps):
        states.append(state.env_state.pipeline_state)
        obs, state, reward, done, _ = step(rng, state, policy(obs))
        ret += float(reward)
        returns.append(ret)
        if bool(done):
            break
    return states, returns


def render_mp4(env, states, returns, path, camera="track", fps=20, width=640, height=480):
    import mujoco  # after MUJOCO_GL is set

    model = env.sys.mj_model
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height, width)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with imageio.get_writer(path, fps=fps, codec="libx264", quality=8) as writer:
        for t, (ps, ret) in enumerate(zip(states, returns)):
            data.qpos[:] = np.asarray(ps.q)
            data.qvel[:] = np.asarray(ps.qd)
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            img = Image.fromarray(renderer.render())
            ImageDraw.Draw(img).text((12, 10), f"step {t:4d}   return {ret:7.1f}", fill=(20, 20, 20))
            writer.append_data(np.asarray(img))
    renderer.close()
    return path


def render_html(env, states, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write(html.render(env.sys.tree_replace({"opt.timestep": env.dt}), states))
    return path


if __name__ == "__main__":
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax.numpy as jnp
    from eqx_marl.env_template_brax_PPO import YourEnv

    env = YourEnv(backend="mjx")
    zero_policy = lambda obs: jnp.zeros((env.action_size,))
    states, returns = rollout(env, zero_policy, jr.PRNGKey(0), num_steps=60)
    render_mp4(env, states, returns, "data/rollouts/zero_policy_test.mp4")
    render_html(env, states, "data/rollouts/zero_policy_test.html")
    print(f"rendered {len(states)} steps, final return {returns[-1]:.1f}")
