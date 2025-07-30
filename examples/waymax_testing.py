from waymax import config, dataloader, datatypes, visualization
import matplotlib.pyplot as plt
# --- Monkey-patch for jax.util.unzip2 ---
import jax.util

# Recreate the removed unzip2 function
def _unzip2(xs):
  if not xs:
    return (), ()
  return tuple(zip(*xs))

# Add the function back to the jax.util module
jax.util.unzip2 = _unzip2
# --- End of patch ---

# Scenario Loading Config
dataset_config = config.DatasetConfig(
    path="gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000",
    max_num_objects=32
)

# Create Data Iteration Object and sample a scenario
data_iter = dataloader.simulator_state_generator(config=dataset_config)
scenario = next(data_iter)

# Visualize using logged trajectory
img = visualization.plot_simulator_state(scenario, use_log_traj=True)
plt.imshow(img)
plt.savefig('examples/output_image.png', bbox_inches='tight', pad_inches=0)

# setting up a multi-agent training env
from waymax import env, config, dynamics, datatypes, dataloader, agents, visualization

import jax, dataclasses, mediapy
import jax.numpy as jnp

# Initialization
maximum_number_of_objects=12
custom_config = config.DatasetConfig(
    path="gs://waymo_open_dataset_motion_v_1_3_0/uncompressed/tf_example/training/training_tfexample.tfrecord-00000-of-01000",
    max_num_objects=maximum_number_of_objects,
)
dynamics_model = dynamics.InvertibleBicycleModel()
# env_config = config.EnvironmentConfig()
scenarios = dataloader.simulator_state_generator(config=custom_config)
# waymax_env = env.MultiAgentEnvironment(dynamics_model, env_config)

dynamics_model = dynamics.StateDynamics()

# Expect users to control all valid object in the scene.
waymax_env = env.MultiAgentEnvironment(
    dynamics_model=dynamics_model,
    config=dataclasses.replace(
        config.EnvironmentConfig(),
        max_num_objects=maximum_number_of_objects,
        controlled_object=config.ObjectType.VALID,
    ),
)


# An actor that doesn't move, controlling all objects with index > 4
obj_idx = jnp.arange(maximum_number_of_objects)
static_actor = agents.create_constant_speed_actor(
    speed=0.0,
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: obj_idx > 4,
)

# IDM actor/policy controlling both object 0 and 1.
# Note IDM policy is an actor hard-coded to use dynamics.StateDynamics().
actor_0 = agents.IDMRoutePolicy(
    is_controlled_func=lambda state: (obj_idx == 0) | (obj_idx == 1) | (obj_idx == 2),
)

# Constant speed actor with predefined fixed speed controlling object 2.
# actor_1 = agents.create_constant_speed_actor(
#     speed=5.0,
#     dynamics_model=dynamics_model,
#     is_controlled_func=lambda state: obj_idx == 2,
# )

# Expert/log actor controlling objects 3 and 4.
actor_2 = agents.create_expert_actor(
    dynamics_model=dynamics_model,
    is_controlled_func=lambda state: (obj_idx == 3) | (obj_idx == 4),
)

actors = [static_actor, actor_0, actor_2]

jit_step = jax.jit(waymax_env.step)
jit_select_action_list = [jax.jit(actor.select_action) for actor in actors]

# Rollout
states = [waymax_env.reset(next(scenarios))]
states = [waymax_env.reset(next(scenarios))]

actions = []
rewards = []
for _ in range(states[0].remaining_timesteps):
    current_state = states[-1]

    outputs = [
        jit_select_action({}, current_state, None, None)
        for jit_select_action in jit_select_action_list
    ]
    action = agents.merge_actions(outputs)
    reward = waymax_env.reward(current_state, action)
    next_state = jit_step(current_state, action)

    actions.append(action)
    rewards.append(reward)
    states.append(next_state)

# Rollout Visualization
imgs = []
for state in states:
  imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
mediapy.write_video("examples/video.mp4", imgs, fps=10)

import pprint as pp

states = [waymax_env.reset(next(scenarios))]

print("CURRENT STATE:")
current_state = states[-1]
img = visualization.plot_simulator_state(current_state, use_log_traj=True)
mediapy.write_image("examples/test1.png", img)

outputs = [
    jit_select_action({}, current_state, None, None)
    for jit_select_action in jit_select_action_list
]
action = agents.merge_actions(outputs)
print("ACTIONS:")
pp.pprint(action)
reward = waymax_env.reward(current_state, action)
print("REWARD:")
pp.pprint(reward)

print("NEXT STATE:")
next_state = jit_step(current_state, action)
img = visualization.plot_simulator_state(next_state, use_log_traj=True)
mediapy.write_image("examples/test2.png", img)

## Ok can I now get the observations to match gpudrive?



pass
