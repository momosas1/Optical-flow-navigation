import os
from time import time
from collections import deque
import random
import numpy as np
import cv2
import torch

import gibson2
import logging
import gym
import flow_vis

from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv
from gibson2.envs.parallel_env import ParallelNavEnvironment
from collections import OrderedDict

from policy.policy import Policy_flow_fusion
from RL.ppo import PPO
from util.utils import update_linear_schedule, ppo_args, batch_obs, RolloutStorage

from action_classifier.liteflownet.model import Network, estimate

train = False
with_reward = False


class RLEnv(NavigateEnv):

    #New env
    def __init__(self,
                 config_file,
                 model_id=None,
                 mode='headless',
                 action_timestep=1 / 5.0,
                 physics_timestep=1 / 240.0,
                 automatic_reset=False,
                 #random_height=False,
                 device_idx=0,
                 render_to_tensor=False,
                 past_flow_window=8):
        super(RLEnv, self).__init__(config_file,
                                    model_id=model_id,
                                    mode=mode,
                                    action_timestep=action_timestep,
                                    physics_timestep=physics_timestep,
                                    automatic_reset=automatic_reset,
                                    #random_height=False,
                                    device_idx=device_idx,
                                    render_to_tensor=render_to_tensor)
        self.flow_stack = []
        self.window_size= past_flow_window
        self.energy_reward = 0
        self.time_start = time()
        self.action_num = 0
        self.energy_loss = 0
        self.turn_num = 0
        self.robot_pre_states = {'action': 0, 'velocity': 0, 'rotational_velocity': 0}
        self.pos = self.robots[0].get_position()

    def load_task_setup(self):
        super(RLEnv, self).load_task_setup()

        # Rs 1
        self.initial_pos = np.array([-0.7, 2.5, 0])
        self.target_pos = np.array([0.5, -3, 0])

        # Rs 2
        self.initial_pos = np.array([0.5, -2.5, 0.1])
        self.target_pos = np.array([-2.5, 3, 0])

        # Rs 3
        self.initial_pos = np.array([-2.7, 2.7, 0])
        self.target_pos = np.array([0.4, -2.5, 0])

        # Rs 4
        self.initial_pos = np.array([-0.7, -3.2, 0])
        self.target_pos = np.array([1, -1, 0])

        # Rs 5
        self.initial_pos = np.array([1, -1, 0])
        self.target_pos = np.array([-1, -3.5, 0])

        # Sawpit 1
        #self.initial_pos = np.array([5.5,0.5,0])
        #self.target_pos = np.array([0,0,0])
        #'''
        # Sawpit 2
        self.initial_pos = np.array([-5.5, -2.2, 0])
        self.target_pos = np.array([0, -3, 0])

        # Sawpit 3

        self.initial_pos = np.array([-2, 0 , 0])
        self.target_pos = np.array([-3.5, -3, 0])

        # Sawpit 4
        self.initial_pos = np.array([5, -4, 0])
        self.target_pos = np.array([0,0,0])

        # Sawpit 5
        self.initial_pos = np.array([2.2, -2.7, 0])
        self.target_pos = np.array([-2, -1, 0])

        # Seward 1
        self.initial_pos = np.array([0, 0, 0])
        self.target_pos = np.array([1, 2.8, 0])

        # Seward 2
        self.initial_pos = np.array([-3, 2, 0])
        self.target_pos = np.array([0, 1, 0])

        # Seward 3
        self.initial_pos = np.array([-2, 2.5, 0])
        self.target_pos = np.array([1.2, 4, 0])

        # Seward 4
        self.initial_pos = np.array([-6,-0.5,0])
        self.target_pos = np.array([-3,1,0])

        # Seward 5
        self.initial_pos = np.array([1.2, 3.5, 0])
        self.target_pos = np.array([-0.5, 1, 0])

        # Ribera
        self.initial_pos = np.array([0, -6, 0])
        self.target_pos = np.array([-2, -3, 0])

        # Ribera 2
        #self.initial_pos = np.array([-1.2, -4.7, 0])
        #self.target_pos = np.array([-4, 0, 0])

        # RIbera 3
        #self.initial_pos = np.array([-1.2, -4.7, 0])
        #self.target_pos = np.array([0, -2, 0])

        # Ribera 4
        #self.initial_pos = np.array([-3.5, -1, 0])
        #self.target_pos = np.array([-1.5, -4.7, 0])

        # Ribera 5
        #self.initial_pos = np.array([-2.5, -5.5, 0])
        #self.target_pos = np.array([-1, -2, 0])
        #'''

    def reset(self):

        state = super(RLEnv, self).reset()
        self.flow_stack.clear()
        self.time_start = time()
        self.action_num = 0
        self.energy_loss = 0
        self.turn_num = 0
        self.robot_pre_states = {'action': 0, 'velocity': 0, 'rotational_velocity': 0, 'collision': 0}
        self.pos = self.robots[0].get_position()
        return state

    def step(self, action):
        state, reward, done, info = super(RLEnv, self).step(action)

        self.action_num += 1
        self.flow_stack.append(action)

        if len(self.flow_stack) == self.window_size+1:
            self.flow_stack.pop(0)
        self.energy_loss += self.get_energy(action, self.robot_pre_states)

        if action == 2 or action == 3:
            self.turn_num+=1
        self.pos = self.robots[0].get_position()
        return state, reward, done, info

    def get_energy(self, action, pre_state):

        velocity = self.robots[0].get_linear_velocity()
        rotational_velocity = self.robots[0].get_angular_velocity()
        velocity = (velocity[0] ** 2 + velocity[1] ** 2) ** 0.5
        rotational_velocity = rotational_velocity[2]

        action_time = self.action_timestep
        friction_energy = 1.0 * 2.7 * 9.8 * (max(abs(velocity), abs(0.16 * rotational_velocity))) * action_time
        sum = 0

        if action == pre_state['action']:

            kinetic_energy = (2.7 * velocity * max((velocity - pre_state['velocity']), 0) +
                              0.002 * rotational_velocity *
                              max((rotational_velocity - pre_state['rotational_velocity']), 0)
                              ) / 2
            control_energy = 1.0 * action_time + (action_time ** 3) / 3.0
            sum = control_energy + friction_energy + kinetic_energy
        else:
            kinetic_energy = (2.7 * velocity ** 2 + 0.002 * rotational_velocity ** 2) / 2
            kinetic_energy_pre = (2.7 * pre_state['velocity'] ** 2 + 0.002 * pre_state['rotational_velocity']) / 2
            control_energy = (1.0 + 0.7 * (velocity + 0.16 * rotational_velocity)) * action_time / 2.0 + (
                        action_time ** 3) / 3.0
            control_energy_pre = (1.0 + 0.7 * (
                        pre_state['velocity'] + 0.16 * pre_state['rotational_velocity'])) * action_time / 2.0 + (
                                             action_time ** 3) / 3.0
            sum = control_energy + kinetic_energy_pre + kinetic_energy + friction_energy + control_energy_pre

        self.robot_pre_states['action'] = action
        self.robot_pre_states['velocity'] = velocity
        self.robot_pre_states['rotational_velocity'] = rotational_velocity
        return sum

def construct_envs():
    #build envs  n
    
    
    scenes = os.listdir("/home/gmh/iGibson/gibson2/dataset")
    
    #change yaml to change building
    
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    random.seed()

    def load_env():
        env=RLEnv(config_file=config_filename, mode='headless')
        #env.config['model_id'] = scenes[random.randint(0, len(config_filename) - 1)]
        return env

    #control number of envs
    envs= ParallelNavEnvironment([load_env] * 1, blocking=False)

    return envs

def get_energy_reward(flow_stack):

        # check the jitter movement
        # if have one jitter movement get -0.1 reward
        count_left = 0
        count_right = 0

        for i in range(0, len(flow_stack)):
            flow_action = flow_stack[i]
            if flow_action == 2:
                count_left += 1
            elif flow_action == 3:
                count_right += 1

        jitter_count = min(count_left, count_right)

        if flow_stack[len(flow_stack) - 1] == 2 or flow_stack[len(flow_stack) - 1] == 3:
            reward = -0.05 * jitter_count
        else:
            reward = 0

        return reward

def convert_action(action):
    if action == 2:
        flow_type = -1
    elif action == 3:
        flow_type = 1
    elif action == 1:
        flow_type = 0
    else:
        flow_type = 2
    return flow_type

def main():
    parser = ppo_args()
    args = parser.parse_args()
    random.seed(args.seed)


    logger = logging.getLogger("AppName")
    file_handler = logging.FileHandler(args.log_file)
    logger.addHandler(file_handler)

    device = torch.device("cuda:{}".format(args.pth_gpu_id))


    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))


    envs = construct_envs()


    actor_critic = Policy(
        observation_space=envs.observation_space,
        action_space=envs.action_space
    )
    print(actor_critic)
    agent = PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,

    )

    #load checkpoint 
    #ckpt = torch.load("ck/ckpt_with_reward_new.pth")
    ckpt = torch.load("ck/ckpt_without_reward_new.pth")
    #ckpt = torch.load("baselines/model/rgbd.pth", torch.device("cpu"))
    trained_weight = list(ckpt["state_dict"].keys())
    new_weight = list(agent.state_dict().keys())
    dict_new=agent.state_dict().copy()

    for i in range(16):
        dict_new[new_weight[i]]=ckpt["state_dict"][trained_weight[i]]
    agent.load_state_dict(dict_new)

    for param in agent.actor_critic.net.cnn.parameters():
        param.requires_grad = False
    #agent.load_state_dict(ckpt["state_dict"])
    actor_critic=agent.actor_critic
    actor_critic.to(device)

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in agent.parameters())
        )
    )


    res = envs.reset()


    #preprocess data     same as habitat
    def build_observations(res):
        observations = list()

        for i in range(envs._num_envs):
            rgb = res[i]['rgb']
            depth = res[i]['depth']
            pointgoal = res[i]['sensor'][:2]

            observations.append(dict(zip(['rgb', 'depth', 'sensor'], [rgb, depth, pointgoal])))
        return observations


    observations = build_observations(res)
    batch = batch_obs(observations)


    #for sensor in batch:
     #   batch[sensor] = batch[sensor].to(device)


    rollouts = RolloutStorage(
        args.num_steps,
        envs._num_envs,
        envs.observation_space,
        envs.action_space,
        args.hidden_size,
    )

    for sensor in rollouts.observations:
        rollouts.observations[sensor][0].copy_(batch[sensor])

    rollouts.to(device)

    episode_rewards = torch.zeros(envs._num_envs, 1)
    episode_counts = torch.zeros(envs._num_envs, 1)
    current_episode_reward = torch.zeros(envs._num_envs, 1)
    window_episode_reward = deque()
    window_episode_counts = deque()

    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = 0
    count_checkpoints = 0
    count_finish  = 0
    spl=[]
    energy = []
    action_num = []
    path_time = []
    turn_num = []
    energy_loss = []
    path_position = []
    for update in range(50):

        if args.use_linear_lr_decay:
            update_linear_schedule(
                agent.optimizer, update, args.num_updates, args.lr
            )

        agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        #collect rollout
        for step in range(args.num_steps):
            t_sample_action = time()


            # sample actions
            with torch.no_grad():
                step_observation = {
                    k: v[step] for k, v in rollouts.observations.items()
                }
                (
                    values,
                    actions,
                    actions_log_probs,
                    recurrent_hidden_states,
                ) = actor_critic.act(
                    step_observation,
                    rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step],
                )
            pth_time += time() - t_sample_action

            t_step_env = time()

            #step 
            outputs = envs.step([a[0].item() for a in actions])
            path_position.append(envs._envs[0].pos)
            states, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            #Add
            for i in range(envs._num_envs):

                if dones[i]:
                    print(
                        "env{} spl:{:.3f}\taction_num:{}\tturn_percentage:{:.3f}\ttime: {:.3f}\tenergy_loss:{:.3f}\tenergy_reward:{:.3f}".format(
                            i, infos[i]['spl'], envs._envs[i].action_num,
                            envs._envs[i].turn_num / envs._envs[i].action_num, time() - envs._envs[i].time_start,
                            envs._envs[i].energy_loss, envs._envs[i].energy_reward
                        ))
                    path_position.clear()


                    if not train:
                        if infos[i]['success']:
                            if infos[i]['spl'] > 0.9:
                                energy.append(envs._envs[0].energy_loss)
                                action_num.append(envs._envs[0].action_num)
                                turn_num.append(envs._envs[0].turn_num / envs._envs[i].action_num)
                                path_time.append(time() - envs._envs[i].time_start)
                                energy_loss.append(envs._envs[0].energy_reward)


                        if len(energy) == 10:
                            print(
                                "average energy:{}    action_num:{:.1f}   turn_num:{:.3f} path_time:{:.3f}    energy_loss:{:.3f}".format(
                                    sum(energy) / len(energy), sum(action_num) / 10, sum(turn_num) / 10,
                                    sum(path_time) / 10, sum(energy_loss) / 10))

                            energy.clear()
                            action_num.clear()
                            turn_num.clear()
                            path_time.clear()
                            energy_loss.clear()

                    path_position.clear()
                    states[i]=envs._envs[i].reset()
                    envs._envs[i].energy_reward = 0
                    count_finish += 1
                    continue

                energy_reward = get_energy_reward(envs._envs[i].flow_stack)
                if with_reward:
                    rewards[i] += energy_reward
                envs._envs[i].energy_reward += energy_reward

            env_time += time() - t_step_env

            t_update_stats = time()

            observations = build_observations(states)
            batch = batch_obs(observations)



            rewards = torch.tensor(rewards, dtype=torch.float)
            rewards = rewards.unsqueeze(1)

            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones], dtype=torch.float
            )

            current_episode_reward += rewards
            episode_rewards += (1 - masks) * current_episode_reward
            episode_counts += 1 - masks
            current_episode_reward *= masks

            rollouts.insert(
                batch,
                recurrent_hidden_states,
                actions,
                actions_log_probs,
                values,
                rewards,
                masks,
            )

            count_steps += envs._num_envs
            pth_time += time() - t_update_stats



        if len(window_episode_reward) == args.reward_window_size:
            window_episode_reward.popleft()
            window_episode_counts.popleft()
        window_episode_reward.append(episode_rewards.clone())
        window_episode_counts.append(episode_counts.clone())


        #update policy
        t_update_model = time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }
            next_value = actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1],
            ).detach()

        rollouts.compute_returns(
            next_value, args.use_gae, args.gamma, args.tau
        )
        if train:
            agent.update(rollouts)

        rollouts.after_update()
        pth_time += time() - t_update_model

        if train:
            # log stats
            if update > 0 and update % args.log_interval == 0:
                logger.info(
                    "update: {}\tfps: {:.3f}\t".format(
                        update, count_steps / (time() - t_start)
                    )
                )

                logger.info(
                    "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                    "frames: {}".format(update, env_time, pth_time, count_steps)
                )

                window_rewards = (
                    window_episode_reward[-1] - window_episode_reward[0]
                ).sum()
                window_counts = (
                    window_episode_counts[-1] - window_episode_counts[0]
                ).sum()

                if window_counts > 0:
                    logger.info(
                        "Average window size {} reward: {:3f}".format(
                            len(window_episode_reward),
                            (window_rewards / window_counts).item(),
                        )
                    )
                else:
                    logger.info("No episodes finish in current window")

        # checkpoint model
        if train:
            if update % 5 == 4:
                checkpoint = {"state_dict": agent.state_dict()}
                torch.save(
                    checkpoint,
                    os.path.join(
                        args.checkpoint_folder,
                        "ckpt_without_reward_new.pth".format(count_checkpoints),
                    ),
                )
                count_checkpoints += 1




if __name__ == "__main__":
    main()
