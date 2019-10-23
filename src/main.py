import argparse
import time
import gym
from gym import wrappers, logger
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from oc_agent import OptionCriticAgent
import gym_pinball
from tqdm import tqdm, trange
from visualizer import Visualizer


def export_csv(file_path, file_name, array):
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    array = pd.DataFrame(array)
    saved_path = os.path.join(file_path, file_name)
    array.to_csv(saved_path)

def moved_average(data, window_size):
    b=np.ones(window_size)/window_size
    return np.convolve(data, b, mode='same')

def learning_loop(env_id, episode_count, model, visual, n_options):
    env = gym.make(env_id)
    outdir = '/tmp/random-agent-results'
    env = wrappers.Monitor(env, directory=outdir, force=True)
    env.seed(0)
    agent = OptionCriticAgent(env.action_space, env.observation_space, n_options)
    option_label = ["opt {}".format(str(i+1)) for i in range(n_options)]
    vis = Visualizer(["ACC_X", "ACC_Y", "DEC_X", "DEC_Y", "NONE"], option_label)
    if model:
        agent.load_model(model)
    reward = 0
    done = False
    total_reward_list = []
    steps_list = []
    max_q_list = []
    max_q_episode_list = []
    max_q = 0.0

    for i in trange(episode_count):
        total_reward = 0
        n_steps = 0
        ob = env.reset()
        option = agent.get_option(ob)
        pre_option = option
        action = agent.act(ob, option)
        pre_action = action
        agent.set_last_q_omega(option, ob)
        is_render = False
        while True:
            if (i+1) % 20 == 0 and visual:
                env.render()
                is_render = True 
            pre_obs = ob
            ob, reward, done, _ = env.step(action)
            n_steps += 1   
            rand_basis = np.random.uniform()
            if agent.get_terminate(ob, option) > rand_basis:
                pre_option = option
                option = agent.get_option(ob)
            pre_action = action
            action = agent.act(ob, option)
            agent.update(pre_option, pre_obs, pre_action, option, ob, action, reward, done)
            total_reward += reward
            tmp_max_q = agent.get_max_q_u(ob, option)
            max_q_list.append(tmp_max_q)
            max_q = tmp_max_q if tmp_max_q > max_q else max_q
            if done:
                print("episode: {}, steps: {}, total_reward: {}, max_q_u: {}".format(i, n_steps, total_reward, max_q_list[-1]))
                total_reward_list.append(total_reward)
                steps_list.append(n_steps)
                break
            
            if is_render:
                vis.set_action_dist(agent.vis_action_dist, action)
                vis.set_option_q(agent.vis_option_q, option)
                vis.pause(.0001)
        
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
        max_q_episode_list.append(max_q)

    date = datetime.now().strftime("%Y%m%d")
    time = datetime.now().strftime("%H%M")
    saved_dir = os.path.join("data", date, time)
    # export process
    saved_res_dir = os.path.join(saved_dir, 'res')
    export_csv(saved_res_dir, "total_reward.csv", total_reward_list)
    td_error_list = agent.td_error_list
    td_error_list_meta = agent.td_error_list_meta
    export_csv(saved_res_dir, "td_error.csv", td_error_list)
    total_reward_list = np.array(total_reward_list)
    steps_list = np.array(steps_list)
    max_q_list = np.array(max_q_list)
    print("Average return: {}".format(np.average(total_reward_list)))

    # save model
    saved_model_dir = os.path.join(saved_dir, 'model')
    agent.save_model(saved_model_dir)

    # output graph
    x = list(range(len(total_reward_list)))
    plt.subplot(4,2,1)
    y = moved_average(total_reward_list, 10)
    plt.plot(x, total_reward_list)
    plt.plot(x, y, 'r--')
    plt.title("total_reward")
    plt.subplot(4,2,2)
    y = moved_average(steps_list, 10)
    plt.plot(x, steps_list)
    plt.plot(x, y, 'r--')
    plt.title("the number of steps until goal")
    plt.subplot(4,1,2)
    y = moved_average(td_error_list, 1000)
    x = list(range(len(td_error_list)))
    plt.plot(x, td_error_list, 'k-')
    plt.plot(x, y, 'r--', label='average')
    plt.title("intra-option Q critic td error")
    plt.legend()
    plt.subplot(4,1,3)
    y = moved_average(td_error_list_meta, 1000)
    x = list(range(len(td_error_list_meta)))
    plt.plot(x, td_error_list_meta, 'k-')
    plt.plot(x, y, 'r--', label='average')
    plt.title("intra-option Q learning td error")
    plt.legend()
    plt.subplot(4,1,4)
    y = moved_average(max_q_episode_list, 1000)
    x = list(range(len(max_q_episode_list)))
    plt.plot(x, max_q_episode_list, 'k-')
    # plt.plot(x, y, 'r--', label='average')
    plt.title("max q_u value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(saved_res_dir, "plot.png"))
    plt.show()
    plt.savefig(os.path.join(saved_res_dir, "res_plot.png"))
    env.close()



def main():
    parser = argparse.ArgumentParser(description='Option-Critic Learning.')
    parser.add_argument('env_id', nargs='?', default='PinBall-v0', help='Select the environment to run.')
    parser.add_argument('--vis', action='store_true', help='Attach when you want to look visual results.')
    parser.add_argument('--model', help='Input model dir path')
    parser.add_argument('--nepisodes', default=250, type=int)
    parser.add_argument('--noptions', default=4, type=int)
    args = parser.parse_args()
    learning_time = time.time()
    learning_loop(args.env_id, args.nepisodes, args.model, args.vis, args.noptions)
        # Close the env and write monitor result info to disk
    duration = time.time() - learning_time
    print("Learning time: {}m {}s".format(int(duration//60), int(duration%60)))
    


if __name__ == '__main__':
    main()