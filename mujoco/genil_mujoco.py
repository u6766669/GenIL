import sys
import os
from pathlib import Path
import argparse
import pickle
from functools import partial
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib
import matplotlib.pylab
from matplotlib import pyplot as plt
from imgcat import imgcat
from model import MyModel
from dataset import gen_traj, TrajDataset, batch_dataset
from random import sample

import os, sys

NP_CONCATENATE = '''np.concatenate'''
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/learner/baselines/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Only show ERROR log
from tf_commons.ops import *

import gym

matplotlib.use('agg')


class PPO2Agent(object):
    def __init__(self, env, env_type, path, stochastic=False, gpu=True):
        from baselines.common.policies import build_policy
        from baselines.ppo2.model import Model

        self.graph = tf.Graph()

        if gpu:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        else:
            config = tf.ConfigProto(device_count={'GPU': 0})

        self.sess = tf.Session(graph=self.graph, config=config)

        with self.graph.as_default():
            with self.sess.as_default():
                ob_space = env.observation_space
                ac_space = env.action_space

                if env_type == 'atari':
                    policy = build_policy(env, 'cnn')
                elif env_type == 'mujoco':
                    policy = build_policy(env, 'mlp')
                else:
                    assert False, ' not supported env_type'

                make_model = lambda: Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=1,
                                           nbatch_train=1,
                                           nsteps=1, ent_coef=0., vf_coef=0.,
                                           max_grad_norm=0.)
                self.model = make_model()

                self.model_path = str(path)
                self.chkpt_num = int(path.name)
                self.model.load(str(path))

        if env_type == 'mujoco':
            with open(str(path) + '.env_stat.pkl', 'rb') as f:
                s = pickle.load(f)
            self.ob_rms = s['ob_rms']
            self.ret_rms = s['ret_rms']
            self.clipob = 10.
            self.epsilon = 1e-8
        else:
            self.ob_rms = None

        self.stochastic = stochastic

    def act(self, obs, reward, done):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)

        with self.graph.as_default():
            with self.sess.as_default():
                if self.stochastic:
                    a, v, state, neglogp = self.model.step(obs)
                else:
                    a = self.model.act_model.act(obs)
        return a


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space
        self.model_path = 'random_agent'

    def act(self, observation, reward, done):
        return self.action_space.sample()[None]


def crossover(args, traj_1_ob, traj_1_ac, step_re_1, traj_2_ob, traj_2_ac, step_re_2):
    # only generate 1 offspring
    exchange_w = np.random.rand()
    interval = np.random.randint(args.crossover_low,args.crossover_high)

    if np.sum(step_re_2) > np.sum(step_re_1):
        off = crossover(args, traj_2_ob, traj_2_ac, step_re_2, traj_1_ob, traj_1_ac, step_re_1)
        return off


    if np.random.random() < args.crossover_rate:
        ob_1, act_1 = traj_1_ob, traj_1_ac
        ob_2, act_2 = traj_2_ob, traj_2_ac

        if np.random.random() < exchange_w:
            off_ob = ob_2[0:interval]
            off_act = act_2[0:interval]
            off_step_r = step_re_2[0:interval]
        else:
            off_ob = ob_1[0:interval]
            off_act = act_1[0:interval]
            off_step_r = step_re_1[0:interval]
        for i in range(interval, min(len(act_1), len(act_2)), interval):
            if np.random.random() < exchange_w:
                off_ob = np.vstack((off_ob, ob_2[i:i + interval]))
                off_act = np.vstack((off_act, act_2[i:i + interval]))
                off_step_r = np.concatenate((off_step_r, step_re_2[i:i + interval]))
            else:
                off_ob = np.vstack((off_ob, ob_1[i:i + interval]))
                off_act = np.vstack((off_act, act_1[i:i + interval]))
                off_step_r = np.concatenate((off_step_r, step_re_1[i:i + interval]))

        if len(act_1) > len(act_2):
            tail = np.random.randint(len(act_1)-len(act_2))
            off_ob = np.vstack((off_ob, ob_1[len(ob_2):len(ob_2)+tail]))
            off_act = np.vstack((off_act, act_1[len(act_2):len(ob_2)+tail]))
            off_step_r = np.concatenate((off_step_r, step_re_1[len(step_re_2):len(ob_2)+tail]))

        if len(act_2) > len(act_1):
            tail = np.random.randint(len(act_2)-len(act_1))
            off_ob = np.vstack((off_ob, ob_2[len(ob_1):len(ob_1)+tail]))
            off_act = np.vstack((off_act, act_2[len(act_1):len(ob_1)+tail]))
            off_step_r = np.concatenate((off_step_r, step_re_2[len(step_re_1):len(ob_1)+tail]))


        return off_ob, off_act, off_step_r
    else:
        return None


def mutation(args, offspring, dataset):
    off_ob, off_act, off_step_r = offspring
    interval = np.random.randint(1,3)
    for i in range(0, len(off_ob)-interval, interval):
        if np.random.random() < args.mutation_rate:

            rantraj_index = np.random.randint(len(dataset.trajs))
            mut_ob, mut_act, step_r = dataset.trajs[rantraj_index]
            randomsample_index = np.random.randint(len(mut_ob)-interval)
            off_ob[i:i + interval] = mut_ob[randomsample_index:randomsample_index + interval]
            off_act[i:i + interval] = mut_act[randomsample_index:randomsample_index + interval]



            mut_rank = np.arange(interval, dtype=np.float64)
            off_step_r[i:i + interval] = list(np.full_like(mut_rank, np.random.randint(0, 7)))


    return off_ob, off_act, off_step_r


def geneticA(dataset, num_ga_off):
    # collect training data

    basic_size = len(dataset.basic_trajs)
    #rank_0,rank_1, rank_2, rank_3,rank_4 =[],[],[],[],[]
    rank_0, rank_1, rank_2 = [], [], []

    #while len(rank_0)*len(rank_1)*len(rank_2)*len(rank_3)*len(rank_4) == 0:
    while min(len(rank_0), len(rank_1), len(rank_2)) != 5:
        traj_pairs = dataset.sample_pair(args.num_samples)
        idx = np.random.randint(len(traj_pairs))
        trajx, trajy = traj_pairs[idx]
        offspring = crossover(args, trajx[0], trajx[1], trajx[2], trajy[0], trajy[1], trajy[2])
        if offspring is None:
            continue
        #print("GET one offspring")
        offspring_ob, offspring_act, offspring_sr = mutation(args, offspring, dataset)

        print("Parents reward are %d and %d. The offspring reward is %d" %(np.sum(trajx[2]),np.sum(trajy[2]),np.sum(offspring_sr)))


        off_rank = np.sum(offspring_sr)/len(offspring_sr)

        #print("@","Find offspring with Rank: ", off_rank, "[ %d,%d,%d,%d,%d]" %(len(rank_0), len(rank_1), len(rank_2),len(rank_3),len(rank_4) ))


        if 0.5<=off_rank<1.5:
            if len(rank_0)< 10:
                #rank_number = round(dataset.min_rank + 0.2*(6-dataset.min_rank))
                rank = np.arange(len(offspring_sr), dtype=np.float64)
                rank = list(np.full_like(rank, 1.0))
                rank_0.append((offspring_ob, offspring_act,rank))
                dataset.insertTraj(offspring_ob, offspring_act, offspring_sr)
                print("@", "Find offspring with Rank: ", off_rank,
                      "[ %d,%d,%d]" % (len(rank_0), len(rank_1), len(rank_2)))
        if 2.5<=off_rank<3.5:
            if len(rank_1) < 10:
                #rank_number = round(dataset.min_rank + 0.5 * (6 - dataset.min_rank))
                rank = np.arange(len(offspring_sr), dtype=np.float64)
                rank = list(np.full_like(rank, 3.0))
                rank_1.append((offspring_ob, offspring_act,rank))
                dataset.insertTraj(offspring_ob, offspring_act, offspring_sr)
                print("@", "Find offspring with Rank: ", off_rank,
                      "[ %d,%d,%d]" % (len(rank_0), len(rank_1), len(rank_2)))
        if 4.5<=off_rank<5.5:
            if len(rank_2) < 10:
                #rank_number = round(dataset.min_rank + 0.8 * (6 - dataset.min_rank))
                rank = np.arange(len(offspring_sr), dtype=np.float64)
                rank = list(np.full_like(rank, 5.0))
                rank_2.append((offspring_ob, offspring_act,rank))
                dataset.insertTraj(offspring_ob, offspring_act, offspring_sr)
                print("@", "Find offspring with Rank: ", off_rank,
                      "[ %d,%d,%d]" % (len(rank_0), len(rank_1), len(rank_2)))
                
    dataset.reset_dataset()
    candidate = []
    while len(candidate) < num_ga_off:
        candidate.append(sample(rank_0, 1)[0])
        candidate.append(sample(rank_1, 1)[0])
        candidate.append(sample(rank_2, 1)[0])


    for i in range(num_ga_off):
        offob,offact,offrank =candidate[i]
        dataset.insertTraj(offob,offact,offrank)

    for i in range(len(dataset.trajs)):
        print("training rank: ", max(set(dataset.trajs[i][2]),key=list(dataset.trajs[i][2]).count),"   Length: ", len(dataset.trajs[i][0]))

    print("Length of training trajectories: ", len(dataset.trajs))


    return dataset


def train(args):
    logdir = Path(args.log_dir)

    if logdir.exists():
        c = input('log dir is already exist. continue to train a preference model? [Y/etc]? ')
        if c in ['YES', 'yes', 'Y','y']:
            import shutil
            shutil.rmtree(str(logdir))  # remove the existing files and dir
        else:
            print('good bye')
            return

    logdir.mkdir(parents=True)  # recursively create dir
    with open(str(logdir / 'args.txt'), 'w') as f:
        f.write(str(args))

    logdir = str(logdir)
    env = gym.make(args.env_id)

    train_agents = []
    models = [p for p in Path(args.learners_path).glob('?????') if (int(p.name) == int(args.max_chkpt) or int(p.name) == int(args.min_chkpt))]

    for path in models:
        agent = PPO2Agent(env, args.env_type, path, stochastic=args.stochastic)
        train_agents.append(agent)


    dataset = TrajDataset(env, include_action=args.include_action, max_chkpt=args.max_chkpt, min_chkpt=args.min_chkpt)

    #load from external
    dataset.load_traj('./external_dataset/'+args.env_id+'/dataset.pkl')


    #with open(os.path.join(logdir, 'dataset.pkl'), 'wb') as f:
    #    pickle.dump(dataset.ori_trajs, f)
    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d' % i):
            models.append(
                MyModel(args.include_action, env.observation_space.shape[0], env.action_space.shape[0],
                      num_layers=args.num_layers, embedding_dims=args.embedding_dims))

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)
    #dataset = geneticA(dataset, num_ga_off=3)

    for i, model in enumerate(models):
        dataset = geneticA(dataset, num_ga_off=12)
        model.train(dataset, iter= args.model_iter,output_size=args.snippets_size, step_size=args.step_size, debug=True)
        model.saver.save(sess, logdir + '/model_%d.ckpt' % (i), write_meta_graph=False)
        dataset.reset_dataset()
        print("Number of Trajectory AFTER reset: ", len(dataset.trajs))



def eval(args):
    logdir = Path(args.log_dir)

    env = gym.make(args.env_id)

    valid_agents = []
    test_agents = []
    models = Path(args.learners_path).glob('?????')
    for path in models:
        if int(path.name) == int(args.max_chkpt) or int(path.name) == int(args.min_chkpt):
            agent = PPO2Agent(env, args.env_type, path, stochastic=args.stochastic)
            valid_agents.append(agent)

        else:
            agent = PPO2Agent(env, args.env_type, path, stochastic=args.stochastic)
            test_agents.append(agent)

    v_dataset = TrajDataset(env, args.include_action, max_chkpt=args.max_chkpt, min_chkpt=args.min_chkpt)
    #v_dataset.generate_traj(valid_agents, args.min_length, num_training_traj=args.num_training_traj)
    v_dataset.load_traj('./external_dataset/'+args.env_id+'/dataset.pkl')

    t_dataset = TrajDataset(env, args.include_action, max_chkpt=args.max_chkpt, min_chkpt=args.min_chkpt)
    t_dataset.generate_traj(test_agents, args.min_length, num_training_traj=5)
    with open(os.path.join(logdir, 'v_dataset.pkl'), 'wb') as f:
        pickle.dump(v_dataset.ori_trajs, f)
    models = []
    for i in range(args.num_models):
        with tf.variable_scope('model_%d' % i):
            models.append(
                MyModel(args.include_action, env.observation_space.shape[0], env.action_space.shape[0],
                      num_layers=args.num_layers, embedding_dims=args.embedding_dims))

    ### Initialize Parameters
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    # Training configuration
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession()

    sess.run(init_op)

    for i, model in enumerate(models):
        with sess.as_default():
            model.saver.restore(sess, str(logdir) + '/model_%d.ckpt' % (i))

    true_r = []
    pre_r = []
    p_t_rate = []
    for j in range(len(v_dataset.ori_trajs)):
        obs, acs, r = v_dataset.ori_trajs[j]
        r_hat = [model.get_reward(obs, acs) for model in models]
        avg_r_hat = np.sum(r_hat)/len(models)
        true_r.append(np.sum(r))
        pre_r.append(avg_r_hat)
        p_t_rate.append(avg_r_hat/(np.sum(r)+0.00001))

    print(true_r)

    test_true_r = []
    test_pre_r = []
    for k in range(len(t_dataset.ori_trajs)):
        obs, acs, r_test = t_dataset.ori_trajs[k]
        r_hat_test = [model.get_reward(obs, acs) for model in models]
        avg_r_hat_test = np.sum(r_hat_test)/len(models)
        test_true_r.append(np.sum(r_test))
        test_pre_r.append(avg_r_hat_test)
        p_t_rate.append(avg_r_hat_test / (np.sum(r)+0.00001))

    print(test_true_r)
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    plt.style.use('ggplot')
    params = {
        'text.color': 'black',
        'axes.labelcolor': 'black',
        'xtick.color': 'black',
        'ytick.color': 'black',
        'legend.fontsize': 'xx-large',
        # 'figure.figsize': (6, 5),
        'axes.labelsize': 'xx-large',
        'axes.titlesize': 'xx-large',
        'xtick.labelsize': 'xx-large',
        'ytick.labelsize': 'xx-large'}
    matplotlib.pylab.rcParams.update(params)


    def _convert_range(x, minimum, maximum, a, b):
        return (x - minimum) / (maximum - minimum) * (b - a) + a


    convert_range = _convert_range

    gt_max, gt_min = max(true_r+test_true_r), min(true_r+test_true_r)
    pred_max, pred_min = max(pre_r+test_pre_r), min(pre_r+test_pre_r)


    fig, ax = plt.subplots()

    true_y_axis = [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in pre_r]
    ax.plot(true_r,true_y_axis,'go')  # seen trajs
    for i, num in enumerate(true_r):
        ax.annotate(num, (true_r[i], true_y_axis[i]))

    ax.plot(test_true_r,
            [convert_range(p, pred_min, pred_max, gt_min, gt_max) for p in test_pre_r],'ro')  # unseen trajs

    ax.plot([gt_min - 5, gt_max + 5], [gt_min - 5, gt_max + 5], 'k--')
    ax.set_xlabel("Ground Truth Returns")
    ax.set_ylabel("Predicted Returns (normalized)")
    fig.tight_layout()


    plt.savefig("./viz/" + args.env_id + "/gt_vs_pred_rewards.png")
    plt.close()
    print(p_t_rate)
    print("avg prediction: ", np.sum(p_t_rate)/len(p_t_rate))
    print("standard deviation: ", np.std(p_t_rate))






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_id', default='', help='Select the environment to run')
    parser.add_argument('--env_type', default='', help='mujoco or atari')
    parser.add_argument('--learners_path', default='', help='path of learning agents')
    parser.add_argument('--max_chkpt', default=220, type=int, help='decide what learner upperbound you want to give')
    parser.add_argument('--min_chkpt', default=40, type=int, help='decide what learner lowerbound you want to give')
    parser.add_argument('--num_training_traj', default=1, type=int, help='number of iter for each checkpoint to '
                                                                          'create trainning set')
    parser.add_argument('--mutation_rate', default=0.05, help="mutation for experiments")
    parser.add_argument('--crossover_rate', default=0.9, help="crossover rate for experiments")
    parser.add_argument('--crossover_low', default=1, type=int, help="crossover minimum step size")
    parser.add_argument('--crossover_high', default=2, type=int, help="crossover maximum step size")

    parser.add_argument('--model_iter', default=3000, type=int, help='number of iterations to train the model')
    parser.add_argument('--step_size', default=100, type=int, help='length of snippets')
    parser.add_argument('--snippets_size', default=4000, type=int, help='number of  snippets')

    parser.add_argument('--min_length', default=1000, type=int,
                        help='minimum length of trajectory generated by each agent')
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=5, type=int, help='number of models to ensemble')

    parser.add_argument('--num_samples', default=500, type=int, help='number of samples of trajectory snippet')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--include_action', default=False, action='store_true', help='whether to include action for the model or not')
    parser.add_argument('--stochastic', action='store_true', help='whether want to use stochastic agent or not')
    parser.add_argument('--random_agent', action='store_true', help='whether to use default random agent')
    parser.add_argument('--infer', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--eval', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--train_rl', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    parser.add_argument('--eval_rl', action='store_true', help='path to log base (env_id will be concatenated at the end)')
    # Args for PPO
    parser.add_argument('--rl_runs', default=5, type=int)
    parser.add_argument('--ppo_log_path', default='ppo2')
    parser.add_argument('--custom_reward', help='preference or preference_normalized')
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--alive_bonus', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    parser.add_argument('--seed', default=3, help="random seed for experiments")
    args = parser.parse_args()

    # set seeds
    seed = int(args.seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    if args.infer:
        train(args)

    if args.eval:
        eval(args)

    if args.train_rl:
        import os, subprocess

        openai_logdir = Path(os.path.abspath(os.path.join(args.log_dir, args.ppo_log_path)))
        if openai_logdir.exists():
            print('openai_logdir is already exist.')
            exit()

        template = 'python -m baselines.run --alg=ppo2 --env={env} --num_timesteps=1e6 --save_interval=10 --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}" --gamma {gamma}'
        kwargs = {
            "num_models": args.num_models,
            "include_action": args.include_action,
            "model_dir": os.path.abspath(args.log_dir),
            "num_layers": args.num_layers,
            "embedding_dims": args.embedding_dims,
            "ctrl_coeff": args.ctrl_coeff,
            "alive_bonus": args.alive_bonus
        }

        openai_logdir.mkdir(parents=True)
        with open(str(openai_logdir / 'args.txt'), 'w') as f:
            f.write(args.custom_reward + '/')
            f.write(str(kwargs))

        cmd = template.format(
            env=args.env_id,
            custom_reward=args.custom_reward,
            gamma=args.gamma,
            kwargs=str(kwargs))

        procs = []
        for i in range(args.rl_runs):
            env = os.environ.copy()
            env["OPENAI_LOGDIR"] = str(openai_logdir / ('run_%d' % i))
            if i == 0:
                env["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'
                p = subprocess.Popen(cmd, cwd='./learner/baselines', stdout=subprocess.PIPE, env=env, shell=True)
            else:
                env["OPENAI_LOG_FORMAT"] = 'log,csv,tensorboard'
                p = subprocess.Popen(cmd, cwd='./learner/baselines', env=env, shell=True)
            procs.append(p)

        for line in procs[0].stdout:
            print(line.decode(), end='')

        for p in procs[1:]:
            p.wait()

    if args.eval_rl:
        import os
        #from performance_checker import gen_traj_dist as get_perf

        from performance_checker import gen_traj_return as get_perf
        from dataset import gen_traj
        from video import VideoRecorder

        env = gym.make(args.env_id)

        agents_dir = Path(os.path.abspath(os.path.join(args.log_dir, args.ppo_log_path)))
        trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))
        print(trained_steps)
        print(str(agents_dir))

        video = VideoRecorder('./viz/'+args.env_id)
        vp=agents_dir / ('run_4') / 'checkpoints/00400'
        vagent = PPO2Agent(env, args.env_type, vp, stochastic=args.stochastic)

        def _create_video(env, agent, video):

            obs = env.reset()
            video.init(enabled=True)
            done = False
            episode_reward = 0
            while not done:
                action = agent.act(obs,None,None)
                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

            video.save('%s.mp4' % args.env_id)

        
        _create_video(env,vagent,video)
        

        for step in trained_steps[::-1]:
            perfs = []
            for i in range(args.rl_runs):
                path = agents_dir / ('run_%d' % i) / 'checkpoints' / step

                if path.exists() == False:
                    continue

                agent = PPO2Agent(env, args.env_type, path, stochastic=args.stochastic)
                perfs += [
                    get_perf(env, agent) for _ in range(5)
                ]
                #print(path)
                print('[%s-%d] %f %f' % (step, i, np.mean(perfs[-5:]), np.std(perfs[-5:])))

            print('[%s] %f %f %f %f' % (step, np.mean(perfs), np.std(perfs), np.max(perfs), np.min(perfs)))

