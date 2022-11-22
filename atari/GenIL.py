import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from baselines.common.trex_utils import preprocess
from run_test import *
from random import sample

# coding: utf-8


def crossover(traj_1, traj_2, step_re_1, step_re_2):
    # only generate 1 offspring
    # crossover length
    interval = np.random.randint(9,10)

    if np.random.random() < crs_R:
        off_ob = []
        off_act = []
        off_step_r = []
        ob_1, act_1 = zip(*traj_1)
        ob_2, act_2 = zip(*traj_2)
        ob_1=list(ob_1)
        ob_2 = list(ob_2)
        act_1 = list(act_1)
        act_2 = list(act_2)

        for i in range(0,min(len(act_1), len(act_2)),interval):
            if np.random.random() < 0.5:
                off_ob+=ob_2[i:i+interval]
                off_act+=act_2[i:i+interval]
                off_step_r += step_re_2[i:i+interval]
            else:
                off_ob+=ob_1[i:i+interval]
                off_act+=act_1[i:i+interval]
                off_step_r += step_re_1[i:i+interval]

        if len(act_1) > len(act_2):
            tail = np.random.randint(len(act_2)-1,len(act_1))
            off_ob = off_ob + ob_1[len(ob_2):tail]
            off_act = off_act + act_1[len(act_2):tail]
            off_step_r = off_step_r + step_re_1[len(step_re_2):tail]
        if len(act_2) > len(act_1):
            tail = np.random.randint(len(act_1)-1, len(act_2))
            off_ob = off_ob + ob_2[len(ob_1):tail]
            off_act = off_act + act_2[len(act_1):tail]
            off_step_r = off_step_r + step_re_2[len(step_re_1):tail]



        return off_ob, off_act, off_step_r
    else:
        return None


def mutation(offspring,candidate_traj,candidate_step_r,num_demos):
    #mutation on one trajectory
    off_ob, off_act, off_step_r = offspring
    interval = np.random.randint(1,3)
    for i in range(0,len(off_ob),interval):
        if np.random.random() < mut_R:


            mut_rank = np.arange(interval, dtype=np.int64)
            off_step_r[i:i+interval] = list(np.full_like(mut_rank,np.random.randint(0,7)))
    return off_ob, off_act, off_step_r




def generate_novice_demos(env, env_name, agent, model_dir,min_length):
    checkpoint_min = 400
    checkpoint_max = 1000
    checkpoint_step = 600
    checkpoints = []
    if env_name == "breakout":
        checkpoint_min = 600
        checkpoint_max = 1200
    elif env_name == "seaquest":
        checkpoint_min = 10
        checkpoint_max = 65
        checkpoint_step = 5
    for i in range(checkpoint_min, checkpoint_max + checkpoint_step, checkpoint_step):
        if i < 10:
            checkpoints.append('0000' + str(i))
        elif i < 100:
            checkpoints.append('000' + str(i))
        elif i < 1000:
            checkpoints.append('00' + str(i))
        elif i < 10000:
            checkpoints.append('0' + str(i))
    print(checkpoints)

    observations = []
    trajectories = []
    learning_returns = []
    step_rank = []
    for checkpoint in checkpoints:

        model_path = model_dir + "/models/" + env_name + "_25/" + checkpoint
        if env_name == "seaquest":
            model_path = model_dir + "/models/" + env_name + "_5/" + checkpoint

        agent.load(model_path)
        episode_count = 1
        for i in range(episode_count):
            done = False
            obsvtion = []
            ob_action_seq = []
            gt_rewards = []
            r = 0

            ob = env.reset()

            steps = 0
            acc_reward = 0
            while True:
                action = agent.act(ob, r, done)


                ob_processed = preprocess(ob, env_name)
                ob_processed = ob_processed[0]  # get rid of first dimension ob.shape = (1,84,84,4)
                obsvtion.append(ob_processed)

                # env.render()

                ob_action_seq.append([ob_processed, action])
                ob, r, done, _ = env.step(action)
                gt_rewards.append(r[0])
                steps += 1
                acc_reward += r[0]

                #if len(obsvtion) > min_length:
                #    break

                if done:
                    #break
                    if len(obsvtion) < min_length:
                        ob =env.reset()

                    else:
                        break

            print("traj length", len(obsvtion), "demo length", len(observations))
            print("checkpoint: {}, steps: {}, return: {}".format(checkpoint, steps, acc_reward))
            observations.append(obsvtion)
            trajectories.append(ob_action_seq)
            #learning_returns.append(acc_reward)

            #modified
            rank = np.arange(len(gt_rewards),dtype=np.int64)
            rank_num = 6*(int(checkpoint)-int(checkpoint_min))/(int(checkpoint_max)-int(checkpoint_min))
            print("rank: ", rank_num)
            rank = list(np.full_like(rank,rank_num))
            learning_returns.append(rank_num)
            step_rank.append(rank)

    return observations, trajectories, learning_returns, step_rank
    #return observations, trajectories, learning_returns, step_rewards


def create_training_data(trajectories, learning_returns, step_rewards, num_new_trajs, num_snippets, min_snippet_length,
                         max_snippet_length):
    # collect training data
    max_traj_length = 0
    training_pairs = []
    training_labels = []
    max_parent_re = max(learning_returns)
    num_demos = len(trajectories)

    candidate_traj = trajectories[:]
    candidate_step_r = step_rewards[:]

    rank_0, rank_1, rank_2 = [], [], []



    while len(rank_0)*len(rank_1)*len(rank_2)== 0:
        i = 0
        j = 0
        while i == j:
            i = np.random.randint(len(candidate_traj))
            j = np.random.randint(len(candidate_traj))

        offspring = crossover(candidate_traj[i], candidate_traj[j], candidate_step_r[i], candidate_step_r[j])
        if offspring is None:
            continue


        offspring_ob, offspring_act, offspring_sr = mutation(offspring,candidate_traj,candidate_step_r,num_demos)

        off_rank = np.sum(offspring_sr) / len(offspring_sr)

        print("@", "Find offspring with Rank: ", off_rank,
              "[ %d,%d,%d]" % (len(rank_0), len(rank_1), len(rank_2)))




        if 0.5<off_rank<1.5:
            if len(rank_0)< 10:
                rank_0.append(list(zip(offspring_ob, offspring_act)))

                candidate_traj.append(list(zip(offspring_ob, offspring_act)))
                candidate_step_r.append(offspring_sr)

        if 2.5<=off_rank<3.5:
            if len(rank_1) < 10:
                rank_1.append(list(zip(offspring_ob, offspring_act)))

                candidate_traj.append(list(zip(offspring_ob, offspring_act)))
                candidate_step_r.append(offspring_sr)

        if 4.5<=off_rank<5.5:
            if len(rank_2) < 10:
                rank_2.append(list(zip(offspring_ob, offspring_act)))

                candidate_traj.append(list(zip(offspring_ob, offspring_act)))
                candidate_step_r.append(offspring_sr)

        del offspring_ob,offspring_act,offspring_sr
    final_train_traj = trajectories[:]
    final_rank = learning_returns[:]
    final_train_length = num_new_trajs+len(trajectories)

    while len(final_train_traj) < final_train_length:
        final_train_traj.append(sample(rank_2, 1)[0])
        final_rank.append(5)
        final_train_traj.append(sample(rank_1, 1)[0])
        final_rank.append(3)
        final_train_traj.append(sample(rank_0, 1)[0])
        final_rank.append(1)

    final_train_traj = final_train_traj[:final_train_length]
    final_rank = final_rank[:final_train_length]

    print("Total training trajectory",len(final_rank),final_rank)

    while len(training_pairs) < num_snippets:
        i = 0
        j = 0
        while i == j:
            i = np.random.randint(len(final_train_traj))
            j = np.random.randint(len(final_train_traj))

        obs_i, act_i = list(zip(*final_train_traj[i]))
        obs_j, act_j = list(zip(*final_train_traj[j]))

        min_length = min(len(final_train_traj[i]), len(final_train_traj[j]))
        rand_length = np.random.randint(min_snippet_length, max_snippet_length)

        if final_rank[i] > final_rank[j]:
            tj_start = np.random.randint(min_length - rand_length)
            ti_start = np.random.randint(tj_start, len(final_train_traj[i]) - rand_length)
            label = 0
            traj_j = obs_j[tj_start:tj_start + rand_length]
            traj_i = obs_i[ti_start:ti_start + rand_length]

        else:
            ti_start = np.random.randint(min_length - rand_length)
            tj_start = np.random.randint(ti_start, len(final_train_traj[j]) - rand_length)
            label = 1
            traj_i = obs_i[ti_start:ti_start + rand_length]
            traj_j = obs_j[tj_start:tj_start + rand_length]

        training_pairs.append((traj_i, traj_j))
        training_labels.append(label)
    return training_pairs, training_labels


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(4, 16, 7, stride=3)
        self.conv2 = nn.Conv2d(16, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 16, 3, stride=1)
        self.conv4 = nn.Conv2d(16, 16, 3, stride=1)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 1)

    def cum_return(self, traj):
        #calculate cumulative return of trajectory
        sum_rewards = 0
        sum_abs_rewards = 0
        x = traj.permute(0, 3, 1, 2)  # get into NCHW format
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = x.contiguous().view(-1, 784)
        x = F.leaky_relu(self.fc1(x))
        r = self.fc2(x)
        sum_rewards += torch.sum(r)
        sum_abs_rewards += torch.sum(torch.abs(r))
        return sum_rewards, sum_abs_rewards

    def forward(self, traj_i, traj_j):
        cum_r_i, abs_r_i = self.cum_return(traj_i)
        cum_r_j, abs_r_j = self.cum_return(traj_j)
        return torch.cat((cum_r_i.unsqueeze(0), cum_r_j.unsqueeze(0)), 0), abs_r_i + abs_r_j,cum_r_i+cum_r_j


# Train the network
def learn_reward(training_inputs, training_outputs, num_iter, l1_reg, checkpoint_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    reward_net = Net()
    reward_net.to(device)

    optimizer = optim.Adam(reward_net.parameters(), lr=lr, weight_decay=weight_decay)
    # check if gpu available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    loss_criterion = nn.CrossEntropyLoss()

    cum_loss = 0.0

    training_data = list(zip(training_inputs, training_outputs))
    for epoch in range(num_iter):
        np.random.shuffle(training_data)
        training_obs,training_labels = zip(*training_data)
        for i in range(len(training_labels)):

            traj_i, traj_j = training_obs[i]
            labels = np.array([training_labels[i]])
            ob_i = np.array(traj_i)
            ob_j = np.array(traj_j)

            ob_i = torch.from_numpy(ob_i).float().to(device)
            ob_j = torch.from_numpy(ob_j).float().to(device)
            labels = torch.from_numpy(labels).to(device)

            optimizer.zero_grad()

            outputs, abs_rewards,cum_r = reward_net.forward(ob_i, ob_j)

            outputs = outputs.unsqueeze(0)
            loss = loss_criterion(outputs, labels) + l1_reg * abs_rewards
            loss.backward()
            optimizer.step()

            item_loss = loss.item()
            cum_loss += item_loss
            if i % 100 == 99:
                print("*"*10)
                print("epoch {}:{} loss {}".format(epoch, i, cum_loss))
                print(abs_rewards.tolist(),"------",cum_r.tolist())
                cum_loss = 0.0
                print("check pointing")
                torch.save(reward_net.state_dict(), checkpoint_dir)
    # save reward network
    torch.save(reward_net.state_dict(), args.reward_model_path)
    print("finished training")
    return reward_net


def calc_accuracy(reward_network, training_inputs, training_outputs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_criterion = nn.CrossEntropyLoss()
    num_correct = 0.
    with torch.no_grad():
        for i in range(len(training_inputs)):
            label = training_outputs[i]
            traj_i, traj_j = training_inputs[i]
            traj_i = np.array(traj_i)
            traj_j = np.array(traj_j)
            traj_i = torch.from_numpy(traj_i).float().to(device)
            traj_j = torch.from_numpy(traj_j).float().to(device)

            # forward to get logits
            outputs, abs_return, cum_re = reward_network.forward(traj_i, traj_j)
            _, pred_label = torch.max(outputs, 0)
            if pred_label.item() == label:
                num_correct += 1.
    return num_correct / len(training_inputs)


def predict_reward_sequence(net, traj):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rewards_from_obs = []
    with torch.no_grad():
        for s in traj:
            r = net.cum_return(torch.from_numpy(np.array([s])).float().to(device))[0].item()
            rewards_from_obs.append(r)
    return rewards_from_obs


def predict_traj_return(net, traj):
    return sum(predict_reward_sequence(net, traj))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--env_name', default='', help='Select the environment name to run, i.e. pong')
    parser.add_argument('--reward_model_path', default='',
                        help="name and location for learned model params, e.g. ./learned_models/breakout.params")
    parser.add_argument('--seed', default=0, help="random seed for experiments")
    parser.add_argument('--mutation_rate', default=0.05, help="mutation for experiments")
    parser.add_argument('--crossover_rate', default=0.9, help="crossover rate for experiments")
    parser.add_argument('--models_dir', default=".",
                        help="path to directory that contains a models directory in which the checkpoint models for demos are stored")
    parser.add_argument('--num_off_trajs', default=3, type=int, help="number of new generated offspring trajectories")
    parser.add_argument('--num_snippets', default=3000, type=int, help="number of short subtrajectories to sample")
    parser.add_argument('--min_length', default=500, type=int, help="number of short subtrajectories to sample")

    args = parser.parse_args()
    env_name = args.env_name
    if env_name == "spaceinvaders":
        env_id = "SpaceInvadersNoFrameskip-v4"
    elif env_name == "mspacman":
        env_id = "MsPacmanNoFrameskip-v4"
    elif env_name == "videopinball":
        env_id = "VideoPinballNoFrameskip-v4"
    elif env_name == "beamrider":
        env_id = "BeamRiderNoFrameskip-v4"
    else:
        env_id = env_name[0].upper() + env_name[1:] + "NoFrameskip-v4"

    env_type = "atari"
    print(env_type)
    # set seeds
    seed = int(args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    print("Training reward for", env_id)
    num_off_trajs = args.num_off_trajs
    num_snippets = args.num_snippets
    mut_R = args.mutation_rate
    crs_R = args.crossover_rate
    min_snippet_length = 200  # min length of trajectory for training comparison
    maximum_snippet_length = 250

    lr = 0.00001
    weight_decay = 0.0
    num_iter = 5  # num times through training data
    ga_epoch = 0
    l1_reg = 0.0
    stochastic = True

    env = make_vec_env(env_id, 'atari', 1, seed,
                       wrapper_kwargs={
                           'clip_rewards': False,
                           'episode_life': False,
                       })

    env = VecFrameStack(env, 4)
    agent = PPO2Agent(env, env_type, stochastic)

    observation_seq, o_a_s, learning_returns, step_rewards = generate_novice_demos(env, env_name, agent,
                                                                                   args.models_dir,args.min_length)

    demo_lengths = [len(d) for d in observation_seq]
    max_snippet_length = min(np.min(demo_lengths), maximum_snippet_length)
    training_obs, training_labels = create_training_data(o_a_s, learning_returns, step_rewards, num_off_trajs, num_snippets,
                                                         min_snippet_length, max_snippet_length)

    learnednetwork = learn_reward(training_obs, training_labels, num_iter, l1_reg, args.reward_model_path)


    with torch.no_grad():
        for i in range(len(observation_seq)):
            pred = predict_traj_return(learnednetwork,observation_seq[i])
            print(i,pred,learning_returns[i])

    print("accuracy", calc_accuracy(learnednetwork, training_obs, training_labels))
