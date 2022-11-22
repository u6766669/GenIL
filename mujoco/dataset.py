import numpy as np
from tqdm import tqdm
import pickle


def gen_traj(env, agent, min_length):
    obs, actions, rewards = [env.reset()], [], []
    while True:
        action = agent.act(obs[-1], None, None)
        ob, reward, done, _ = env.step(action)


        obs.append(ob)
        actions.append(action)
        rewards.append(reward)


        if done:
            if len(obs) < min_length:
                done = False
                obs.pop()
                obs.append(env.reset())
            else:
                obs.pop()
                break



    return (np.stack(obs, axis=0), np.array(actions), np.array(rewards))

class TrajDataset(object):
    def __init__(self,env,include_action,max_chkpt,min_chkpt):
        self.env = env
        self.include_action = include_action
        self.max_chkpt = max_chkpt
        self.min_chkpt = min_chkpt

    def insertTraj(self,obs,actions,rewards):
        traj = (obs,actions,rewards)
        self.trajs.append(traj)
        #if np.sum(rewards) > self.max_reward:
            #self.max_reward = sum(rewards)

    def replaceTraj(self,obs,actions,rewards,rank_x):
        traj = (obs, actions, rewards)
        index_t = np.random.randint(len(self.trajs))
        self.trajs[index_t]=traj

        index_rank = np.random.randint(len(rank_x))
        del(rank_x[index_rank])
        return rank_x

    def load_traj(self,filename):
        r_trajs = []
        with open(filename, 'rb') as f:
            trajs = pickle.load(f)
        ori_reward_traj = trajs

        i = 0
        max_r = max(np.sum(traj[2]) for traj in trajs)
        min_r = min(np.sum(traj[2]) for traj in trajs)
        print("Loading successful!!! %d trajectories are loaded."%(len(trajs)))
        print("maximum reward is %d, minimum reward is %d" %(max_r,min_r))

        for traj in trajs:
            true_reward = np.sum(traj[2])
            rank = np.arange(len(traj[2]), dtype=np.float64)
            rank_num = 6 * (true_reward - min_r) / (max_r - min_r)
            rank = list(np.full_like(rank, rank_num))
            traj = (traj[0], traj[1], np.array(rank))
            tqdm.write('model: %d length: %d true reward: %d avg reward: %f' % (
            i, len(traj[2]), true_reward, np.sum(traj[2]) / len(traj[2])))
            i +=1
            r_trajs.append(traj)
            # if np.sum(traj[2])>max_reward:
            #    max_reward=np.sum(traj[2])

        self.trajs = r_trajs
        self.basic_trajs = r_trajs[:]
        self.ori_trajs = ori_reward_traj

    #add parameter args.num_training_traj
    def generate_traj(self,agents, min_length,num_training_traj=1):

        trajs = []
        ori_reward_traj = []
        #max_reward = -999999
        for agent in tqdm(agents):
            for _ in range(num_training_traj):
                traj = gen_traj(self.env, agent, min_length)
                ori_reward_traj.append(traj)
                true_reward = np.sum(traj[2])
                if agent.chkpt_num == self.max_chkpt or agent.chkpt_num == self.min_chkpt:

                    rank = np.arange(len(traj[2]), dtype=np.float64)
                    rank_num = 6 * (agent.chkpt_num - self.min_chkpt) / (self.max_chkpt - self.min_chkpt)
                    rank = list(np.full_like(rank, rank_num))
                    traj=(traj[0],traj[1], np.array(rank))
                    tqdm.write('model: %s length: %d true reward: %d avg reward: %f' % (agent.chkpt_num, len(traj[2]), true_reward, np.sum(traj[2]) / len(traj[2])))
                trajs.append(traj)
                #if np.sum(traj[2])>max_reward:
                #    max_reward=np.sum(traj[2])

        self.trajs = trajs
        self.basic_trajs = trajs[:]
        self.ori_trajs = ori_reward_traj
        #self.max_reward = max_reward

    def reset_dataset(self):
        self.trajs = self.basic_trajs[:]
        return True

    #generate trajectory pairs
    def sample_pair(self, num_samples):
        sample_dataset = []

        for _ in range(num_samples):
            x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)
            sample_dataset.append((self.trajs[x_idx],self.trajs[y_idx]))

        return sample_dataset

    def generate_joint_input(self,output_size,step_size):
        misorder = 0

        output = []
        label = []
        z_count = 0
        one_count =0
        for _ in tqdm(range(output_size)):
            rank_x,rank_y = 0,0

            while rank_x == rank_y:
                x_idx, y_idx = np.random.choice(len(self.trajs), 2, replace=False)

                ob_x, action_x, r_x = self.trajs[x_idx]
                ob_y, action_y, r_y = self.trajs[y_idx]

                rank_x = max(set(r_x),key=list(r_x).count)
                rank_y = max(set(r_y),key=list(r_y).count)

            min_len = min(len(ob_x), len(ob_y))
            x_ptr = np.random.randint(len(ob_x) - step_size + 1)
            y_ptr = np.random.randint(len(ob_y) - step_size + 1)
            if rank_x> rank_y:
                if np.sum(r_x[x_ptr:x_ptr+step_size]) < np.sum(r_y[y_ptr:y_ptr+step_size]):
                    misorder +=1

                label.append(0)
                z_count +=1
            else:
                if np.sum(r_x[x_ptr:x_ptr+step_size]) > np.sum(r_y[y_ptr:y_ptr+step_size]):
                    misorder +=1

                label.append(1)
                one_count+=1

            sub_ob_x, sub_a_x = ob_x[x_ptr:x_ptr + step_size], action_x[x_ptr:x_ptr + step_size]
            sub_ob_y, sub_a_y = ob_y[y_ptr:y_ptr + step_size], action_y[y_ptr:y_ptr + step_size]

            if self.include_action:
                sub_traj_x = np.concatenate((sub_ob_x,sub_a_x),axis=1)
                sub_traj_y = np.concatenate((sub_ob_y,sub_a_y),axis=1)
            else:
                sub_traj_x = sub_ob_x
                sub_traj_y = sub_ob_y
            output.append([sub_traj_x,sub_traj_y])

        #output ------->>  list(  np.concatenate((sub_ob_x,sub_a_x),axis=1) , ...  )
        # OR
        # output ------->>  list(  sub_ob_x , sub_ob_y  )
        print("# 0's:", z_count,"@"*10,"# 1's",one_count, "Misorder rate:", misorder/output_size)

        return output, label


    # output ------->>  list(  (sub_ob_x, sub_a_x) , (sub_ob_y, sub_a_y)  )
    def output_train_valid(self, output, label):
        assert len(output)==len(label), "input size mismatch"
        idxes = np.random.permutation(len(output))
        train_idxes = idxes[:int(len(output) * 0.8)]
        valid_idxes = idxes[int(len(output) * 0.8):]

        train_set = []
        valid_set = []

        for ti in train_idxes:
            train_set.append((output[ti],label[ti]))

        for vi in valid_idxes:
            valid_set.append((output[vi],label[vi]))

        self.train_set = train_set
        self.vaid_set = valid_set

        return train_set, valid_set

def batch_dataset(dataset,batch_size):
    batch = []
    if len(dataset)< batch_size:
        return dataset
    for _ in range(batch_size):
        idx = np.random.randint(len(dataset)) #replace = false
        batch.append(dataset[idx])
    return batch

