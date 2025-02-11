#This code contrains portions originating from neka-nat and is licensed under the MIT License. See: https://github.com/neka-nat/distributed_rl
#Code, Functions and structures were adopted from the original repository. Modified code is marked with [+]. 

# -*- coding: utf-8 -*-
import numpy as np
from itertools import count
from collections import deque
import redis
import torch
from ..libs import replay_memory, utils
from PIL import Image
from skimage.transform import resize


class Actor(object):
    """Actor of Ape-X

    Args:
        actor_no (int): number of the actor process
        env (gym.Env): Open AI gym environment object
        policy_net (torch.nn.Module): Q-function network
        vis (visdom.Visdom): visdom object
        hostname (str, optional): host name of redis server
        batch_size (int, optional): batch data size when sending to learner
        nstep_return (int, optional): number of multi-step boostrapping
        gamma (float, optional): discount factor
        clip (function, optional): reward clipping function
        target_update (int, optional): update frequency of the target network
        num_total_actors (int, optional): number of total actors
        device (torch.device, optional): calculation device
    """
    EPS_BASE = 0.4
    EPS_ALPHA = 7.0
    def __init__(self, actor_no, env, policy_net, vis, hostname='localhost',
                 batch_size=50, nstep_return=3, gamma=0.999, buf_size=4,
                 clip=lambda x: min(max(-1.0, x), 1.0),
                 target_update=200, num_total_actors=4, 
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self._env = env
        self._actor_no = actor_no
        self._name = "actor_" + str(actor_no)
        self._vis = vis
        self._batch_size = batch_size
        self._nstep_return = nstep_return
        self._gamma = gamma
        self._clip = clip
        self._target_update = target_update
        self._num_total_actors = num_total_actors
        self._policy_net = policy_net
        self._policy_net.eval()
        self._device = device
        self._local_memory = replay_memory.ReplayMemory(1000)
        self._connect = redis.StrictRedis(host=hostname)
        self._img_buf = deque(maxlen = buf_size)
        self.no_op_steps = 30

    def _pull_params(self):
        params = self._connect.get('params')
        if not params is None:
            print("[%s] Sync params." % self._name)
            self._policy_net.load_state_dict(utils.loads(params))

    def preproc_state(self, screen): #[+]

        screen_Gray_max = np.amax(np.dstack((screen[:,:,0], screen[:,:,1])), axis=2)
        transformed_image = resize(screen_Gray_max, output_shape=(84, 84), anti_aliasing=None, preserve_range=True) 
        int_image = np.asarray(transformed_image, dtype=np.float32)/255.0

        return np.ascontiguousarray(int_image, dtype=np.float32)

    def _initialize(self):#[+]
        nx_st = self._env.reset()
        nx_st_gray = np.zeros((210,160), dtype=np.uint8) 
        self._env.ale.getScreenGrayscale(nx_st_gray) 
        nx_screen = np.empty((210, 160, 2), dtype=np.uint8)
        nx_screen[:,:,0] = nx_st_gray
        nx_screen[:,:,1] = nx_st_gray
        for _ in range(self._img_buf.maxlen):
            self._img_buf.append(self.preproc_state(nx_screen))
        for _ in range(np.random.randint(1, self.no_op_steps)):
            self._env.step(0)

    def reset(self):#[+]
        self._img_buf.clear()
        self._initialize()
        return np.array(list(self._img_buf))

    def run(self):
        state = self.reset()
        step_buffer = deque(maxlen=self._nstep_return)
        gamma_nsteps = [self._gamma ** i for i in range(self._nstep_return + 1)]
        sum_rwd = 0
        n_episode = 0
        length_epsiode = 0 #[+]
        next_grayscreen = np.zeros((210, 160), dtype=np.uint8) #[+]

        if self._num_total_actors == 1: 
            eps = self.EPS_BASE
        else:
            eps = self.EPS_BASE ** (1.0 + (self._actor_no - 1.0) / (self._num_total_actors - 1.0) * self.EPS_ALPHA)

        for t in count():
                
            action = utils.epsilon_greedy(torch.from_numpy(state).unsqueeze(0).to(self._device),
                                          self._policy_net, eps)
            
            reward = 0 #[+]
            screen_temp = np.empty((210, 160, 2), dtype=np.uint8) #[+]
            for i in range(4): #[+]
                length_epsiode +=1 #[+]
                _, curr_reward, done, _ = self._env.step(action.item()) #[+]
                self._env.ale.getScreenGrayscale(next_grayscreen) #[+]
                reward += curr_reward #[+]
                if i==2: #[+]
                    screen_temp[:,:,0] = next_grayscreen #[+]
                if i==3: #[+]
                    screen_temp[:,:,1] = next_grayscreen #[+]
                    
            self._img_buf.append(self.preproc_state(screen_temp)) #[+]
            next_state = np.array(list(self._img_buf)) #[+]
            done = torch.tensor([float(done)])
            reward = torch.tensor([self._clip(reward)])

            step_buffer.append(utils.Transition(torch.from_numpy(state), action, reward,
                                                torch.from_numpy(next_state), done))
            if len(step_buffer) == step_buffer.maxlen:
                r_nstep = sum([gamma_nsteps[-(i + 2)] * step_buffer[i].reward for i in range(step_buffer.maxlen)])
                self._local_memory.push(utils.Transition(step_buffer[0].state, step_buffer[0].action, r_nstep,
                                                         step_buffer[-1].next_state, step_buffer[-1].done))
            state = next_state.copy() #[+]
            if done or length_epsiode > 50000: #[+]
                state = self.reset() #[+]
                n_episode += 1 #[+]
                step_buffer.clear() #[+]
                length_epsiode = 0 #[+]
            if len(self._local_memory) >= self._batch_size:
                samples = self._local_memory.sample(self._batch_size)
                _, prio = self._policy_net.calc_priorities(self._policy_net, samples,
                                                           gamma=gamma_nsteps[-1],
                                                           device=self._device)
                print("[%s] Publish experience." % self._name)
                self._connect.rpush('experience',
                                    utils.dumps((samples, prio.squeeze(1).cpu().numpy().tolist())))
                self._local_memory.clear()

            if t > 0 and t % self._target_update == 0:
                self._pull_params()
