# Generates predictors for the GLM
#Input: Videos seen by the subjects
#Output: time series of Q-values 
#python3 Feature_Generation.py -g breakout -lm 0 -hm 180000

#This code contrains portions originating from neka-nat and is licensed under the MIT License. See: https://github.com/neka-nat/distributed_rl
#Code, Functions and structures were adopted from the original repository. Original code is marked with [*]. 

import numpy as np
from PIL import Image
import torch
from torch import nn
import copy
import gym
from skimage.transform import resize
from collections import deque
import time
import fcntl
import torch.nn.functional as F
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-g', '--game', help='Game name', type=str,
                    required=True)
parser.add_argument('-hm', '--highest_model', help='highest model weights', 
                    type=int, required=True)
parser.add_argument('-lm', '--lowest_model', help='lowest model weights', 
                    type=int, required=True)
args = parser.parse_args()

###################################
# APEX
###################################

class DuelingDQNApeX(nn.Module): #[*]
    def __init__(self, n_action, input_shape=(4, 84, 84)):
        super(DuelingDQNApeX, self).__init__()
        self.n_action = n_action
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        r = int((int(input_shape[1] / 4) - 1) / 2) - 3
        c = int((int(input_shape[2] / 4) - 1) / 2) - 3
        self.adv1 = nn.Linear(r * c * 64, 512)
        self.adv2 = nn.Linear(512, self.n_action)
        self.val1 = nn.Linear(r * c * 64, 512)
        self.val2 = nn.Linear(512, 1)

    def forward(self, x): #[*]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        adv = F.relu(self.adv1(x))
        adv = self.adv2(adv)
        val = F.relu(self.val1(x))
        val = self.val2(val)
        return val + adv - adv.mean(1, keepdim=True)
		
        
class APEX(object):
    
    def __init__(self, env):
        self.env = env
        self.NumberActions = env.action_space.n
        self.policy_net = DuelingDQNApeX(self.NumberActions).to('cuda')
        self.buf_size = 4
        self.img_buf = deque(maxlen=self.buf_size)

    def preproc_state(self, screen):

        screen_Gray_max = np.amax(np.dstack((screen[:,:,0], screen[:,:,1])), axis=2)
        transformed_image = resize(screen_Gray_max, output_shape=(84, 84), 
                                   anti_aliasing=None, preserve_range=True) 
        int_image = np.asarray(transformed_image, dtype=np.float32)/255.0

        return np.ascontiguousarray(int_image, dtype=np.float32)

    def _initialize(self): 
        nx_st_gray = np.zeros((210,160), dtype=np.uint8)  
        nx_screen = np.empty((210, 160, 2), dtype=np.uint8)
        nx_screen[:,:,0] = nx_st_gray
        nx_screen[:,:,1] = nx_st_gray
        for _ in range(self.img_buf.maxlen):
            self.img_buf.append(self.preproc_state(nx_screen))

    def reset(self): 
        self.img_buf.clear()
        self._initialize()

    def from_screen_to_state_tensor(self, frame):

        self.img_buf.append(frame)
        next_state = np.array(list(self.img_buf))

        state_tensor = torch.from_numpy(next_state).unsqueeze(0).to('cuda')
        return state_tensor


###################################
# END APEX
###################################

def get_activation_value(name):
    
    def hook(m, input, output):
        actIn[name] = input[0].detach()
        actOut[name] = output.detach()
    return hook
    
AnzahlSessions = 7
NumberOfFrames15Hz = 5000

if args.game == 'breakout':
    game_name = 'Breakout'
elif args.game == 'space_invaders':
    game_name = 'SpaceInvaders'
elif args.game == 'enduro':
    game_name = 'Enduro'
else:
    print('Wrong game name')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SelfEnv = game_name + 'NoFrameskip-v0' 
env = gym.make(SelfEnv)

prob_file = open('/workspace/container_mount/code/Code_Subjects','r')
prob_code = prob_file.read().splitlines()

for participant in prob_code:

    pseudo_code = participant

    for j in range(args.lowest_model,args.highest_model+1,20000):

        print('Model weights', j)
        model_weights = j

        GamePlayer = APEX(env)
        GamePlayer.policy_net.load_state_dict(torch.load('/workspace/container_mount/checkpoints/' 
                                            + args.game + '/model_' + str(model_weights) + '.pth'))
        GamePlayer.policy_net.eval()
        
        for sn in range(1, AnzahlSessions + 1): 

            print('Session number:', sn)
            #load video
            screen_path = '/workspace/data/Raw_Data/' + pseudo_code + '/' + args.game 
                          + '/' + pseudo_code + '_' + args.game  + '_E_' + str(sn) + '_screen_15hz_Gray.npy'
            Video15Hz = np.load(screen_path)
            episode_path = '/workspace/data/Raw_Data/' + pseudo_code + '/' + args.game 
                           + '/' + pseudo_code +  '_' + args.game + '_E_' + str(sn) + '_episode_vec.csv'
            episode_vec = np.loadtxt(episode_path, dtype=np.int)
            ActValuesLayerOutput = np.zeros((NumberOfFrames15Hz,1,GamePlayer.NumberActions))
        
            GamePlayer.reset()
            current_episode = 0

            for i in range(NumberOfFrames15Hz): 
                print('Frame number:', i)
                Screen15HzCurrent = Video15Hz[:,:,:,i]

                if episode_vec[i*4+3] != current_episode:
                    GamePlayer.reset()
                    current_episode += 1

                ProcCurScreen = GamePlayer.preproc_state(Screen15HzCurrent)

                state_tensor = GamePlayer.from_screen_to_state_tensor(ProcCurScreen)

                actIn = {}
                actOut = {}
                handles = {}

                for name, module in GamePlayer.policy_net.named_children(): 
                    handles[name] = module.register_forward_hook(get_activation_value(name))
                
                with torch.no_grad():    
                    out = GamePlayer.policy_net(state_tensor)    
                    for k, v in handles.items():
                        handles[k].remove() 
                        
                ActValuesLayerOutput[i] = out.cpu().detach().numpy()

        print(ActValuesLayerOutput)


        del GamePlayer
