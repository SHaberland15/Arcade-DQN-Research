# Baseline DQN generates features (predictors) for GLM

#python3 Feature_Generation.py -g breakout -lm 0 -hm 240
#python3 Feature_Generation.py -g space_invaders -lm 0 -hm 240
#python3 Feature_Generation.py -g enduro -lm 0 -hm 240

#Probleme: stimmt action_space, funktioniert es ohne gym?

import numpy as np
from PIL import Image
import torch
from torch import nn
import copy
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
# SIMPLE DQN
###################################
class DqnNN(nn.Module):
    
    def __init__(self, no_of_actions):
        super(DqnNN, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, no_of_actions))
    
    def forward(self, x):
        conv_out = self.conv(x)
        return conv_out

class SimpleDQN():
    
    def __init__(self, action_space):
        
        self.NumberActions = action_space
        self.policy_net = DqnNN(self.NumberActions).to('cuda')
        self.state = np.zeros((4, 84, 84), dtype=np.float32)
  
    def new_state(self, screen):
        
        state_new = self.update_state(screen)
        self.state = state_new
        
        return state_new
    
    def update_state(self, screen): 
    
        state_new = np.concatenate((np.expand_dims(screen, axis=0), self.state[0:3, :,:]))
        
        return state_new

    def preproc_state(self, screen):

        screen_Gray_max = np.amax(np.dstack((screen[:,:,0], screen[:,:,1])), axis=2)
        transformed_image = resize(screen_Gray_max, output_shape=(84, 84), 
                                   anti_aliasing=None, preserve_range=True) 
        int_image = np.asarray(transformed_image, dtype=np.float32)/255.0

        return int_image

    def from_screen_to_state_tensor(self, frame):

        next_state = self.new_state(frame) 
        state_tensor = torch.from_numpy(next_state).cuda().unsqueeze(0) #ready for NN

        return state_tensor

    def reset(self):
        self.state = np.zeros((4, 84, 84), dtype=np.float32)



###################################
# END SIMPLE DQN
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
    action_space = 5
elif args.game == 'space_invaders':
    game_name = 'SpaceInvaders'
    action_space = 5
elif args.game == 'enduro':
    game_name = 'Enduro'
    action_space = 8
else:
    print('Wrong game name')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Einlesen der Probandenliste
prob_file = open('/workspace/container_mount/code/Code_Subjects','r')
prob_code = prob_file.read().splitlines()

for participant in prob_code:

    pseudo_code = participant

    for j in range(args.lowest_model,args.highest_model+1,20):

        print('Model weights', j)
        model_weights = j

        
        model_weight_zero = str(j).zfill(3)
        GamePlayer = SimpleDQN(action_space)
        GamePlayer.policy_net.load_state_dict(torch.load('/workspace/container_mount/checkpoints/' + args.game 
                                              + '/' + args.game + '_DNN_epoch_' + model_weight_zero + '.pt'))
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
