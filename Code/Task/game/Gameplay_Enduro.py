#Gameplay for Enduro black and white
#python3 Gameplay_Enduro.py -sp /workspace/Task/ramdisk/ -pc 1 -g enduro -sn 0 -rp /workspace/ROMs/


import sys
from ale_python_interface import ALEInterface
import numpy as np
import pygame
import time
import fcntl
from PIL import Image, ImageDraw, ImageFont
import argparse
from skimage.transform import resize


parser = argparse.ArgumentParser()

parser.add_argument('-sp', '--save_path', help='Save path', type=str,
                    required=True)
parser.add_argument('-pc', '--pseudo_code', help='Pseudonymization subject', 
                    type=str, required=True)
parser.add_argument('-g', '--game', help='Game name', type=str,
                    required=True)
parser.add_argument('-sn', '--session_no', help='Session number', type=str,
                    required=True)
parser.add_argument('-rp', '--rom_path', help='Rom path', type=str,
                    required=True)
args = parser.parse_args()

key_action_tform_table = (
0, #00000 none
2, #00001 up
5, #00010 down
2, #00011 up/down (invalid)
4, #00100 left
7, #00101 up/left
9, #00110 down/left
7, #00111 up/down/left (invalid)
3, #01000 right
6, #01001 up/right
8, #01010 down/right
6, #01011 up/down/right (invalid)
3, #01100 left/right (invalid)
6, #01101 left/right/up (invalid)
8, #01110 left/right/down (invalid)
6, #01111 up/down/left/right (invalid)
1, #10000 fire
10, #10001 fire up
13, #10010 fire down
10, #10011 fire up/down (invalid)
12, #10100 fire left
15, #10101 fire up/left
17, #10110 fire down/left
15, #10111 fire up/down/left (invalid)
11, #11000 fire right
14, #11001 fire up/right
16, #11010 fire down/right
14, #11011 fire up/down/right (invalid)
11, #11100 fire left/right (invalid)
14, #11101 fire left/right/up (invalid)
16, #11110 fire left/right/down (invalid)
14  #11111 fire up/down/left/right (invalid)
)

def preproc_screen(screen):

   transformed_image = resize(screen, output_shape=(pixel_screen, pixel_screen), 
                                  anti_aliasing=None, preserve_range=True)
   int_image = np.asarray(transformed_image, dtype=np.float32)

   return np.ascontiguousarray(int_image, dtype=np.float32)

def preproc_score(screen_np_in, pixel_score):  

    transformed_image = resize(screen_np_in, output_shape=(pixel_score, pixel_score),
                                anti_aliasing=True, preserve_range=True) 
    int_image = np.array(transformed_image, dtype=np.float32)

    return np.ascontiguousarray(int_image, dtype=np.float32)


ale = ALEInterface()
max_frames_per_episode = ale.getInt(b"max_num_frames_per_episode");
np.random.seed()  
ale_seed = np.random.randint(2^32)
ale.setInt(b"random_seed",ale_seed)

random_seed = ale.getInt(b"random_seed")
print("random_seed: " + str(random_seed))

ale.setFloat(b'repeat_action_probability', 0.0)
action_repeat_prob = ale.getFloat(b'repeat_action_probability')
print('action repeat prob.: ' + str(action_repeat_prob))

ale.loadROM(str.encode(args.rom_path + args.game + '.bin'))
legal_actions = ale.getMinimalActionSet()
print(legal_actions)

(screen_width,screen_height) = ale.getScreenDims()
print("width/height: " +str(screen_width) + "/" + str(screen_height))

(display_width,display_height) = (1280,1024)

pygame.display.init()
pygame.font.init()
screen = pygame.display.set_mode((display_width,display_height))
pygame.display.set_caption("Arcade Learning Environment Player Agent Display")

pixel_score = 300
pixel_height_score = 80
pixel_height_screen = 64
pixel_screen = 84
index_array_screen = 220
game_height = 782

game_surface = pygame.Surface((pixel_screen,pixel_height_screen)) 
game_surface_score = pygame.Surface((pixel_score,pixel_height_score)) 
pygame.mouse.set_visible(False)

pygame.display.flip()

clock = pygame.time.Clock()

episode = 0
total_reward = 0.0
total_total_reward = 0.0

n_frames = 20000 

loop_count = 0
loop_count_intro = 0
loop_15hz_count = 0
mod_4_count = 0
a = 0
a_old = 0

screen_temp = np.zeros((210, 160, 3), dtype=np.uint8)
screen_temp_Gray = np.zeros((210, 160), dtype=np.uint8)
screen_15hz_RGB = np.zeros((210, 160, 3, 2, int(n_frames/4)), dtype=np.uint8)
screen_15hz_Gray = np.zeros((210, 160, 2, int(n_frames/4)), dtype=np.uint8)
responses_vec = np.zeros(n_frames, dtype=np.uint8)
reward_vec = np.zeros(n_frames, dtype=np.double)
episode_vec = np.zeros(n_frames, dtype=np.int32)

counter, text = 3, '3'.rjust(3)
pygame.time.set_timer(pygame.USEREVENT, 1000)
screen.fill((0,0,0))
font = pygame.font.SysFont("Ubuntu Mono",60)
run = True

while run:
    for e in pygame.event.get():
        if e.type == pygame.USEREVENT: 
            counter -= 1
            if counter > 0:
                text = str(counter).rjust(3)
            else:
                text = 'Start!'
                run = False
        if e.type == pygame.QUIT: 
            run = False
    screen.fill((0,0,0))
    screen.blit(font.render(text, True, (255,255,255)), (600,500))
    pygame.display.flip()
    clock.tick(60)
time.sleep(1)

screen_preproc_old  = np.zeros((pixel_screen,pixel_screen))  

while(loop_count < n_frames):

    mod_4_count += 1
   
    keys = 0
    pressed = pygame.key.get_pressed()
    keys |= pressed[pygame.K_u]
    keys |= pressed[pygame.K_c] <<1
    keys |= pressed[pygame.K_m] <<2
    keys |= pressed[pygame.K_l] <<3
    keys |= pressed[pygame.K_s] <<4
    a = key_action_tform_table[keys]

    responses_vec[loop_count] = a
    reward = ale.act(a);
    lives = ale.lives()
    total_reward += reward
    total_total_reward += reward

    reward_vec[loop_count] = total_reward
    episode_vec[loop_count] = episode

    screen.fill((0,0,0))

    numpy_surface = np.frombuffer(game_surface.get_buffer(),dtype=np.uint8)
    ale.getScreenRGB(screen_temp)
    ale.getScreenGrayscale(screen_temp_Gray)
    
    screen_preproc_current = preproc_screen(screen_temp_Gray)
    screen_preproc_max = np.maximum(screen_preproc_current, 
                                    screen_preproc_old) 

    screen_reversed = np.reshape(np.dstack((screen_preproc_max[:,:],
                                            screen_preproc_max[:,:], 
                                            screen_preproc_max[:,:], 
                                            np.zeros((pixel_screen, pixel_screen), 
                                            dtype=np.uint8)))[:pixel_height_screen,:],
                                            pixel_height_screen*pixel_screen*4) 
    numpy_surface[:] = screen_reversed
    screen.blit(pygame.transform.scale(game_surface, 
                                      (display_width,game_height)),
                                      (0,0))

    #get RAM
    ram_size = ale.getRAMSize()
    ram = np.zeros((ram_size),dtype=np.uint8)
    ale.getRAM(ram)

    current_day = "%01X "%ram[45]

    passed_cars = (int(current_day)-1)*300+200-total_reward

    icon = Image.open('enduro_icon.jpg')
    new_scoreboard_txt = Image.new('L',(800,600), color=(0)) #800x600 pixel
    icon.thumbnail((50,50))
    new_scoreboard_txt.paste(icon, (375,515))
    draw = ImageDraw.Draw(new_scoreboard_txt)
    font = ImageFont.load_default()
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/"
                              "DejaVuSans-Bold.ttf",23)
    font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/"
                                  "DejaVuSans-Bold.ttf",40)
    daytext = current_day
    scoretext = str(int(total_reward)).zfill(3)
    carstext = str(int(passed_cars))

    if (total_reward == 200 and int(current_day) ==1) or (total_reward == 500):
        carstext = '\u2691 \u2691'
    
    draw.text((291,513), daytext, fill='white', font=font)
    draw.text((460,513), carstext, fill='white', font=font)
    draw.text((460,465), scoretext, fill='white', font=font)

    draw.rectangle([(250,450),(550,550)], outline='white', width=1) 
    draw.rectangle([(280,460),(520,500)], outline='white', width=1) 
    draw.rectangle([(280,510),(320,540)], outline='white', width=1) 
    draw.rectangle([(357,510),(520,540)], outline='white', width=1) 

    new_scoreboard_img = np.array(new_scoreboard_txt)[:,:]
    score_preproc = preproc_score(new_scoreboard_img,pixel_score) 
    numpy_surface_score = np.frombuffer(game_surface_score.get_buffer(),
                                        dtype=np.uint8)
    score_reversed = np.reshape(np.dstack((score_preproc[:,:], 
                                           score_preproc[:,:], 
                                           score_preproc[:,:], 
                                           np.zeros((pixel_score,pixel_score),
                                           dtype=np.uint8)))[index_array_screen:,:], 
                                           pixel_height_score*pixel_score*4) 
    numpy_surface_score[:] = score_reversed
    
    screen.blit(pygame.transform.scale(game_surface_score, 
                                      (display_width,display_height-game_height)),
                                      (0,game_height)) 
    

    if(mod_4_count == 3):
        screen_15hz_Gray[:,:,0,loop_15hz_count] = screen_temp_Gray
        screen_15hz_RGB[:,:,:,0,loop_15hz_count] = screen_temp
    if(mod_4_count == 4):
        mod_4_count = 0        
        screen_15hz_Gray[:,:,1,loop_15hz_count] = screen_temp_Gray
        screen_15hz_RGB[:,:,:,1,loop_15hz_count] = screen_temp
        loop_15hz_count += 1

    screen_preproc_old  = screen_preproc_current

    del numpy_surface
    del numpy_surface_score
   
    pygame.display.flip()

    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    if(pressed[pygame.K_LALT]) and (pressed[pygame.K_F4]):
        exit = True

    if(exit):
        break

    clock.tick(45.)

    if(ale.game_over()):
        episode_frame_number = ale.getEpisodeFrameNumber()
        frame_number = ale.getFrameNumber()
        print("Frame Number: " + str(frame_number) 
                               + " Episode Frame Number: "
                               + str(episode_frame_number))
        print("Episode " + str(episode) 
                         + " ended with score: " 
                         + str(total_reward))
        ale.reset_game()
        total_reward = 0.0
        episode = episode + 1

    loop_count += 1

result_count = 0

while(result_count < 120):

    screen.fill((0,0,0))

    font = pygame.font.SysFont("Ubuntu Mono",40)
    text = font.render("Block " + args.session_no + " completed!", 1, 
                      (255,255,255))
    screen.blit(text,(500,300))
    text = font.render("Total score: " + str(total_total_reward), 1, 
                      (255,255,255))
    height = font.get_height()*1.2
    screen.blit(text,(500,400 + height))

    pygame.display.flip()

    exit=False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit=True
            break;

    if(pressed[pygame.K_LALT]) and (pressed[pygame.K_F4]):
        exit = True
    if(exit):
        break

    clock.tick(30.)

    result_count += 1

screen.fill((0,0,0))

font = pygame.font.SysFont("Ubuntu Mono",35)
text = font.render("Done, writing files...", 1, (255,255,255))
screen.blit(text,(500,500))

pygame.display.flip()

pygame.event.pump()

clock.tick(60.)


np.save(args.save_path + args.pseudo_code + '_' + args.game 
                       + '_' + args.session_no 
                       + '_screen_15hz_Gray', screen_15hz_Gray)
np.save(args.save_path + args.pseudo_code + '_' + args.game 
                       + '_' + args.session_no 
                       + '_screen_15hz_RGB', screen_15hz_RGB)

np.savetxt(args.save_path + args.pseudo_code + '_' + args.game 
                          + '_' + args.session_no 
                          + '_responses_vec.csv', responses_vec, 
                          delimiter=",", fmt='%01.0u')
np.savetxt(args.save_path + args.pseudo_code + '_' + args.game 
                          + '_' + args.session_no 
                          + '_reward_vec.csv', reward_vec, 
                          delimiter=",", fmt='%01.1f')
np.savetxt(args.save_path + args.pseudo_code + '_' + args.game 
                          + '_' + args.session_no 
                          + '_episode_vec.csv', episode_vec, 
                          delimiter=",", fmt='%01.0u')
