function [net_probs_3D_conv, responses_mat_bin]...
     = preproc_data(folder_name_data, folder_name_DQN, game_name, pcode, gauss_sigma, zscore_flag)

net_out_mat = csvread(strcat(folder_name_DQN, '1.csv'));
responses_vec = csvread(strcat(folder_name_data,'/',pcode,'/',game_name,'/',pcode,'_',...
                game_name,'_E_1_responses_vec.csv'));

n_sessions = 7;
n_frames_per_session = size(responses_vec,1);
n_passes_per_session = n_frames_per_session/4;

if n_passes_per_session ~= size(net_out_mat,1)   
    error('Something wrong with the sampling rates!');  
end

n_resp_options = size(net_out_mat,2);  
net_out_3D = NaN(n_passes_per_session, n_resp_options, n_sessions);
responses_mat = NaN(n_frames_per_session, n_sessions);

for sess_no = 1:n_sessions
    clear net_out_mat
    clear responses_vec

    net_out_mat = csvread(strcat(folder_name_DQN, num2str(sess_no), '.csv'));    
    net_out_3D(:,:,sess_no) = net_out_mat;
    responses_vec = csvread(strcat(folder_name_data,'/',pcode,'/',game_name,'/',pcode,'_',game_name,...
                    '_E_' , num2str(sess_no) , '_responses_vec.csv'));
    responses_mat(:,sess_no) = responses_vec;

end

clear net_out_mat
clear responses_vec

net_out_3D_60hz = NaN(n_frames_per_session,  n_resp_options, n_sessions);

for sess_no = 1:n_sessions
    for pass_no = 1:n_passes_per_session
        net_out_3D_60hz(4*pass_no-3:4*pass_no,:,sess_no) = repmat(net_out_3D(pass_no,:,sess_no), 4, 1, 1);
    end
end

clear net_out_3D
clear n_passes_per_session
clear pass_no

net_out_3D_60hz([1:12, end-3:end],:,:) = [];
responses_mat(1:16,:) = [];
n_frames_per_session = size(net_out_3D_60hz,1);

if strcmp(game_name, 'enduro')
    responses_mat_bin = resp_recode_enduro(n_frames_per_session, n_sessions, responses_mat);
elseif strcmp(game_name, 'space_invaders')
    responses_mat_bin = resp_recode_space_inv(n_frames_per_session, n_sessions, responses_mat);
elseif strcmp(game_name, 'breakout')
    responses_mat_bin = resp_recode_breakout(n_frames_per_session, n_sessions, responses_mat);
else
    error('folder_name not found!');
end

clear responses_mat

net_probs_3D_zscore = NaN(size(net_out_3D_60hz));

for sess_no = 1:n_sessions
    for frame_no = 1:n_frames_per_session
        net_out_curr = net_out_3D_60hz(frame_no,:,sess_no);
        if zscore_flag == 1
            net_out_curr = zscore(net_out_curr);
        end
        net_probs_3D_zscore(frame_no,:,sess_no) = net_out_curr;
    end
end

clear net_out_3D_60hz
clear net_out_curr
clear frame_no

if gauss_sigma ~= 0
    net_probs_3D_conv = conv_with_gauss(net_probs_3D_zscore, gauss_sigma);
else
    net_probs_3D_conv = net_probs_3D_zscore;
end
clear net_probs_3D_zscore