function [corr_conv, corr_no_conv, MSE_conv, MSE_no_conv]...
     = analyze_information_loss(folder_name_data, folder_name_DQN, game_name, pcode, conv_param, zscore_flag)

maxNumCompThreads(1);

[net_probs_3D_conv, responses_mat_bin] = preproc_data(folder_name_data, folder_name_DQN,...
                                         game_name, pcode, conv_param, zscore_flag);

n_sessions = size(net_probs_3D_conv,3);
n_frames_per_session = size(responses_mat_bin,1);
n_frames_per_session_cut = size(net_probs_3D_conv,1);

if conv_param ~= 0
    responses_conv = conv_with_gauss(responses_mat_bin, conv_param);
    p = 0.9995;  
    gauss_radius = ceil(norminv(p, 0, conv_param));
    gauss_kernel = normpdf((-gauss_radius:gauss_radius), 0, conv_param)';
    n_cut = size(gauss_kernel,1) - 1;
    responses_mat_bin_cut = responses_mat_bin(n_cut/2+1:size(responses_mat_bin,1)-n_cut/2,:,:);
else
    responses_conv = responses_mat_bin;
    responses_mat_bin_cut = responses_mat_bin;
end
  
%% run GLM cross-validation
corr_conv = NaN(n_sessions, size(responses_conv,2));
corr_no_conv = NaN(n_sessions, size(responses_conv,2));
MSE_conv = NaN(n_sessions, size(responses_conv,2));
MSE_no_conv = NaN(n_sessions, size(responses_conv,2));
betas = NaN(n_sessions, size(responses_conv,2),size(net_probs_3D_conv,2));

for resp_ind = 1:size(responses_conv,2)

    y_all_sess = squeeze(responses_conv(:,resp_ind,:));
    y_all_sess_without_conv = squeeze(responses_mat_bin_cut(:,resp_ind,:));

    for sess_no = 1:n_sessions

        train_bin = true(n_sessions,1);
        train_bin(sess_no,1) = false;    
        test_bin = ~train_bin;

        net_probs_train = NaN(n_frames_per_session_cut*(n_sessions-1), size(net_probs_3D_conv,2));
        y_train = NaN(n_frames_per_session_cut*(n_sessions-1),1);

        train_count = 0;

        for sess_ind = 1:n_sessions

            if train_bin(sess_ind)

                train_count = train_count + 1;
                net_probs_train(n_frames_per_session_cut*(train_count-1)+1:n_frames_per_session_cut*train_count,:) ...
                                                                                 = net_probs_3D_conv(:,:,sess_ind);
                y_train(n_frames_per_session_cut*(train_count-1)+1:n_frames_per_session_cut*train_count,1) ...
                                                                                 = y_all_sess(:,sess_ind);

            end
        end

        net_probs_train(:,end)  = 1;
        net_probs_test = net_probs_3D_conv(:,:,test_bin);
        net_probs_test(:,end) = 1;

        y_train(y_train > 1) = 1;
        betas(sess_no,resp_ind,:) = glmfit(net_probs_train, y_train, 'binomial', 'constant', 'off');
        y_pred = 1./(1+exp(-net_probs_test*squeeze(betas(sess_no,resp_ind,:))));

        y_test_conv = y_all_sess(:,test_bin);
        corr_temp_conv = corrcoef([y_pred, y_test_conv]);
        corr_conv(sess_no,resp_ind) = corr_temp_conv(2,1);
        MSE_conv(sess_no,resp_ind) = mean((y_pred - y_test_conv).^2);
        
        y_test_no_conv = y_all_sess_without_conv(:,test_bin);
        corr_temp_no_conv = corrcoef([y_pred, y_test_no_conv]);
        corr_no_conv(sess_no,resp_ind) = corr_temp_no_conv(2,1);
        MSE_no_conv(sess_no,resp_ind) = mean((y_pred - y_test_no_conv).^2);

    end
end

