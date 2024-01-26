%Enter the path where the repository is located:
YOUR_REPO_PATH = %; repopath/Arcade-DQN-Research
%Enter the path where the OSF files are located:
YOUR_DATA_PATH = %;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation smoothing parameters 
%Training: model was fitted using smoothed (in)dependend variables
%Prediction smoothed and unsmoothed human time series
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DQN_Liste = ["BaselineDQN" "ApeX" "SeedRL"];
game_name_Liste = ["breakout" "enduro" "space_invaders"];
conv_param_vec = [0 1 5 10 15 20 30 40 50 60 80 100 120];
zscore_flag = 1;

for dqn = 1:length(DQN_Liste)

    DQN_name = DQN_Liste(dqn);

    for game = 1:length(game_name_Liste)

        game_name = game_name_Liste(game);

        fid = fopen('Code_Subjects');
        tline = fgetl(fid);
        tlines = cell(0,1);
        while true
            if ~ischar(tline); break; end 
            tlines{end+1,1} = tline;
            tline = fgetl(fid);
        end
        fclose(fid);
        
        if strcmp(game_name,'breakout') 
            action_size = 4;
            tlines(5) = [];
            tlines(17) = [];
        elseif strcmp(game_name,'space_invaders')
            action_size = 4;
        else 
            action_size = 4;
        end 
        
        if strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'breakout') 
            model_weight_vec = ["180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'enduro')
            model_weight_vec = ["180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'space_invaders') 
            model_weight_vec = ["180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'breakout') 
            model_weight_vec = ["600000" "620000" "640000" "660000"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'enduro')
            model_weight_vec = ["420000" "440000" "460000" "480000"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'space_invaders')
            model_weight_vec = ["480000" "500000" "520000" "540000"];
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'breakout')
            model_weight_vec = ["0-ckpt-91" "0-ckpt-93" "0-ckpt-96" "0-ckpt-98"];
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'enduro')
            model_weight_vec = ["0-ckpt-90" "0-ckpt-93" "0-ckpt-96" "0-ckpt-98"]; 
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'space_invaders')
            model_weight_vec = ["0-ckpt-123" "0-ckpt-126" "0-ckpt-128" "0-ckpt-130"]; 
        end

        corr_conv_mean = zeros(length(tlines), length(conv_param_vec), length(model_weight_vec), action_size);
        corr_no_conv_mean = zeros(length(tlines), length(conv_param_vec), length(model_weight_vec), action_size);
        MSE_conv_mean = zeros(length(tlines), length(conv_param_vec), length(model_weight_vec), action_size);
        MSE_no_conv_mean = zeros(length(tlines), length(conv_param_vec), length(model_weight_vec), action_size);

        for model = 1: length(model_weight_vec)
                model_weight = model_weight_vec(model);
                for conv_param_no = 1: length(conv_param_vec)
                    conv_param = conv_param_vec(conv_param_no);
                    parfor subject = 1:length(tlines)
                        pcode = tlines{subject};
                        folder_name_DQN = strcat(YOUR_DATA_PATH, '/osfstorage-archive/DQN_Output/',...
                                                 DQN_name , '/' , pcode , '/' ,game_name, '/Output_' ,...
                                                 DQN_name , '_model_' , model_weight , '_' , game_name , '_' , pcode , '_Session_');
                        folder_name_data = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Raw_Data');
                        [corr_conv, corr_no_conv, MSE_conv, MSE_no_conv] = analyze_information_loss(folder_name_data,...
                                                     folder_name_DQN, game_name, pcode, conv_param, zscore_flag);
                        corr_conv_mean(subject, conv_param_no, model, :) = mean(corr_conv);
                        corr_no_conv_mean(subject, conv_param_no, model, :) = mean(corr_no_conv);
                        MSE_conv_mean(subject, conv_param_no, model, :) = mean(MSE_conv,1);
                        MSE_no_conv_mean(subject, conv_param_no, model, :) = mean(MSE_no_conv,1);
                    end
                end
        end

        file_name_corr = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/smoothing_parameter/', DQN_name , '/' , game_name, '/Correlation_LgRg_LgRg_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_corr, 'corr_conv_mean')
        file_name_corr = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/smoothing_parameter/', DQN_name , '/' , game_name, '/Correlation_LgRg_LrRg_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_corr, 'corr_no_conv_mean')
        file_name_MSE = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/smoothing_parameter/', DQN_name , '/' , game_name, '/MSE_LgRg_LgRg_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_MSE, 'MSE_conv_mean');
        file_name_corr = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/smoothing_parameter/', DQN_name , '/' , game_name, '/MSE_LgRg_LrRg_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_corr, 'MSE_no_conv_mean')
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation of prediction accuracy (correlations and MSE) with and without zscoring
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DQN_Liste = ["BaselineDQN" "ApeX" "SeedRL"];
game_name_Liste = ["breakout" "enduro" "space_invaders"];
conv_param = 15;

for dqn = 1:length(DQN_Liste)

    DQN_name = DQN_Liste(dqn);

    for game = 1:length(game_name_Liste)

        game_name = game_name_Liste(game);

        fid = fopen('Code_Subjects');
        tline = fgetl(fid);
        tlines = cell(0,1);
        while true
            if ~ischar(tline); break; end 
            tlines{end+1,1} = tline;
            tline = fgetl(fid);
        end
        fclose(fid);
        
        if strcmp(game_name,'breakout') 
            action_size = 4;
            tlines(5) = [];
            tlines(17) = [];
        elseif strcmp(game_name,'space_invaders')
            action_size = 4;
        else 
            action_size = 4;
        end 
        
        if strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'breakout') 
            model_weight_vec = ["180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'enduro')
            model_weight_vec = ["180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'space_invaders') 
            model_weight_vec = ["180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'breakout') 
            model_weight_vec = ["600000" "620000" "640000" "660000"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'enduro')
            model_weight_vec = ["420000" "440000" "460000" "480000"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'space_invaders')
            model_weight_vec = ["480000" "500000" "520000" "540000"];
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'breakout')
            model_weight_vec = ["0-ckpt-91" "0-ckpt-93" "0-ckpt-96" "0-ckpt-98"];
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'enduro')
            model_weight_vec = ["0-ckpt-90" "0-ckpt-93" "0-ckpt-96" "0-ckpt-98"]; 
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'space_invaders')
            model_weight_vec = ["0-ckpt-123" "0-ckpt-126" "0-ckpt-128" "0-ckpt-130"]; 
        end

        corr_vec = zeros(length(tlines), length(model_weight_vec), action_size, 2);
        MSE_vec = zeros(length(tlines), length(model_weight_vec), action_size, 2);

        for zscore_flag = 1:2
            for model = 1: length(model_weight_vec)
                    model_weight = model_weight_vec(model);
                    parfor subject = 1:length(tlines)
                        pcode = tlines{subject};
                        folder_name_DQN = strcat(YOUR_DATA_PATH, '/osfstorage-archive/DQN_Output/',...
                                          DQN_name , '/' , pcode , '/' ,game_name, '/Output_' , DQN_name , '_model_' ,...
                                          model_weight , '_' , game_name , '_' , pcode , '_Session_');
                        folder_name_data = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Raw_Data');
                        [corr, MSE] = get_prediction_accuracy(folder_name_data,...
                                          folder_name_DQN, game_name, pcode, conv_param, zscore_flag-1);
                        corr_vec(subject, model, :, zscore_flag) = mean(corr,1); 
                        MSE_vec(subject, model, :, zscore_flag) = mean(MSE,1);
                    end 
            end
        end

        corr_subject = squeeze(mean(corr,1));
        corr_subject_actions = squeeze(mean(corr_subject,2));
        corr_subject_actions_models = squeeze(mean(corr_subject_actions,1)); 

        MSE_subject = squeeze(mean(MSE,1));
        MSE_subject_actions = squeeze(mean(MSE_subject,2)); 
        MSE_subject_actions_models = squeeze(mean(MSE_subject_actions,1)); 

        file_name_corr = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_accuracy/', DQN_name , '/' , game_name, '/Correlation_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_corr, 'corr')
        file_name_MSE = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_accuracy/', DQN_name , '/' , game_name, '/MSE_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_MSE, 'MSE');

        file_name_corr = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_accuracy/', DQN_name , '/' , game_name, '/Correlation_analysis_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_corr, 'corr_subject_actions_models')
        file_name_MSE = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_accuracy/', DQN_name , '/' , game_name, '/MSE_analysis_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_MSE, 'MSE_subject_actions_models');

    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Evaluation of prediction accuracy (correlations and MSE) in terms of training time
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

DQN_Liste = ["BaselineDQN" "ApeX" "SeedRL"];
game_name_Liste = ["breakout" "enduro" "space_invaders"];
conv_param = 15;
zscore_flag = 1;

for dqn = 1:length(DQN_Liste)

    DQN_name = DQN_Liste(dqn);

    for game = 1:length(game_name_Liste)

        game_name = game_name_Liste(game);

        fid = fopen('Code_Subjects');
        tline = fgetl(fid);
        tlines = cell(0,1);
        while true
            if ~ischar(tline); break; end 
            tlines{end+1,1} = tline;
            tline = fgetl(fid);
        end
        fclose(fid);
        
        if strcmp(game_name,'breakout') 
            action_size = 4;
            tlines(5) = [];
            tlines(17) = [];
        elseif strcmp(game_name,'space_invaders')
            action_size = 4;
        else
            action_size = 4;
        end 
        
        if strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'breakout') 
            model_weight_vec = ["20" "40" "60" "80" "100" "120" "140" "160" "180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'enduro')
            model_weight_vec = ["20" "40" "60" "80" "100" "120" "140" "160" "180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'BaselineDQN') && strcmp(game_name, 'space_invaders') 
            model_weight_vec = ["20" "40" "60" "80" "100" "120" "140" "160" "180" "200" "220" "240"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'breakout') 
            model_weight_vec = ["0" "80000" "160000" "240000" "320000" "400000" "480000" "560000" "660000"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'enduro')
            model_weight_vec = ["0" "80000" "160000" "240000" "320000" "400000" "480000"];
        elseif strcmp(DQN_name, 'ApeX') && strcmp(game_name, 'space_invaders')
            model_weight_vec = ["0" "80000" "160000" "240000" "320000" "400000" "480000" "540000"];
       elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'breakout')
            model_weight_vec = ["0-ckpt-91" "0-ckpt-93" "0-ckpt-96" "0-ckpt-98"];
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'enduro')
            model_weight_vec = ["0-ckpt-90" "0-ckpt-93" "0-ckpt-96" "0-ckpt-98"]; 
        elseif strcmp(DQN_name, 'SeedRL') && strcmp(game_name, 'space_invaders')
            model_weight_vec = ["0-ckpt-123" "0-ckpt-126" "0-ckpt-128" "0-ckpt-130"]; 
        end

        corr_vec = zeros(length(tlines), length(model_weight_vec), action_size);
        MSE_vec = zeros(length(tlines), length(model_weight_vec), action_size);

        for model = 1: length(model_weight_vec)
            model_weight = model_weight_vec(model);
                parfor proband = 1:length(tlines)
                    pcode = tlines{proband};
                    folder_name_DQN = strcat(YOUR_DATA_PATH, '/osfstorage-archive/DQN_Output/',...
                                     DQN_name , '/' , pcode , '/' ,game_name, '/Output_' , DQN_name , '_model_' , model_weight,...
                                     '_' , game_name , '_' , pcode , '_Session_');
                    folder_name_data = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Raw_Data');
                    [corr, MSE] = get_prediction_accuracy(folder_name_data, ...
                                     folder_name_DQN, game_name, pcode, conv_param, zscore_flag);
                    corr_vec(proband, model, :) = mean(corr,1);  
                    MSE_vec(proband, model, :) = mean(MSE,1);
                end
        end

        corr_subject = squeeze(mean(corr,1)); 
        corr_subject_actions = squeeze(mean(corr_subject,2)); 

        MSE_subject = squeeze(mean(MSE,1));
        MSE_subject_actions = squeeze(mean(MSE_subject,2)); %mean über Aktions

        file_name_corr = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_checkpoints/', DQN_name , '/' , game_name, '/Correlation_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_corr, 'corr')
        file_name_MSE = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_checkpoints/', DQN_name , '/' , game_name, '/MSE_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_MSE, 'MSE');

        file_name_corr = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_checkpoints/', DQN_name , '/' , game_name, '/Correlation_analysis_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_corr, 'corr_subject_actions')
        file_name_MSE = strcat(YOUR_DATA_PATH, '/osfstorage-archive/Analysis_Results/prediction_checkpoints/', DQN_name , '/' , game_name, '/MSE_analysis_' , DQN_name , '_' , game_name , '.mat'); 
        save(file_name_MSE, 'MSE_subject_actions');

    end
end


