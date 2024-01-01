close all; clc; clear;
addpath(genpath(pwd))

%% path configs
mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets

results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)]; % A folder path for saving results
addpath([mydir(1:idcs(end-1)-1),'/chmm-lsim-matlab-toolbox']) % download from https://github.com/sajjadkarimi91/chmm-lsim-matlab-toolbox
mkdir(results_dir)

%% model config
model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint'};
channel_num = 3; % can be 2 or 3 for channel fusion

sleepedf_num = 20;
feature_sel = 1;
load(['data_split_scratch_trainingchk_',num2str(sleepedf_num),'.mat'])

for km = 1:length(model_name_all)

    % testing 2or3 -channel LSIM
    model_name = model_name_all{km};
    load(['output_',model_name,'.mat'])
    load(['flbss_',num2str(channel_num),'ch_',model_name,'.mat'])


    %LSIM test on all models
    max_r = size(lbss,3); % repeat number for LSIM that was set 1 for saving time
    num_lsim = size(lbss,1);
    CV_number = size(lbss,2);

    y_test = [];
    y_true = [];
    for i = 1:CV_number
        clear set_features
        disp([km,i])
        response = true_label{1,i};
        response= response(:);
        if sum(response==0)>0
            response = response+1;
        end
        this_fold_number = fold_number{1,i};
        test_set = this_fold_number==i;
        train_all_set = this_fold_number~=i;
        eval_set = ismember(this_fold_number,eval_sub{i});
        train_set = ismember(this_fold_number,train_sub{i});

        kappa_cv = [];
        predictors = [];
        for repeat_num = 1:max_r
            for ss = 1:num_lsim
                predictors = [predictors,lbss{ss,i,repeat_num}'];
            end
        end


        for k = 1:10

            Mdl = fitcknn(predictors(train_set, :), response(train_set, :),'NumNeighbors',10*k,'Distance','euclidean','NSMethod','exhaustive');
            yhat = predict(Mdl, predictors(eval_set, :));
            % a correction for kappa computation
            if length(unique(yhat)) < length(unique(response))
                d_set = setdiff(unique(response),unique(yhat));
                ind_rand = randperm(length(yhat));
                yhat(ind_rand(1:length(d_set))) = d_set;
            end
            [ acc_cv(k,1), kapp, f1, sens, spec] = calculate_overall_metrics(response(eval_set), yhat);

        end


        M = acc_cv;
        [CC,I1] = max(acc_cv);

        k = I1;

        Mdl = fitcknn(predictors(train_all_set, :), response(train_all_set, :),'NumNeighbors',10*k,'Distance','euclidean','NSMethod','exhaustive');
        yhat = predict(Mdl, predictors(test_set,:));

        y_test = [y_test;yhat];
        y_true = [y_true;response(test_set)];

    end

    [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true, y_test);
    save(['.\results\lsim fusion\poolres_',num2str(channel_num),'ch_',model_name,'_',num2str(feature_sel),'.mat'],'kappa','acc',"y_true","y_test")


    disp(['poolres_',num2str(channel_num),'ch_',model_name])
    disp([acc,kappa,f1])

end



