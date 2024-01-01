close all; clc; clear;
addpath(genpath(pwd))

%% path configs
mydir = pwd;
idcs = strfind(mydir,filesep);

results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)]; % your folder path for saving results
lsim_path = [mydir(1:idcs(end-1)-1),'/chmm-lsim-matlab-toolbox'];  % download from https://github.com/sajjadkarimi91/chmm-lsim-matlab-toolbox

addpath(lsim_path)
mkdir(results_dir)

%% model config

model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint'};
channel_num = 3; % can be 2 or 3 for channel fusion

sleepedf_num = 20;
feature_sel = 0; % for paper
load(['data_split_scratch_trainingchk_',num2str(sleepedf_num),'.mat'])

for km = 1:length(model_name_all)

    % testing 2or3 -channel LSIM
    model_name = model_name_all{km};
    load(['output_',model_name,'.mat'])
    load(['flbss_',num2str(channel_num),'ch_',model_name,'.mat'])

    % LSIM test on best likelihood model
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

        clear acc_cv kappa_cv 
        for repeat_num = 1:max_r
            for ss = 1:num_lsim
                predictors = lbss{ss,i,repeat_num}';

                if feature_sel > 0
                    [fea, score] = mRMR(predictors(train_set, :), response(train_set, :), size(predictors,2));

                    score(1)=max(score);
                    score_sort = sort(score,'descend') ;

                    ind_select = find(score_sort<score_sort(1)*0.25,1,"first");
                    if isempty(ind_select)
                        ind_select = length(fea);
                    end

                    features_index = fea(1:ind_select);
                else
                    features_index = 1:size(predictors,2);
                end
                set_features(repeat_num,ss).features_index = features_index;

                for k = 1:10


                    Mdl = fitcknn(predictors(train_set, :), response(train_set, :),'NumNeighbors',5*k,'Distance','euclidean','NSMethod','exhaustive');
                    yhat = predict(Mdl, predictors(eval_set, :));
                    % a correction for kappa computation
                    if length(unique(yhat)) < length(unique(response))
                        d_set = setdiff(unique(response),unique(yhat));
                        ind_rand = randperm(length(yhat));
                        yhat(ind_rand(1:length(d_set))) = d_set;
                    end
                    [ acc_cv(k,ss), kappa, f1, sens, spec] = calculate_overall_metrics(response(eval_set), yhat);

                end
            end
        end

        M = acc_cv;
        [CC,I2] = max(mean(acc_cv));
        [CC,I1] = max(mean(acc_cv'));
        %         [I1,I2,I3] = ind2sub(size(M),I);

        ss = I2;
        k = I1;

        para_best(i,1) = 1;
        para_best(i,2) = ss;
        para_best(i,3) = k;

        features_index = set_features(repeat_num,ss).features_index ;

        predictors = lbss{ss,i,repeat_num}';
        Mdl = fitcknn(predictors(train_all_set, features_index), response(train_all_set, :),'NumNeighbors',5*k,'Distance','euclidean','NSMethod','exhaustive');
        yhat = predict(Mdl, predictors(test_set,:));

        y_test = [y_test;yhat];
        y_true = [y_true;response(test_set)];

    end

    [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true, y_test);
    save(['.\results\lsim fusion\res_',num2str(channel_num),'ch_',model_name,'_',num2str(feature_sel),'.mat'],'kappa','acc',"y_true","y_test","para_best")


    disp(['res_',num2str(channel_num),'ch_',model_name])
    disp([acc,kappa,f1])

end



