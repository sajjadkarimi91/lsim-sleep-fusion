
clear
close all

close all
clc
clear
addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets

results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];
addpath([mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'])
mkdir(results_dir)

model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint'};
model_name_all = { 'x_joint'};
channel_num = 3;
sleepedf_num = 20;
feature_sel = 1;
load(['data_split_scratch_trainingchk_',num2str(sleepedf_num),'.mat'])

for km = 1:length(model_name_all)

    %% testing 2or3 -channel LSIM
    model_name = model_name_all{km};
    load(['output_',model_name,'.mat'])
    load(['flbss_',num2str(channel_num),'ch_',model_name,'.mat'])


    %% LSIM test on best likelihood model


    max_r = size(lbss,3);
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

        [fea, score] = mRMR(predictors(train_all_set, :), response(train_all_set, :), 100);

        clear kappa_cv
        for j=1:12
            features_index = fea(1:j*5);
            set_features(j).features_index = features_index;

            for k = 1:10

                Mdl = fitcknn(predictors(train_set, features_index), response(train_set, :),'NumNeighbors',5*k,'Distance','euclidean','NSMethod','exhaustive');
                yhat = predict(Mdl, predictors(eval_set, features_index));
                % a correction for kappa computation
                if length(unique(yhat)) < length(unique(response))
                    d_set = setdiff(unique(response),unique(yhat));
                    ind_rand = randperm(length(yhat));
                    yhat(ind_rand(1:length(d_set))) = d_set;
                end
                [ acc_cv(k,j), kapp, f1, sens, spec] = calculate_overall_metrics(response(eval_set), yhat);

            end

        end
        M = acc_cv;
        %         [CC,I] = max(M(:));
        %         [I1,I2,I3] = ind2sub(size(M),I);

        [CC,I2] = max(mean(acc_cv));
        [CC,I1] = max(mean(acc_cv'));

        k = I1;
        j = I2;


        para_best(i,1) = 1;
        para_best(i,2) = j;
        para_best(i,3) = k;

        features_index = set_features(j).features_index ;

        Mdl = fitcknn(predictors(train_all_set, features_index), response(train_all_set, :),'NumNeighbors',5*k,'Distance','euclidean','NSMethod','exhaustive');
        yhat = predict(Mdl, predictors(test_set,features_index));

        y_test = [y_test;yhat];
        y_true = [y_true;response(test_set)];

    end

    [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true, y_test);
    save(['pres_',num2str(channel_num),'ch_',model_name,'_',num2str(feature_sel),'.mat'],'kappa','acc',"y_true","y_test","para_best")


    disp(['pres_',num2str(channel_num),'ch_',model_name])
    disp([acc,kappa,f1])

end



