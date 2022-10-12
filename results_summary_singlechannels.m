close all
clc
clear
addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets


model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint', 'x_jointfuse','seqfused'};% {'dgdss', 'tiny', 'seq', 'x_joint'};
channel_num = 2;
sleepedf_num = 20;


for k = 1:length(model_name_all)

    model_name = model_name_all{1,k};
    load(['output_',model_name,'.mat'])


    % single channel fusion
    for ch = 1:3

        y_test_org = [];
        y_true_org = [];
        for i = 1:CV_number

            response = true_label{ch,i};
            response = response(:);
            if sum(response==0)>0
                response = response+1;
            end
            this_fold_number = fold_number{1,i};
            test_set = this_fold_number==i;
            [~,yhat]= max( softmax( hingeloss_traintest{ch,i}(:,test_set)')');
            y_test_org = [y_test_org;yhat(:)];
            y_true_org = [y_true_org;response(test_set)];
        end

        [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true_org, y_test_org);
        acc_single(k,ch) = acc;
        f1_single(k,ch) = f1;
        kappa_single(k,ch) = kappa;

    end

    num_test(k,1)=length(response);
    %arithmatic mean fusion
    y_test_org = [];
    y_true_org = [];
    for i = 1:CV_number

        response = true_label{ch,i};
        response = response(:);
        if sum(response==0)>0
            response = response+1;
        end
        this_fold_number = fold_number{1,i};
        test_set = this_fold_number==i;
        if channel_num==2
            [~,yhat]= max(softmax( hingeloss_traintest{1,i}(:,test_set)')'+softmax( hingeloss_traintest{2,i}(:,test_set)')');
        else
            [~,yhat]= max(softmax( hingeloss_traintest{1,i}(:,test_set)')'+softmax( hingeloss_traintest{2,i}(:,test_set)')'+softmax( hingeloss_traintest{3,i}(:,test_set)')');
        end
        y_test_org = [y_test_org;yhat(:)];
        y_true_org = [y_true_org;response(test_set)];
    end

    [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true_org, y_test_org);

    acc_mean(k,1) = acc;
    f1_mean(k,1) = f1;
    kappa_mean(k,1) = kappa;

    %geomean fusion
    y_test_org = [];
    y_true_org = [];
    for i = 1:CV_number

        response = true_label{ch,i};
        response = response(:);
        if sum(response==0)>0
            response = response+1;
        end
        this_fold_number = fold_number{1,i};
        test_set = this_fold_number==i;
        if channel_num==2
            [~,yhat]= max(softmax( hingeloss_traintest{1,i}(:,test_set)')'.*softmax( hingeloss_traintest{2,i}(:,test_set)')');
        else
            [~,yhat]= max(softmax( hingeloss_traintest{1,i}(:,test_set)')'.*softmax( hingeloss_traintest{2,i}(:,test_set)')'.*softmax( hingeloss_traintest{3,i}(:,test_set)')');
        end
        y_test_org = [y_test_org;yhat(:)];
        y_true_org = [y_true_org;response(test_set)];
    end
    [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true_org, y_test_org);
    acc_mean(k,2) = acc;
    f1_mean(k,2) = f1;
    kappa_mean(k,2) = kappa;

end
acc_single