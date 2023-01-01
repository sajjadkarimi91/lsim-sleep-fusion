close all
clc
clear
addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets

model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint'};
type_fusion_all = {'', 'pool'};
channel_num = 3;
sleepedf_num = 20;


for k = 1:length(model_name_all)

    model_name = model_name_all{1,k};
    load(['output_',model_name,'.mat'])

    for i = 1:length(type_fusion_all)
        load([type_fusion_all{i},'res_',num2str(channel_num),'ch_',model_name,'.mat'],"y_true","y_test")
        [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true, y_test);
        acc_lsim(k,i) = acc;
        f1_lsim(k,i) = f1;
        kappa_lsim(k,i) = kappa;
    end

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


%% plot confusion matrix

model_name_all = {'tiny'};
type_fusion_all = {'pool'};
channel_num = 3;
sleepedf_num = 20;

for k = 1:length(model_name_all)

    model_name = model_name_all{1,k};
    load(['output_',model_name,'.mat'])

    for i = 1:length(type_fusion_all)
        load([type_fusion_all{i},'res_',num2str(channel_num),'ch_',model_name,'.mat'],"y_true","y_test")
        [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true, y_test);
    end

end

plotconfusion(categorical(y_true),categorical(y_test))

%% plot hypnogram
close all
clear y_test_singles

model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint'};
type_fusion_all = {'pool'};
channel_num = 3;
sleepedf_num = 20;


for k = 1:length(model_name_all)

    model_name = model_name_all{1,k};
    load(['output_',model_name,'.mat'])
    if k==2
        load([type_fusion_all{1},'res_',num2str(channel_num),'ch_',model_name,'.mat'],"y_true","y_test")
        y_test_model = y_test(1:10000)';
    end
    % single channel fusion
    ch = 1;
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

    y_test_org_copy = y_test_org;
    if k==1
        y_test_org(y_test_org_copy==5)=4;
        y_test_org(y_test_org_copy==4)=3;
        y_test_org(y_test_org_copy==3)=2;
        y_test_org(y_test_org_copy==2)=5;
    end

    y_test_singles(k,1:10000) = y_test_org(1:10000)';

end

plot(y_test_singles(:,1971:2470)')
hold on
grid on
plot(y_test_model(1971:2470),'b','LineWidth',2.5)
plot(y_true(1971:2470),'m','LineWidth',2)
ylim([0.5,5.5])

legend({'DGDSS','TinySleepNet','SeqSleepNet', 'XSleepNet', 'LSIM fusion', 'True stages'},'FontSize',13,'Interpreter' ,'latex')

set(gca, 'FontWeight','bold','FontSize',11,'Ytick', [1:5],'YTickLabel',{'W','N1','N2','N3','REM'});
xlabel('Samples' ,'FontSize',14,'Interpreter' ,'latex' )
ylabel('Hypnogram' ,'FontSize',14,'Interpreter' ,'latex' )


