close all; clc; clear;
addpath(genpath(pwd))

%% path setting

lbss_save_dir = './results/lbss features'; % A folder path for saving results
model_save_dir = './results/lsim models';
lsim_path = './chmm-lsim-matlab-toolbox';  % download from https://github.com/sajjadkarimi91/chmm-lsim-matlab-toolbox

addpath(lsim_path)
mkdir(lbss_save_dir)
mkdir(model_save_dir)

%% model config

model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint'};% tiny, seq, dgdss, x_joint
channel_num = 2;
sleepedf_num = 20;
force_train = 0; % if force_train =1 then LSIM models always are retrained that needs more than 24 hours time
preprocess_apply = 0;

%% train & test 2or3 -channel LSIM

for km = 1:length(model_name_all)

    model_name = model_name_all{km};

    load(['output_',model_name,'.mat'])

    CV_number = size(hingeloss_traintest,2);

    %% simple EDA

    y_test_org = [];
    y_est_org = [];
    y_true_org = [];
    ch = 1;

    for i = 1:CV_number

        response = true_label{ch,i};
        response = response(:);
        if sum(response==0)>0
            response = response+1;
        end
        this_fold_number = fold_number{1,i};
        test_set = this_fold_number==i;
        [~,yhat]= max( softmax( hingeloss_traintest{ch,i}(:,test_set)')');
        y_est_org = [y_est_org;  hingeloss_traintest{ch,i}(:,test_set)'];
        y_test_org = [y_test_org;yhat(:)];
        y_true_org = [y_true_org;response(test_set)];
    end

    [acc, kappa , f1, sens, spec] = calculate_overall_metrics(y_true_org, y_test_org);

    y_est =y_est_org;
    close all
    figure
    plot(hingeloss_traintest{1,15}')
    hold on
    max_val = max(hingeloss_traintest{1,10}(:));
    min_val = min(hingeloss_traintest{1,10}(:));
    plot((max_val-min_val)*this_fold_number/20+min_val,'k','LineWidth',1.5)

    for ch = 1:3
        y_est_org=[];
        for i = 1:CV_number


            this_fold_number = fold_number{1,i};
            for j=1:CV_number
                test_set = this_fold_number==j;
                this_sub = hingeloss_traintest{ch,i}(:,test_set);
                this_min = quantile(this_sub(:),0.05);
                this_max =quantile(this_sub(:),0.95);
                if preprocess_apply>0
                    hingeloss_traintest{ch,i}(:,test_set) = (hingeloss_traintest{ch,i}(:,test_set)-this_min)/(this_max-this_min);
                    hingeloss_traintest{ch,i}(:,test_set) = hingeloss_traintest{ch,i}(:,test_set)-max(hingeloss_traintest{ch,i}(:,test_set));
                end
                if j==i
                    y_est_org = [y_est_org;  hingeloss_traintest{ch,i}(:,test_set)'];
                end
            end

        end

    end

    figure
    plot(hingeloss_traintest{1,10}')
    hold on
    max_val = max(hingeloss_traintest{1,10}(:));
    min_val = min(hingeloss_traintest{1,10}(:));
    plot((max_val-min_val)*this_fold_number/20+min_val,'k','LineWidth',1.5)

    figure
    as(1)=subplot(2,1,1);
    plot(y_est_org)
    hold on
    plot((max_val-min_val)*this_fold_number/20+min_val,'k','LineWidth',1.5)
    as(2)=subplot(2,1,2);
    plot(y_est)
    linkaxes(as,'x')

    %% training 2or3-channel LSIM
    close all
    C = channel_num;

    state_numbers_grid = [5,10,15,20,25]; % number of states

    num_gmm_component_grid = [1*ones(1,length(state_numbers_grid)),2*ones(1,length(state_numbers_grid))]; % number of GMMs
    state_numbers_grid = [state_numbers_grid,state_numbers_grid];

    max_r = 1;

    if ~exist([model_save_dir,'/lsim_',num2str(channel_num),'ch_',model_name,'.mat'],'file') || force_train == 1

        clear lsim_gmm_para_all transitions_matrices_all coupling_tetha_all pi_0_all AIC_all log_likelihood_all BIC_all
        clear lsim_hingeloss_traintest


        if C==2 && sleepedf_num==20
            for i = 1:CV_number
                for ch=1:C
                    %to prevent out of memory error, data is splitted in to smaller groups
                    t_half = round(size(hingeloss_traintest{ch,i},2)/2);
                    lsim_hingeloss_traintest{ch, i,1}= hingeloss_traintest{ch,i}(:,1:t_half);
                    lsim_hingeloss_traintest{ch, i,2}= hingeloss_traintest{ch,i}(:,t_half:end);
                end
            end
        elseif C==3 && sleepedf_num==20
            for i = 1:CV_number
                for ch=1:C
                    %to prevent out of memory error, data is splitted in to smaller groups
                    t_third = round(size(hingeloss_traintest{ch,i},2)/3);
                    lsim_hingeloss_traintest{ch, i,1}= hingeloss_traintest{ch,i}(:,1:t_third);
                    lsim_hingeloss_traintest{ch, i,2}= hingeloss_traintest{ch,i}(:,t_third:2*t_third);
                    lsim_hingeloss_traintest{ch, i,3}= hingeloss_traintest{ch,i}(:,2*t_third:end);
                end
            end
        end


        max_itration = 100;
        extra.plot = 0;
        extra.check_convergence=0;
        extra.sigma_diag = 1;


        counter = 0;

        for repeat_num = 1:max_r

            for i = 1:CV_number
                close all
                clc
                counter = counter+1;
                disp(round(counter*100/(CV_number*max_r)))

                parfor ss = 1:length(state_numbers_grid)

                    channel_num_states = ones(1,C)*state_numbers_grid(ss);
                    num_gmm_component = ones(1,C)*num_gmm_component_grid(ss);

                    [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                        em_lsim( squeeze(lsim_hingeloss_traintest(:, i, :)) , channel_num_states , num_gmm_component , max_itration , extra);

                    lsim_gmm_para_all{ss,i,repeat_num} =  lsim_gmm_para;
                    transitions_matrices_all{ss,i,repeat_num} = transition_matrices_convex_comb;
                    coupling_tetha_all{ss,i,repeat_num} = coupling_tetha_convex_comb;
                    pi_0_all{ss,i,repeat_num} = pi_steady ;
                    AIC_all{ss,i,repeat_num} = AIC;
                    log_likelihood_all{ss,i,repeat_num} =log_likelihood;
                    BIC_all{ss,i,repeat_num} = BIC;
                end
                save([model_save_dir,'lsim_',num2str(channel_num),'ch_',model_name,'.mat'],'lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all')

            end

            save([model_save_dir,'/lsim_',num2str(channel_num),'ch_',model_name,'.mat'],'lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all')
        end

    end

    %% test lsim

    load([model_save_dir,'/lsim_',num2str(channel_num),'ch_',model_name,'.mat'])

    clear lsim_hingeloss_traintest
    for i = 1:CV_number
        for ch=1:C
            lsim_hingeloss_traintest{ch, i}= hingeloss_traintest{ch,i};
        end
    end

    counter=0;
    for repeat_num = 1:max_r

        for i = 1:CV_number
            clc
            disp([num2str(channel_num),'-channel test'])
            counter = counter+1;

            disp(round(counter*100/(CV_number*max_r)))

            for ss = 1:length(state_numbers_grid)

                lsim_gmm_para = lsim_gmm_para_all{ss,i,repeat_num} ;
                transition_matrices_convex_comb = transitions_matrices_all{ss,i,repeat_num};
                coupling_tetha_convex_comb = coupling_tetha_all{ss,i,repeat_num} ;
                pi_0 = pi_0_all{ss,i,repeat_num};

                [~ ,~ , ~ , alpha_T_all]  = forward_backward_lsim( pi_0 , coupling_tetha_convex_comb  , transition_matrices_convex_comb ,  lsim_gmm_para , lsim_hingeloss_traintest(:, i) );

                predictors = log(cell2mat(alpha_T_all)+10^-7);

                lbss{ss,i,repeat_num} = round(predictors,3);

            end

        end

        save([lbss_save_dir,'/flbss_',num2str(channel_num),'ch_',model_name,'.mat'],'lbss')
    end

end


