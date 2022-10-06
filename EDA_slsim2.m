
clc
clear
close all

addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets

results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];
addpath([mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'])
mkdir(results_dir)


model_name_all = {'dgdss', 'tiny', 'x_joint'};% tiny, seq, dgdss, x_joint
model_name = 'x_joint';
channel_num = 3;
sleepedf_num = 20;
preprocess_apply = 0;
feature_sel =0;

%% testing 2or3 -channel LSIM
sigma_diag_loop = 1;



load(['output_',model_name,'.mat'])

CV_number = size(hingeloss_traintest,2);

%% training 2or3-channel LSIM

clear lsim_gmm_para_all transitions_matrices_all coupling_tetha_all pi_0_all AIC_all log_likelihood_all BIC_all
clear lsim_hingeloss_traintest

C = channel_num;


for ch = 1:C
    y_est_org=[];
    for fold_num = 1:CV_number
        this_fold_number = fold_number{1,fold_num};
        counter = 0;
        for j=1:CV_number
            this_set = this_fold_number==j;

            if fold_num==j
                continue
            end
            counter = counter+1;
            temp_label = true_label{ch,fold_num}(this_set)' ;
            temp_label = temp_label(:)';

            lsim_hingeloss_traintest{ch, fold_num,counter} = hingeloss_traintest{ch,fold_num}(:,this_set) +ch*0* randn(size( hingeloss_traintest{ch,fold_num}(:,this_set))) ;
            %             lsim_hingeloss_traintest{ch, fold_num,counter} = temp_label +2*(ch)* randn(size(temp_label)) ;

            if ch>2
                lsim_hingeloss_traintest{ch, fold_num,counter} = randn(size(lsim_hingeloss_traintest{ch, fold_num,counter}));
            end

            lsim_hingeloss_traintest{C+ch, fold_num,counter} =[lsim_hingeloss_traintest{ch, fold_num,counter}(:,2:end),lsim_hingeloss_traintest{ch, fold_num,counter}(:,end)];

            %             temp_label = [temp_label(1),temp_label(1:end-1)];

            channel_states{ch, fold_num,counter} = temp_label;
            channel_states{C+ch, fold_num,counter} = [temp_label(2:end),temp_label(end)];
        end

    end

end


max_r = 1;
max_itration = 200;
extra.plot = 1;
extra.check_convergence=0;
extra.sigma_diag = sigma_diag_loop;
sigma_diag = num2str(extra.sigma_diag);
extra.sup_learn_flag =1;

num_gmm_component_grid = 3;
extra.auto_gmm = 1;
counter = 0;


fold_num = 1;
close all
clc
counter = counter+1;
disp(round(counter*100/(CV_number*max_r)))

ss = 1;

num_gmm_component = ones(1,2*C)*num_gmm_component_grid(ss);

[pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady,coupling_all, kl_dist, acc] = ...
    lsim_supervised_fast_embed( squeeze(lsim_hingeloss_traintest(:, fold_num, :)) , squeeze(channel_states(:, fold_num, :)) , num_gmm_component , max_itration , extra);

lsim_gmm_para_all{ss,fold_num} =  lsim_gmm_para;
transitions_matrices_all{ss,fold_num} = transition_matrices_convex_comb;
coupling_tetha_all{ss,fold_num} = coupling_tetha_convex_comb;
pi_0_all{ss,fold_num} = pi_0_lsim ;
AIC_all{ss,fold_num} = acc;
log_likelihood_all{ss,fold_num} = log_likelihood;
BIC_all{ss,fold_num} = kl_dist;

close all

subplot(3,1,1)
plot(log_likelihood,'LineWidth',2)
grid on
title('loglik')

subplot(3,1,2)
plot(acc,'LineWidth',2)
grid on
title('ACC')

subplot(3,1,3)
plot(kl_dist,'LineWidth',2)
grid on
title('KL')
%% test lsim
clear lsim_hingeloss_traintest

for ch = 1:C
    for i = 1:CV_number
        this_fold_number = fold_number{1,i};
        total_epochs =0;
        for j=1:CV_number
            this_set = this_fold_number==j;
            total_epochs = total_epochs +sum(this_set);
            temp_label = true_label{ch,fold_num}(this_set)' ;
            temp_label = temp_label(:)';

            lsim_hingeloss_traintest{ch, i,j}=hingeloss_traintest{ch,i}(:,this_set) +ch*0* randn(size( hingeloss_traintest{ch,fold_num}(:,this_set))) ;
            %             lsim_hingeloss_traintest{ch, i,j} = temp_label +2*(ch)* randn(size(temp_label)) ;
            lsim_hingeloss_traintest{C+ch, fold_num,counter} =[lsim_hingeloss_traintest{ch, fold_num,counter}(:,2:end),lsim_hingeloss_traintest{ch, fold_num,counter}(:,end)];

            %             temp_label = [temp_label(1),temp_label(1:end-1)];
            channel_states{ch, fold_num,j} = temp_label;
            channel_states{C+ch, fold_num,counter} = [temp_label(2:end),temp_label(end)];

        end

    end

end



repeat_num = 1;
lsim_gmm_para = lsim_gmm_para_all{ss,fold_num} ;
transition_matrices_convex_comb = transitions_matrices_all{ss,fold_num};
coupling_tetha_convex_comb = coupling_tetha_all{ss,fold_num} ;
pi_0 = pi_0_all{ss,fold_num,repeat_num};
lbss{ss,fold_num,repeat_num}=[];
lbss_v{ss,fold_num,repeat_num}=[];

k=1;

[~ ,alpha_all, ~ , alpha_T_all]  = forward_backward_lsim( pi_0 , coupling_tetha_convex_comb  , transition_matrices_convex_comb ,  lsim_gmm_para , squeeze(lsim_hingeloss_traintest(:, fold_num,k)) );

figure
plot(alpha_all{1, 1}(1,:))
hold on
plot(alpha_T_all{1, 1}(1,:))
plot(channel_states{1, fold_num,k}==1)
% plot(lsim_hingeloss_traintest{1, fold_num,k}(1,:)+2)
plot(alpha_T_all{3, 1}(1,:))

predictors = log(cell2mat(alpha_T_all)+10^-7);
lbss{ss,fold_num,repeat_num} = [lbss{ss,fold_num,repeat_num},round(predictors,3)];





