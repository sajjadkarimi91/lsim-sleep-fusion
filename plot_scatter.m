%% metrics

close all
clear channels_observations
COM_new = [];
for i = 1:20
    
    COM_new = [COM_new;[validationScores_all_1{1,i},validationScores_all_2{1,i}]];
    
end

mfcc_time_series_eeg = COM_new';

TT = size(mfcc_time_series_eeg,2);


train_index = 1:TT;

num_CC1 = size(mfcc_time_series_eeg,1)/2;
for c=1:2
    channels_observations{c,1}= mfcc_time_series_eeg( (c-1)*num_CC1+1 : c*num_CC1,train_index);
end



clear Log Model_t Model_rep BIC_S Model_lsim


state_numbers_all = 3;
num_gmm_component_all = 3*ones(1,length(state_numbers_all));

for s_search = 1: length(state_numbers_all)
    
    
    'lsim'
    s_search
    close all
    max_itration = 200;
    channel_num_states(1:size(channels_observations,1)) = state_numbers_all(s_search);
    num_gmm_component(1:size(channels_observations,1)) = num_gmm_component_all(s_search);
    extra.plot = 1;
    extra.check_convergence=0;
    
    parfor replicate_number = 1:4
        
        
        [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
            em_lsim( channels_observations , channel_num_states , num_gmm_component , max_itration , extra);
        
        Model_rep{replicate_number}.CHMM_GMM_Param = lsim_gmm_para;
        Model_rep{replicate_number}.Transitions_matrices = transition_matrices_convex_comb;
        Model_rep{replicate_number}.Coupling_Tetha = coupling_tetha_convex_comb;
        Model_rep{replicate_number}.PI_0 = pi_steady;
        Log(replicate_number) = log_likelihood(end);
        BIC_Rep(replicate_number) =  BIC(end);
        AIC_Rep(replicate_number) =  AIC(end);
        
        
    end
    
    [~,Index_max] = max(Log);
    Model_lsim{s_search}.CHMM_GMM_Param =    Model_rep{Index_max}.CHMM_GMM_Param ;
    Model_lsim{s_search}.Transitions_matrices = Model_rep{Index_max}.Transitions_matrices;
    Model_lsim{s_search}.Coupling_Tetha = Model_rep{Index_max}.Coupling_Tetha;
    Model_lsim{s_search}.PI_0 = Model_rep{Index_max}.PI_0 ;
    BIC_lsim_S(s_search) = BIC_Rep(Index_max);
    AIC_lsim_S(s_search) = AIC_Rep(Index_max);
    
end


[~,Index_min] = min(BIC_lsim_S);

%%


Index_min = 1;

lsim_gmm_para_1 =  Model_lsim{Index_min}.CHMM_GMM_Param ;
transitions_matrices_1 = Model_lsim{Index_min}.Transitions_matrices;
coupling_tetha_1 = Model_lsim{Index_min}.Coupling_Tetha;
pi_0_1 = Model_lsim{Index_min}.PI_0 ;

%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%---- train policy ----%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

train_index = 1:TT;


for c=1:length(sel_elec)
    channels_observations{c,1}= mfcc_time_series_eeg( (c-1)*num_CC1+1 : c*num_CC1,train_index);
end

[~ ,~ , ~ , alpha_T_all]  = forward_backward_lsim( pi_0_1 , coupling_tetha_1  , transitions_matrices_1 ,  lsim_gmm_para_1 , channels_observations );

if knn_type == 0
    predictors = cell2mat(alpha_T_all)+10^-7;
    %predictors = predictors(21:end,:);
    predictors = predictors./sum(predictors);
    predictors = predictors';
else
    predictors = log(cell2mat(alpha_T_all)+10^-7);
    predictors = predictors';
end

%%
close all

% tempGT(find(annotation=='M'))=0;
% tempGT(find(annotation=='W'))=1;
% tempGT(find(annotation=='R'))=2;
% tempGT(find(annotation=='1'))=3;
% tempGT(find(annotation=='2'))=4;
% tempGT(find(annotation=='3'))=5;
% tempGT(find(annotation=='4'))=5;

color_set = {'gs','rx','ko','b.','m<'};
color_set = {'s','x','o','.','<'};

figure
for cls =1:5
    temp1 = 0.2*randn(sum(cls==response),1)+predictors(cls==response, 1);
    temp2 = 0.2*randn(sum(cls==response),1)+predictors(cls==response, 2);
    temp3 = 0.2*randn(sum(cls==response),1)+predictors(cls==response, 3);
    
    temp1(temp1>0)=0;
    temp2(temp2>0)=0;
    temp3(temp3>0)=0;
    
    scatter3(temp1,temp2,temp3,color_set{cls});
    hold on
    grid on
    
end

title('Channel 1: Fpz-Cz','FontSize',15,'Interpreter' ,'latex')
legend({'W','R','N1','N2','N3'},'FontSize',12,'Interpreter' ,'latex')
set(gca, 'FontWeight','bold','FontSize',11);
set(gcf,'renderer','painters')

figure
for cls =1:5
    
    temp1 = 0.2*randn(sum(cls==response),1)+predictors(cls==response, 4);
    temp2 = 0.2*randn(sum(cls==response),1)+predictors(cls==response, 5);
    temp3 = 0.2*randn(sum(cls==response),1)+predictors(cls==response, 6);
    
    temp1(temp1>0)=0;
    temp2(temp2>0)=0;
    temp3(temp3>0)=0;
    
    scatter3(temp1,temp2,temp3,color_set{cls});
    hold on
    grid on
    
end

% title('Channel 2')
title('Channel 2: Pz-Oz','FontSize',15,'Interpreter' ,'latex')
legend({'W','R','N1','N2','N3'},'FontSize',12,'Interpreter' ,'latex')
set(gca, 'FontWeight','bold','FontSize',11);
set(gcf,'renderer','painters')