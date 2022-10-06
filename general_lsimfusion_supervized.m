
clc
clear

addpath(genpath(pwd))

mydir = pwd;
idcs = strfind(mydir,filesep);
% second parent folder contains the datasets

results_dir = [mydir(1:idcs(end-1)-1),'/Results/',mydir(idcs(end-1)+1:end)];
addpath([mydir(1:idcs(end-1)-1),'/lsim karimi toolbox'])
mkdir(results_dir)


model_name_all = {'dgdss', 'tiny', 'seq','x_joint','output_feembd'};% tiny, seq, dgdss, x_joint

channel_num = 3;
sleepedf_num = 20;


%% testing 2or3 -channel LSIM
for sigma_diag_loop =0:1

    for km = 1:length(model_name_all)

        model_name = model_name_all{km};

        load(['output_',model_name,'.mat'])

        CV_number = size(hingeloss_traintest,2);

        %% training 2or3-channel LSIM

        clear lsim_gmm_para_all transitions_matrices_all coupling_tetha_all pi_0_all AIC_all log_likelihood_all BIC_all
        clear lsim_hingeloss_traintest

        C = channel_num;


        for ch = 1:C
            y_est_org=[];
            for i = 1:CV_number
                this_fold_number = fold_number{1,i};
                counter = 0;
                for j=1:CV_number
                    this_set = this_fold_number==j;

                    if i==j
                        continue
                    end
                    counter = counter+1;

                    lsim_hingeloss_traintest{ch, i,counter} = hingeloss_traintest{ch,i}(:,this_set) ;
                    %             if ch>2
                    %                 lsim_hingeloss_traintest{ch, i,counter} = randn(size(lsim_hingeloss_traintest{ch, i,counter}));
                    %             end
                    temp_label = true_label{ch,i}(this_set)' +1;
                    channel_states{ch, i,counter} = temp_label(:)';
                end

            end

        end


        max_r = 1;
        max_itration = 100;
        extra.plot = 0;
        extra.check_convergence=0;
        extra.sigma_diag = sigma_diag_loop;
        sigma_diag = num2str(extra.sigma_diag);
        extra.sup_learn_flag =1;
        extra.auto_gmm = 1;
        num_gmm_component_grid = 6;

        counter = 0;


        for i = 1:CV_number
            close all
            clc
            counter = counter+1;
            disp(round(counter*100/(CV_number*max_r)))

            for ss = 1:length(num_gmm_component_grid)

                num_gmm_component = ones(1,C)*num_gmm_component_grid(ss);

                [pi_0_lsim , coupling_tetha_convex_comb , transition_matrices_convex_comb ,  lsim_gmm_para ,  AIC , log_likelihood , BIC ,pi_steady] = ...
                    lsim_supervised( squeeze(lsim_hingeloss_traintest(:, i, :)) , squeeze(channel_states(:, i, :)) , num_gmm_component , max_itration , extra);

                lsim_gmm_para_all{ss,i,1} =  lsim_gmm_para;
                transitions_matrices_all{ss,i,1} = transition_matrices_convex_comb;
                coupling_tetha_all{ss,i,1} = coupling_tetha_convex_comb;
                pi_0_all{ss,i,1} = pi_0_lsim ;
                AIC_all{ss,i,1} = AIC;
                log_likelihood_all{ss,i,1} =log_likelihood;
                BIC_all{ss,i,1} = BIC;
            end
            save(['slsim_',num2str(channel_num),'ch_',model_name,num2str(sigma_diag_loop),'.mat'],'lsim_gmm_para_all','transitions_matrices_all','coupling_tetha_all','pi_0_all','AIC_all','log_likelihood_all','BIC_all')

        end




        %% test lsim
        clear lsim_hingeloss_traintest

        for ch = 1:C
            for i = 1:CV_number
                this_fold_number = fold_number{1,i};
                total_epochs =0;
                for j=1:CV_number
                    this_set = this_fold_number==j;
                    total_epochs = total_epochs +sum(this_set);
                    lsim_hingeloss_traintest{ch, i,j}=hingeloss_traintest{ch,i}(:,this_set) ;
                end

            end

        end


        counter=0;
        for repeat_num = 1:max_r

            for i = 1:CV_number
                clc
                disp([num2str(channel_num),'-channel test'])
                counter = counter+1;

                disp(round(counter*100/(CV_number*max_r)))

                for ss = 1:length(num_gmm_component_grid)

                    lsim_gmm_para = lsim_gmm_para_all{ss,i,repeat_num} ;
                    transition_matrices_convex_comb = transitions_matrices_all{ss,i,repeat_num};
                    coupling_tetha_convex_comb = coupling_tetha_all{ss,i,repeat_num} ;
                    pi_0 = pi_0_all{ss,i,repeat_num};
                    lbss{ss,i,repeat_num}=[];
                    lbss_v{ss,i,repeat_num}=[];

                    for k=1:CV_number
                        %                         in_hmm = cell2mat(squeeze(lsim_hingeloss_traintest(:, i, k)));
                        %                         hmm_hingloss{1,1} = in_hmm;
                        %                         [pi_0_ehmm , coupling_tetha_ehmm ,  transition_ehmm  ,ehmm_gmm_para, index_matrix, pi_0_chmm ,  transition_chmm  , chmm_gmm_para] = im_para_eqhmm(pi_0, lsim_gmm_para, coupling_tetha_convex_comb, transition_matrices_convex_comb);
                        %                         [P_star_model , X_star]  = viterbi_chmm( pi_0_ehmm, coupling_tetha_ehmm,  transition_ehmm, ehmm_gmm_para, hmm_hingloss);
                        %
                        %                         predictors_viterbi = zeros(length(cell2mat(pi_0)),size(X_star,2));
                        %                         for t = 1:size(X_star,2)
                        %                             this_index = index_matrix(:,X_star(t));
                        %
                        %                             for c=1:channel_num
                        %                                 predictors_viterbi((c-1)*5+this_index(c),t) = 1;
                        %                             end
                        %                         end

                        [~ ,~ , ~ , alpha_T_all]  = forward_backward_lsim( pi_0 , coupling_tetha_convex_comb  , transition_matrices_convex_comb ,  lsim_gmm_para , squeeze(lsim_hingeloss_traintest(:, i,k)) );

                        predictors = log(cell2mat(alpha_T_all)+10^-7);

                        lbss{ss,i,repeat_num} = [lbss{ss,i,repeat_num},round(predictors,3)];
                        lbss_v{ss,i,repeat_num} =[ lbss_v{ss,i,repeat_num},  round(predictors,3)];
                    end
                end

            end

            save(['slbss_',num2str(channel_num),'ch_',model_name,num2str(sigma_diag_loop),'.mat'],'lbss','lbss_v')
        end



        %% testing 2or3 -channel LSIM

        load(['output_',model_name,'.mat'])
        load(['data_split_scratch_trainingchk_',num2str(sleepedf_num),'.mat'])


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

            clear acc_cv kappa_cv
            for repeat_num = 1:max_r
                for ss = 1:num_lsim
                    predictors = lbss{ss,i,repeat_num}';

                    features_index = 1:size(predictors,2);

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
                        [ acc_cv(k,ss), kappa_cv(k,ss)] = calculate_overall_metrics(response(eval_set), yhat);

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

            features_index = set_features(ss).features_index ;

            predictors = lbss{ss,i,repeat_num}';
            Mdl = fitcknn(predictors(train_all_set, features_index), response(train_all_set, :),'NumNeighbors',5*k,'Distance','euclidean','NSMethod','exhaustive');
            yhat = predict(Mdl, predictors(test_set,:));

            y_test = [y_test;yhat];
            y_true = [y_true;response(test_set)];

        end


        [acc, kappa , f1] = calculate_overall_metrics(y_true, y_test);

        disp(['lsim_res_',num2str(channel_num),'ch_',model_name])
        disp([acc,kappa,f1])


        % LSIM test on best likelihood model


        max_r = size(lbss,3);
        num_lsim = size(lbss,1);
        CV_number = size(lbss,2);

        y_test2 = [];
        y_true2 = [];
        y_test3 = [];
        y_true3 = [];
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

            [fea, score] = mRMR(predictors(train_all_set, :), response(train_all_set, :), 10);

            clear kappa_cv acc_cv
            for j=1:5
                features_index = fea(1:j*2);
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
                    [ acc_cv(k,j), kappa_cv(k,j)] = calculate_overall_metrics(response(eval_set), yhat);

                end

            end

            [CC,I2] = max(mean(acc_cv));
            [CC,I1] = max(mean(acc_cv'));

            k = I1;
            j = I2;


            para_best2(i,1) = 1;
            para_best2(i,2) = j;
            para_best2(i,3) = k;

            features_index = set_features(j).features_index ;

            Mdl = fitcknn(predictors(train_all_set, features_index), response(train_all_set, :),'NumNeighbors',5*k,'Distance','euclidean','NSMethod','exhaustive');
            yhat = predict(Mdl, predictors(test_set,features_index));

            y_test2 = [y_test2;yhat];
            y_true2 = [y_true2;response(test_set)];

            Mdl = fitcknn(predictors(train_all_set, :), response(train_all_set, :),'NumNeighbors',5*k,'Distance','euclidean','NSMethod','exhaustive');
            yhat = predict(Mdl, predictors(test_set,:));

            y_test3 = [y_test3;yhat];
            y_true3 = [y_true3;response(test_set)];

        end

        [acc2, kappa2 , f12, sens, spec] = calculate_overall_metrics(y_true2, y_test2);
        [acc3, kappa3 , f13, sens, spec] = calculate_overall_metrics(y_true3, y_test3);
        save(['sres_',num2str(channel_num),'ch_',model_name,'_',num2str(sigma_diag_loop),'.mat'],'kappa','acc','f1','acc2', 'kappa2' , 'f12','acc3', 'kappa3' , 'f13',"y_true","y_test","para_best","y_true2","y_test2","para_best2","y_true3","y_test3")


        disp(['sres_',num2str(channel_num),'ch_',model_name])
        disp([acc2,kappa2,f12])

    end

end
