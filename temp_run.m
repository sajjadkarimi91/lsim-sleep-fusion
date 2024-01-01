
% git remote add chmm-lsim-matlab-toolbox https://github.com/sajjadkarimi91/chmm-lsim-matlab-toolbox
% git subtree add --prefix=chmm-lsim-matlab-toolbox/ chmm-lsim-matlab-toolbox main --squash  

model_name_all = {'dgdss', 'tiny', 'seq', 'x_joint'};

for km = 1:length(model_name_all)

    % testing 2or3 -channel LSIM
    model_name = model_name_all{km};
    load(['output_',model_name,'.mat'])
    for i = 1:size(hingeloss_traintest,1)
        for j = 1:size(hingeloss_traintest,2)
            hingeloss_traintest{i,j} = round(hingeloss_traintest{i,j},6);
        end
    end
    save(['output_',model_name,'.mat'],'hingeloss_traintest','fold_number','CV_number','true_label')
end



load('res_2ch_seq.mat', 'para_best')
x3 = para_best(:,3);
load('res_2ch_dgdss.mat', 'para_best')
x1 = para_best(:,3);
load('res_2ch_tiny.mat', 'para_best')
x2 = para_best(:,3);
load('res_2ch_x_joint_0.mat', 'para_best')
x4 = para_best(:,3);
yy = [x1;x2;x3;x4];


state_nums = xx;
state_nums(state_nums>5) = state_nums(state_nums>5)-5;
histogram(state_nums*5)
grid on
xlabel('M')

histogram(yy*5)
grid on
xlabel('K')

load('res_3ch_seq.mat', 'para_best')
x3 = para_best(:,3);
load('res_3ch_dgdss.mat', 'para_best')
x1 = para_best(:,3);
load('res_3ch_tiny.mat', 'para_best')
x2 = para_best(:,3);
load('res_3ch_x_joint_0.mat', 'para_best')
x4 = para_best(:,3);
yy3 = [x1;x2;x3;x4];

load('res_3ch_seq.mat', 'para_best')
x3 = para_best(:,2);
load('res_3ch_dgdss.mat', 'para_best')
x1 = para_best(:,2);
load('res_3ch_tiny.mat', 'para_best')
x2 = para_best(:,2);
load('res_3ch_x_joint_0.mat', 'para_best')
x4 = para_best(:,2);
xx3 = [x1;x2;x3;x4];

histogram(yy3*5)


state_nums = xx3;
state_nums(state_nums>5) = state_nums(state_nums>5)-5;
histogram(state_nums*5)
grid on
xlabel('M')
