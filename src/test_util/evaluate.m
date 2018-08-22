
test_subject_id = [1,2,3,4,5,6];


sequencewise_activity_labels = cell(6,1);
for i = 1:length(test_subject_id)
   dat = load(['../../../mpi_inf_3dhp/test/TS' int2str(test_subject_id(i)) filesep 'annot_data.mat']);
   sequencewise_activity_labels{i} = dat.activity_annotation(dat.valid_frame == 1);
end


%%
index = [11     9    15    16    17    12    13    14     2     3     4     5     6     7     1     8    10];  %reorder the joint
err = importdata('../Result.txt');
err = err';
err = err(index,:);
err = reshape(err,[17,1,size(err,2)]);
error{1} = err(:,:,1:1143);
% error{2} = err(:,:,1144:2201);
% error{3} = err(:,:,2202:2929);
error{2} = err(:,:,1144:2207);
error{3} = err(:,:,2208:2935);

[sequencewise_table, activitywise_table] = mpii_evaluate_errors(error,sequencewise_activity_labels);
