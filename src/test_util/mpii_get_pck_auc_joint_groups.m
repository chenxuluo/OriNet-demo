function [joint_groups] = mpii_get_pck_auc_joint_groups()

joint_groups = { %'Head', [1,17];
                'Head', [1];
                 'Neck', [2];
                 'Shou', [3,6];
                 'Elbow', [4,7];
                'Wrist', [5,8];
                 %'spine', [16];
                'Hip', [9,12];
                 'Knee', [10,13];
                 'Ankle', [11,14];
                };
%joint_groups = { 
%                 'Head', [11];
%                 'Neck', [9];
%                 'Shou', [12,15];
 %                'Elbow', [13,16];
 %                'Wrist', [14,17];
 %                'Hip', [2,5];
 %                'Knee', [3,6];
 %                'Ankle', [4,7];
 %                };
end