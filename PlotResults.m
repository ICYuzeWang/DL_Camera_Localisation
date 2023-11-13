clear
clc
close all

CLIndex2read = [0,2,5,8,33];
% state:[FrameNo,x,y,z,Fx,Fy,Fz,Vx,Vy,Vz,theta,CLid,CLIndex]  not sure
% maybe force and velocity or faces vertices?
load('Airway_Phantom_Adjust.mat');
figure,
patch('Faces',tri,'Vertices',ver,'facecolor',[1 0 0],'facealpha',0.3,'edgecolor','none');
hold on;


poseXYZ=cell(length(CLIndex2read),1);
w=0;
for CLIndex = CLIndex2read
    w=w+1;
    itersize=length(dir("CTCLDepthfiles/depthmapsCL"+string(CLIndex)+"Tangent/*.pose.txt"));
    for i=1:itersize
        poseMat=readmatrix("CTCLDepthfiles/depthmapsCL"+string(CLIndex)+"Tangent/frame-"+pad(string(i-1),6,'left','0')+".pose.txt");
        poseXYZ{w}=[poseXYZ{w}, poseMat(1:3,4)];
    end
end

for w=1:length(CLIndex2read)
    pose=poseXYZ{w}';
    plot3(pose(:,1),pose(:,2),pose(:,3),'.'); drawnow;
end

%% plot EM as well%%
% load('EM9dataReg.mat');
% EMxyz=tformd_pts;
% EMxyz = EMxyz(all((EMxyz>-1000000 & EMxyz<1000000),2),:);
% 
% plot3(EMxyz(:,1),EMxyz(:,2),EMxyz(:,3),'.'); drawnow;

%% plot results
% pred_pose=readmatrix("7Scenes_CTCLDepthfiles_mapnet_pred_poses_nonorm_200.txt");
pred_pose=readmatrix("7Scenes_CTCLDepthfiles_mapnet_posenet_500_pred_poses.txt");
pred_loc=pred_pose(1:802,1:3);
for i = 1:802
plot3(pred_loc(i,1),pred_loc(i,2),pred_loc(i,3),'+')
% pause
end
%plot3(pred_loc(:,1),pred_loc(:,2),pred_loc(:,3),'+')
%% plot checker
% target_pose=readmatrix("7Scenes_CTCLDepthfiles_mapnet_targ_poses_nonorm_300.txt");
target_pose=readmatrix("7Scenes_CTCLDepthfiles_mapnet_posenet_500_targ_poses.txt");
target_loc=target_pose(1:end,1:3);
plot3(target_loc(:,1),target_loc(:,2),target_loc(:,3),'o')


