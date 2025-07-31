%% Evaluate the systems-level temporal organization of a circuit/network of 
%% interest using "temporal unwrapping"/edge functional connectivity (eFC).
%% From Korponay et al (2022), "The Temporal Orgaization of Corticostriatal
%% Communications". 


% clean up env for clean run
clearvars; close all; clc

%%%%% INPUTS and DOWNLOADS %%%%%

    %1. Download the right striatum mask and the 21 right frontal cortex masks:
    %https://neurovault.org/collections/21499/
    
    %2. Requires denoised fMRI BOLD timeseries data in NIFTI (.nii/.nii.gz)
    %format.
    %Run the following two AFNI command scripts in the terminal to extract
    %each subject's denoised BOLD timeseries at each voxel/ROI in the structures of interest
    %(e.g., striatum and frontal cortex). The resulting CSV files are the inputs to this MATLAB script:
    
    %Structure 1 (e.g., striatum; voxel-wise timeseries extraction) 
    
      % #!/bin/tcsh -xef
      % set subjList = (subj1 subj2 subj3 etc.)
      % foreach subj ($subjList)
      % 3dmaskdump -noijk  -mask rStriatum_Mask.nii.gz {$subj}_Clean_rfMRI_REST1.nii.gz > {$subj}_rStriatum_Clean_TimeSeries_.1D
      % 1dcat -csvout {$subj}_rStriatum_Clean_TimeSeries_.1D > {$subj}_rStriatum_Clean_TimeSeries.csv
      % end
      
    %Structure 2 (e.g., frontal cortex; ROI-wise timeseries extration)
    
      % #!/bin/tcsh -xef
      % set subjList = (subj1 subj2 subj3 etc.)
      % set ROIList = (1 2 3 etc.)
      % foreach subj ($subjList)
      % foreach ROI ($ROIList)
      % 3dmaskave -quiet -overwrite -mask {$ROI}.nii {$subj}_Clean_rfMRI_REST1.nii.gz > {$subj}_Clean_rfMRI_REST1_{$ROI}_ts.1D
      % end
      % 1dcat -csvout -overwrite {$subj}_*_Clean_TimeSeries.1D > {$subj}_CorticalROIs_Clean_TimeSeries.csv
      % end
    
    %3.
    %Download the eFC package from
    %https://github.com/brain-networks/edge-centric_demo 
    %Faskowitz et al (2020) 

    %4.
    % Download the NIFTI and ANALYZE image toolbox for Matlab:
    %(https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image)

    %5.
    % Download the Brain Connectivity Toolbox for MATLAB from:
    % https://sites.google.com/site/bctnet/

% add helper functions to path
addpath(genpath('fcn'));
addpath '/path/to/Downloads/eFC/fcn'
addpath '/path/to/Downloads/NIfTI_20140122'
addpath '/path/to/Downloads/BCT'
      

%% load node time series CSV files for each subject
P = '/path/to/striatumCSVs';
S = dir(fullfile(P,'*csv*')); 

P2 = '/path/to/corticalCSVs';
S2 = dir(fullfile(P2,'*.csv'));


%% I).

%% Compute frame-wise cortical coactivation profiles for all striatal voxels
%% across all subejcts

I=1;

Subjects=396;           %User sets parameter 
TRs=253;                %User sets parameter 
Striatal_Voxels=1710;   %User sets parameter
Total_Cortical_ROIs=21; %User sets parameter
Target_Cortical_ROIs=5; %User sets parameter

Final_EdgeTimeseries_Top5=zeros(Subjects*TRs*Striatal_Voxels,Target_Cortical_ROIs);

for i = 1:Subjects
    i=i

%load striatal voxel-wise timeseries

    F = fullfile(P,S(i).name);
    S(i).data = readtable(F,'NumHeaderLines', 1);
    S(i).data = table2array(S(i).data); 
    S(i).data = double(S(i).data);
    S(i).data = S(i).data';
    S(i).data = S(i).data(any(S(i).data,2),:);   %remove any motion-censored TRs (i.e. rows of all zeros)
    z1 = zscore(S(i).data);                
    S(i).data = [];

%load cortical ROI timeseries
    F2 = fullfile(P2,S2(i).name);
    S2(i).data = readtable(F2,'NumHeaderLines', 1);
    S2(i).data = table2array(S2(i).data); 
    S2(i).data = double(S2(i).data);
    S2(i).data = S2(i).data(any(S2(i).data,2),:);   %remove any motion-censored TRs (i.e. rows of all zeros)
    z2 = zscore(S2(i).data);           
    S2(i).data = [];

%compute the coactivation time series of each striatal voxel-cortical ROI
%pair
for x = 1:Striatal_Voxels                                  
    for j = 1:Total_Cortical_ROIs
       S(i).Final_EdgeTimeseries(1:TRs,j) = z1(:,x).*z2(:,j);
    end

%Limit each striatal voxel's cortical connectivity profile to only its 5 strongest connections%
StaticFC(x,:,i) = mean(S(i).Final_EdgeTimeseries);   
[B,maxIndx]=maxk(StaticFC(x,:,i) ,5);
Final_EdgeTimeseries_Top5(I:I+TRs-1,:)=S(i).Final_EdgeTimeseries(:,maxIndx);
%ID_top5_CorticalInput(x,:,i)=maxIndx;
S(i).Final_EdgeTimeseries=[];
I=I+TRs;
%%%%%%%%%%
end

end

%% We now have "Final_EdgeTimeseries_Top5", a Voxels*TRs*Subjects-by-5 matrix where each row encodes
%% the coactivation magnitudes of a striatal voxel and its 5 strongest
%% cortical inputs at a given TR (i.e., frame), across all frames, voxels and subjects

%% II).

%% Next, identify (cluster) recurring coactivation states. Start by identifying
%% the coactivation "burst" frames and separating them from the coactivation "rest" frames:

Rand_TRs=randsample(TRs*Subjects*Striatal_Voxels,25000);
Final_EdgeTimeseries_Top5_Rand=Final_EdgeTimeseries_Top5(Rand_TRs,:);
dbscan_output=dbscan(abs(Final_EdgeTimeseries_Top5_Rand),1,20);

class_Burst=zeros(TRs*Subjects*Striatal_Voxels,1);
i=1;
while i<TRs*Subjects*Striatal_Voxels-1
class_Burst(i:i+TRs-1)=classify(Final_EdgeTimeseries_Top5(i:i+TRs-1,:),Final_EdgeTimeseries_Top5_Rand,dbscan_output);
i=i+TRs;
end

indx_Rest=find(class_Burst==1);
indx_Burst=find(class_Burst==-1);

%% Having now separated out the burst and rest frames, identify (cluster)
%% subtypes of rest frames:

Rand_Rest_indx=randsample(length(indx_Rest),25000);
Rand_Rest_IDs=indx_Rest(Rand_Rest_indx);
Rand_Rest_CPs=Final_EdgeTimeseries_Top5(Rand_Rest_IDs,:);

Rand_Rest_CPs_corrMatrix = fcn_edgets2edgecorr(Rand_Rest_CPs'); 

  n  = size(Rand_Rest_CPs_corrMatrix,1);             % number of nodes
       M  = 1:n;                   % initial community affiliations
       Q0 = -1; Q1 = 0;            % initialize modularity values
       while Q1-Q0>1e-5;           % while modularity increases
           Q0 = Q1;                % perform community detection
           [M, Q1] = community_louvain(Rand_Rest_CPs_corrMatrix, [], M,'negative_asym');
       end

  Final_EdgeTimeseries_Top5_Rest=Final_EdgeTimeseries_Top5(indx_Rest,:);

class_Rest=zeros(TRs*Subjects*Striatal_Voxels,1);
i=1;
while i<TRs*Subjects*Striatal_Voxels-1
class_Rest(i:i+TRs-1)=classify(Final_EdgeTimeseries_Top5(i:i+TRs-1,:), Rand_Rest_CPs,M);
i=i+TRs;
end

Number_of_Rest_States=M;

%Compile the cluster labels across all TRs/frames, voxels and subjects:

class_All=class_Rest;
class_All(indx_Burst)=4;


%% III). Compute properties of each coactivation state

%% Compute the average coactivation profile of each state

  Avg_CP_State(1,:)=mean(Final_EdgeTimeseries_Top5(class_All==1,:));
  Avg_CP_State(2,:)=mean(Final_EdgeTimeseries_Top5(class_All==2,:));
  Avg_CP_State(3,:)=mean(Final_EdgeTimeseries_Top5(class_All==3,:));
  Avg_CP_State(4,:)=mean(Final_EdgeTimeseries_Top5(class_All==4,:));

%% Compute the occupancy liklihood (i.e., % of all TRs spent in a given state across all voxels and subjects) of each state: 

  TRs_in_CP_State(1)=((sum(class_All==1))/(TRs*Subjects*Striatal_Voxels))*100;
  TRs_in_CP_State(2)=((sum(class_All==2))/(TRs*Subjects*Striatal_Voxels))*100;
  TRs_in_CP_State(3)=((sum(class_All==3))/(TRs*Subjects*Striatal_Voxels))*100;
  TRs_in_CP_State(4)=((sum(class_All==4))/(TRs*Subjects*Striatal_Voxels))*100;


%% Compute the the number of times each state transitions to each other state

h=1;
b=1;
while h<(TRs*Subjects*Striatal_Voxels)-1

Transitions1_1(b) = numel(strfind(class_All(h:h+TRs-1)',[1 1]));
Transitions1_2(b) = numel(strfind(class_All(h:h+TRs-1)',[1 2]));
Transitions1_3(b) = numel(strfind(class_All(h:h+TRs-1)',[1 3]));
Transitions1_4(b) = numel(strfind(class_All(h:h+TRs-1)',[1 4]));
Transitions2_1(b) = numel(strfind(class_All(h:h+TRs-1)',[2 1]));
Transitions2_2(b) = numel(strfind(class_All(h:h+TRs-1)',[2 2]));
Transitions2_3(b) = numel(strfind(class_All(h:h+TRs-1)',[2 3]));
Transitions2_4(b) = numel(strfind(class_All(h:h+TRs-1)',[2 4]));
Transitions3_1(b) = numel(strfind(class_All(h:h+TRs-1)',[3 1]));
Transitions3_2(b) = numel(strfind(class_All(h:h+TRs-1)',[3 2]));
Transitions3_3(b) = numel(strfind(class_All(h:h+TRs-1)',[3 3]));
Transitions3_4(b) = numel(strfind(class_All(h:h+TRs-1)',[3 4]));
Transitions4_1(b) = numel(strfind(class_All(h:h+TRs-1)',[4 1]));
Transitions4_2(b) = numel(strfind(class_All(h:h+TRs-1)',[4 2]));
Transitions4_3(b) = numel(strfind(class_All(h:h+TRs-1)',[4 3]));
Transitions4_4(b) = numel(strfind(class_All(h:h+TRs-1)',[4 4]));

b=b+1;
h=h+TRs;
end 

Transitions(1,1)=sum(Transitions1_1);
Transitions(1,2)=sum(Transitions1_2);
Transitions(1,3)=sum(Transitions1_3);
Transitions(1,4)=sum(Transitions1_4);
Transitions(2,1)=sum(Transitions2_1);
Transitions(2,2)=sum(Transitions2_2);
Transitions(2,3)=sum(Transitions2_3);
Transitions(2,4)=sum(Transitions2_4);
Transitions(3,1)=sum(Transitions3_1);
Transitions(3,2)=sum(Transitions3_2);
Transitions(3,3)=sum(Transitions3_3);
Transitions(3,4)=sum(Transitions3_4);
Transitions(4,1)=sum(Transitions4_1);
Transitions(4,2)=sum(Transitions4_2);
Transitions(4,3)=sum(Transitions4_3);
Transitions(4,4)=sum(Transitions4_4);


%% Quantify the number of voxels in each state during each TR/frame in each subject

Reshaped_class_All=reshape(class_All,TRs,Striatal_Voxels,Subjects);

for j=1:Subjects
    for i=1:TRs
Total_RestState1_Voxels(i,j)=sum(Reshaped_class_All(i,:,j)==1);
    end
end

for j=1:Subjects
    for i=1:TRs
Total_RestState2_Voxels(i,j)=sum(Reshaped_class_All(i,:,j)==2);
    end
end

for j=1:Subjects
    for i=1:TRs
Total_RestState3_Voxels(i,j)=sum(Reshaped_class_All(i,:,j)==3);
    end
end

for j=1:Subjects
    for i=1:TRs
Total_BurstState_Voxels(i,j)=sum(Reshaped_class_All(i,:,j)==4);
    end
end

Percent_Striatum_in_RestState1=Total_RestState1_Voxels/Striatal_Voxels*100;
Percent_Striatum_in_RestState2=Total_RestState2_Voxels/Striatal_Voxels*100;
Percent_Striatum_in_RestState3=Total_RestState3_Voxels/Striatal_Voxels*100;
Percent_Striatum_in_BurstState=Total_BurstState_Voxels/Striatal_Voxels*100;

%hist(reshape(Percent_Striatum_in_RestState1,Subjects*TRs,1))
%hist(reshape(Percent_Striatum_in_RestState2,Subjects*TRs,1))
%hist(reshape(Percent_Striatum_in_RestState3,Subjects*TRs,1))
%hist(reshape(Percent_Striatum_in_BurstState,Subjects*TRs,1))



%% Compute the average dwell time in each state

for k = 1:4
k=k
h=1;
b=1;
while h<(TRs*Subjects*Striatal_Voxels)-1

TotalDwellTime_inTRs(k) = sum(class_All==k);

    TR_IDs_in_State_K = find(class_All(h:(h+TRs-1)) == k);
    TR_IDs_in_State_K(end+1)=2;   
    I_1=find(diff(TR_IDs_in_State_K)~=1);  % finds where sequences of consecutive numbers end
    [m,n]=size(I_1);   % finds dimensions of I_1 i.e. how many sequences of consecutive numbers you have
    startpoint=1;    
    seq=cell(1,n);  
    for w=1:m
        End_Idx=I_1(w);   %set end index
        seq{w}=TR_IDs_in_State_K(startpoint:End_Idx);  %finds sequences of consecutive numbers and assigns to cell array
        startpoint=End_Idx+1;   %update start index for the next consecutive sequence
    end
    for w=1:m
        TR_Durations(w) = cellfun(@numel, seq(w));
    end 
    TR_Durations(1) = []; 
    TR_IDs_in_State_K = [];
    I_1= [];
    seq=[];
    
AverageDuration_inTRs(k,b) = mean(TR_Durations);
    TR_Durations =[];
    b = b+1;
h=h+TRs;
end
end

AverageDuration_inTRs_Mean=mean(AverageDuration_inTRs,2,"omitnan");
AverageDuration_inTRs_std=std(AverageDuration_inTRs',"omitnan");