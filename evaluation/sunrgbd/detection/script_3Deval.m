

% Important: Change directory to the following first before continuing
% cd Transferable3D/evaluation/sunrgbd/detection/
clear all

homeDir = '/home/yewsiang/Transferable3D/';
predDir = homeDir + 'sunrgbd/predictions3D/';
toolboxpath = homeDir + '/sunrgbd/SUNRGBDtoolbox';
addpath(genpath(toolboxpath));
split = load(fullfile(toolboxpath,'/traintestSUNRGBD/allsplit.mat'));
testset_path = split.alltest;
%%

% Configurations
testOn = "B" % Set of classes to test on ("A", "B", "AB")
% Predictions that we want to evaluation
predictionsDir = predDir + "PREDATest/";

classNamesA = {'bed','chair','toilet','desk','bathtub'};
classNamesB = {'table', 'sofa','dresser','night_stand','bookshelf'};
classNamesAB = {'bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub'};
nClassesA = size(classNamesA,2);
nClassesB = size(classNamesB,2);
nClassesAB = size(classNamesAB,2);
nClasses = 0;
if testOn == 'AB'
    nClasses = nClassesAB;
elseif testOn == 'A' || testOn == 'B'
    nClasses = nClassesA;
end
apScores(nClasses) = 0;

for c = 1:nClasses
    if testOn == 'AB'
        className = classNamesAB{c};
    elseif testOn == 'A'
        className = classNamesA{c};
    elseif testOn == 'B'
        className = classNamesB{c};
    end
    classPredictionFile = strcat(predictionsDir, className, '_pred.txt');
    [allTestImgIds, allBb3dtight] = parse_class_predictions(classPredictionFile);
    disp(['Number of predictions for ', upper(className), ': ', int2str(size(allBb3dtight, 1))]);
    
    % Load Ground Truth
    [groundTruthBbs,all_sequenceName] = benchmark_groundtruth(className,fullfile(toolboxpath,'Metadata/'),testset_path);
    
    img  = allTestImgIds;
    pred = allBb3dtight;
    gt = groundTruthBbs;
       
    % Calculate AP
    [apScore,precision,recall,isTp,isFp,isMissed,gtAssignment,maxOverlaps] = computePRCurve3D(className,allBb3dtight,allTestImgIds,groundTruthBbs,zeros(length(groundTruthBbs),1));
    result_all = struct('apScore',apScore,'precision',precision,'recall',recall,'isTp',isTp,'isFp',isFp,'isMissed',isMissed,'gtAssignment',gtAssignment);
    disp(['AP Score for ', upper(className), ': [', num2str(apScore * 100.), ']']);
    apScores(c) = apScore;

end
disp(['Mean AP Score: [', num2str(mean(apScores) * 100.), ']']);
