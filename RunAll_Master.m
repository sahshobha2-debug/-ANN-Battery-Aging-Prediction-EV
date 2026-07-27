%% ============================================================
%% RUN ALL — executes Steps 1-9 in order
%% Use this to run the whole pipeline at once. Use the individual
%% Step0X_*.m files to run/show one stage at a time to your supervisor.
%% ============================================================
clear; clc;

steps = {
    'Step01_DataCollection.m'
    'Step02_DataPreprocessing.m'
    'Step03_CorrelationHeatmap.m'
    'Step04_PCA.m'
    'Step05_FeatureSelection.m'
    'Step06_DataSplit.m'
    'Step07_HyperparameterTuning.m'
    'Step08_ANN_Training.m'
    'Step09_EvaluationMetrics.m'
    'Step10_SOH_Conversion.m'
};

for i = 1:length(steps)
    fprintf('\n\n########## RUNNING %s ##########\n\n', steps{i});
    run(steps{i});
end

fprintf('\n\nAll steps complete.\n');
