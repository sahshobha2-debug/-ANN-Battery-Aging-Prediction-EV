%% ============================================================
%% STEP 6 — SPLIT DATASET
%% Methodology item 6
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   We split the data into three separate buckets so we can trust our
%   results:
%     TRAIN (70%)      - the network actually learns from these
%     VALIDATION (15%) - used DURING training to check progress and
%                        decide when to stop (Step 8's early stopping)
%     TEST (15%)       - touched only ONCE, at the very end, to report
%                        an honest, unbiased performance number
%
%   Analogy: TRAIN is the homework, VALIDATION is a practice quiz you
%   use to know when to stop studying, and TEST is the real final exam
%   you take only once.
% ============================================================

load('Step05_data.mat');
fprintf('=== STEP 6: TRAIN / VALIDATION / TEST SPLIT ===\n\n');

rng(1);   % fixed seed -> same split every time you re-run this script
N   = size(X_final, 1);
idx = randperm(N);

train_end = round(0.70 * N);
val_end   = round(0.85 * N);

train_idx = idx(1:train_end);
val_idx   = idx(train_end+1:val_end);
test_idx  = idx(val_end+1:end);

fprintf('Total samples : %d\n', N);
fprintf('  Train      : %d (%.0f%%)\n', length(train_idx), 100*length(train_idx)/N);
fprintf('  Validation : %d (%.0f%%)\n', length(val_idx),   100*length(val_idx)/N);
fprintf('  Test       : %d (%.0f%%)\n', length(test_idx),  100*length(test_idx)/N);

%% Standardize the TARGET using only the training set's statistics
% We compute mean/std from the TRAINING data only, then apply the same
% transform to validation and test. This avoids "peeking" at data the
% model isn't supposed to know about yet (a common student mistake
% called data leakage).
y_mu    = mean(y_capacity(train_idx));
y_sigma = std(y_capacity(train_idx));

y_train = (y_capacity(train_idx) - y_mu) / y_sigma;
y_val   = (y_capacity(val_idx)   - y_mu) / y_sigma;
y_test  = (y_capacity(test_idx)  - y_mu) / y_sigma;

X_train = X_final(train_idx, :);
X_val   = X_final(val_idx,   :);
X_test  = X_final(test_idx,  :);

fprintf('\nTarget standardized using TRAIN-set mean=%.4f, std=%.4f\n', y_mu, y_sigma);

%% Save checkpoint for Step 7 / Step 8
save('Step06_data.mat', 'X_train','X_val','X_test', ...
    'y_train','y_val','y_test', 'y_mu','y_sigma', ...
    'train_idx','val_idx','test_idx', 'final_names', 'y_capacity');
fprintf('\nSaved -> Step06_data.mat\n');
fprintf('Next: run Step07_HyperparameterTuning.m\n');
