%% ============================================================
%% STEP 5 — FEATURE SELECTION
%% Methodology item 5: (a) use Principal Component scores, OR
%%                     (b) use selected original features
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   Now we decide exactly what goes into the ANN as its inputs. There
%   are two valid approaches, and this script lets you switch between
%   them with a single flag. Try both, and report whichever gives
%   better validation results (or show both for a stronger report).
%
%   Option A - PC SCORES:
%     Feed the top PCs from Step 4 (the ones covering >=80% of
%     variance) directly into the ANN. Pro: fewer inputs, no
%     redundancy at all. Con: harder to explain physically ("PC2" has
%     no direct real-world meaning).
%
%   Option B - SELECTED RAW FEATURES:
%     Use the PCA loadings to identify which ORIGINAL features
%     contribute most to the top PCs, then feed those raw features
%     (e.g. "Discharge Time", "Rct") into the ANN. Pro: every input has
%     a clear physical meaning you can explain to your supervisor. Con:
%     slightly more inputs, some residual correlation.
% ============================================================

load('Step04_data.mat');
fprintf('=== STEP 5: FEATURE SELECTION ===\n\n');

%% ---- CHOOSE YOUR METHOD HERE ----
USE_PCA_SCORES = true;    % true  = Option A (PC scores)  <- default, per
                          %         supervisor: most papers use PC data
                          % false = Option B (selected raw features)
N_FEATURES_B   = 4;       % how many raw features to keep for Option B
%% ----------------------------------

if USE_PCA_SCORES
    fprintf('--- Option A: using PC scores as ANN inputs ---\n');
    X_final       = score(:, 1:n_pc_80);
    final_names   = arrayfun(@(i) sprintf('PC%d',i), 1:n_pc_80, 'UniformOutput', false);
    fprintf('Using top %d PCs (%.1f%% variance explained).\n', ...
        n_pc_80, sum(explained(1:n_pc_80)));
else
    fprintf('--- Option B: using selected raw features as ANN inputs ---\n');
    % Rank candidate features by their loading on PC1 (the component
    % most strongly tied to aging / capacity)
    [~, load_order] = sort(abs(coeff(:,1)), 'descend');
    top_local_idx   = load_order(1:N_FEATURES_B);
    final_feat_idx  = candidate_idx(top_local_idx);   % map back to original 13
    final_names     = feature_names(final_feat_idx);

    fprintf('Top %d features by |PC1 loading|:\n', N_FEATURES_B);
    for i = 1:N_FEATURES_B
        fprintf('  %d. %-15s (|PC1 loading| = %.4f)\n', ...
            i, final_names{i}, abs(coeff(top_local_idx(i),1)));
    end

    X_final = X_std(:, final_feat_idx);   % standardized values, Step 2d
end

fprintf('\nFinal ANN input matrix: %d samples x %d features\n', ...
    size(X_final,1), size(X_final,2));

%% Save checkpoint for Step 6
save('Step05_data.mat', 'X_final', 'final_names', 'y_capacity', ...
    'batt_label', 'USE_PCA_SCORES');
fprintf('\nSaved -> Step05_data.mat\n');
fprintf('Next: run Step06_DataSplit.m\n');
