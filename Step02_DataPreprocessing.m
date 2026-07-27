%% ============================================================
%% STEP 2 — DATA PREPROCESSING
%% Methodology item 2: (a) missing values (b) duplicates
%%                     (c) outlier detection (d) standard normalization
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   Raw data is never perfectly clean. Before training, we fix four
%   kinds of problems, in this order:
%     (a) Missing values  - some rows have no impedance reading at all
%     (b) Duplicates      - the exact same row appearing twice
%     (c) Outliers        - rows that look like sensor glitches, not
%                           real battery behaviour
%     (d) Normalization   - putting every feature on the same numeric
%                           scale, so the network doesn't think
%                           "temperature in degrees" matters more than
%                           "voltage in volts" just because its numbers
%                           are bigger
% ============================================================
load('Step01_data.mat');
fprintf('=== STEP 2: DATA PREPROCESSING ===\n\n');

X          = all_features;   % working copy
y_capacity = all_capacity;
batt_label = battery_label;
n_start    = size(X,1);

%% ------------------------------------------------------------
%% 2a) HANDLE MISSING VALUES: remove rows with missing data
%% ------------------------------------------------------------
% Columns 10-12 (Re, Rct, Bat Imp) come from the impedance test, which
% NASA only ran partway through each experiment. Earlier cycles have
% no impedance reading at all -> NaN in those columns.
%
% Per your supervisor's instruction, we REMOVE those rows rather than
% guessing a replacement value.
fprintf('--- 2a: Missing Values ---\n');
missing_rows = any(isnan(X(:,10:12)), 2);
fprintf('Rows missing impedance data : %d of %d (%.1f%%)\n', ...
    sum(missing_rows), n_start, 100*sum(missing_rows)/n_start);

X(missing_rows,:)          = [];
y_capacity(missing_rows)   = [];
batt_label(missing_rows)   = [];
fprintf('Rows removed  : %d\n', sum(missing_rows));
fprintf('Rows remaining: %d\n\n', size(X,1));

% NOTE FOR YOUR REPORT: this removes every early-experiment cycle
% (the ones without an impedance test), which is a meaningful chunk of
% data. If your supervisor later asks "why not just fill in the
% average instead?", the trade-off is:
%   Remove  -> cleaner data, but throws away real cycles
%   Impute  -> keeps all cycles, but invents values that were never
%              measured (riskier for a physical health measurement)
% This script follows the "remove" instruction as given.

%% ------------------------------------------------------------
%% 2b) REMOVE DUPLICATES
%% ------------------------------------------------------------
fprintf('--- 2b: Duplicate Rows ---\n');
[X_unique, keep_idx] = unique(X, 'rows', 'stable');
n_dup = size(X,1) - size(X_unique,1);
fprintf('Duplicate rows found and removed: %d\n\n', n_dup);

X          = X_unique;
y_capacity = y_capacity(keep_idx);
batt_label = batt_label(keep_idx);

%% ------------------------------------------------------------
%% 2c) OUTLIER DETECTION: z-score, IQR, and Isolation Forest
%% ------------------------------------------------------------
% We run three different outlier detectors and only remove a row if at
% least TWO of the three methods flag it. This avoids over-cleaning the
% data based on any single method's assumptions.
fprintf('--- 2c: Outlier Detection ---\n');

% Method 1: z-score (flags a value as an outlier if it is more than
% 3 standard deviations from the mean of its own column)
Z = zscore(X);
outlier_zscore = any(abs(Z) > 3, 2);

% Method 2: IQR (flags a value outside 1.5x the interquartile range,
% the same rule used to draw the whiskers on a box plot)
Q1  = prctile(X, 25);
Q3  = prctile(X, 75);
IQR_val = Q3 - Q1;
lower_bound = Q1 - 1.5*IQR_val;
upper_bound = Q3 + 1.5*IQR_val;
outlier_iqr = any(X < lower_bound | X > upper_bound, 2);

% Method 3: Isolation Forest (flags points that are "easy to isolate"
% from the rest of the data with random splits - a point that gets
% separated in very few splits is probably an anomaly). Requires
% Statistics and Machine Learning Toolbox R2021b or newer.
try
    forest         = iforest(X, 'ContaminationFraction', 0.05);
    outlier_iforest = isanomaly(forest, X);
catch ME
    warning(['Isolation Forest unavailable (%s). This needs MATLAB ' ...
        'R2021b+ with Statistics and Machine Learning Toolbox. ' ...
        'Continuing with z-score + IQR only.'], ME.identifier);
    outlier_iforest = false(size(X,1), 1);
end

vote          = outlier_zscore + outlier_iqr + outlier_iforest;
outlier_final = vote >= 2;   % majority vote across the 3 methods

fprintf('Flagged by z-score        : %d\n', sum(outlier_zscore));
fprintf('Flagged by IQR            : %d\n', sum(outlier_iqr));
fprintf('Flagged by Isolation Forest: %d\n', sum(outlier_iforest));
fprintf('Flagged by >=2 methods (removed): %d\n\n', sum(outlier_final));

X(outlier_final,:)        = [];
y_capacity(outlier_final) = [];
batt_label(outlier_final) = [];

%% ------------------------------------------------------------
%% 2d) DATA NORMALIZATION (standard / z-score)
%% ------------------------------------------------------------
% "Standard" normalization rescales each feature to have mean = 0 and
% standard deviation = 1. Analogy: instead of comparing raw scores from
% different tests (one out of 10, one out of 100), we convert everyone
% to "how many standard deviations above/below average" - now every
% feature is judged on the same scale.
%
% (Your earlier scripts used mapminmax, which squeezes every feature
% into a fixed [-1, 1] range instead. Both are valid; this step follows
% your supervisor's specific request for STANDARD normalization.)
fprintf('--- 2d: Standard Normalization (z-score) ---\n');
[X_std, mu_X, sigma_X] = zscore(X);
fprintf('Standardized %d features to mean=0, std=1.\n', size(X_std,2));
fprintf('(Target capacity is NOT normalized here - that happens later,\n');
fprintf(' per-split, in Step 6, to avoid leaking test-set statistics.)\n\n');

fprintf('Preprocessing summary: %d -> %d rows (%.1f%% kept)\n', ...
    n_start, size(X,1), 100*size(X,1)/n_start);

%% Save checkpoint for Step 3
save('Step02_data.mat', 'X', 'X_std', 'mu_X', 'sigma_X', ...
    'y_capacity', 'batt_label', 'feature_names');
fprintf('\nSaved -> Step02_data.mat\n');
fprintf('Next: run Step03_CorrelationHeatmap.m\n');
