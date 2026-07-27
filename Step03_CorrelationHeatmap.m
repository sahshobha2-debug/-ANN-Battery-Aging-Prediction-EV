%% ============================================================
%% STEP 3 — CORRELATION HEAT MAP (redundancy check)
%% Methodology item 3
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   Two features are "redundant" if they move together almost
%   perfectly - knowing one tells you the other for free. Feeding both
%   into the network wastes an input slot and can make training less
%   stable. A correlation heat map shows, at a glance, which feature
%   pairs are strongly related (close to +1 or -1) versus unrelated
%   (close to 0).
% ============================================================
load('Step02_data.mat');
fprintf('=== STEP 3: CORRELATION HEAT MAP ===\n\n');

n_feat = length(feature_names);

%% Correlation of each feature WITH the target (capacity)
fprintf('--- Feature vs Target (Capacity) correlation ---\n');
target_corr = zeros(n_feat,1);
for i = 1:n_feat
    target_corr(i) = corr(X(:,i), y_capacity);
    fprintf('%-15s : r = %+.4f\n', feature_names{i}, target_corr(i));
end

[sorted_corr, sort_idx] = sort(abs(target_corr), 'descend');
figure('Name','Feature-Target Correlation','Position',[100 100 1000 500]);
b = bar(sorted_corr, 'FaceColor','flat');
for i = 1:n_feat
    if sorted_corr(i) > 0.7,      b.CData(i,:) = [0.18 0.65 0.18];
    elseif sorted_corr(i) > 0.4,  b.CData(i,:) = [0.95 0.60 0.10];
    else,                         b.CData(i,:) = [0.85 0.20 0.20];
    end
end
set(gca,'XTick',1:n_feat,'XTickLabel',feature_names(sort_idx));
xtickangle(45);
ylabel('|Pearson r| with Capacity');
title('Feature vs Target Correlation');
yline(0.7,'r--','Strong (>0.7)','LineWidth',1.5);
yline(0.4,'b--','Moderate (>0.4)','LineWidth',1.5);
grid on;

%% Inter-feature correlation heat map (redundancy)
C = corr(X);
figure('Name','Correlation Heat Map','Position',[100 100 900 800]);
imagesc(C); colorbar; colormap('jet'); clim([-1 1]);
set(gca,'XTick',1:n_feat,'XTickLabel',feature_names, ...
        'YTick',1:n_feat,'YTickLabel',feature_names);
xtickangle(45);
title('Inter-Feature Correlation Heat Map');
for i = 1:n_feat
    for j = 1:n_feat
        text(j,i,sprintf('%.2f',C(i,j)), ...
            'HorizontalAlignment','center','FontSize',6,'Color','k');
    end
end

%% Flag redundant pairs (|r| > 0.85) and decide which to drop
fprintf('\n--- Redundancy Warnings (|r| > 0.85 between features) ---\n');
to_drop = false(n_feat,1);
warned  = false;
for i = 1:n_feat
    for j = i+1:n_feat
        if abs(C(i,j)) > 0.85
            if abs(target_corr(i)) < abs(target_corr(j))
                drop = i; keep = j;
            else
                drop = j; keep = i;
            end
            fprintf('  %-15s <-> %-15s : r=%.4f -> drop %s, keep %s\n', ...
                feature_names{i}, feature_names{j}, C(i,j), ...
                feature_names{drop}, feature_names{keep});
            to_drop(drop) = true;
            warned = true;
        end
    end
end
if ~warned, fprintf('  None found.\n'); end

%% Build the candidate feature list: drop weak (|r|<0.4) AND redundant
weak = abs(target_corr) < 0.4;
remove_mask   = weak | to_drop;
candidate_idx = find(~remove_mask);

fprintf('\n--- Candidate features carried forward to PCA (Step 4) ---\n');
for i = 1:length(candidate_idx)
    fprintf('  %d. %-15s (|r| with capacity = %.4f)\n', ...
        i, feature_names{candidate_idx(i)}, abs(target_corr(candidate_idx(i))));
end

%% Save checkpoint for Step 4
save('Step03_data.mat', 'X', 'X_std', 'mu_X', 'sigma_X', 'y_capacity', ...
    'batt_label', 'feature_names', 'target_corr', 'C', 'candidate_idx');
fprintf('\nSaved -> Step03_data.mat\n');
fprintf('Next: run Step04_PCA.m\n');
