%% ============================================================
%% STEP 4 — PRINCIPAL COMPONENT ANALYSIS (PCA)
%% Methodology item 4
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   PCA finds new "combined" directions in the data (called Principal
%   Components, or PCs) that capture as much of the original
%   information (variance) as possible, using as few numbers as
%   possible. Think of it like photographing a 3-D object from the one
%   angle that shows its shape most clearly in a single 2-D photo -
%   PC1 is that "best angle", PC2 is the next-best angle at a right
%   angle to it, and so on.
%
%   We use PCA two ways later (Step 5 lets you choose):
%     Option A - feed the top PC SCORES directly into the ANN
%     Option B - use PCA just to see which ORIGINAL features matter
%                most (via "loadings"), then feed those original
%                features into the ANN (easier to explain physically)
% ============================================================

load('Step03_data.mat');
fprintf('=== STEP 4: PCA ===\n\n');

cand_names = feature_names(candidate_idx);
X_cand_std = X_std(:, candidate_idx);   % PCA runs on STANDARDIZED data

%% Run PCA
[coeff, score, ~, ~, explained] = pca(X_cand_std);

fprintf('--- Variance explained by each Principal Component ---\n');
for i = 1:length(explained)
    fprintf('PC%d : %.2f%%  (cumulative %.2f%%)\n', ...
        i, explained(i), sum(explained(1:i)));
end

n_pc_80 = find(cumsum(explained) >= 80, 1, 'first');
fprintf('\n%d of %d components explain >= 80%% of total variance.\n', ...
    n_pc_80, length(explained));

%% Correlation of each PC score with the target (which PCs matter?)
r_pc = zeros(size(score,2),1);
for i = 1:size(score,2)
    r_pc(i) = corr(score(:,i), y_capacity);
end
fprintf('\n--- PC score correlation with Capacity ---\n');
for i = 1:length(r_pc)
    fprintf('PC%d : r = %+.4f\n', i, r_pc(i));
end

%% Figure 1: Scree plot
figure('Name','PCA Scree Plot','Position',[100 100 700 450]);
bar(explained, 'FaceColor',[0.4 0.6 0.9]); hold on;
yyaxis right;
plot(1:length(explained), cumsum(explained), 'ro-', 'LineWidth',1.5,'MarkerSize',5);
yline(80,'k--','80%');
ylabel('Cumulative Variance (%)'); ylim([0 105]);
yyaxis left;
xlabel('Principal Component'); ylabel('Individual Variance (%)');
title('Scree Plot — Candidate Features');
grid on;

%% Figure 2: Loadings heat map (which raw features drive each PC)
figure('Name','PCA Loadings','Position',[100 100 760 420]);
imagesc(abs(coeff)'); colorbar; colormap('hot'); clim([0 max(abs(coeff(:)))]);
set(gca,'XTick',1:length(cand_names),'XTickLabel',cand_names, ...
        'YTick',1:size(coeff,2), ...
        'YTickLabel',arrayfun(@(i) sprintf('PC%d',i),1:size(coeff,2),'UniformOutput',false));
xtickangle(30);
title('|PCA Loadings| — which raw features drive each component');
xlabel('Original Feature'); ylabel('Principal Component');
for i = 1:size(coeff,2)
    for j = 1:length(cand_names)
        text(j,i,sprintf('%.2f',abs(coeff(j,i))), ...
            'HorizontalAlignment','center','FontSize',8);
    end
end

%% Figure 2b: PC SCORE correlation heat map
% This is the key evidence for why papers prefer feeding PC SCORES
% into the ANN instead of raw features: unlike the raw feature
% correlation heat map (Step 3), which shows real off-diagonal color
% (redundancy between features), the PCs are mathematically guaranteed
% to be uncorrelated with each other - this heat map should come out
% looking like a clean diagonal (1's down the middle, ~0 everywhere
% else). Put this figure next to Step 3's heat map when presenting.
C_pc = corr(score);
figure('Name','PC Score Correlation (redundancy check)','Position',[100 100 650 580]);
imagesc(C_pc); colorbar; colormap('jet'); clim([-1 1]);
pc_labels = arrayfun(@(i) sprintf('PC%d',i), 1:size(score,2), 'UniformOutput', false);
set(gca,'XTick',1:size(score,2),'XTickLabel',pc_labels, ...
        'YTick',1:size(score,2),'YTickLabel',pc_labels);
title('PC Score Correlation Matrix (should be ~diagonal — no redundancy)');
for i = 1:size(score,2)
    for j = 1:size(score,2)
        text(j,i,sprintf('%.2f',C_pc(i,j)), ...
            'HorizontalAlignment','center','FontSize',7,'Color','k');
    end
end
fprintf('\n--- PC Score Redundancy Check ---\n');
off_diag = C_pc - eye(size(C_pc));
fprintf('Largest off-diagonal |correlation| between any two PCs: %.4f\n', ...
    max(abs(off_diag(:))));
fprintf('(Compare this to Step 3''s raw feature heat map, where off-\n');
fprintf(' diagonal values were much larger - that difference IS the\n');
fprintf(' argument for using PC scores as ANN inputs.)\n');

%% Figure 3: Biplot (PC1 vs PC2)
figure('Name','PCA Biplot','Position',[100 100 600 550]);
biplot(coeff(:,1:2), 'scores', score(:,1:2), 'varlabels', cand_names);
xlabel(sprintf('PC1 (%.1f%% variance)', explained(1)));
ylabel(sprintf('PC2 (%.1f%% variance)', explained(2)));
title('PCA Biplot');
grid on;

%% Save checkpoint for Step 5
save('Step04_data.mat', 'X', 'X_std', 'mu_X', 'sigma_X', 'y_capacity', ...
    'batt_label', 'feature_names', 'candidate_idx', 'cand_names', ...
    'coeff', 'score', 'explained', 'r_pc', 'n_pc_80');
fprintf('\nSaved -> Step04_data.mat\n');
fprintf('Next: run Step05_FeatureSelection.m\n');
