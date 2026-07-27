%% ============================================================
%% STEP 9 — EVALUATION METRICS
%% Methodology item 17
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   Now that the network is trained and has never seen the TEST set,
%   we ask: how good are its predictions, really? Four standard
%   regression metrics, each answering a slightly different question:
%     MSE  - average squared error (penalizes big misses heavily)
%     RMSE - same as MSE but back in the original units (Ah), easier
%            to interpret ("on average, off by about this many Ah")
%     MAE  - average absolute error (less sensitive to rare big misses)
%     R^2  - fraction of the variation in capacity the model explains
%            (1.0 = perfect, 0.0 = no better than just guessing the
%            average every time)
% ============================================================

load('Step08_data.mat');
fprintf('=== STEP 9: EVALUATION METRICS (Test Set) ===\n\n');

err     = y_pred_test - y_actual_test;
mse_val  = mean(err.^2);
rmse_val = sqrt(mse_val);
mae_val  = mean(abs(err));

ss_res = sum((y_actual_test - y_pred_test).^2);
ss_tot = sum((y_actual_test - mean(y_actual_test)).^2);
r2_val = 1 - ss_res/ss_tot;

fprintf('MSE  : %.6f Ah^2\n', mse_val);
fprintf('RMSE : %.6f Ah\n', rmse_val);
fprintf('MAE  : %.6f Ah\n', mae_val);
fprintf('R^2  : %.6f\n\n', r2_val);

%% Baseline comparison: simple linear regression on the same inputs
% Always worth showing your supervisor that the ANN is earning its
% complexity by beating a much simpler model.
load('Step06_data.mat', 'X_train','X_test','y_capacity','train_idx','test_idx');
mdl        = fitlm(X_train, y_capacity(train_idx));
y_pred_lr  = predict(mdl, X_test);

err_lr   = y_pred_lr - y_actual_test;
rmse_lr  = sqrt(mean(err_lr.^2));
mae_lr   = mean(abs(err_lr));
ss_res_lr = sum((y_actual_test - y_pred_lr).^2);
r2_lr    = 1 - ss_res_lr/ss_tot;

fprintf('=== ANN vs LINEAR REGRESSION BASELINE ===\n');
fprintf('%-12s | %-10s | %-10s | %-8s\n','Model','RMSE (Ah)','MAE (Ah)','R^2');
fprintf('%s\n', repmat('-',1,50));
fprintf('%-12s | %-10.5f | %-10.5f | %-8.5f\n', 'ANN',     rmse_val, mae_val, r2_val);
fprintf('%-12s | %-10.5f | %-10.5f | %-8.5f\n', 'Lin Reg', rmse_lr,  mae_lr,  r2_lr);

%% Plot 1: Predicted vs Actual over test samples
figure('Name','Predicted vs Actual','Position',[100 100 800 450]);
plot(y_actual_test, 'b-', 'LineWidth', 1.5); hold on;
plot(y_pred_test,   'r--','LineWidth', 1.5);
xlabel('Test Sample Index'); ylabel('Capacity (Ah)');
title('ANN: Predicted vs Actual Capacity (Test Set)');
legend('Actual','Predicted','Location','best');
grid on;

%% Plot 2: Regression scatter
figure('Name','Regression Plot','Position',[100 100 500 500]);
scatter(y_actual_test, y_pred_test, 30, 'filled', 'MarkerFaceColor',[0.2 0.5 0.8]);
hold on;
lims = [min(y_actual_test)*0.98, max(y_actual_test)*1.02];
plot(lims, lims, 'r-', 'LineWidth', 2);
xlabel('Actual Capacity (Ah)'); ylabel('Predicted Capacity (Ah)');
title(sprintf('Regression   R^2 = %.5f', r2_val));
grid on;

%% Plot 3: Error histogram
figure('Name','Error Histogram','Position',[100 100 600 400]);
histogram(err, 20, 'FaceColor',[0.2 0.5 0.8], 'EdgeColor','white');
xline(0, 'r--', 'LineWidth', 1.5);
xlabel('Prediction Error (Ah)'); ylabel('Frequency');
title(sprintf('Error Distribution   MAE = %.5f Ah', mae_val));
grid on;

fprintf('\n=== PIPELINE COMPLETE ===\n');
fprintf('Steps 1-9 map to your supervisor''s full methodology list.\n');
fprintf('See PIPELINE_README.md for the full mapping.\n');
