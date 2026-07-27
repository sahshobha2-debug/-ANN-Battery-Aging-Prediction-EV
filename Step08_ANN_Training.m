%% ============================================================
%% STEP 8 — FULL ANN TRAINING
%% Methodology items 8-16, using the best hyperparameters from Step 7
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   This is the "real" training run. It uses trainManualNN.m - the
%   from-scratch network that explicitly shows:
%     weight init -> forward propagation -> MSE loss -> backprop via
%     chain rule -> Adam weight update -> validation check -> repeat
%     for many epochs -> early stopping if validation stops improving
%
%   Open trainManualNN.m alongside this script when presenting to your
%   supervisor - each numbered STEP comment in that file corresponds
%   directly to one line item on the methodology list.
% ============================================================

load('Step06_data.mat');
load('Step07_data.mat');
fprintf('=== STEP 8: FULL ANN TRAINING ===\n\n');

fprintf('Using tuned hyperparameters from Step 7:\n');
fprintf('  Activation     : %s\n', best_activation);
fprintf('  Hidden neurons : %d\n', best_hidden_size);
fprintf('  Learning rate  : %.4f\n\n', best_learning_rate);

opts               = struct();
opts.activation    = best_activation;
opts.hidden_size   = best_hidden_size;
opts.learning_rate = best_learning_rate;
opts.max_epochs    = 2000;   % full budget this time
opts.patience      = 100;    % more patience for the real run
opts.seed          = 1;
opts.verbose       = true;

net_result = trainManualNN(X_train, y_train, X_val, y_val, opts);

fprintf('\nTraining stopped at epoch %d (best epoch was %d)\n', ...
    net_result.stopped_epoch, net_result.best_epoch);

%% Plot: training vs validation loss curve (shows early stopping point)
figure('Name','Training Curve','Position',[100 100 800 450]);
plot(net_result.train_loss_history, 'b-', 'LineWidth', 1.5); hold on;
plot(net_result.val_loss_history,    'r-', 'LineWidth', 1.5);
xline(net_result.best_epoch, 'k--', 'Best epoch (early stop point)', ...
    'LabelVerticalAlignment','bottom');
xlabel('Epoch'); ylabel('MSE Loss (standardized units)');
title('Training vs Validation Loss (early stopping)');
legend('Training loss','Validation loss','Location','best');
grid on;

%% Predict on the TEST set (touched here for the first and only time)
y_pred_test_std = net_result.predict(X_test);      % standardized scale
y_pred_test     = y_pred_test_std * y_sigma + y_mu;  % back to real Ah
y_actual_test   = y_capacity(test_idx);

fprintf('\nTest-set predictions generated (%d samples).\n', length(y_actual_test));
fprintf('Proceed to Step 9 for full metrics (RMSE, MAE, MSE, R^2).\n');

%% Save checkpoint for Step 9
save('Step08_data.mat', 'net_result', 'y_pred_test', 'y_actual_test', ...
    'y_mu', 'y_sigma', 'best_activation', 'best_hidden_size', 'best_learning_rate');
fprintf('\nSaved -> Step08_data.mat\n');
fprintf('Next: run Step09_EvaluationMetrics.m\n');
