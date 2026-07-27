%% ============================================================
%% STEP 7 — HYPERPARAMETER TUNING
%% Methodology item 7 (and item 15, revisited)
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   "Hyperparameters" are the settings YOU choose before training even
%   starts (as opposed to weights, which the network learns on its
%   own). Here we search over:
%     - activation function : tanh / sigmoid / relu
%                              (the "shape" each neuron uses to react)
%     - hidden layer size    : how many neurons in the hidden layer
%     - learning rate (eta)  : how big a step Adam takes each update
%                              (too big = overshoots, too small = slow)
%
%   We try every combination on a SHORT training budget (fewer epochs,
%   tight patience) purely to compare them fairly and cheaply. The
%   winning combination is then trained properly, for real, in Step 8.
% ============================================================
load('Step06_data.mat');
fprintf('=== STEP 7: HYPERPARAMETER TUNING (grid search) ===\n\n');

%% Define the search grid
activations    = {'tanh', 'sigmoid', 'relu'};
hidden_sizes   = [5, 10, 15, 20];
learning_rates = [0.001, 0.01, 0.05];

n_combos = length(activations) * length(hidden_sizes) * length(learning_rates);
fprintf('Testing %d combinations (%d activations x %d hidden sizes x %d learning rates)\n\n', ...
    n_combos, length(activations), length(hidden_sizes), length(learning_rates));

results = struct('activation',{}, 'hidden_size',{}, 'learning_rate',{}, 'val_rmse',{});
combo = 0;

fprintf('%-10s | %-12s | %-14s | %-10s\n', 'Activation','Hidden Size','Learning Rate','Val RMSE');
fprintf('%s\n', repmat('-',1,55));

for a = 1:length(activations)
    for h = 1:length(hidden_sizes)
        for lr = 1:length(learning_rates)
            combo = combo + 1;

            opts = struct();
            opts.activation    = activations{a};
            opts.hidden_size   = hidden_sizes(h);
            opts.learning_rate = learning_rates(lr);
            opts.max_epochs    = 150;   % short budget - just for comparison
            opts.patience      = 20;
            opts.seed          = 1;
            opts.verbose       = false;

            net_result = trainManualNN(X_train, y_train, X_val, y_val, opts);

            % Convert standardized validation loss back to real RMSE (Ah)
            val_rmse = sqrt(net_result.best_val_loss) * y_sigma;

            results(combo).activation    = activations{a};
            results(combo).hidden_size   = hidden_sizes(h);
            results(combo).learning_rate = learning_rates(lr);
            results(combo).val_rmse      = val_rmse;

            fprintf('%-10s | %-12d | %-14.4f | %-10.5f\n', ...
                activations{a}, hidden_sizes(h), learning_rates(lr), val_rmse);
        end
    end
end

%% Find and report the best combination
[~, best_i] = min([results.val_rmse]);
best_activation    = results(best_i).activation;
best_hidden_size   = results(best_i).hidden_size;
best_learning_rate = results(best_i).learning_rate;

fprintf('\n=== BEST HYPERPARAMETERS ===\n');
fprintf('Activation     : %s\n', best_activation);
fprintf('Hidden neurons : %d\n', best_hidden_size);
fprintf('Learning rate  : %.4f\n', best_learning_rate);
fprintf('Val RMSE       : %.5f Ah (short-budget search)\n', results(best_i).val_rmse);

%% Plot: hyperparameter comparison
figure('Name','Hyperparameter Search','Position',[100 100 900 450]);
val_rmse_all = [results.val_rmse];
bar(val_rmse_all);
xlabel('Combination index (see table printed above)');
ylabel('Validation RMSE (Ah)');
title('Hyperparameter Grid Search Results');
grid on;

%% Save checkpoint for Step 8
save('Step07_data.mat', 'results', 'best_activation', 'best_hidden_size', ...
    'best_learning_rate');
fprintf('\nSaved -> Step07_data.mat\n');
fprintf('Next: run Step08_ANN_Training.m\n');
