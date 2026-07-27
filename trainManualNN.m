function result = trainManualNN(X_train, y_train, X_val, y_val, opts)
%% ============================================================
%% FROM-SCRATCH NEURAL NETWORK TRAINING
%% Covers methodology items 8-16:
%%   8  Initialize weights
%%   9  Forward propagation
%%   10 Calculate loss (MSE)
%%   11 Backpropagation (chain rule)
%%   12 Update weights (Adam optimizer)
%%   13 Validate on validation set
%%   14 Repeat for many epochs
%%   16 Early stopping to prevent overfitting
%% ============================================================
% This is a plain, single-hidden-layer network written by hand (no
% toolbox training function) specifically so every step of the math
% is visible and explainable - unlike fitnet/trainlm, which hides all
% of this inside compiled code.
%
% INPUTS
%   X_train, X_val : N x d matrices (rows = samples, cols = features)
%   y_train, y_val : N x 1 column vectors (standardized target)
%   opts           : struct with fields (all optional, defaults shown)
%       .hidden_size    (10)      number of hidden neurons
%       .activation     ('tanh')  'tanh' | 'sigmoid' | 'relu'
%       .learning_rate  (0.01)    Adam step size (eta)
%       .max_epochs     (500)
%       .patience       (30)      epochs to wait for improvement
%       .beta1          (0.9)     Adam momentum term
%       .beta2          (0.999)   Adam RMS term
%       .adam_eps       (1e-8)
%       .seed           (1)
%       .verbose        (false)
%
% OUTPUT (struct)
%   .W1,.b1,.W2,.b2      best weights (lowest validation loss seen)
%   .train_loss_history, .val_loss_history
%   .stopped_epoch, .best_epoch
%   .predict             function handle: predict(X) -> y_hat (N x 1)
% ============================================================

%% ---- Step 7 (carried over): read hyperparameters ----
if nargin < 5, opts = struct(); end
hidden_size   = getOpt(opts, 'hidden_size', 10);
activation    = getOpt(opts, 'activation', 'tanh');
learning_rate = getOpt(opts, 'learning_rate', 0.01);
max_epochs    = getOpt(opts, 'max_epochs', 500);
patience      = getOpt(opts, 'patience', 30);
beta1         = getOpt(opts, 'beta1', 0.9);
beta2         = getOpt(opts, 'beta2', 0.999);
adam_eps      = getOpt(opts, 'adam_eps', 1e-8);
seed          = getOpt(opts, 'seed', 1);
verbose       = getOpt(opts, 'verbose', false);

[act_fn, act_deriv] = getActivation(activation);

d = size(X_train, 2);   % number of input features
N = size(X_train, 1);   % number of training samples

Xt      = X_train';     % d x N   (transpose once, reuse every epoch)
y_row   = y_train(:)';  % 1 x N
Xt_val  = X_val';
y_val_r = y_val(:)';

%% ------------------------------------------------------------
%% STEP 8: INITIALIZE WEIGHTS
%% ------------------------------------------------------------
% Weights start as small random numbers (not zero - if every neuron
% started identical, they'd all learn the exact same thing forever).
% Scale depends on activation: He init for ReLU, Xavier for tanh/sigmoid.
rng(seed);
if strcmp(activation, 'relu')
    scale1 = sqrt(2/d);
else
    scale1 = sqrt(1/d);
end
W1 = randn(hidden_size, d) * scale1;
b1 = zeros(hidden_size, 1);
W2 = randn(1, hidden_size) * sqrt(1/hidden_size);
b2 = 0;

% Adam "memory" of past gradients (starts at zero)
mW1 = zeros(size(W1)); vW1 = zeros(size(W1));
mb1 = zeros(size(b1)); vb1 = zeros(size(b1));
mW2 = zeros(size(W2)); vW2 = zeros(size(W2));
mb2 = 0;               vb2 = 0;

train_loss_history = zeros(max_epochs, 1);
val_loss_history    = zeros(max_epochs, 1);

best_val_loss = inf;
best_epoch    = 0;
best_W1 = W1; best_b1 = b1; best_W2 = W2; best_b2 = b2;
fails_since_best = 0;

if verbose
    fprintf('Initialized: %d inputs -> %d hidden (%s) -> 1 output\n', ...
        d, hidden_size, activation);
end

%% ------------------------------------------------------------
%% STEPS 9-16: EPOCH LOOP
%% (forward prop -> loss -> backprop -> Adam update -> validate ->
%%  repeat -> early stop)
%% ------------------------------------------------------------
for epoch = 1:max_epochs

    %% STEP 9: FORWARD PROPAGATION (training data)
    Z1 = W1 * Xt + b1;          % hidden_size x N
    A1 = act_fn(Z1);            % activated hidden layer
    Z2 = W2 * A1 + b2;          % 1 x N  (linear output -> regression)
    y_hat = Z2;

    %% STEP 10: CALCULATE LOSS (Mean Squared Error, for regression)
    % MSE = (1/N) * sum( (prediction - actual)^2 )
    err = y_hat - y_row;
    train_loss = mean(err.^2);
    train_loss_history(epoch) = train_loss;

    %% STEP 11: BACKPROPAGATION - gradients via the chain rule
    % Each gradient below is literally "how much does the loss change
    % if I nudge this weight?", found by chaining derivatives back
    % from the output to each layer.
    dZ2 = 2 * err / N;                 % dLoss/dZ2         (1 x N)
    dW2 = dZ2 * A1';                   % dLoss/dW2         (1 x hidden)
    db2 = sum(dZ2, 2);                 % dLoss/db2         (scalar)

    dA1 = W2' * dZ2;                   % dLoss/dA1         (hidden x N)
    dZ1 = dA1 .* act_deriv(Z1);        % dLoss/dZ1 (chain rule through activation)
    dW1 = dZ1 * Xt';                   % dLoss/dW1         (hidden x d)
    db1 = sum(dZ1, 2);                 % dLoss/db1         (hidden x 1)

    %% STEP 12: UPDATE WEIGHTS - Adam optimizer
    % Adam keeps a running average of the gradient (m, like momentum)
    % and of its squared size (v, to auto-adjust step size per
    % weight), then updates:  w_new = w_old - eta * mhat / (sqrt(vhat)+eps)
    t = epoch;
    [W1, mW1, vW1] = adamStep(W1, dW1, mW1, vW1, t, beta1, beta2, adam_eps, learning_rate);
    [b1, mb1, vb1] = adamStep(b1, db1, mb1, vb1, t, beta1, beta2, adam_eps, learning_rate);
    [W2, mW2, vW2] = adamStep(W2, dW2, mW2, vW2, t, beta1, beta2, adam_eps, learning_rate);
    [b2, mb2, vb2] = adamStep(b2, db2, mb2, vb2, t, beta1, beta2, adam_eps, learning_rate);

    %% STEP 13: VALIDATE ON VALIDATION SET
    % Forward pass only (no weight update) using the CURRENT weights,
    % on data the network has never trained on.
    Z1_val = W1 * Xt_val + b1;
    A1_val = act_fn(Z1_val);
    y_hat_val = W2 * A1_val + b2;
    val_loss = mean((y_hat_val - y_val_r).^2);
    val_loss_history(epoch) = val_loss;

    if verbose && mod(epoch, 50) == 0
        fprintf('  Epoch %4d | train MSE = %.5f | val MSE = %.5f\n', ...
            epoch, train_loss, val_loss);
    end

    %% STEP 16: EARLY STOPPING
    % If validation loss improves, save these weights as "best so far"
    % and reset the patience counter. If it goes `patience` epochs
    % without improving, stop - further training would just be
    % memorizing the training set (overfitting) rather than learning
    % anything that generalizes.
    if val_loss < best_val_loss
        best_val_loss    = val_loss;
        best_epoch       = epoch;
        best_W1 = W1; best_b1 = b1; best_W2 = W2; best_b2 = b2;
        fails_since_best = 0;
    else
        fails_since_best = fails_since_best + 1;
    end

    if fails_since_best >= patience
        if verbose
            fprintf('  Early stopping at epoch %d (best was epoch %d, val MSE = %.5f)\n', ...
                epoch, best_epoch, best_val_loss);
        end
        train_loss_history = train_loss_history(1:epoch);
        val_loss_history    = val_loss_history(1:epoch);
        break;
    end
    %% STEP 14: repeat (the `for` loop itself does this)
end

%% ---- Package results, using the BEST weights seen (not the last) ----
result.W1 = best_W1; result.b1 = best_b1;
result.W2 = best_W2; result.b2 = best_b2;
result.train_loss_history = train_loss_history;
result.val_loss_history    = val_loss_history;
result.best_epoch    = best_epoch;
result.stopped_epoch  = length(train_loss_history);
result.best_val_loss  = best_val_loss;
result.activation     = activation;
result.predict = @(Xq) predictWithNet(Xq, best_W1, best_b1, best_W2, best_b2, act_fn);

end

%% ================= local helper functions =================

function v = getOpt(opts, field, default)
    if isfield(opts, field)
        v = opts.(field);
    else
        v = default;
    end
end

function [act_fn, act_deriv] = getActivation(name)
    switch lower(name)
        case 'tanh'
            act_fn    = @(z) tanh(z);
            act_deriv = @(z) 1 - tanh(z).^2;
        case 'sigmoid'
            act_fn    = @(z) 1 ./ (1 + exp(-z));
            act_deriv = @(z) (1./(1+exp(-z))) .* (1 - 1./(1+exp(-z)));
        case 'relu'
            act_fn    = @(z) max(0, z);
            act_deriv = @(z) double(z > 0);
        otherwise
            error('Unknown activation "%s". Use tanh, sigmoid, or relu.', name);
    end
end

function [param, m, v] = adamStep(param, grad, m, v, t, beta1, beta2, eps_, eta)
    m = beta1*m + (1-beta1)*grad;
    v = beta2*v + (1-beta2)*(grad.^2);
    m_hat = m / (1 - beta1^t);
    v_hat = v / (1 - beta2^t);
    param = param - eta * m_hat ./ (sqrt(v_hat) + eps_);
end

function y_hat = predictWithNet(Xq, W1, b1, W2, b2, act_fn)
    Zt = Xq';
    Z1 = W1 * Zt + b1;
    A1 = act_fn(Z1);
    Z2 = W2 * A1 + b2;
    y_hat = Z2';
end
