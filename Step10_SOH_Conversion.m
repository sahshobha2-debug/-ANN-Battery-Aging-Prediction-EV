%% ============================================================
%% STEP 10 — CONVERT RESULTS TO SOH (%)
%% Extends methodology item 17 into the practical, presentable unit
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   Capacity in Ah is precise but not intuitive - "1.87 Ah" doesn't
%   mean much on its own. State of Health (SOH) fixes that by
%   expressing capacity as a PERCENTAGE of what the battery could hold
%   brand new:
%       SOH (%) = (capacity now / capacity on cycle 1) x 100
%   A battery at 100% SOH is like new; most EVs are considered due for
%   replacement around 80% SOH; below ~60% is usually only fit for
%   recycling. This script re-expresses your Step 9 test-set results
%   in those terms and adds the standard health "traffic light" zones.
%
%   IMPORTANT: each battery has its OWN initial capacity (they don't
%   all start at exactly the same Ah), so SOH must be calculated
%   per-battery, using that battery's own first-cycle capacity as the
%   100% reference point - not one shared number for every battery.
% ============================================================

load('Step01_data.mat', 'all_capacity', 'battery_label', 'battery_names');
load('Step05_data.mat', 'batt_label');
load('Step06_data.mat', 'test_idx');
load('Step08_data.mat', 'y_pred_test', 'y_actual_test');
fprintf('=== STEP 10: SOH (%%) CONVERSION ===\n\n');

%% Find each battery's initial (cycle-1) capacity from the RAW,
%% unfiltered Step 1 data - this must come from before any rows were
%% removed in Step 2, so "cycle 1" really is that battery's first cycle.
n_batteries  = length(battery_names);
initial_cap  = zeros(n_batteries, 1);
for b = 1:n_batteries
    first_row = find(battery_label == b, 1, 'first');
    initial_cap(b) = all_capacity(first_row);
    fprintf('%s initial capacity (cycle 1): %.4f Ah\n', battery_names{b}, initial_cap(b));
end

%% Map each TEST sample to its battery, then to that battery's SOH scale
test_batt_id = batt_label(test_idx);           % which battery each test row came from
ref_cap      = initial_cap(test_batt_id);       % that battery's 100% reference

SOH_actual = (y_actual_test ./ ref_cap) * 100;
SOH_pred   = (y_pred_test   ./ ref_cap) * 100;

%% Metrics in SOH (%) units
err_soh   = SOH_pred - SOH_actual;
rmse_soh  = sqrt(mean(err_soh.^2));
mae_soh   = mean(abs(err_soh));
ss_res    = sum((SOH_actual - SOH_pred).^2);
ss_tot    = sum((SOH_actual - mean(SOH_actual)).^2);
r2_soh    = 1 - ss_res/ss_tot;

fprintf('\n--- SOH (%%) metrics on test set ---\n');
fprintf('RMSE : %.3f %% SOH\n', rmse_soh);
fprintf('MAE  : %.3f %% SOH\n', mae_soh);
fprintf('R^2  : %.5f\n', r2_soh);

%% Reuse/second-life/recycle zone agreement
zone_actual = arrayfun(@soh_zone, SOH_actual, 'UniformOutput', false);
zone_pred   = arrayfun(@soh_zone, SOH_pred,   'UniformOutput', false);
zone_match  = strcmp(zone_actual, zone_pred);
fprintf('\nCorrect health zone predicted: %d of %d test samples (%.1f%%)\n', ...
    sum(zone_match), length(zone_match), 100*sum(zone_match)/length(zone_match));

%% Plot: predicted vs actual SOH with classification zones
n = length(SOH_actual);
figure('Name','SOH Classification','Position',[100 100 850 460]);
fill([0 n n 0],[80 80 100 100],'g','FaceAlpha',0.08,'EdgeColor','none'); hold on;
fill([0 n n 0],[60 60 80  80 ],'y','FaceAlpha',0.08,'EdgeColor','none');
fill([0 n n 0],[0  0  60  60 ],'r','FaceAlpha',0.08,'EdgeColor','none');
plot(SOH_actual,'b-', 'LineWidth',1.5);
plot(SOH_pred,  'r--','LineWidth',1.5);
yline(80,'g-','LineWidth',1.2);
yline(60,'k--','LineWidth',1.2);
text(2,91,'Reuse in EV (>=80%)','FontSize',9,'Color',[0 0.5 0]);
text(2,70,'Second-life (60-80%)','FontSize',9,'Color',[0.6 0.5 0]);
text(2,50,'Recycle (<60%)','FontSize',9,'Color',[0.7 0 0]);
xlabel('Test Sample Index'); ylabel('SOH (%)');
title(sprintf('Predicted vs Actual SOH   (RMSE = %.2f%%, R^2 = %.4f)', rmse_soh, r2_soh));
legend('Actual SOH','Predicted SOH','Location','best');
grid on;

%% Plot: SOH regression scatter
figure('Name','SOH Regression','Position',[100 100 500 500]);
scatter(SOH_actual, SOH_pred, 28, 'filled', 'MarkerFaceColor',[0.2 0.5 0.8]);
hold on;
lims = [min(SOH_actual)*0.98, max(SOH_actual)*1.02];
plot(lims, lims, 'r-', 'LineWidth', 2);
xlabel('Actual SOH (%)'); ylabel('Predicted SOH (%)');
title(sprintf('SOH Regression   R^2 = %.4f', r2_soh));
grid on;

fprintf('\nSaved SOH results -> Step10_data.mat\n');
save('Step10_data.mat', 'SOH_actual', 'SOH_pred', 'rmse_soh', 'mae_soh', ...
    'r2_soh', 'initial_cap', 'battery_names');

%% ---- local function ----
function z = soh_zone(soh_val)
    if soh_val >= 80
        z = 'reuse';
    elseif soh_val >= 60
        z = 'second-life';
    else
        z = 'recycle';
    end
end
