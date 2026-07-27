%% ============================================================
%% STEP 1 — DATA COLLECTION
%% Methodology item 1: define INPUT FEATURES and OUTPUT TARGET
%% ============================================================
% PLAIN-ENGLISH SUMMARY
%   Before a neural network can learn anything, we need two things:
%     INPUT  = the measurements we feed IN     (the "clues")
%     OUTPUT = the thing we want to PREDICT    (the "answer")
%
%   The raw data comes from NASA's battery aging experiments. Each
%   battery is repeatedly charged, then discharged, sometimes with an
%   impedance (resistance) test in between. Each of these is one "cycle".
%
%   INPUT FEATURES (13 numbers calculated per cycle):
%     1  Dis Mean V   - average voltage during discharge
%     2  Dis Min V    - lowest voltage reached during discharge
%     3  Dis Mean T   - average temperature during discharge
%     4  Dis Max T    - peak temperature during discharge
%     5  Dis Time     - how long the discharge lasted
%     6  Chg Mean V   - average voltage during charge
%     7  Chg End V    - final voltage at end of charge
%     8  Chg Mean T   - average temperature during charge
%     9  Chg Time     - how long the charge lasted
%     10 Re           - ohmic resistance (electrolyte health)
%     11 Rct          - charge-transfer resistance (electrode health)
%     12 Bat Imp      - overall battery impedance magnitude
%     13 Cycle Idx    - which cycle number this is (aging proxy)
%
%   OUTPUT TARGET:
%     Capacity (Ah) - how much charge the battery could hold on that
%     cycle. This is later converted to SOH (%) = capacity / initial
%     capacity x 100.
%
%   NOTE: "Dis Mean I" (mean discharge current) is deliberately NOT
%   collected as a feature. It would leak the answer directly, because
%   capacity is calculated FROM current (cap = integral of current over
%   time), so using it as an input would be like using the answer key
%   as a clue.
% ============================================================

clc; close all;
% NOTE: no 'clear' here on purpose - this script can be launched by
% RunAll_Master.m, and 'clear' would wipe that script's own loop
% variables mid-run. If you're running this file by itself and want a
% clean workspace first, just type `clear` in the command window
% before running it.
fprintf('=== STEP 1: DATA COLLECTION ===\n\n');

%% Load raw NASA battery files
load('B0005.mat');
load('B0006.mat');
load('B0018.mat');
load('B0007.mat');

datasets      = {B0005, B0006, B0018, B0007};
battery_names = {'B0005', 'B0006', 'B0018', 'B0007'};

feature_names = {
    'Dis Mean V',  'Dis Min V',  'Dis Mean T', ...
    'Dis Max T',   'Dis Time',   'Chg Mean V', ...
    'Chg End V',   'Chg Mean T', 'Chg Time',   ...
    'Re',          'Rct',        'Bat Imp',    ...
    'Cycle Idx'
};

all_features   = [];   % N x 13 matrix (input features)
all_capacity   = [];   % N x 1 vector  (output target, Ah)
battery_label  = [];   % which battery each row came from (1/2/3)

%% Extract features from each battery
% Each charge cycle is matched to the discharge that follows it, and
% (if present) the impedance test in between. This is necessary because
% NASA's logging order changes partway through each experiment:
%   Phase 1 (early cycles): charge -> discharge            (no impedance)
%   Phase 2 (later cycles): charge -> impedance -> discharge
for d = 1:length(datasets)
    data   = datasets{d};
    cycles = data.cycle;
    n      = length(cycles);

    aligned_charge = [];
    aligned_dis    = [];
    aligned_imp    = [];

    i = 1;
    while i <= n
        if strcmp(cycles(i).type, 'charge')
            charge_idx = i; imp_idx = 0; dis_idx = 0;
            j = i + 1;
            while j <= n && ~strcmp(cycles(j).type, 'charge')
                if strcmp(cycles(j).type, 'impedance') && imp_idx == 0
                    imp_idx = j;
                elseif strcmp(cycles(j).type, 'discharge') && dis_idx == 0
                    dis_idx = j;
                end
                j = j + 1;
            end
            if dis_idx > 0
                aligned_charge(end+1) = charge_idx; %#ok<AGROW>
                aligned_dis(end+1)    = dis_idx;    %#ok<AGROW>
                aligned_imp(end+1)    = imp_idx;    %#ok<AGROW>
            end
            i = j;
        else
            i = i + 1;
        end
    end

    fprintf('Battery %s: %d aligned charge-discharge pairs (%d with impedance)\n', ...
        battery_names{d}, length(aligned_charge), sum(aligned_imp > 0));

    for k = 1:length(aligned_charge)
        %--- discharge (source of the output target) ---
        dis   = cycles(aligned_dis(k)).data;
        V_dis = dis.Voltage_measured;
        I_dis = dis.Current_measured;
        T_dis = dis.Temperature_measured;
        t_dis = dis.Time;

        cap = abs(trapz(t_dis, abs(I_dis))) / 3600;  % Ah, from current integration
        if cap < 0.5, continue; end  % discard clearly-corrupt readings

        f1 = mean(V_dis); f2 = min(V_dis);
        f3 = mean(T_dis); f4 = max(T_dis);
        f5 = max(t_dis);

        %--- charge ---
        chg   = cycles(aligned_charge(k)).data;
        V_chg = chg.Voltage_measured;
        T_chg = chg.Temperature_measured;
        t_chg = chg.Time;

        f6 = mean(V_chg); f7 = V_chg(end);
        f8 = mean(T_chg); f9 = max(t_chg);

        %--- impedance (may be missing -> NaN, handled in Step 2) ---
        if aligned_imp(k) > 0
            imp = cycles(aligned_imp(k)).data;
            f10 = mean(imp.Re);
            f11 = mean(imp.Rct);
            f12 = mean(abs(imp.Battery_impedance));
        else
            f10 = NaN; f11 = NaN; f12 = NaN;
        end

        f13 = k;

        row = [f1 f2 f3 f4 f5 f6 f7 f8 f9 f10 f11 f12 f13];
        all_features  = [all_features; row];        %#ok<AGROW>
        all_capacity  = [all_capacity; cap];         %#ok<AGROW>
        battery_label = [battery_label; d];          %#ok<AGROW>
    end
end

fprintf('\nTotal cycles collected : %d\n', size(all_features,1));
fprintf('Input features          : %d  (see list above)\n', size(all_features,2));
fprintf('Output target           : Capacity (Ah)\n');

%% Save checkpoint for Step 2
save('Step01_data.mat', 'all_features', 'all_capacity', 'battery_label', ...
    'feature_names', 'battery_names');
fprintf('\nSaved -> Step01_data.mat\n');
fprintf('Next: run Step02_DataPreprocessing.m\n');
