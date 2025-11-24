

% Load data
filename_1 = 'sensitivity_results/sensitivity_RODIU_20251101_231508.csv';  % ensure path is correct
delay_error_tab = readtable(filename_1);

% --- Check that the table has the expected column ---
if ~ismember('total_cost', delay_error_tab.Properties.VariableNames)
    error('Table does not contain a ''total_cost'' column. Columns found: %s', strjoin(delay_error_tab.Properties.VariableNames, ', '));
end

% Extract psi values
psi_error = 0.4:0.1:1.4;      % length = 11
psi_delay = 0.01:0.02:0.09;  % length = 5

delay_error_cost = delay_error_tab.total_cost;
delay_error_rental = delay_error_tab.C1_rental;
nExpected = numel(psi_error) * numel(psi_delay);

if numel(delay_error_cost) ~= nExpected
    error('Number of cost entries (%d) does not match psi grid size (%d).', numel(delay_error_cost), nExpected);
end

% Reshape: rows = length(psi_error), columns = length(psi_delay)
delay_error_cost_reshape = reshape(delay_error_cost, [numel(psi_error), numel(psi_delay)]);
delay_error_rental_reshape = reshape(delay_error_rental, [numel(psi_error), numel(psi_delay)]);

% If your CSV was exported row-wise (i.e., first varying psi_delay then psi_error),
% you may need to transpose:
% delay_error_cost_reshape = reshape(delay_error_cost, [numel(psi_error), numel(psi_delay)])';
% Use the following quick visual check:
% imagesc(delay_error_cost_reshape); colorbar; xlabel('psi\_delay index'); ylabel('psi\_error index');

% Plot cost
figure;
hold on; box on;
markers = {'-x','-s','-*','-^','-+'};
markerSizes = [14,10,10,10,14];

h = gobjects(1,numel(psi_delay));
for k = 1:numel(psi_delay)
    % plot column k vs psi_error
    h(k) = plot(psi_error, delay_error_cost_reshape(:,k), markers{k},'LineWidth', 2, 'MarkerSize', markerSizes(k));
    % fill marker faces
    set(h(k), 'MarkerFaceColor', get(h(k), 'Color'));
end

xlabel('\Psi_{\epsilon}', 'FontSize', 16);   % fixed typo
ylabel('Total Cost', 'FontSize', 16);

% dynamic legend using psi_delay values
legendStrings = arrayfun(@(x) sprintf('\\Psi_{\\Delta} = %.2f', x), psi_delay, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);

set(gca, 'FontSize', 16);
grid on
xlim([min(psi_error), max(psi_error)]);
xticks(min(psi_error):0.2:max(psi_error));
hold off;


%%

figure;
hold on; box on;
markers = {'-x','-s','-*','-^','-+'};
markerSizes = [14,10,10,10,14];

h = gobjects(1,numel(psi_delay));
for k = 1:numel(psi_delay)
    % plot column k vs psi_error
    h(k) = plot(psi_error, delay_error_rental_reshape(:,k), markers{k}, ...
                'LineWidth', 2, 'MarkerSize', markerSizes(k));
    % fill marker faces
    set(h(k), 'MarkerFaceColor', get(h(k), 'Color'));
end

xlabel('\Psi_{\epsilon}', 'FontSize', 16);   % fixed typo
ylabel('GPU rental Cost', 'FontSize', 16);

% dynamic legend using psi_delay values
legendStrings = arrayfun(@(x) sprintf('\\Psi_{\\Delta} = %.2f', x), psi_delay, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);

set(gca, 'FontSize', 16);
grid on
xlim([min(psi_error), max(psi_error)]);
xticks(min(psi_error):0.2:max(psi_error));
hold off;


%%



% Load data
filename_2 = 'sensitivity_results/sensitivity_memory_error_20251102_000559.csv';  % ensure path is correct
mem_error_tab = readtable(filename_2);

% --- Check that the table has the expected column ---
if ~ismember('total_cost', mem_error_tab.Properties.VariableNames)
    error('Table does not contain a ''total_cost'' column. Columns found: %s', strjoin(delay_error_tab.Properties.VariableNames, ', '));
end

% Extract psi values
psi_gpu = 0.6:0.2:2.0;    
psi_error = 0.6:0.2:2.0; 

gpu_error_cost = mem_error_tab.total_cost;
gpu_error_rental = mem_error_tab.C1_rental;
nExpected = numel(psi_gpu) * numel(psi_error);

if numel(gpu_error_cost) ~= nExpected
    error('Number of cost entries (%d) does not match psi grid size (%d).', numel(delay_error_cost), nExpected);
end

% Reshape: rows = length(psi_error), columns = length(psi_delay)
gpu_error_cost_reshape = reshape(gpu_error_cost, [numel(psi_gpu), numel(psi_error)]);
gpu_error_rental_reshape = reshape(gpu_error_rental, [numel(psi_gpu), numel(psi_error)]);

% If your CSV was exported row-wise (i.e., first varying psi_delay then psi_error),
% you may need to transpose:
% delay_error_cost_reshape = reshape(delay_error_cost, [numel(psi_error), numel(psi_delay)])';
% Use the following quick visual check:
% imagesc(delay_error_cost_reshape); colorbar; xlabel('psi\_delay index'); ylabel('psi\_error index');

% Plot cost
figure;
hold on; box on;
markers = {'-x','-s','-*','-^','-+'};
markerSizes = [14,10,10,10,14];

h = gobjects(1,numel(psi_gpu));
for k = 1:numel(psi_gpu)
    % plot column k vs psi_error
    h(k) = plot(psi_error, delay_error_cost_reshape(:,k), markers{k},'LineWidth', 2, 'MarkerSize', markerSizes(k));
    % fill marker faces
    set(h(k), 'MarkerFaceColor', get(h(k), 'Color'));
end

xlabel('\Psi_{\epsilon}', 'FontSize', 16);   % fixed typo
ylabel('Total Cost', 'FontSize', 16);

% dynamic legend using psi_delay values
legendStrings = arrayfun(@(x) sprintf('\\Psi_{gpu} = %.2f', x), psi_gpu, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);

set(gca, 'FontSize', 16);
grid on
xlim([min(psi_error), max(psi_error)]);
xticks(min(psi_error):0.2:max(psi_error));
hold off;




%%

% Plot memory/GPU sensitivity - fixed version
clear; close all; clc;

% Load data
filename_2 = 'sensitivity_results/sensitivity_memory_error_20251102_000559.csv';  % ensure path is correct
mem_error_tab = readtable(filename_2);

% --- Check that the table has the expected columns ---
expectedCols = {'total_cost','C1_rental'};
missing = setdiff(expectedCols, mem_error_tab.Properties.VariableNames);
if ~isempty(missing)
    error('Table is missing required column(s): %s. Columns found: %s', ...
          strjoin(missing, ', '), strjoin(mem_error_tab.Properties.VariableNames, ', '));
end

% Extract psi values (x-axis and series)
psi_gpu   = 0.6:0.2:2.0;    % series values (each line corresponds to a psi_gpu)
psi_error = 0.6:0.2:2.0;    % x-axis values
nExpected = numel(psi_gpu) * numel(psi_error);

% Extract numeric columns
gpu_error_cost   = mem_error_tab.total_cost;
gpu_error_rental = mem_error_tab.C1_rental;

% Sanity check counts
if numel(gpu_error_cost) ~= nExpected
    error('Number of cost entries (%d) does not match psi grid size (%d).', numel(gpu_error_cost), nExpected);
end
if numel(gpu_error_rental) ~= nExpected
    error('Number of rental entries (%d) does not match psi grid size (%d).', numel(gpu_error_rental), nExpected);
end

% Reshape so rows = length(psi_error) (x-axis), columns = length(psi_gpu) (lines)
gpu_error_cost_reshape   = reshape(gpu_error_cost,   [numel(psi_error), numel(psi_gpu)]);
gpu_error_rental_reshape = reshape(gpu_error_rental, [numel(psi_error), numel(psi_gpu)]);

% Quick check (uncomment to visually inspect):
% imagesc(gpu_error_cost_reshape); colorbar; xlabel('psi\_gpu index'); ylabel('psi\_error index');

% Prepare markers/styles (auto-expand if needed)
baseMarkers = {'-o','-s','-d','-^','-v','-<','->','-p','-*','-+'};
nSeries = numel(psi_gpu);
if nSeries <= numel(baseMarkers)
    markers = baseMarkers(1:nSeries);
else
    % repeat/extend marker list if more series than base markers
    markers = repmat(baseMarkers, 1, ceil(nSeries/numel(baseMarkers)));
    markers = markers(1:nSeries);
end
markers = {'-x','-s','-*','-^','-+','-p'};
markerSizes = [14,10,10,10,14,10];
% Plot cost vs psi_error for each psi_gpu
figure;
hold on; box on;

n_select = [1,2,3,4,5,6];
h = gobjects(1, length(n_select));

for idx = 1:length(n_select)
    k = n_select(idx);
    h(idx) = plot(psi_error, gpu_error_cost_reshape(:, k), markers{k}, ...
                  'LineWidth', 2, 'MarkerSize', markerSizes(k));
    % fill marker faces where supported
    try
        set(h(idx), 'MarkerFaceColor', get(h(idx), 'Color'));
    catch
        % some marker types may not support MarkerFaceColor — ignore quietly
    end
end

% Make legend only for the selected ones
legendStrings = arrayfun(@(x) sprintf('\\Psi_{p_c} = %.2f', psi_gpu(x)), n_select, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);

xlabel('\Psi_{\epsilon}', 'FontSize', 16);
ylabel('Total Cost', 'FontSize', 16);

set(gca, 'FontSize', 16);
grid on;
xlim([min(psi_error), max(psi_error)]);
xticks(min(psi_error):0.2:max(psi_error));
hold off;



%%

% --- Robust plotting: psi_gpu on x-axis, selected psi_error rows ---
n_select = [1,2,3,4,5,6];  % select ψ_error indices to plot
nSeries = length(n_select);
h = gobjects(1, nSeries);

% Basic sanity checks
if any(n_select < 1) || any(n_select > numel(psi_error))
    error('n_select contains invalid indices. Valid range is 1..%d (numel(psi_error)).', numel(psi_error));
end
% Ensure gpu_error_cost_reshape has shape [numel(psi_error), numel(psi_gpu)]
expectedSize = [numel(psi_error), numel(psi_gpu)];
if ~isequal(size(gpu_error_cost_reshape), expectedSize)
    if isequal(size(gpu_error_cost_reshape), fliplr(expectedSize))
        gpu_error_cost_reshape = gpu_error_cost_reshape';
    else
        error('gpu_error_cost_reshape has unexpected size %s; expected %s or its transpose.', mat2str(size(gpu_error_cost_reshape)), mat2str(expectedSize));
    end
end

% Marker settings (base set — will cycle if more series)
baseMarkers = {'-x','-s','-*','-^','-+','-p'};   % base marker styles
baseMarkerSizes = [14,10,10,10,14,10];        % base sizes (can be any length)

figure;
hold on; box on;

for idx = 1:nSeries
    j = n_select(idx);                         % actual row index in gpu_error_cost_reshape
    mIndex = mod(idx-1, numel(baseMarkers)) + 1;  % cycle through baseMarkers
    markerStyle = baseMarkers{mIndex};
    ms = baseMarkerSizes(mod(idx-1, numel(baseMarkerSizes))+1);

    % Plot using psi_gpu on x-axis, row j across columns (psi_gpu values)
    h(idx) = plot(psi_gpu, gpu_error_cost_reshape(j, :), markerStyle, ...
                  'LineWidth', 2, 'MarkerSize', ms);
    % Try to fill marker face where supported
    try
        set(h(idx), 'MarkerFaceColor', get(h(idx), 'Color'));
    catch
        % ignore if not supported
    end
end

xlabel('\Psi_{gpu}', 'FontSize', 16);
ylabel('Total Cost', 'FontSize', 16);

% Legend: show psi_error values corresponding to plotted rows
legendStrings = arrayfun(@(x) sprintf('\\Psi_{\\epsilon} = %.2f', psi_error(x)), n_select, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);

set(gca, 'FontSize', 14);
grid on;
xlim([min(psi_gpu), max(psi_gpu)]);
xticks(min(psi_gpu):0.2:max(psi_gpu));
hold off;





%% Sensitivity analysis on Budget and rental cost (fixed)

clearvars -except n_select;  % keep n_select if you set it earlier, otherwise we'll set a default below
close all;

% Load data
filename_3 = 'sensitivity_results/sensitivity_cost_budget_20251102_220702.csv';
cost_budget_tab = readtable(filename_3);

% --- Check that the table has the expected columns ---
expectedCols = {'total_cost','C1_rental'};
missing = setdiff(expectedCols, cost_budget_tab.Properties.VariableNames);
if ~isempty(missing)
    error('Table is missing required column(s): %s. Columns found: %s', ...
          strjoin(missing, ', '), strjoin(cost_budget_tab.Properties.VariableNames, ', '));
end

% Define psi vectors
psi_rental = 0.4:0.05:0.65;    % series values (each line corresponds to one rental psi)
psi_budget = 0.3:0.05:0.5;     % x-axis values (budget)

% Expected count and extract numeric columns
nExpected = numel(psi_rental) * numel(psi_budget);
budget_cost   = cost_budget_tab.total_cost;
budget_rental = cost_budget_tab.C1_rental;

% Sanity checks on lengths
if numel(budget_cost) ~= nExpected
    error('Number of cost entries (%d) does not match psi grid size (%d).', numel(budget_cost), nExpected);
end
if numel(budget_rental) ~= nExpected
    error('Number of rental entries (%d) does not match psi grid size (%d).', numel(budget_rental), nExpected);
end

% Reshape: rows = length(psi_budget) (x axis), cols = length(psi_rental) (series)
budget_cost_reshape   = reshape(budget_cost,   [numel(psi_budget), numel(psi_rental)]);
budget_rental_reshape = reshape(budget_rental, [numel(psi_budget), numel(psi_rental)]);

% If reshape produced unexpected orientation, attempt transpose (robust check)
expectedSize = [numel(psi_budget), numel(psi_rental)];
if ~isequal(size(budget_cost_reshape), expectedSize)
    if isequal(size(budget_cost_reshape), fliplr(expectedSize))
        budget_cost_reshape   = budget_cost_reshape';
        budget_rental_reshape = budget_rental_reshape';
    else
        error('Reshaped data has unexpected size %s; expected %s or its transpose.', ...
              mat2str(size(budget_cost_reshape)), mat2str(expectedSize));
    end
end

% Select which rental indices to plot (indices refer to psi_rental)
if ~exist('n_select','var') || isempty(n_select)
    % default: plot up to 5 evenly spaced rental indices
    n_select = round(linspace(1, numel(psi_rental), min(5, numel(psi_rental))));
end

% Validate n_select
if any(n_select < 1) || any(n_select > numel(psi_rental))
    error('n_select contains invalid indices. Valid range is 1..%d (numel(psi_rental)).', numel(psi_rental));
end

% Marker settings (will cycle if needed)
baseMarkers = {'-o','-s','-d','-^','-v','-<','->','-p','-*','-+'};
baseSizes   = [10,9,9,9,10,9,9,9,10,10];
nSeries = length(n_select);

figure;
hold on; box on;

h = gobjects(1, nSeries);
for idx = 1:nSeries
    j = n_select(idx);                     % actual rental index
    mIndex = mod(idx-1, numel(baseMarkers)) + 1;
    markerStyle = baseMarkers{mIndex};
    ms = baseSizes(mod(idx-1, numel(baseSizes)) + 1);

    % plot budget (x-axis) vs total cost for rental index j
    h(idx) = plot(psi_budget, budget_cost_reshape(:, j), markerStyle, ...
                  'LineWidth', 2, 'MarkerSize', ms);
    % try to fill marker face
    try
        set(h(idx), 'MarkerFaceColor', get(h(idx), 'Color'));
    catch
        % ignore unsupported marker face settings
    end
end

xlabel('\Psi_{\delta}', 'FontSize', 16);
ylabel('Total Cost', 'FontSize', 16);

% Legend labels correspond to psi_rental values (series)
legendStrings = arrayfun(@(x) sprintf('\\Psi_{p_c} = %.2f', psi_rental(x)), n_select, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);

set(gca, 'FontSize', 14);
grid on;
xlim([min(psi_budget), max(psi_budget)]);
xticks(min(psi_budget):0.05:max(psi_budget));
hold off;



%% Sensitivity analysis: GPU rental vs error (fixed & robust)
clear; close all; clc;

% Load data
filename_4 = 'sensitivity_results/sensitivity_gpu_cost_error_20251105_155657.csv';
gpu_rental_tab = readtable(filename_4);

% --- Check that the table has the expected columns (on the correct table) ---
expectedCols = {'total_cost','C1_rental'};
missing = setdiff(expectedCols, gpu_rental_tab.Properties.VariableNames);
if ~isempty(missing)
    error('Table is missing required column(s): %s. Columns found: %s', ...
          strjoin(missing, ', '), strjoin(gpu_rental_tab.Properties.VariableNames, ', '));
end

% Define psi vectors (adjust if your intended ranges differ)
psi_gpu   = 0.4:0.1:1.4;       % GPU rental values (one axis)
psi_error = 0.4:0.1:1.4;       % error values (other axis)

% Extract numeric columns
gpu_error_cost   = gpu_rental_tab.total_cost;
gpu_error_rental = gpu_rental_tab.C1_rental;

% Expected count
nExpected = numel(psi_gpu) * numel(psi_error);

% Sanity checks on lengths
if numel(gpu_error_cost) ~= nExpected
    error('Number of total_cost entries (%d) does not match psi grid size (%d).', numel(gpu_error_cost), nExpected);
end
if numel(gpu_error_rental) ~= nExpected
    error('Number of C1_rental entries (%d) does not match psi grid size (%d).', numel(gpu_error_rental), nExpected);
end

% Reshape - choice: rows = psi_error, cols = psi_gpu
% so that gpu_error_cost_reshape(i,j) is cost at psi_error(i), psi_gpu(j).
gpu_error_cost_reshape   = reshape(gpu_error_cost,   [numel(psi_error), numel(psi_gpu)]);
gpu_error_rental_reshape = reshape(gpu_error_rental, [numel(psi_error), numel(psi_gpu)]);

% If orientation seems swapped, try transpose (robust check)
expectedSize = [numel(psi_error), numel(psi_gpu)];
if ~isequal(size(gpu_error_cost_reshape), expectedSize)
    if isequal(size(gpu_error_cost_reshape), fliplr(expectedSize))
        gpu_error_cost_reshape   = gpu_error_cost_reshape';
        gpu_error_rental_reshape = gpu_error_rental_reshape';
    else
        error('Reshaped data has unexpected size %s; expected %s or its transpose.', mat2str(size(gpu_error_cost_reshape)), mat2str(expectedSize));
    end
end

% --- Plotting: Option A (psi_gpu on x-axis; each line = different psi_error) ---
% Choose which psi_error indices to plot
n_select = 1:1:11;  % indices into psi_error (change as needed)
if any(n_select < 1) || any(n_select > numel(psi_error))
    error('n_select indices must be in 1..%d (numel(psi_error)).', numel(psi_error));
end

% Marker styles (will cycle if needed)
baseMarkers = {'-x','-s','-*','-^','-+','-p'};   % base marker styles
baseSizes = [14,10,10,10,14,10];        % base sizes (can be any length)


figure;
hold on; box on;
h = gobjects(1, length(n_select));
for idx = 1:length(n_select)
    i = n_select(idx);                          % row index (psi_error index)
    mIndex = mod(idx-1, numel(baseMarkers)) + 1;
    markerStyle = baseMarkers{mIndex};
    ms = baseSizes(mod(idx-1, numel(baseSizes)) + 1);

    % x = psi_gpu, y = total cost for psi_error(i) across psi_gpu
    h(idx) = plot(psi_gpu, gpu_error_cost_reshape(i, :), markerStyle, ...
                  'LineWidth', 2, 'MarkerSize', ms);
    try
        set(h(idx), 'MarkerFaceColor', get(h(idx), 'Color'));
    catch
    end
end
xlabel('\Psi_{gpu}', 'FontSize', 14);
ylabel('Total cost', 'FontSize', 14);
legendStrings = arrayfun(@(x) sprintf('\\Psi_{error} = %.2f', psi_error(x)), n_select, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);
grid on;
xlim([min(psi_gpu), max(psi_gpu)]);
xticks(psi_gpu);
set(gca,'FontSize',12);
hold off;


%% --- Plotting: Option B (psi_error on x-axis; each line = different psi_gpu) ---
% Uncomment this block if you prefer psi_error on the x-axis and lines for psi_gpu
figure;
% Marker styles (will cycle if needed)
baseMarkers = {'-x','-s','-*','-^','-+','-p'};   % base marker styles
baseSizes = [14,10,10,10,14,10];        % base sizes (can be any length)


% Select psi_gpu indices to plot
n_select_gpu = 1:2:11;   % indices into psi_gpu
if any(n_select_gpu < 1) || any(n_select_gpu > numel(psi_gpu))
    error('n_select_gpu indices must be in 1..%d (numel(psi_gpu)).', numel(psi_gpu));
end

hold on; box on;
h2 = gobjects(1, length(n_select_gpu));
for idx = 1:length(n_select_gpu)
    j = n_select_gpu(idx);                     % column index (psi_gpu index)
    mIndex = mod(idx-1, numel(baseMarkers)) + 1;
    markerStyle = baseMarkers{mIndex};
    ms = baseSizes(mod(idx-1, numel(baseSizes)) + 1);

    % x = psi_error, y = total cost for this psi_gpu across psi_error
    h2(idx) = plot(psi_error, gpu_error_cost_reshape(:, j), markerStyle, ...
                   'LineWidth', 2, 'MarkerSize', ms);
    try
        set(h2(idx), 'MarkerFaceColor', get(h2(idx), 'Color'));
    catch
    end
end
xlabel('\Psi_{\epsilon}', 'FontSize', 14);
ylabel('Total cost', 'FontSize', 14);
legendStrings2 = arrayfun(@(x) sprintf('\\Psi_{p_c} = %.2f', psi_gpu(x)), n_select_gpu, 'UniformOutput', false);
legend(h2, legendStrings2, 'Location', 'northeast', 'FontSize', 16);
grid on;
xlim([min(psi_error), max(psi_error)]);
xticks(psi_error);
set(gca,'FontSize',16);
hold off;



%% Sensitivity analysis: GPU rental vs error (fixed & robust)
clear; close all; clc;

% Load data
filename_4 = 'sensitivity_results/sensitivity_gpu_cost_error_20251105_155657.csv';
gpu_rental_tab = readtable(filename_4);

% --- Check that the table has the expected columns (on the correct table) ---
expectedCols = {'total_cost','C1_rental'};
missing = setdiff(expectedCols, gpu_rental_tab.Properties.VariableNames);
if ~isempty(missing)
    error('Table is missing required column(s): %s. Columns found: %s', ...
          strjoin(missing, ', '), strjoin(gpu_rental_tab.Properties.VariableNames, ', '));
end

% Define psi vectors (adjust if your intended ranges differ)
psi_gpu   = 0.4:0.1:1.4;       % GPU rental values (one axis)
psi_error = 0.4:0.1:1.4;       % error values (other axis)

% Extract numeric columns
gpu_error_cost   = gpu_rental_tab.total_cost;
gpu_error_rental = gpu_rental_tab.C1_rental;

% Expected count
nExpected = numel(psi_gpu) * numel(psi_error);

% Sanity checks on lengths
if numel(gpu_error_cost) ~= nExpected
    error('Number of total_cost entries (%d) does not match psi grid size (%d).', numel(gpu_error_cost), nExpected);
end
if numel(gpu_error_rental) ~= nExpected
    error('Number of C1_rental entries (%d) does not match psi grid size (%d).', numel(gpu_error_rental), nExpected);
end

% Reshape - choice: rows = psi_error, cols = psi_gpu
% so that gpu_error_cost_reshape(i,j) is cost at psi_error(i), psi_gpu(j).
gpu_error_cost_reshape   = reshape(gpu_error_cost,   [numel(psi_error), numel(psi_gpu)]);
gpu_error_rental_reshape = reshape(gpu_error_rental, [numel(psi_error), numel(psi_gpu)]);

% If orientation seems swapped, try transpose (robust check)
expectedSize = [numel(psi_error), numel(psi_gpu)];
if ~isequal(size(gpu_error_cost_reshape), expectedSize)
    if isequal(size(gpu_error_cost_reshape), fliplr(expectedSize))
        gpu_error_cost_reshape   = gpu_error_cost_reshape';
        gpu_error_rental_reshape = gpu_error_rental_reshape';
    else
        error('Reshaped data has unexpected size %s; expected %s or its transpose.', mat2str(size(gpu_error_cost_reshape)), mat2str(expectedSize));
    end
end

% --- Plotting: Option A (psi_gpu on x-axis; each line = different psi_error) ---
% Choose which psi_error indices to plot
n_select = 3:2:11;  % indices into psi_error (change as needed)
if any(n_select < 1) || any(n_select > numel(psi_error))
    error('n_select indices must be in 1..%d (numel(psi_error)).', numel(psi_error));
end

% Marker styles (will cycle if needed)
baseMarkers = {'-x','-s','-*','-^','-+','-p'};   % base marker styles
baseSizes = [14,10,10,10,14,10];        % base sizes (can be any length)


figure;
hold on; box on;
h = gobjects(1, length(n_select));
for idx = 1:length(n_select)
    i = n_select(idx);                          % row index (psi_error index)
    mIndex = mod(idx-1, numel(baseMarkers)) + 1;
    markerStyle = baseMarkers{mIndex};
    ms = baseSizes(mod(idx-1, numel(baseSizes)) + 1);

    % x = psi_gpu, y = total cost for psi_error(i) across psi_gpu
    h(idx) = plot(psi_gpu, gpu_error_rental_reshape(i, :), markerStyle, ...
                  'LineWidth', 2, 'MarkerSize', ms);
    try
        set(h(idx), 'MarkerFaceColor', get(h(idx), 'Color'));
    catch
    end
end
xlabel('\Psi_{p_c}', 'FontSize', 14);
ylabel('GPU Rental cost', 'FontSize', 14);
legendStrings = arrayfun(@(x) sprintf('\\Psi_{\\epsilon} = %.2f', psi_error(x)), n_select, 'UniformOutput', false);
legend(h, legendStrings, 'Location', 'northeast', 'FontSize', 16);
grid on;
xlim([min(psi_gpu), max(psi_gpu)]);
xticks(psi_gpu);
set(gca,'FontSize',12);
hold off;