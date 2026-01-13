close all;
clear;
clc;
%% Variable initialization
theta = (-90:0.1:90-0.1)*pi/180; % rad
lambda = 1; % Wavelength
M = 12; % Number of array elements
A = generateSteeringVector(theta, M, lambda);
desDirs_c = 0.0;

%% Array response for the equivalent directions
Q = 160; % With Q=40, 12 ULA curve is exactly as in the paper
phi = 1; % Quantization step
eqDir = -1:phi/Q:1-phi/Q; % Equivalent scanning directions
Aq = generateQuantizedArrResponse(M, eqDir);

%% Reference beam - Generate desired and reference patterns
[PdM, P_refGen, W0] = generateDesPattern(eqDir, sin(desDirs_c), Aq);
P_init = ones(size(eqDir));
PM = P_init;
alpha = sort([find(ismember(eqDir, eqDir(1:4:end))), find(PdM)]);

%% Hyperparameter Tuning: Define 3 scenarios
scenarios = struct();

% Scenario 1: Fast (few wolves, fewer iterations)
scenarios(1).name = 'Fast';
scenarios(1).SearchAgents = 20;
scenarios(1).MaxIter = 100;

% Scenario 2: Balanced (moderate wolves and iterations)
scenarios(2).name = 'Balanced';
scenarios(2).SearchAgents = 30;
scenarios(2).MaxIter = 300;

% Scenario 3: Thorough (many wolves, many iterations)
scenarios(3).name = 'Thorough';
scenarios(3).SearchAgents = 50;
scenarios(3).MaxIter = 500;

num_scenarios = length(scenarios);

% Storage for results
results = struct();
for s = 1:num_scenarios
    results(s).name = scenarios(s).name;
    results(s).ILS = struct();
    results(s).GWO = struct();
    results(s).IGWO = struct();
    results(s).CGWO = struct();
end

%% Run all algorithms for each scenario
for s = 1:num_scenarios
    fprintf('\n');
    fprintf('========================================================\n');
    fprintf('SCENARIO %d: %s (Wolves=%d, Iterations=%d)\n', s, ...
        scenarios(s).name, scenarios(s).SearchAgents, scenarios(s).MaxIter);
    fprintf('========================================================\n');
    
    Max_iter = scenarios(s).MaxIter;
    SearchAgents_no = scenarios(s).SearchAgents;
    
    %% Two-Step ILS
    fprintf('\n--- Running Two-Step ILS ---\n');
    tic;
    [W_ILS, convergence_ILS] = twoStepILS(Max_iter, alpha, Aq, W0, PM, PdM);
    time_ILS = toc;
    fitness_ILS = sum(abs(abs(W_ILS' * Aq(:, alpha)) - PdM(:, alpha)));
    fprintf('Two-Step ILS completed in %.2f seconds\n', time_ILS);
    fprintf('Final fitness: %.6f\n', fitness_ILS);
    
    results(s).ILS.W = W_ILS;
    results(s).ILS.fitness = fitness_ILS;
    results(s).ILS.convergence = convergence_ILS;
    results(s).ILS.time = time_ILS;
    
    %% Standard GWO
    fprintf('\n--- Running Standard GWO ---\n');
    tic;
    [W_GWO, fitness_GWO, convergence_GWO] = GWO(alpha, Aq, W0, PM, PdM, Max_iter, SearchAgents_no);
    time_GWO = toc;
    fprintf('Standard GWO completed in %.2f seconds\n', time_GWO);
    fprintf('Final fitness: %.6f\n', fitness_GWO);
    
    results(s).GWO.W = W_GWO;
    results(s).GWO.fitness = fitness_GWO;
    results(s).GWO.convergence = convergence_GWO;
    results(s).GWO.time = time_GWO;
    
    %% Improved GWO (IGWO with DLH)
    fprintf('\n--- Running IGWO (DLH) ---\n');
    tic;
    [W_IGWO, fitness_IGWO, convergence_IGWO] = IGWO(alpha, Aq, W0, PM, PdM, Max_iter, SearchAgents_no);
    time_IGWO = toc;
    fprintf('IGWO completed in %.2f seconds\n', time_IGWO);
    fprintf('Final fitness: %.6f\n', fitness_IGWO);
    
    results(s).IGWO.W = W_IGWO;
    results(s).IGWO.fitness = fitness_IGWO;
    results(s).IGWO.convergence = convergence_IGWO;
    results(s).IGWO.time = time_IGWO;
    
    %% Chaotic GWO
    fprintf('\n--- Running Chaotic GWO ---\n');
    tic;
    [W_CGWO, fitness_CGWO, convergence_CGWO] = ChaoticGWO(alpha, Aq, W0, PM, PdM, Max_iter, SearchAgents_no);
    time_CGWO = toc;
    fprintf('Chaotic GWO completed in %.2f seconds\n', time_CGWO);
    fprintf('Final fitness: %.6f\n', fitness_CGWO);
    
    results(s).CGWO.W = W_CGWO;
    results(s).CGWO.fitness = fitness_CGWO;
    results(s).CGWO.convergence = convergence_CGWO;
    results(s).CGWO.time = time_CGWO;
end

%% Plot Convergence Comparison for All Scenarios
figure('Position', [50, 50, 1800, 1000]);

for s = 1:num_scenarios
    % Convergence curves (logarithmic scale)
    subplot(2, 3, s);
    Max_iter_s = scenarios(s).MaxIter;
    
    semilogy(1:Max_iter_s, results(s).ILS.convergence, 'g-', 'LineWidth', 2); hold on;
    semilogy(1:Max_iter_s, results(s).GWO.convergence, 'b--', 'LineWidth', 2);
    semilogy(1:Max_iter_s, results(s).IGWO.convergence, 'r-', 'LineWidth', 2);
    semilogy(1:Max_iter_s, results(s).CGWO.convergence, 'Color', [0.9, 0.7, 0], 'LineStyle', '-.', 'LineWidth', 2);
    grid on;
    xlabel('Iteration', 'FontSize', 11);
    ylabel('Fitness (log scale)', 'FontSize', 11);
    title(sprintf('%s\n(Wolves=%d, Iter=%d)', scenarios(s).name, ...
        scenarios(s).SearchAgents, scenarios(s).MaxIter), ...
        'FontSize', 12, 'FontWeight', 'bold');
    if s == 1
        legend('ILS', 'GWO', 'IGWO', 'CGWO', 'Location', 'northeast', 'FontSize', 9);
    end
    xlim([1 Max_iter_s]);
    
    % Beam Pattern Comparison
    subplot(2, 3, s+3);
    plot(eqDir, 10*log10(PdM/max(PdM)), 'm-*', 'LineWidth', 1.5, 'MarkerSize', 3); hold on;
    plot(eqDir, 10*log10(abs(results(s).ILS.W'*Aq)/max(abs(results(s).ILS.W'*Aq))), 'g-', 'LineWidth', 1.8);
    plot(eqDir, 10*log10(abs(results(s).GWO.W'*Aq)/max(abs(results(s).GWO.W'*Aq))), 'b--', 'LineWidth', 1.8);
    plot(eqDir, 10*log10(abs(results(s).IGWO.W'*Aq)/max(abs(results(s).IGWO.W'*Aq))), 'r-', 'LineWidth', 1.8);
    plot(eqDir, 10*log10(abs(results(s).CGWO.W'*Aq)/max(abs(results(s).CGWO.W'*Aq))), 'Color', [0.9, 0.7, 0], 'LineStyle', '-.', 'LineWidth', 1.8);
    grid on;
    xlabel('Equivalent Directions', 'FontSize', 11);
    ylabel('Magnitude (dB)', 'FontSize', 11);
    title(sprintf('Beam Pattern - %s', scenarios(s).name), 'FontSize', 12, 'FontWeight', 'bold');
    if s == 1
        legend('Desired', 'ILS', 'GWO', 'IGWO', 'CGWO', 'Location', 'northeast', 'FontSize', 9);
    end
    xlim([-1 1]);
    ylim([-35, 1]);
end

sgtitle('Hyperparameter Tuning: Comparison Across Scenarios', 'FontSize', 16, 'FontWeight', 'bold');

%% Performance Summary Table for All Scenarios
fprintf('\n\n');
fprintf('============================================================================\n');
fprintf('                    HYPERPARAMETER TUNING SUMMARY                          \n');
fprintf('============================================================================\n');

for s = 1:num_scenarios
    fprintf('\n--- SCENARIO %d: %s (Wolves=%d, Iterations=%d) ---\n', s, ...
        scenarios(s).name, scenarios(s).SearchAgents, scenarios(s).MaxIter);
    fprintf('Algorithm          | Fitness   | Time (s)  | vs ILS\n');
    fprintf('--------------------------------------------------------\n');
    fprintf('Two-Step ILS       | %.6f | %8.2f | Baseline\n', ...
        results(s).ILS.fitness, results(s).ILS.time);
    fprintf('Standard GWO       | %.6f | %8.2f | %.2f%%\n', ...
        results(s).GWO.fitness, results(s).GWO.time, ...
        (results(s).GWO.fitness - results(s).ILS.fitness) / results(s).ILS.fitness * 100);
    fprintf('IGWO (DLH)         | %.6f | %8.2f | %.2f%%\n', ...
        results(s).IGWO.fitness, results(s).IGWO.time, ...
        (results(s).IGWO.fitness - results(s).ILS.fitness) / results(s).ILS.fitness * 100);
    fprintf('Chaotic GWO        | %.6f | %8.2f | %.2f%%\n', ...
        results(s).CGWO.fitness, results(s).CGWO.time, ...
        (results(s).CGWO.fitness - results(s).ILS.fitness) / results(s).ILS.fitness * 100);
end
fprintf('\n============================================================================\n');

%% Calculate Peak Sidelobe Level (PSLL) for all scenarios
main_lobe_region = abs(eqDir) < 0.1;

fprintf('\n\n');
fprintf('============================================================================\n');
fprintf('                    PEAK SIDELOBE LEVEL (PSLL) ANALYSIS                    \n');
fprintf('============================================================================\n');

for s = 1:num_scenarios
    fprintf('\n--- SCENARIO %d: %s ---\n', s, scenarios(s).name);
    
    pattern_ILS = 10*log10(abs(results(s).ILS.W'*Aq)/max(abs(results(s).ILS.W'*Aq)));
    pattern_GWO = 10*log10(abs(results(s).GWO.W'*Aq)/max(abs(results(s).GWO.W'*Aq)));
    pattern_IGWO = 10*log10(abs(results(s).IGWO.W'*Aq)/max(abs(results(s).IGWO.W'*Aq)));
    pattern_CGWO = 10*log10(abs(results(s).CGWO.W'*Aq)/max(abs(results(s).CGWO.W'*Aq)));
    
    PSLL_ILS = max(pattern_ILS(~main_lobe_region));
    PSLL_GWO = max(pattern_GWO(~main_lobe_region));
    PSLL_IGWO = max(pattern_IGWO(~main_lobe_region));
    PSLL_CGWO = max(pattern_CGWO(~main_lobe_region));
    
    fprintf('Two-Step ILS: %.2f dB (Baseline)\n', PSLL_ILS);
    fprintf('Standard GWO: %.2f dB (Δ=%.2f dB)\n', PSLL_GWO, PSLL_GWO - PSLL_ILS);
    fprintf('IGWO (DLH):   %.2f dB (Δ=%.2f dB)\n', PSLL_IGWO, PSLL_IGWO - PSLL_ILS);
    fprintf('Chaotic GWO:  %.2f dB (Δ=%.2f dB)\n', PSLL_CGWO, PSLL_CGWO - PSLL_ILS);
    
    results(s).PSLL_ILS = PSLL_ILS;
    results(s).PSLL_GWO = PSLL_GWO;
    results(s).PSLL_IGWO = PSLL_IGWO;
    results(s).PSLL_CGWO = PSLL_CGWO;
end
fprintf('\n============================================================================\n');

%% Create comparison summary figure
figure('Position', [100, 100, 1400, 800]);

% Subplot 1: Best Fitness Comparison
subplot(2, 2, 1);
scenario_names = {scenarios.name};
fitness_comparison = zeros(4, num_scenarios);
for s = 1:num_scenarios
    fitness_comparison(1, s) = results(s).ILS.fitness;
    fitness_comparison(2, s) = results(s).GWO.fitness;
    fitness_comparison(3, s) = results(s).IGWO.fitness;
    fitness_comparison(4, s) = results(s).CGWO.fitness;
end
bar(fitness_comparison');
set(gca, 'XTickLabel', scenario_names);
xlabel('Scenario', 'FontSize', 12);
ylabel('Final Fitness', 'FontSize', 12);
title('Fitness Comparison Across Scenarios', 'FontSize', 13, 'FontWeight', 'bold');
legend('ILS', 'GWO', 'IGWO', 'CGWO', 'Location', 'best');
grid on;

% Subplot 2: Computation Time Comparison
subplot(2, 2, 2);
time_comparison = zeros(4, num_scenarios);
for s = 1:num_scenarios
    time_comparison(1, s) = results(s).ILS.time;
    time_comparison(2, s) = results(s).GWO.time;
    time_comparison(3, s) = results(s).IGWO.time;
    time_comparison(4, s) = results(s).CGWO.time;
end
bar(time_comparison');
set(gca, 'XTickLabel', scenario_names);
xlabel('Scenario', 'FontSize', 12);
ylabel('Time (seconds)', 'FontSize', 12);
title('Computation Time Comparison', 'FontSize', 13, 'FontWeight', 'bold');
legend('ILS', 'GWO', 'IGWO', 'CGWO', 'Location', 'best');
grid on;

% Subplot 3: PSLL Comparison
subplot(2, 2, 3);
psll_comparison = zeros(4, num_scenarios);
for s = 1:num_scenarios
    psll_comparison(1, s) = results(s).PSLL_ILS;
    psll_comparison(2, s) = results(s).PSLL_GWO;
    psll_comparison(3, s) = results(s).PSLL_IGWO;
    psll_comparison(4, s) = results(s).PSLL_CGWO;
end
bar(psll_comparison');
set(gca, 'XTickLabel', scenario_names);
xlabel('Scenario', 'FontSize', 12);
ylabel('PSLL (dB)', 'FontSize', 12);
title('Peak Sidelobe Level Comparison', 'FontSize', 13, 'FontWeight', 'bold');
legend('ILS', 'GWO', 'IGWO', 'CGWO', 'Location', 'best');
grid on;

% Subplot 4: Convergence Speed
subplot(2, 2, 4);
iters_to_threshold = zeros(4, num_scenarios);
for s = 1:num_scenarios
    threshold = results(s).ILS.fitness * 1.1;
    % ILS
    idx = find(results(s).ILS.convergence <= threshold, 1);
    iters_to_threshold(1, s) = ifempty(idx, scenarios(s).MaxIter, idx);
    % GWO
    idx = find(results(s).GWO.convergence <= threshold, 1);
    iters_to_threshold(2, s) = ifempty(idx, scenarios(s).MaxIter, idx);
    % IGWO
    idx = find(results(s).IGWO.convergence <= threshold, 1);
    iters_to_threshold(3, s) = ifempty(idx, scenarios(s).MaxIter, idx);
    % CGWO
    idx = find(results(s).CGWO.convergence <= threshold, 1);
    iters_to_threshold(4, s) = ifempty(idx, scenarios(s).MaxIter, idx);
end
bar(iters_to_threshold');
set(gca, 'XTickLabel', scenario_names);
xlabel('Scenario', 'FontSize', 12);
ylabel('Iterations to Convergence', 'FontSize', 12);
title('Iterations to Reach 110% of ILS fitness', 'FontSize', 13, 'FontWeight', 'bold');
legend('ILS', 'GWO', 'IGWO', 'CGWO', 'Location', 'best');
grid on;

sgtitle('Hyperparameter Tuning Summary', 'FontSize', 16, 'FontWeight', 'bold');

%% Helper function
function result = ifempty(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

%% Save all results
save('optimization_results.mat', 'scenarios', 'results', 'eqDir', 'Aq', 'PdM');

fprintf('\n✓ All results saved to optimization_results.mat\n');
fprintf('✓ Total scenarios tested: %d\n', num_scenarios);
fprintf('✓ Total runs: %d (each scenario runs 4 algorithms)\n', num_scenarios * 4);
