function [Best_W, Best_fitness, Convergence_curve] = IGWO(alpha, V_pattern, W0, PM, PdM, Max_iter, SearchAgents_no)
% Improved Grey Wolf Optimization with Dimension Learning-based Hunting (DLH)
% Inputs:
%   alpha: Indices of the eq. directions to be approximated
%   V_pattern: Array response matrix
%   W0: Initial beamforming vector
%   PM: Initial pattern magnitude
%   PdM: Desired pattern magnitude
%   Max_iter: Maximum number of iterations
%   SearchAgents_no: Number of search agents (wolves)
% Outputs:
%   Best_W: Best beamforming vector found
%   Best_fitness: Best fitness value
%   Convergence_curve: Convergence history

% Problem dimensions
dim = length(W0);

% Initialize alpha, beta, and delta positions
Alpha_pos = zeros(1, dim);
Alpha_score = inf;

Beta_pos = zeros(1, dim);
Beta_score = inf;

Delta_pos = zeros(1, dim);
Delta_score = inf;

% Initialize the positions of search agents (complex values)
Positions = zeros(SearchAgents_no, dim);
for i = 1:SearchAgents_no
    Positions(i, :) = (rand(1, dim) - 0.5) + 1j*(rand(1, dim) - 0.5);
end

Convergence_curve = zeros(1, Max_iter);

% Fitness function
fitness_func = @(W) sum(abs(abs(W' * V_pattern(:, alpha)) - PdM(:, alpha)));

% Main loop
for iter = 1:Max_iter
    % Calculate fitness for all agents
    fitness_values = zeros(SearchAgents_no, 1);
    for i = 1:SearchAgents_no
        fitness_values(i) = fitness_func(Positions(i, :)');
    end
    
    % Update Alpha, Beta, and Delta
    [sorted_fitness, sorted_indices] = sort(fitness_values);
    
    Alpha_score = sorted_fitness(1);
    Alpha_pos = Positions(sorted_indices(1), :);
    
    Beta_score = sorted_fitness(2);
    Beta_pos = Positions(sorted_indices(2), :);
    
    Delta_score = sorted_fitness(3);
    Delta_pos = Positions(sorted_indices(3), :);
    
    a = 2 - iter * ((2) / Max_iter); % a decreases linearly from 2 to 0
    
    % Update the Position of search agents
    for i = 1:SearchAgents_no
        % Standard GWO update
        X_GWO = zeros(1, dim);
        for j = 1:dim
            % Update position based on Alpha
            r1 = rand();
            r2 = rand();
            
            A1 = 2 * a * r1 - a;
            C1 = 2 * r2;
            
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
            X1 = Alpha_pos(j) - A1 * D_alpha;
            
            % Update position based on Beta
            r1 = rand();
            r2 = rand();
            
            A2 = 2 * a * r1 - a;
            C2 = 2 * r2;
            
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
            X2 = Beta_pos(j) - A2 * D_beta;
            
            % Update position based on Delta
            r1 = rand();
            r2 = rand();
            
            A3 = 2 * a * r1 - a;
            C3 = 2 * r2;
            
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
            X3 = Delta_pos(j) - A3 * D_delta;
            
            % GWO position
            X_GWO(j) = (X1 + X2 + X3) / 3;
        end
        
        % DLH (Dimension Learning-based Hunting) mechanism
        X_DLH = zeros(1, dim);
        
        % Find nearest neighbor (based on fitness)
        distances = abs(fitness_values - fitness_values(i));
        distances(i) = inf; % Exclude self
        [~, neighbor_idx] = min(distances);
        
        % Random wolf
        random_idx = randi([1, SearchAgents_no]);
        while random_idx == i
            random_idx = randi([1, SearchAgents_no]);
        end
        
        for j = 1:dim
            % DLH update: learn from neighbor and random wolf
            X_DLH(j) = Positions(i, j) + rand() * (Positions(neighbor_idx, j) - Positions(random_idx, j));
        end
        
        % Greedy selection between GWO and DLH
        fitness_GWO = fitness_func(X_GWO');
        fitness_DLH = fitness_func(X_DLH');
        
        if fitness_GWO < fitness_DLH
            Positions(i, :) = X_GWO;
        else
            Positions(i, :) = X_DLH;
        end
    end
    
    Convergence_curve(iter) = Alpha_score;
    
    if mod(iter, 10) == 0
        fprintf('IGWO Iteration %d: Best fitness = %f\n', iter, Alpha_score);
    end
end

Best_W = Alpha_pos';
Best_fitness = Alpha_score;

end
