function [Best_W, Best_fitness, Convergence_curve] = ChaoticGWO(alpha, V_pattern, W0, PM, PdM, Max_iter, SearchAgents_no)
% Chaotic Grey Wolf Optimization using Logistic Map
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

% Chaotic map initialization (Logistic Map)
Ch = rand(); % Initial chaotic value
mu = 4; % Chaotic parameter (mu=4 for full chaos)

% Fitness function
fitness_func = @(W) sum(abs(abs(W' * V_pattern(:, alpha)) - PdM(:, alpha)));

% Main loop
for iter = 1:Max_iter
    for i = 1:SearchAgents_no
        % Calculate fitness
        fitness = fitness_func(Positions(i, :)');
        
        % Update Alpha, Beta, and Delta
        if fitness < Alpha_score
            Alpha_score = fitness;
            Alpha_pos = Positions(i, :);
        end
        
        if fitness > Alpha_score && fitness < Beta_score
            Beta_score = fitness;
            Beta_pos = Positions(i, :);
        end
        
        if fitness > Alpha_score && fitness > Beta_score && fitness < Delta_score
            Delta_score = fitness;
            Delta_pos = Positions(i, :);
        end
    end
    
    a = 2 - iter * ((2) / Max_iter); % a decreases linearly from 2 to 0
    
    % Update the Position of search agents using chaotic values
    for i = 1:SearchAgents_no
        for j = 1:dim
            % Update chaotic value using Logistic Map
            Ch = mu * Ch * (1 - Ch);
            
            % Use chaotic value instead of random value
            r1 = Ch;
            
            % Update chaotic value again for C
            Ch = mu * Ch * (1 - Ch);
            r2 = Ch;
            
            % Update position based on Alpha (with chaotic A and C)
            A1 = 2 * a * r1 - a;
            C1 = 2 * r2;
            
            D_alpha = abs(C1 * Alpha_pos(j) - Positions(i, j));
            X1 = Alpha_pos(j) - A1 * D_alpha;
            
            % Update position based on Beta
            Ch = mu * Ch * (1 - Ch);
            r1 = Ch;
            Ch = mu * Ch * (1 - Ch);
            r2 = Ch;
            
            A2 = 2 * a * r1 - a;
            C2 = 2 * r2;
            
            D_beta = abs(C2 * Beta_pos(j) - Positions(i, j));
            X2 = Beta_pos(j) - A2 * D_beta;
            
            % Update position based on Delta
            Ch = mu * Ch * (1 - Ch);
            r1 = Ch;
            Ch = mu * Ch * (1 - Ch);
            r2 = Ch;
            
            A3 = 2 * a * r1 - a;
            C3 = 2 * r2;
            
            D_delta = abs(C3 * Delta_pos(j) - Positions(i, j));
            X3 = Delta_pos(j) - A3 * D_delta;
            
            % Update position
            Positions(i, j) = (X1 + X2 + X3) / 3;
        end
    end
    
    Convergence_curve(iter) = Alpha_score;
    
    if mod(iter, 10) == 0
        fprintf('Chaotic GWO Iteration %d: Best fitness = %f\n', iter, Alpha_score);
    end
end

Best_W = Alpha_pos';
Best_fitness = Alpha_score;

end
