function [solution, time, off, ofit, site, paretoAVE, tempVar, bitImportance] = PRDH(train_F, train_L, cnti)
    fprintf('PRDH\n');                                      
    tic
    global maxFES
    global sizep
    FES = 1;
    dim = size(train_F, 2);
    ofit = zeros(sizep, 2); % Objective function values for the population
    initThres = 1;
    thres = 0.1; % Exponential decay constant
    paretoAVE = zeros(1, 2); % To save final result of the Pareto front
    
    %% Initialization
    Problem.D = dim;
    Problem.N = sizep;
    Problem.lower = zeros(1, Problem.D);
    Problem.upper = ones(1, Problem.D);
    Problem.encoding = 4; % Binary encoding for all variables
    Population = InitializePopulation(Problem);
    [~, FrontNo, CrowdDis] = EnvironmentalSelection(Population, Problem.N);
    
    %% Evaluate
    for i = 1:Problem.N
        [ofit(i, 1), ofit(i, 2)] = FSKNNfeixiang(Population(i).decs, train_F, train_L);
    end
    site = find(FrontNo == 1);
    solution = ofit(site, :);
    solution(:, 2) = solution(:, 2) / dim;
    disp('Solution:');
    disp(solution);
    erBestParetoAVE = 1;  % To save the history best
    paretoAVE(1) = mean(solution(:, 1));
    paretoAVE(2) = mean(solution(:, 2));
    
    %% Optimization
    while FES <= maxFES
        MatingPool = TournamentSelection(2, Problem.N, FrontNo, -CrowdDis);
        Offspring = OffspringReproduction(Population(MatingPool), Problem);
        [Population, FrontNo, CrowdDis] = EnvironmentalSelection([Population, Offspring], Problem.N);
        
        % Evaluate new population
        for i = 1:length(Offspring)
            [ofit(end+1, 1), ofit(end, 2)] = FSKNNfeixiang(Offspring(i).decs, train_F, train_L);
        end
        
        % Update solution and pareto front
        site = find(FrontNo == 1);
        solution = ofit(site, :);
        paretoAVE(1) = mean(solution(:, 1));
        paretoAVE(2) = mean(solution(:, 2));
        
        FES = FES + 1;
    end
    
    %% Finalization
    tempVar{1} = ofit; % All objective function values
    tempVar{2} = FrontNo; % All front numbers
    tempVar{3} = CrowdDis; % All crowding distances
    tempVar{4} = []; % Other temporary variables if needed
    
    clear tAveError;
    clear tAveFea;
    clear tErBest;
    clear tThres;
    toc
    time = toc;
    off = Population.decs; % Ensure off is assigned
end

function Population = InitializePopulation(Problem)
    T = min(Problem.D, Problem.N * 3);
    PopDec = zeros(Problem.N, Problem.D); % 决策变量
    PopObj = zeros(Problem.N, 2); % 假设有两个目标，初始化为0
    PopCon = zeros(Problem.N, 1); % 假设没有约束违反，初始化为0
    
    for i = 1 : Problem.N
        k = randperm(T, 1);
        j = randperm(Problem.D, k);
        PopDec(i, j) = 1;
    end
    
    % 调用SOLUTION构造函数时提供决策变量、目标值和约束违反情况
    Population = SOLUTION(PopDec, PopObj, PopCon);
end
