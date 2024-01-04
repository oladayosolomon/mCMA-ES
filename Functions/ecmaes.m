function [u_opt, J_opt] = ecmaes(initial_param,obj_func,const_func, LB, UB)
  % --------------------  Initialization --------------------------------  
  % User defined input parameters (need to be edited)
  strfitnessfct = obj_func;  % name of objective/fitness function
  strconsfct = const_func;
  N = size(initial_param,1)*size(initial_param,2);               % number of objective variables/problem dimension
  xmean = initial_param(:);  % objective variables initial point
  
  sigma = 0.001;          % coordinate wise standard deviation (step size)
  stopfitness = 1e-2;  % stop if fitness < stopfitness (minimization)
  stopeval = 1e5;   % stop after stopeval number of function evaluations
  
  % Strategy parameter setting: Selection  
  lambda = 4+floor(3*log(N));  % population size, offspring number
  mu = 1;               % number of parents/points for recombination
  weights = log(mu+1/2)-log(1:mu)'; % muXone array for weighted recombination
  mu = floor(mu);        
  weights = weights/sum(weights);     % normalize recombination weights array
  mueff=sum(weights)^2/sum(weights.^2); % variance-effectiveness of sum w_i x_i

  % Strategy parameter setting: Adaptation
  cc = (4 + mueff/N) / (N+4 + 2*mueff/N); % time constant for cumulation for C
  cs = (mueff+2) / (N+mueff+5);  % t-const for cumulation for sigma control
  c1 = 2 / ((N+1.3)^2+mueff);    % learning rate for rank-one update of C
  cmu = min(1-c1, 2 * (mueff-2+1/mueff) / ((N+2)^2+mueff));  % and for rank-mu update
  damps = 1 + 2*max(0, sqrt((mueff-1)/(N+1))-1) + cs; % damping for sigma 
                                                      % usually close to 1
  % Initialize dynamic (internal) strategy parameters and constants
  pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
  B = eye(N,N);                       % B defines the coordinate system
  D = ones(N,1);                      % diagonal D defines the scaling
  C = B * diag(D.^2) * B';            % covariance matrix C
  invsqrtC = B * diag(D.^-1) * B';    % C^-1/2 
  eigeneval = 0;                      % track update of B and D
  chiN=N^0.5*(1-1/(4*N)+1/(21*N^2));  % expectation of 
                                      %   ||N(0,I)|| == norm(randn(N,1))
  out.dat = []; out.datx = [];  % for plotting output

  % -------------------- Generation Loop --------------------------------
  counteval = 0;  % the next 40 lines contain the 20 lines of interesting code
  bestfitness = inf;

  xb = ones(N,1)*inf;
  xf = inf;
  xcv = inf;


  while counteval < stopeval
      clear arfitness
    % Generate and evaluate lambda offspring
    for k=1:lambda
        if k > lambda/2
             newpopy(:,k) = xmean + sigma * B * (D .* cauchy(N,1)); % m + sig * Normal(0,C)
        else
             newpopy(:,k) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C)
        end
      %newpopy(:,k) = xmean + sigma * B * (D .* randn(N,1)); % m + sig * Normal(0,C)
      arx(:,k)  = keep_range(newpopy(:,k),LB(:),UB(:));
      dec_vars = reshape(arx(:,k),size(initial_param));
      arfitness(k) = feval(strfitnessfct, dec_vars); % objective function call
      g(k,:) = feval(strconsfct, dec_vars);
      counteval = counteval+1;
    end
   
   

    g(g<0)=0;
    CV = sum(g');
    CV = [CV,xcv];
    arfitness = [arfitness,xf];
    arx =[arx,xb];
    

    %% sof constriant  % implement 
    %sapfitness = SaPenalty(arfitness,CV,lambda);
    [~, arindex] = sortrows([abs(CV)',arfitness']);
    arindex=arindex';

    %%
    % Sort by fitness and compute weighted mean into xmean
    %[arfitness, arindex] = sort(arfitness);  % minimization
    xold = xmean;
    xmean = arx(:,arindex(1:mu)) * weights;  % recombination, new mean value
    
    % Cumulation: Update evolution paths
    ps = (1-cs) * ps ... 
          + sqrt(cs*(2-cs)*mueff) * invsqrtC * (xmean-xold) / sigma; 
    hsig = sum(ps.^2)/(1-(1-cs)^(2*counteval/lambda))/N < 2 + 4/(N+1);
    pc = (1-cc) * pc ...
          + hsig * sqrt(cc*(2-cc)*mueff) * (xmean-xold) / sigma; 

    % Adapt covariance matrix C
    artmp = (1/sigma) * (arx(:,arindex(1:mu)) - repmat(xold,1,mu));  % mu difference vectors
    C = (1-c1-cmu) * C ...                   % regard old matrix  
         + c1 * (pc * pc' ...                % plus rank one update
                 + (1-hsig) * cc*(2-cc) * C) ... % minor correction if hsig==0
         + cmu * artmp * diag(weights) * artmp'; % plus rank mu update 

    % Adapt step size sigma
    sigma = sigma * exp((cs/damps)*(norm(ps)/chiN - 1)); 


     if any(isnan(C), 'all')  % restart
        pc = zeros(N,1); ps = zeros(N,1);   % evolution paths for C and sigma
        B = eye(N,N);                       % B defines the coordinate system
        D = ones(N,1);                      % diagonal D defines the scaling
        C = B * diag(D.^2) * B'; 
    end
    
    % Update B and D from C
    if counteval - eigeneval > lambda/(c1+cmu)/N/10  % to achieve O(N^2)
      eigeneval = counteval;
      C = triu(C) + triu(C,1)'; % enforce symmetry
      [B,D] = eig(C);           % eigen decomposition, B==normalized eigenvectors
      D = sqrt(diag(D));        % D contains standard deviations now
      invsqrtC = B * diag(D.^-1) * B';
    end

    if bestfitness > arfitness(arindex(1))
        bestfitness = arfitness(arindex(1));
        bestsoln = arx(:,arindex(1));
    end
 
    % Break, if fitness is good enough or condition exceeds 1e14, better termination methods are advisable 
    if bestfitness <= stopfitness %|| max(D) > 1e7 * min(D)
      break;
    end
  end % while, end generation loop

  J_opt = bestfitness;
  %u_opt = arx(:, arindex(1));
  u_opt = bestsoln;
  u_opt  = reshape(u_opt,size(initial_param));
  
end
