function beta = leastSquaresGD(y, tX, alpha)

  % learning parametes
  maxIters = 1000;
  converged = 0;

  % initialize
  beta = zeros(length(y), 1);

  for k = 1:maxIters
    
    % error
    e = y - tX * beta;  
    % gradient
    g = - 1 / N * tX' * e;
    % MSE
    L = e' * e / 2 / N;
    % learning beta
    beta = beta - alpha .* g;

    % check convergence
    if k > 1
        if abs(L_all(k - 1) - L) <= converged
            fprintf('Got convergence, quit interation\n');
            break;
        end
    end
    
    % store L for convergence check
    L_all(k) = L;

    % print
    fprintf('Interation %d, %.2f  %.2f %.2f\n', k, L, beta(1), beta(2));

  end

