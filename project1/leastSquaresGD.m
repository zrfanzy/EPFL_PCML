function beta = leastSquaresGD(y, tX, alpha)

  % learning parametes
  maxIters = 10000;
  converged = 1e-10;

  % initialize
  beta = zeros(size(tX, 2), 1);
  N = length(y);

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
    fprintf('Interation %d, got rmse: %.2f\n', k, sqrt(L));

  end

