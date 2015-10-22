function beta = penLogisticRegression(y,tX,alpha,lambda)
% Penalized logistic regression using gradient descent or Newton's method.

% algorithm parametes
sigmoid = @(x) exp(x)./(1+exp(x));
maxIters = 1000;
converged = 0.01;
N = length(y);

% initilize
beta = zeros(size(tX, 2), 1);

%% start iterate
fprintf('Starting iterations');

 for k = 1:maxIters
    % compute gradient:
    g =  (tX)' * (exp(tX*beta)./(1 + exp(tX*beta)) - y);
    g = g + lambda *beta;

    % cost function
    L = (y)' * tX * beta - sum(log(1 + exp(tX * beta)));
    L = -L + lambda * ((beta)'*beta);
    L = L/N;
    % check convergence
    if k > 1
        if abs(L_all(k - 1) - L) <= converged
            fprintf('Got convergence, quit interation\n');
            break;
        end
    end
%%   if use gradient descent
    % gradient update to find beta
    %   beta = beta - alpha * g;
    
%%  if use newton method:
    % compute Hessian
    S =  diag(sigmoid(tX*beta)).*diag([1-sigmoid(tX*beta)]);
    H = (tX)' * S *tX;
    lambdaI = lambda * eye(length(H));
    lambdaI(1,1) = 0;
    H = H + lambdaI;

    d = H\g; %solve function H*d = g
    % newton method update beta
    beta = beta - alpha * d;
    
    % store beta and L
    beta_all(:,k) = beta;
    L_all(k) = L;

    % print
    fprintf('L: %.2f \n',L);

    % Overlay on the contour plot
 end
 
fprintf('run %d iteration beta is: \n', k);
display(beta);
 
end