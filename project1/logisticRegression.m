function beta = logisticRegression(y,tX,alpha)
% Logistic regression using gradient descent or Newton's method.

% algorithm parametes
sigmoid = @(x) exp(x)./(1+exp(x));
maxIters = 1000;
converged = 0.01;
N = length(y);

% initilization of beta:
beta = zeros(size(tX, 2), 1);

% start iterate
fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 beta1 beta2\n');

 for k = 1:maxIters
    % compute gradient:
    g =  (tX)' * (exp(tX*beta)./(1 + exp(tX*beta)) - y);

    % cost function
    L = (y)' * tX * beta - sum(log(1 + exp(tX * beta))); % log likelyhood of beta
    L = -L ; % negetaive as cost function
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

    d = H\g; %solve function H*d = g
    % newton method update beta
    beta = beta - alpha * d;
    
%%  if use IRLS  
%     sig = sigmoid(tX*beta);
%     s = sig.*(1-sig);
%     z = tX*beta + (y-sig)./s;
%     beta = weightedLeastSquares(z,tX,s);
%     L = computeCost_logl(y,tX,beta);
    
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
