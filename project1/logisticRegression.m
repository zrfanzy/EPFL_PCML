function beta = logisticRegression(y,tX,alpha)
% Logistic regression using gradient descent or Newton's method.

%% algorithm parametes
sigmoid = @(x) exp(x)./(1+exp(x));
maxIters = 1000;
convergent = 0.2;
%% initilization of beta:
beta = [1;3;-8];

%% start iterate
fprintf('Starting iterations, press Ctrl+c to break\n');
fprintf('L  beta0 beta1 beta2\n');

 for k = 1:maxIters
    % compute gradient:
    g =  (tX)' * (exp(tX*beta)./(1 + exp(tX*beta)) - y);

    % cost function
    L = (y)' * tX * beta - sum(log(1 + exp(tX * beta))); % log likelyhood of beta
    L = -L ; % negetaive as cost function
    
    % check convergent
    if L < convergent
        break
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
