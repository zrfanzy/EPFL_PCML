function X_poly = mypoly(X,degree)
% build matrix Phi for polynomial regression of a given degree

    X_poly = X;
    if degree >1
        for i = 1:degree
            X_poly = [X_poly X.^i];
        end
    else 
        X_poly = [X_poly X.^degree];
    end
    
     fprintf('size of xpoly is %d',size(Xpoly));
end
