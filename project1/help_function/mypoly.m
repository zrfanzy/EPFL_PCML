function X_poly = mypoly(X,degree)
% build matrix Phi for polynomial regression of a given degree

        sub_cols = size(X,2);
        X_poly = zeros(size(X,1),sub_cols*degree);
        for i = 1:sub_cols
            sub_Xpoly = mypoly_single(X(:,i),degree);
            X_poly(:,1+(i-1)*degree:i*degree) = sub_Xpoly;
        end
    
    
%     fprintf('size of xpoly is %d',size(Xpoly));
end

function Xpoly_single = mypoly_single(X,degree)
        % init
        Xpoly_single = zeros(size(X,1),degree);
        % 
        for col = 1:degree
            k = col;
            Xpoly_single(:,col) = X.^k;
        end
end