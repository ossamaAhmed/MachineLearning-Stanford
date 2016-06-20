function y = lwlr(X_train, y_train, x, tau)


%%% YOUR CODE HERE
lamd = 0.0001;
m = size(X_train,1); %training set size
n = size(X_train,2); %features
theta = zeros(n,1); % weights

w = exp(-sum((X_train-repmat(x', m, 1)).^2, 2) / (2*(tau.^2))) %calculate weights

%calculate newton's method

derivative = ones(n,1);

while (norm(derivative) > 1e-6)

    h = 1./(1+exp(-X_train*theta));

    derivative = (X_train' * (w.*(y_train-h))) - (lamd*theta);

    diagonal_matrix = diag(-w.*h.*(1-h));

    Hessian = (X_train' * diagonal_matrix * X_train) - (lamd*eye(n)); 

    theta = theta - (Hessian \ derivative);

end

y = double(x'*theta>0);



