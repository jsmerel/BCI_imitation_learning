function [F,G,bias] = fitSSVKF(set_of_states, set_of_neural, set_of_oracleUpdates,dt,p,rp)
    [N,S] = size(set_of_neural);
    S = S-1;
    Y = set_of_oracleUpdates;
    X = [set_of_neural(:,1:(end-1)); set_of_states(4:end,1:(end-1)); ones(1,S)];
    weights = (Y*X')/(X*X' + rp*eye(N+p+1));
    Fv = weights(:,1:N);
    Gv = weights(:,(N+1):(N+3));
    biasv = weights(:,end);
    
    F = [zeros(3,N); Fv];
    G = [eye(3) dt*eye(3); zeros(3,3) Gv]; 
    bias = [zeros(3,1); biasv];
    
end
