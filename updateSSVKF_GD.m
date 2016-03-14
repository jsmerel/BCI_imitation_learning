function [F,G,bias] = updateSSVKF_GD(F,G,bias,lr,set_of_states, set_of_neural, set_of_oracleUpdates,dt,rp)
    [N,S] = size(set_of_neural);
    
    Y = set_of_oracleUpdates;
    X = [set_of_neural(:,1:end); set_of_states(4:end,1:end); ones(1,S)];
    
    weights_init = zeros(3,N+3+1);
    weights_init(:,1:N) = F((3+1):end,:);
    weights_init(:,(N+1):(N+3)) = G((3+1):end,(3+1):end);
    weights_init(:,end) = bias((3+1):end);
    
    weights = weights_init - lr*( (-2)*(Y-weights_init*X)*X'/S + 2*rp*weights_init);
    
    Fv = weights(:,1:N);
    Gv = weights(:,(N+1):(N+3));
    biasv = weights(:,end);
    
    F = [zeros(3,N); Fv];
    G = [eye(3) dt*eye(3); zeros(3,3) Gv]; 
    bias = [zeros(3,1); biasv];
    
end
