
for oi = 1%:100

%% define task space and task parameters
bounds{1} = [-10 10];
bounds{2} = [-10 10];
bounds{3} = [-10 10];

N = 10; % number of neurons
dt = .1;
intendedSpeed = 3;
K = 20;

%% update using GD or batch retrain
useGD = 0;
useMA = 0;

%% init state (position + velocity)
state = [0 0 0 0 0 0]';

%% init decoder - SS velocity KF
% decoder is some function of state and neural activity
% here, we use: state_{t+1} = G state_{t} + F neural_{t}
% F = [P_params; V_params] w/ P_params probably fixed to zero
% G = [I dt*I; 0 V_dynamics] w/ 0 b/c pos doesn't affect velocity and V_dynamics being the smoothing on velocity

F = [zeros(3,N); .1*randn(3,N)]; %more stable if this starts out smaller
G = [eye(3) dt*eye(3); zeros(3,3) zeros(3,3)]; %bottom right will likely be non-zero in reality
bias = zeros(6,1);

%% encoding model
A = randn(N,3); %project to neurons from velocities.

noiseMag = 3;

%% Pinball task (loop over number of targets)
set_of_states = [];
set_of_neural = [];
set_of_oracleUpdates = [];
set_of_actualUpdates = []; %only need to save for evaluation of performance

set_of_states = [set_of_states state]; % 6 x S
set_of_neural = [set_of_neural zeros(N,1)]; % N x S

data_index = 1;
goal_acquired_indices = [data_index];
goal_positions = [];

clear M
mi = 1;
for k = 1:K
    
    %% select goal for reach
    goal = [diff(bounds{1})*rand+bounds{1}(1) diff(bounds{2})*rand+bounds{2}(1) diff(bounds{3})*rand+bounds{3}(1)]';
    goal_positions = [goal_positions goal];
    
    iter = 1;
    %% simulate a reach
    while (sum(abs(state(1:3) - goal)) > 5e-1) && (iter<200)
        
        %% what is the user's intention -- the oracle will do exactly this
        % max(unit,dist(state,goal)) size step towards goal
        intention = goal-state(1:3);
        remainingDistance = norm(intention);
        if remainingDistance<intendedSpeed
        else
            intention = intendedSpeed*intention/remainingDistance;
        end
        
        %% Get neural data for current timestep
        % treat optimal velocity at this time as intended and simulate neural data based on goal and intention
        % this will essentially be a noisy observation of the optimal update
        y = A*intention + noiseMag*randn(N,1); %if not zero-mean, will need a bias in the decoder

        %% Determine optimal expert update given oracle
        % we've already computed the optimal 1-timestep because we needed it above
        oracleUpdate = intention;
        
        %% collect the pairs for the current state and the oracle update
        set_of_states = [set_of_states state]; % 6 x S
        set_of_neural = [set_of_neural y]; % N x S
        set_of_oracleUpdates = [set_of_oracleUpdates oracleUpdate]; % 3 x S
        data_index = data_index + 1;
        
        %% update the state using the current decoder and the neural data
        if k==1 
            state_new = [eye(3) dt*eye(3); zeros(3,3) zeros(3,3)]*[state(1:3);intention] + 1/sqrt(iter)*randn(6,1);
            %add mild noise if using assisstive training
        else
            state_new = F*y + G*state + bias;
        end
        actualUpdate = state_new - state;
        set_of_actualUpdates = [set_of_actualUpdates actualUpdate]; % 6 x S
        state = state_new;
        state(1) = max(min(state(1),bounds{1}(2)),bounds{1}(1)); 
        state(2) = max(min(state(2),bounds{2}(2)),bounds{2}(1)); 
        state(3) = max(min(state(3),bounds{3}(2)),bounds{3}(1)); 

        %% visualize
        cursorVis(state(1:3), goal, 3*actualUpdate/norm(actualUpdate), oracleUpdate, bounds, 10)
        title(num2str(k))
        
        iter = iter + 1;
    end
    
    % goal has been acquired
    goal_acquired_indices = [goal_acquired_indices data_index];
        
    %% update the decoder (i.e. re-train the model based on all aggregated state-action pairs.
    if useGD
        rp = 1e2/K;
        lr = .005; %current final
%         lr = .01/sqrt(k); % or this
        data_subset = goal_acquired_indices(k):(goal_acquired_indices(k+1)-1);
        [F,G,bias] = updateSSVKF_GD(F,G,bias,lr,set_of_states(:,data_subset), set_of_neural(:,data_subset), set_of_oracleUpdates(:,data_subset),dt,rp);
    elseif useMA
        rp = 1e2;
        lambda = .9;
        if k == 1
            data_subset = goal_acquired_indices(k):(goal_acquired_indices(k+1)-1);
        else
            data_subset = goal_acquired_indices(k):(goal_acquired_indices(k+1)-1);
        end
        [F_sub,G_sub,bias_sub] = fitSSVKF(set_of_states(:,data_subset), set_of_neural(:,data_subset), set_of_oracleUpdates(:,data_subset(1:end-1)),dt,3,rp);
        F = lambda*F + (1-lambda)*F_sub;
        G = lambda*G + (1-lambda)*G_sub;
        bias = lambda*bias + (1-lambda)*bias_sub;
    else
        rp = 1e2;
        [F,G,bias] = fitSSVKF(set_of_states, set_of_neural, set_of_oracleUpdates,dt,3,rp);
    end
end

end
