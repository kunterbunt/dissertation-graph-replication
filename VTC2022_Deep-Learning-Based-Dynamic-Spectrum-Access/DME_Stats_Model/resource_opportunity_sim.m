function [pmf_mat, ccdf_mat, expectation_mat] = resource_opportunity_sim(n_DME, t_LDACS, period_DME, n_urns, n_users, n_runs)                
    pmf_mat = zeros(n_urns, n_users);
    for ni=1:n_users
        n=n_DME(ni);
        util_vecs = [];
        for i=1:n_runs
            util_vecs = [util_vecs resource_opportunity_sim_run(n, t_LDACS, period_DME)];
        end

        % no. of empty urns per run
        num_empty_urns_per_run = sum(util_vecs == 0);
        % go through all no. of urns i
        for i=1:n_urns
            % go over each run j
            for j=1:n_runs
                % check if in this run, exactly i urns were empty
                if num_empty_urns_per_run(j) == i
                    % if so, increment counter
                    pmf_mat(i, ni) = pmf_mat(i, ni) + 1;
                end
            end
            % get fraction of runs where i urns were empty
            pmf_mat(i, ni) = pmf_mat(i, ni) / n_runs;
        end                           
    end

    % we have the PMF P(X=x)
    % now to get the CCDF P(X>x), we flip the axes with users on x-axis
    % and since it's >x, remove the last element
    ccdf_mat = zeros(n_users, n_urns-1);
    for i=1:n_users        
        for j=1:n_urns-1
            for k=j+1:n_urns % were we do iterate to the last element
                ccdf_mat(i,j) = ccdf_mat(i,j) + pmf_mat(k,i); % sum up
            end
        end        
    end
    
    expectation_mat = zeros(n_users, 1);
    for i=1:n_users                        
        for j=1:n_urns
            expectation_mat(i) = expectation_mat(i) + pmf_mat(j,i)*j / n_urns;
        end        
    end
end

function utilization_vec = resource_opportunity_sim_run(n_DME, t_LDACS, period_DME)            
    remainder = period_DME/t_LDACS - floor(period_DME/t_LDACS);
    
    % construct probability distribution
    n_idle_ldacs_slots = floor(period_DME / t_LDACS);    
    n_urns = n_idle_ldacs_slots + 1;    
    P = zeros(n_urns, 1);
    P(1) = remainder/(period_DME/t_LDACS);
    P(2:end) = (1 - P(1)) / n_idle_ldacs_slots;    
         
    % prepare simulation result container
    % this is just a counter of balls placed into each urn
    utilization_vec = zeros(n_urns, 1);
    utilization_vec(1) = 1;  % first urn is always occupied
    % each user
    for n=1:n_DME  
        % draw a random number between 0 and 1
        r = rand();
        % figure out which urn this goes to
        i = 1;
        if r > P(1)            
            % find the largest index of the element in the prob. distr.
            % that is smaller than r
            % then take the next urn
            % e.g. if P=[.33 .33 .33]
            % and r=0.78
            % then cumsum(P)=[.33 .66 1]
            % and P(2) is the largest element <0.78
            % and it should go to bin 3 which covers the range 0.66-1
            i = find(r>cumsum(P), 1, 'last' ) + 1;
        end
        % increment counter
        utilization_vec(i) = utilization_vec(i) + 1;
    end       
end