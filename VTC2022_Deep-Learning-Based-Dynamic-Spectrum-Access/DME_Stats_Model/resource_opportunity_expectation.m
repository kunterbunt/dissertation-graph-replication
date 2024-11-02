function expectation_vec = resource_opportunity_model(N, m, t_LDACS, period_DME)                    
    r = period_DME/t_LDACS - floor(period_DME/t_LDACS);    
    q = r/(period_DME/t_LDACS);
    num_users = size(N, 2);
    expectation_vec = zeros(num_users, 1);    
    for i=1:num_users
        n=N(i);              
        expectation_vec(i) = m * ((m-1+q)/m)^n / m;        
    end
end