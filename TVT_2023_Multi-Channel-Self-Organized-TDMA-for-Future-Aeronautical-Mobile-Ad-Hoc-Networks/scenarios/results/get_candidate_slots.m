function k = get_candidate_slots(target_error_prob, num_contenders)
%GET_CANDIDATE_SLOTS Returns candidate slot set size to achieve target error probability. 
    p = target_error_prob;
    n = num_contenders;
    if n > 0
        k = ceil(1 / (1 - nthroot(1-p, n)));
    else
        k = 1;
    end
end

