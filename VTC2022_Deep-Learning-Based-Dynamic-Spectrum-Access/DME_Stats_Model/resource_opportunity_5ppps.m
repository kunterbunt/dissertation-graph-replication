%% configure
n_runs = 5000;
n_DME = 0:50;
n_users = size(n_DME, 2);
T_LDACS = [6, 12, 24];
t_LDACS = T_LDACS(2);
n_durations = size(T_LDACS, 2);
period_DME = 200;
expectation_mats_sim = [];

% simulate
for t_LDACS=T_LDACS    
    n_urns = floor(period_DME / t_LDACS);

    disp(strcat(['Sampling ', num2str(n_runs*n_users), ' values for t=', num2str(t_LDACS), '...']))
    [pmf_mat__sim, ccdf_mat__sim, expectation_mat__sim] = resource_opportunity_sim(n_DME, t_LDACS, period_DME, n_urns, n_users, n_runs);
    expectation_mats_sim = [expectation_mats_sim, expectation_mat__sim];
end
% t_LDACS = T_LDACS(2);

% compute analytically
expectation_mats_mdl = [];

for t_LDACS=T_LDACS
    n_urns = floor(period_DME / t_LDACS);

    disp(strcat(['Evaluating analytical model for t=', num2str(t_LDACS), '...']))
    expectation_mats_mdl = [expectation_mats_mdl, resource_opportunity_expectation(n_DME, n_urns, t_LDACS, period_DME)];
end
expectation_legend_vec__mdl = {};
for i=1:n_durations
    expectation_legend_vec__mdl{end + 1} = strcat('$t=', num2str(T_LDACS(i)), '$ms');                
end


%% plot both
FILENAME = strcat('imgs/both_expectation_n-', num2str(min(n_DME)), '_', num2str(max(n_DME)));
% fig = figure('visible', 'off');
fig = figure();
ax = axes;
ax.ColorOrder = [0, 0.4470, 0.7410; 0.8500, 0.3250, 0.0980; 0.4660, 0.6740, 0.1880];
hold on;
for i=1:n_durations
    p = plot(n_DME+1, expectation_mats_mdl(:,i)*100, 'LineWidth', 2, 'DisplayName', strcat('model $l=', num2str(T_LDACS(i)), '\,$ms'));      
    p.Color(4) = .5;
end
for i=1:n_durations
    p = plot(n_DME+1, expectation_mats_sim(:,i)*100, 'LineStyle', '--', 'LineWidth', 2, 'DisplayName', strcat('simulation $l=', num2str(T_LDACS(i)), '\,$ms'));    
    p.Color(4) = .75;
end
xlabel('no. of DME users $n$', 'Interpreter', 'latex');
ylabel('Expected idle time slots [\%]', 'Interpreter', 'latex');
legend('Interpreter', 'latex');
hold off;
xlim([1, n_DME(end)+1]);
set(gca, 'FontSize', 16);
set(gcf, 'color', 'w');
set(fig, 'Units', 'Inches');
pos = get(fig, 'Position');
set(fig, 'PaperPositionMode', 'Auto', 'PaperUnits', 'Inches', 'PaperSize', [pos(3), pos(4)])
% print(fig, FILENAME, '-dpdf', '-r0')
% disp(['Saved to ', FILENAME, '.pdf.']);
writematrix(expectation_mats_mdl, 'data/expectation_mats_model_5ppps.csv');
disp(['Saved data to data/expectation_mats_model_5ppps.csv']);
writematrix(expectation_mats_sim, 'data/expectation_mats_sim_5ppps.csv');
disp(['Saved data to data/expectation_mats_sim_5ppps.csv']);
disp(['Hint: with the saved .csv files, you can call plot.py to plot using Python (which looks nicer)!']);