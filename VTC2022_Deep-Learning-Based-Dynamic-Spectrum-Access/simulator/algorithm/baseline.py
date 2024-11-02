from environment.dme_environment import *
from environment.simple_environment import get_env
from environment.dme_environment import DMEEnvironmentBase
from plot.prediction import plot_roc


def evaluate_pattern(tuple_list, k):
    for t in tuple_list:
        if (k % int(t[0])) == int(t[1]):
            return 0.0
    return 1.0


def create_candidate_tuples(d_min, d_max, anchor):
    candidate_tuples = []
    for d in range(d_min, d_max+1):
        candidate_tuples.append((d,anchor))
    return candidate_tuples


def baseline_algorithm(obs_vec, d_min, d_max):
    candidate_tuples = []
    for k, obs in enumerate(obs_vec):
        if k < d_max and obs == 0.0:
            candidate_tuples += create_candidate_tuples(max(d_min, k+1), d_max, k)
        if obs == 1.0:
            candidate_tuples[:] = [t for t in candidate_tuples if evaluate_pattern([t], k) == obs]
    return candidate_tuples


def train(env, steps):
    observations = []
    for _ in range(steps + 34):
        observations.append(env.step(0)[1])
    baseline_tuples = baseline_algorithm(observations[34:], 10, 33)

    return baseline_tuples


def predict(env, baseline_tuples, current_step, steps):
    labels_pattern = []
    predicted_pattern = []
    for i in range(steps):
        labels_pattern.append(env.step(0)[1])
        predicted_pattern.append(evaluate_pattern(baseline_tuples, current_step + i))

    return labels_pattern, predicted_pattern


if __name__ == "__main__":
    steps_training = 500

    data = {}
    for integer in [True, False]:
        env = DMEEnvironment(get_env(num_users=1, num_obs=1, integer=integer))
        env.reset()

        baseline_tuples = train(env, steps_training)
        labels_pattern, predicted_pattern = predict(env, baseline_tuples, steps_training, 500)

        if integer:
            data[r'Baseline $\frac{d}{l} \in \mathbb{N}$'] = (labels_pattern, predicted_pattern, "black")
        else:
            data[r'Baseline $\frac{d}{l} \notin \mathbb{N}$'] = (labels_pattern, predicted_pattern, "grey")

    plot_roc("roc_baseline", data)
