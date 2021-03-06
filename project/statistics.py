import matplotlib.pyplot as plt
import pickle
import os


def update_statistics_file(file_path, param_name, param_value):
    with open(file_path, 'rb') as f:
        # data = f.read()
        # print("len(data):", len(data))
        # data_dict = pickle.loads(data)
        data_dict = pickle.load(f)
        data_dict['param_name'] = param_name
        data_dict['param_value'] = param_value

    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)


def plot_single_statistics(file_path, mean_ax=None, best_mean_ax=None):
    """
    Plot single statistics.
    :param file_path: Path to statistics file.
    :param mean_ax: Axis to plot the mean episode rewards.
    :param best_mean_ax: Axis to plot the best mean episode rewards.
    """
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
        mean_episode_reward = data_dict['mean_episode_rewards']
        best_mean_episode_rewards = data_dict['best_mean_episode_rewards']
        param_name = data_dict['param_name']
        param_value = data_dict['param_value']

    if mean_ax is None:
        fig, (mean_ax, best_mean_ax) = plt.subplots(2, 1, sharex=True)
        fig.suptitle("Learning Curve")

    n_time_steps = len(mean_episode_reward)
    t = range(n_time_steps)

    mean_ax.plot(t, mean_episode_reward, label="{}={}".format(param_name, param_value))
    mean_ax.set_title("mean episode reward")
    mean_ax.set_xlabel('t')
    mean_ax.set_ylabel('value')
    mean_ax.legend()

    best_mean_ax.plot(t, best_mean_episode_rewards, label="{}={}".format(param_name, param_value))
    best_mean_ax.set_title("best mean episode reward")
    best_mean_ax.set_xlabel('t')
    best_mean_ax.set_ylabel('value')
    best_mean_ax.legend()


def plot_multiple_statistics(dir_path):
    fig, (mean_ax, best_mean_ax) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Learning Curve")

    for filename in os.listdir(dir_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(dir_path, filename)
            plot_single_statistics(file_path, mean_ax=mean_ax, best_mean_ax=best_mean_ax)


def plot_best_experiment(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f)
        mean_episode_reward = data_dict['mean_episode_rewards']
        best_mean_episode_rewards = data_dict['best_mean_episode_rewards']
        param_name = data_dict['param_name']
        param_value = data_dict['param_value']

    fig, ax = plt.subplots()
    fig.suptitle("Learning Curve - {}={}".format(param_name, param_value))

    n_time_steps = len(mean_episode_reward)
    t = range(n_time_steps)

    ax.plot(t, mean_episode_reward, label="mean reward")
    ax.plot(t, best_mean_episode_rewards, label="best mean reward")

    ax.set_xlabel('t')
    ax.set_ylabel('mean reward')
    ax.ticklabel_format(style='scientific', axis='x')
    ax.legend()


def plot_hyperparameter_experiments(dir_path):
    fig, ax = plt.subplots()
    fig.suptitle("Hyperparameter Experiments")

    for filename in os.listdir(dir_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'rb') as f:
                data_dict = pickle.load(f)
                mean_episode_reward = data_dict['mean_episode_rewards']
                param_name = data_dict['param_name']
                param_value = data_dict['param_value']

            n_time_steps = len(mean_episode_reward)
            t = range(n_time_steps)

            ax.plot(t, mean_episode_reward, label="{}={}".format(param_name, param_value))

            ax.set_xlabel('t')
            ax.set_ylabel('mean reward')
            ax.ticklabel_format(style='scientific', axis='x')
            ax.legend()

# update_statistics_file("statistics/test_statistics_gamma0999.pkl", "gamma", 0.999)
# update_statistics_file("statistics/experiments/test_statistics_alon_default.pkl", "learning_rate", 0.00025)
# update_statistics_file("statistics/test_statistics_alon_lr0005.pkl", "learning_rate", 0.0005)
# update_statistics_file("statistics/test_statistics_batch16.pkl", "batch_size", 16)
# plot_single_statistics("statistics/test_statistics_gamma0999.pkl")
# update_statistics_file("statistics/test_statistics_lr025.pkl", "learning_rate", 0.25)
# update_statistics_file("statistics/test_statistics_lr_005.pkl", "learning_rate", 0.05)
# update_statistics_file("statistics/test_statistics_lr0025.pkl", "learning_rate", 0.0025)
# update_statistics_file("statistics/experiments/test_statistics_lr001.pkl", "learning_rate", 0.001)
# update_statistics_file("statistics/experiments/test_statistics_lr0001.pkl", "learning_rate", 0.0001)
# update_statistics_file("statistics/other_experiments/test_statistics_batch16.pkl", "batch_size", 16)
# update_statistics_file("statistics/other_experiments/test_statistics_default.pkl", "params", "default")
# update_statistics_file("statistics/other_experiments/test_statistics_gamma999.pkl", "gamma", 0.999)

# plot_multiple_statistics('statistics')
# plt.show()

# plot_best_experiment("statistics/test_statistics_alon_default.pkl")
# plot_best_experiment("statistics/experiments/test_statistics_lr001.pkl")
# plt.savefig("statistics/experiments/best_hyperparameter_lr001")

# plot_hyperparameter_experiments("statistics/experiments")
# plt.savefig("statistics/experiments/learning_rate_experiments")

# plot_hyperparameter_experiments("statistics/other_experiments")
# plt.savefig("statistics/experiments/other_experiments")
