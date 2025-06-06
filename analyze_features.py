import numpy as np
import matplotlib.pyplot as plt
import os
from collections import defaultdict, Counter
import itertools


'''
Data structure
    HRTF: original, arma, averaged, N1N2P1P2
    Convolutional Layers: conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8, conv9, conv10
    Sound Categories: knock, drawer, clear throat, phone, keys drop, speech, keyboard, page turn, cough, door slam, laughter
    Units: 32 units (conv1, conv3, conv4) / 64 units (conv2, conv5, conv6) / 128 untis (conv7, conv8) / 256 units (conv9, conv10)
    DOA(elevation angle): 0°, 30°, 60°, 90°, 120°, 150°, 180°
    List of Average Activity values of a certain DOA

Evaluation process for each unit
    Calculate the mean value of its average activity for each DOA
    Calculate the difference d between the maximum and minimum
    Randomize the average activites and recalculate the difference d’ for 1000 times
    Decide whether d is bigger than 99% of d’
Possibilities
    sensitive: d is bigger than 99% of d’
    non-sensitive: otherwise

'''
def analyze_multiple_datasets(npz_paths, dataset_labels, layers_to_plot,
                              shuffle_times=1000, threshold_percentile=99,
                              save_fig_path=None, save_dist_path=None):
    all_avg_results = {}   # [dataset][layer] = avg_percent
    all_passed_indices = {}  # [dataset][layer][class] = list of passed neuron indices
    neuron_total = {}  # [dataset][layer] = total neurons

    for npz_path, label in zip(npz_paths, dataset_labels):
        print(f"\n🔍 Analyzing dataset: {label}")
        data = np.load(npz_path, allow_pickle=True)
        dataset_layer_percent = {}
        dataset_passed_per_class = defaultdict(lambda: defaultdict(list))
        dataset_total_per_layer = {}

        for layer in layers_to_plot:
            layer_data = data[layer].item()
            total_neurons_set = set()
            class_pass_ratios = []

            for class_name, neuron_dict in layer_data.items():
                passed_neurons = set()
                for neuron_idx, angle_dict in neuron_dict.items():
                    total_neurons_set.add(neuron_idx)

                    # # check and remove duplicated values
                    # all_values = []
                    # for values in angle_dict.values():
                    #     all_values.extend(values)
                    # value_counts = Counter(all_values)
                    # duplicate_values = {v for v, c in value_counts.items() if c > 1}
                    #
                    # cleaned_angle_dict = {}
                    # for angle, values in angle_dict.items():
                    #     cleaned_values = [v for v in values if v not in duplicate_values]
                    #     if cleaned_values:
                    #         cleaned_angle_dict[angle] = cleaned_values
                    #
                    # if len(cleaned_angle_dict) < 2:
                    #     continue
                    #
                    # angle_order = sorted(cleaned_angle_dict.keys())
                    # grouped = [cleaned_angle_dict[a] for a in angle_order]

                    angle_order = sorted(angle_dict.keys())
                    grouped = [angle_dict[a] for a in angle_order]
                    n_per_angle = [len(x) for x in grouped]

                    # check if each angle has the same number of activities
                    # if len(set(n_per_angle)) != 1:
                    #     continue

                    flat = np.concatenate(grouped)
                    n_angles = len(angle_order)
                    n_samples = n_per_angle[0]
                    means = [np.mean(x) for x in grouped]
                    orig_range = max(means) - min(means)

                    greater = 0
                    for _ in range(shuffle_times):
                        shuffled = np.random.permutation(flat)
                        shuffled_groups = [shuffled[i * n_samples:(i + 1) * n_samples] for i in range(n_angles)]
                        shuffled_range = max(map(np.mean, shuffled_groups)) - min(map(np.mean, shuffled_groups))
                        if orig_range > shuffled_range:
                            greater += 1

                    if greater >= shuffle_times * (threshold_percentile / 100):
                        passed_neurons.add(neuron_idx)

                if len(neuron_dict) > 0:
                    percent = len(passed_neurons) / len(neuron_dict) * 100
                    class_pass_ratios.append(percent)
                    dataset_passed_per_class[layer][class_name] = sorted(passed_neurons)

            avg_percent = np.mean(class_pass_ratios) if class_pass_ratios else 0
            dataset_layer_percent[layer] = avg_percent
            dataset_total_per_layer[layer] = len(total_neurons_set)

        all_avg_results[label] = dataset_layer_percent
        all_passed_indices[label] = dataset_passed_per_class
        neuron_total[label] = dataset_total_per_layer

    plot_percentage_comparison(all_avg_results, layers_to_plot, save_fig_path)
    plot_neuron_dot_distribution_per_layer(all_passed_indices, layers_to_plot, save_dist_path)


def plot_percentage_comparison(results, layers, save_path=None):
    for label, layer_result in results.items():
        percentages = [layer_result.get(layer, 0) for layer in layers]
        plt.plot(layers, percentages, label=label, marker='o')

    plt.xlabel("Layer")
    plt.ylabel("Average Sensitive Unit Percentage (%)")
    plt.title("DOA Sensitivity: Class-Averaged per Layer")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_neuron_dot_distribution_per_layer(neuron_data, layers, save_path=None):
    all_classes = sorted({cls for dataset in neuron_data.values()
                          for layer_dict in dataset.values()
                          for cls in layer_dict})
    datasets = list(neuron_data.keys())
    class_colors = plt.cm.get_cmap('tab10', len(all_classes))

    n_layers = len(layers)
    n_cols = 2
    n_rows = (n_layers + 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(14, 3 * n_rows), sharey=False)
    axs = axs.flatten()

    for layer_idx, layer in enumerate(layers):
        ax = axs[layer_idx]
        max_neuron_index = 0

        for d_idx, dataset in enumerate(datasets):
            y = d_idx  # 每个数据集在一条水平线上
            class_dict = neuron_data[dataset].get(layer, {})
            for c_idx, cls in enumerate(all_classes):
                neuron_indices = class_dict.get(cls, [])
                if not neuron_indices:
                    continue
                jittered_y = y + np.random.normal(0, 0.05, size=len(neuron_indices))  # 添加轻微扰动
                ax.scatter(neuron_indices, jittered_y,
                           label=cls,
                           color=class_colors(c_idx),
                           alpha=0.7,
                           s=25)
                if neuron_indices:
                    max_neuron_index = max(max_neuron_index, max(neuron_indices))

        ax.set_title(layer)
        ax.set_xlim(-1, max_neuron_index + 5)
        ax.set_xticks(np.arange(0, max_neuron_index + 10, step=20))
        ax.set_xlabel("Unit Index")
        ax.set_yticks(range(len(datasets)))
        ax.set_yticklabels(datasets)

    handles = [plt.Line2D([0], [0], marker='o', color='w', label=cls,
                          markerfacecolor=class_colors(i), markersize=6)
               for i, cls in enumerate(all_classes)]
    fig.legend(handles, all_classes, bbox_to_anchor=(1, 1), loc='upper right')

    fig.suptitle("Distribution of DOA-Sensitive Units by Layer and Dataset", fontsize=14)

    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    if save_path:
        plt.savefig(save_path)
    plt.show()

def main():
    npz_paths = [
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_original/val_feature_data_split1.npz",
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_arma/val_feature_data_split1.npz",
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_averaged/val_feature_data_split1.npz",
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_N1N2P1P2/val_feature_data_split1.npz"
    ]
    npz_paths_new = [
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_original/val_feature_data_split1_new_v2.npz",
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_arma/val_feature_data_split1_new_v2.npz",
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_averaged/val_feature_data_split1_new_v2.npz",
        "/home/wu/bc_learning_sound-master/save_model/samrai_original_envnetstereov2_11categories/samrai_test_N1N2P1P2/val_feature_data_split1_new_v2.npz"
    ]
    dataset_labels = ["original", "arma", "averaged", "N1N2P1P2"]
    layers = ["conv1", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "conv10"]

    # analyze_multiple_datasets(
    #     npz_paths=npz_paths,
    #     dataset_labels=dataset_labels,
    #     layers_to_plot=layers,
    # )
    analyze_multiple_datasets(
        npz_paths=npz_paths_new,
        dataset_labels=dataset_labels,
        layers_to_plot=layers,
    )

if __name__ == "__main__":
    main()
