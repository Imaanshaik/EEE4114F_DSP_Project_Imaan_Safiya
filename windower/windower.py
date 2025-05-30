import random
import numpy as np
import pandas as pd
from collections import defaultdict

from data.reader import get_data
from function_cache.function_cache import DEFAULT_CACHE

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
SAMPLES_PER_WINDOW = 150  # 3 seconds * 50 Hz

# ---------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------

@DEFAULT_CACHE.memoize(tag="WINDOWER")
def window_count_per_action() -> dict[np.uint8, int]:
    data = get_data()
    trials = data["trial"].unique()
    users = data["id"].unique()
    result = defaultdict(int)

    for trial in trials:
        for uid in users:
            subset = data[(data["trial"] == trial) & (data["id"] == uid)]
            if len(subset) < SAMPLES_PER_WINDOW:
                continue
            action = subset["act"].iloc[0]
            result[action] += len(subset) // SAMPLES_PER_WINDOW

    return dict(result)


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def max_windows_per_action() -> int:
    windows = window_count_per_action()
    return max(windows.values())


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def trial_count_per_action() -> dict[np.uint8, int]:
    data = get_data()
    result = defaultdict(int)
    for trial in data["trial"].unique():
        for uid in data["id"].unique():
            idx = data[(data["trial"] == trial) & (data["id"] == uid)].index
            if not idx.empty:
                action = data["act"].iloc[idx[0]]
                result[action] += 1
    return dict(result)


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def samples_per_trial() -> dict[np.uint8, dict[np.uint8, int]]:
    data = get_data()
    counts = {}
    for trial in data["trial"].unique():
        counts[trial] = {}
        for uid in data["id"].unique():
            counts[trial][uid] = ((data["trial"] == trial) & (data["id"] == uid)).sum()
    return counts


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def samples_per_action() -> dict[np.uint8, int]:
    data = get_data()
    return {act: (data["act"] == act).sum() for act in data["act"].unique()}


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def action_per_trial() -> dict[np.uint8, dict[np.uint8, int]]:
    data = get_data()
    result = {}
    for trial in data["trial"].unique():
        result[trial] = {}
        for uid in data["id"].unique():
            idx = data[(data["trial"] == trial) & (data["id"] == uid)].index
            if not idx.empty:
                result[trial][uid] = data["act"].iloc[idx[0]]
    return result


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def windows_per_trial():
    data = get_data()
    trial_samples = samples_per_trial()
    total_per_action = samples_per_action()
    trial_labels = action_per_trial()
    max_windows = max_windows_per_action()

    output = {}
    for trial in data["trial"].unique():
        output[trial] = {}
        for uid in data["id"].unique():
            act = trial_labels[trial][uid]
            ratio = trial_samples[trial][uid] / total_per_action[act]
            output[trial][uid] = int(np.floor(ratio * max_windows))
    return output


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def start_index_per_trial() -> dict[np.uint8, dict[np.uint8, int]]:
    data = get_data()
    result = {}
    for trial in data["trial"].unique():
        result[trial] = {}
        for uid in data["id"].unique():
            idx = data[(data["trial"] == trial) & (data["id"] == uid)].index
            if not idx.empty:
                result[trial][uid] = idx[0]
    return result


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def window_starts_per_trial() -> dict[np.uint8, dict[np.uint8, list[int]]]:
    start_idx = start_index_per_trial()
    num_samples = samples_per_trial()
    num_windows = windows_per_trial()

    result = {}
    for trial in start_idx:
        result[trial] = {}
        for uid in start_idx[trial]:
            base = start_idx[trial][uid]
            count = num_windows[trial][uid]
            max_start = num_samples[trial][uid] - SAMPLES_PER_WINDOW
            starts = np.linspace(0, max_start, num=count, dtype=int)
            result[trial][uid] = [base + s for s in starts]
    return result


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def window_starts_per_action() -> dict[np.uint8, list[int]]:
    trials = window_starts_per_trial()
    labels = action_per_trial()
    result = defaultdict(list)

    for trial in trials:
        for uid, starts in trials[trial].items():
            action = labels[trial][uid]
            result[action] += starts
    return dict(result)


@DEFAULT_CACHE.memoize(tag="WINDOWER")
def split_indices():
    random.seed(1234)
    train, val, test = [], [], []
    for action, starts in window_starts_per_action().items():
        indices = starts.copy()
        random.shuffle(indices)
        n = len(indices)
        split_train = int(n * 0.6)
        split_val = int((n - split_train) * 0.5)

        train += indices[:split_train]
        val += indices[split_train:split_train + split_val]
        test += indices[split_train + split_val:]
    return train, val, test


def train_window_start_indices():
    return split_indices()[0]

def validate_window_start_indices():
    return split_indices()[1]

def test_window_start_indices():
    return split_indices()[2]


def split_indices_by_users():
    data = get_data()
    np.random.seed(1234)
    all_starts = np.concatenate(split_indices())
    user_index_map = defaultdict(list)

    user_ids = list(data["id"].unique())
    np.random.shuffle(user_ids)

    split_75 = int(len(user_ids) * 0.75)
    train_users = user_ids[:split_75]
    test_users = user_ids[split_75:]

    for idx in all_starts:
        user = data.at[idx, "id"]
        user_index_map[user].append(idx)

    train = [i for user in train_users for i in user_index_map[user]]
    test = [i for user in test_users for i in user_index_map[user]]

    return train, test
