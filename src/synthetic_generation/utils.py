import numpy as np
import torch


def generate_spikes(
    size: int,
    spikes_type: str = "choose_randomly",
    spike_intervals: int | None = None,
    n_spikes: int | None = None,
    to_keep_rate: float = 0.4,
):
    spikes = np.zeros(size)
    if size < 120:
        build_up_points = 1
    elif size < 250:
        build_up_points = np.random.choice([2, 1], p=[0.3, 0.7])
    else:
        build_up_points = np.random.choice([3, 2, 1], p=[0.15, 0.45, 0.4])

    spike_duration = build_up_points * 2

    if spikes_type == "choose_randomly":
        spikes_type = np.random.choice(["regular", "patchy", "random"], p=[0.4, 0.5, 0.1])

    if spikes_type == "patchy" and size < 64:
        spikes_type = "regular"

    if spikes_type in ["regular", "patchy"]:
        if spike_intervals is None:
            upper_bound = np.ceil(spike_duration / 0.05)  ## at least 1 spike every 24 periods (120 if 5 spike duration)
            lower_bound = np.ceil(spike_duration / 0.15)  ## at most 3 spikes every 24 periods
            spike_intervals = np.random.randint(lower_bound, upper_bound)
        n_spikes = np.ceil(size / spike_intervals)
        spike_intervals = np.arange(spike_intervals, size, spike_intervals)
        if spikes_type == "patchy":
            patch_size = np.random.randint(2, max(n_spikes * 0.7, 3))
            to_keep = np.random.randint(np.ceil(patch_size * to_keep_rate), patch_size)
    else:
        n_spikes = (
            n_spikes if n_spikes is not None else np.random.randint(4, min(max(size // (spike_duration * 3), 6), 20))
        )
        spike_intervals = np.sort(np.random.choice(np.arange(spike_duration, size), size=n_spikes, replace=False))

    constant_build_rate = False
    if spikes_type in ["regular", "patchy"]:
        random_ = np.random.random()
        constant_build_rate = True

    patch_count = 0
    spike_intervals -= 1
    for interval in spike_intervals:
        interval = np.round(interval).astype(int)
        if spikes_type == "patchy":
            if patch_count >= patch_size:
                patch_count = 0
            if patch_count < to_keep:
                patch_count += 1
            else:
                patch_count += 1
                continue
        if not constant_build_rate:
            random_ = np.random.random()
        build_up_rate = np.random.uniform(0.5, 2) if random_ < 0.7 else np.random.uniform(2.5, 5)

        spike_start = interval - build_up_points + 1
        for i in range(build_up_points):
            if 0 <= spike_start + i < len(spikes):
                spikes[spike_start + i] = build_up_rate * (i + 1)

        for i in range(1, build_up_points):
            if (interval + i) < len(spikes):
                spikes[interval + i] = spikes[interval - i]

    # randomly make it positive or negative
    spikes += 1
    spikes = spikes * np.random.choice([1, -1], 1, p=[0.7, 0.3])

    return torch.Tensor(spikes)


def generate_peak_spikes(ts_size, peak_period, spikes_type="regular"):
    return generate_spikes(ts_size, spikes_type=spikes_type, spike_intervals=peak_period)
