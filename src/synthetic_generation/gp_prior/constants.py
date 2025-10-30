KERNEL_BANK = {
    0: ("matern_kernel", 3),
    1: ("linear_kernel", 2),
    2: ("rbf_kernel", 2),
    3: ("periodic_kernel", 5),
    4: ("polynomial_kernel", 1),
    5: ("rational_quadratic_kernel", 1),
    6: ("spectral_mixture_kernel", 2),
}


KERNEL_PERIODS_BY_FREQ = {
    "min": [5, 15, 30, 60, 120, 240, 360],
    "H": [3, 6, 12, 24, 48, 72, 168],
    "D": [7, 14, 28, 30, 90, 180, 365],
    "W": [2, 4, 8, 12, 24, 52],
    "MS": [3, 4, 6, 12, 24, 36, 60],
}
