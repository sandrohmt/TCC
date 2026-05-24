# preços por hora (USD)
prices = {
    "c6i.xlarge": 0.17,
    "c7i.2xlarge": 0.357,
    "c8a.4xlarge": 0.86216,
    "z1d.3xlarge": 1.116,
    "g4dn.xlarge (T4)": 0.526,
    "g5.2xlarge (A10G)": 1.212,
    "g6e.2xlarge (L40S)": 2.24208,
    "inf1.2xlarge": 0.362,
    "inf2.xlarge": 0.7582,
    "r8id.4xlarge": 1.33056
}

# tempos (CPU + GPU)
times = {
    "c6i.xlarge": {"small": None, "medium": None, "large": None},
    "c7i.2xlarge": {"small": None, "medium": None, "large": None},
    "c8a.4xlarge": {"small": None, "medium": None, "large": None},

    "z1d.3xlarge": {"small": 764.92, "medium": 4817.89, "large": None},

    "g4dn.xlarge (T4) - CPU": {"small": None, "medium": None, "large": None},
    "g5.2xlarge (A10G) - CPU": {"small": None, "medium": None, "large": None},
    "g6e.2xlarge (L40S) - CPU": {"small": 2541.11, "medium": None, "large": None},

    "g4dn.xlarge (T4) - GPU": {"small": 240.88, "medium": 1293.44, "large": 3685.19},
    "g5.2xlarge (A10G) - GPU": {"small": 68.41, "medium": 423.65, "large": 1162.25},
    "g6e.2xlarge (L40S) - GPU": {"small": 46.97, "medium": 260.47, "large": 784.54},

    "inf1.2xlarge": {"small": None, "medium": None, "large": None},
    "inf2.xlarge": {"small": None, "medium": None, "large": None},

    "r8id.4xlarge": {"small": 574.65, "medium": 3326.73, "large": None}
}

FACTOR = 1000 / 3600  # 0.277777...

def format_br(value):
    return f"{value:.4f}".replace(".", ",")

def calc(time, price):
    return time * price * FACTOR

# execução
for instance, size_times in times.items():
    # pegar nome base da instância (sem " - CPU/GPU")
    base_instance = instance.split(" - ")[0]
    price = prices[base_instance]

    results = {}
    for size, t in size_times.items():
        if t is None:
            results[size] = "N/A"
        else:
            results[size] = format_br(calc(t, price))

    print(instance)
    print(f"small: {results['small']} / medium: {results['medium']} / large: {results['large']}")
    print()