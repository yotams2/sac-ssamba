"""
Configuration file for standardizing the optimal sigma values
used across the SAC codebase (loss calculation, evaluation, and sweeps).
"""

OPTIMAL_SIGMAS = {
    16: {
        'Prosody': 0.2560,
        'Scene': 0.3393,
        'Timbre': 0.4498,
        'Vocal_Tract': 0.2947,
        'Voice_Quality': 0.1677
    },
    32: {
        'Prosody': 0.2223,
        'Scene': 0.2947,
        'Timbre': 0.3907,
        'Vocal_Tract': 0.2560,
        'Voice_Quality': 0.1265
    },
    64: {
        'Prosody': 0.1931,
        'Scene': 0.2947,
        'Timbre': 0.3907,
        'Vocal_Tract': 0.2223,
        'Voice_Quality': 0.0954
    }
}

# Winning hyperparameter configuration from Optuna hyperparameter study (v5_1 Trial #49)
OPTUNA_TAU = 0.8502

OPTUNA_OPTIMAL_SIGMAS = {
    16: {
        'Prosody': 0.3207,
        'Vocal_Tract': 0.6171,
        'Timbre': 0.5827,
        'Voice_Quality': 0.2850,
        'Scene': 0.4336
    },
    32: {
        'Prosody': 0.3207,
        'Vocal_Tract': 0.6171,
        'Timbre': 0.5827,
        'Voice_Quality': 0.2850,
        'Scene': 0.4336
    },
    64: {
        'Prosody': 0.3207,
        'Vocal_Tract': 0.6171,
        'Timbre': 0.5827,
        'Voice_Quality': 0.2850,
        'Scene': 0.4336
    }
}


