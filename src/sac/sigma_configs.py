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
