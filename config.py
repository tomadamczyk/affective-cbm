import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

for d in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

ZUCO_N_CHANNELS = 128
ZUCO_MAX_WORDS = 40

EEG_BANDS = {
    "theta1": (4, 6),
    "theta2": (6, 8),
    "alpha1": (8, 10),
    "alpha2": (10, 13),
    "beta1": (13, 18),
    "beta2": (18, 30),
    "gamma1": (30, 40),
    "gamma2": (40, 50),
}
N_EEG_BANDS = len(EEG_BANDS)

SENTIMENT_LABELS = {
    "negative": 0,
    "positive": 1,
    "neutral": 2,
}
NUM_CLASSES = 3

EMBEDDING_DIM = 256
TEXT_MODEL_NAME = "roberta-base"
EEG_INPUT_CHANNELS = 128
EEG_SEQUENCE_LENGTH = 512
EEG_TEMPORAL_DIM = 64

NEURAL_CONCEPT_NAMES = [
    "frontal_alpha_asymmetry",
    "parietal_alpha_asymmetry",
    "frontal_theta_power",
    "parietal_alpha_power",
    "central_beta_power",
    "temporal_gamma_power",
    "theta_beta_ratio",
    "alpha_theta_ratio",
    "frontal_midline_theta",
    "posterior_alpha_suppression",
    "beta_gamma_ratio",
    "left_temporal_gamma",
    "global_alpha_power",
    "frontal_beta_asymmetry",
    "occipital_alpha_power",
]
N_NEURAL_CONCEPTS = len(NEURAL_CONCEPT_NAMES)

NUM_CONCEPTS = 80
CONCEPT_CATEGORIES = {
    "dimensional": ["valence", "arousal", "dominance"],
    "discrete_emotions": ["happiness", "sadness", "anger", "fear", "surprise", "disgust"],
    "cognitive_affective": ["engagement", "confusion", "boredom", "frustration", "cognitive_load"],
    "linguistic_affective": ["irony", "emotional_intensity", "sentiment_ambiguity"],
}

BATCH_SIZE = 16
LEARNING_RATE = 1e-4
CBL_LEARNING_RATE = 5e-4
PREDICTOR_LEARNING_RATE = 1e-3
NUM_EPOCHS_CBL = 30
NUM_EPOCHS_PREDICTOR = 30
WEIGHT_DECAY = 1e-5
DROPOUT = 0.3
WARMUP_STEPS = 100
MAX_GRAD_NORM = 1.0
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

PREDICTOR_L1_LAMBDA = 0.01

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
