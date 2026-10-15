# Image preprocessing (ImageNet statistics used by the pretrained backbone)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# Labels and roles
NORMAL_CLASS = "good"
ROLE_FIT, ROLE_VAL, ROLE_TEST = "fit", "val_normal", "test"
EVAL_DEV, EVAL_FINAL = "dev", "final"

# Artifact file names (so a name is changed in one place only)
RECORDS_FILE = "records.csv"
SPLITS_FILE = "splits.csv"
FIT_EMBEDDINGS_FILE = "fit_embeddings.pt"
MEMORY_BANK_FILE = "memory_bank.pt"
THRESHOLD_FILE = "threshold.json"

EXPERIMENT_INFO_FILE = "experiment_info.json"