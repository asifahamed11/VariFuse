"""Stage 08 - TDA + fuzzy KNORA-E ensemble: feature build, training, and evaluation."""

from matplotlib import colors
import pandas as pd
from config import DATA_DIR, STAGE07_OUT, STAGE08_OUT
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import sem, chi2_contingency
import time
import copy
import warnings
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from joblib import Parallel, delayed
import multiprocessing
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    learning_curve,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_classif

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    cohen_kappa_score,
    brier_score_loss,
)
from sklearn.manifold import TSNE

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except ImportError:
    print("Warning: LightGBM not available")
    LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except ImportError:
    print("Warning: CatBoost not available")
    CATBOOST_AVAILABLE = False

try:
    from gtda.homology import VietorisRipsPersistence
    from gtda.diagrams import (
        PersistenceEntropy,
        Amplitude,
        BettiCurve,
        PersistenceLandscape,
    )

    TDA_AVAILABLE = True
except ImportError:
    print("Warning: giotto-tda not available. Install with: pip install giotto-tda")
    TDA_AVAILABLE = False

try:
    from pyfaidx import Fasta

    PYFAIDX_AVAILABLE = True
except ImportError:
    print("Warning: pyfaidx not available. Install with: pip install pyfaidx")
    PYFAIDX_AVAILABLE = False

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not available. Install with: pip install shap")
    SHAP_AVAILABLE = False

warnings.filterwarnings("ignore")


class Config:
    """Model, feature-engineering, and training hyper-parameters."""

    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_JOBS = -1

    KNORA_K = 11
    MIN_COMPETENCE_THRESHOLD = 0.55  # was 0.70 (diversity patch)
    TDA_N_NEIGHBORS = 75
    SEQUENCE_WINDOW = 75
    FCGR_K_VALUES = [3, 4]
    # TDA homology dims kept at (0, 1); the extractor now produces
    # ~24 features instead of 6, all computed over the FCGR representation.
    TDA_HOMOLOGY_DIMS = (0, 1)

    FCGR_PCA_COMPONENTS = 20

    WEAK_LEARNER_MARGIN = 0.22  # was 0.10 (diversity patch: keep specialists)

    THRESHOLD_SEARCH_LOW = 0.45
    THRESHOLD_SEARCH_HIGH = 0.65

    # Selective prediction — abstain when neighbourhood purity is
    # below this threshold. Set to 0.0 to disable (returns all predictions).
    SELECTIVE_ABSTAIN_THRESHOLD = 0.60  # neighbourhood purity ≥ 0.60 to predict

    # Consensus margin — variants whose |CONSENSUS_SCORE − 0.5| is
    # below this value are treated as ambiguous; their sample weight during
    # base-learner training is scaled down to AMBIGUOUS_SAMPLE_WEIGHT.
    AMBIGUOUS_MARGIN = 0.10
    AMBIGUOUS_SAMPLE_WEIGHT = 1.0  # was 0.30 (diversity patch: disable down-weighting)

    # --- NEW: random-subspace + bagging diversity controls (diversity patch) ---
    BOOTSTRAP_FRACTION = 0.75      # per-model row bootstrap sample size
    SUBSPACE_BIO_FRACTION = 0.60   # fraction of bio cols each member keeps
    SUBSPACE_OTHER_FRACTION = 0.70 # fraction of FCGR/TDA cols each member keeps
    DIRICHLET_ALPHA = 0.60         # <1 => sharper per-model reweighting
    POOL_REPLICAS = 2              # bagging replicas per base architecture

    DATA_PATH = STAGE07_OUT / "Final_Dataset_Balanced.csv"
    GENOME_PATH = DATA_DIR / "hg19.fa"
    OUTPUT_DIR = Path(STAGE08_OUT)

    LEAKAGE_COLS = ["chr", "pos", "ref", "alt", "CONSENSUS_SCORE", "TIER"]


Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
np.random.seed(Config.RANDOM_STATE)

FIG_SINGLE_COL_WIDTH = 85 / 25.4
FIG_DOUBLE_COL_WIDTH = 170 / 25.4
FIG_DPI = 300
FIG_FONT_SIZE = 10
FIG_TITLE_SIZE = 12
FIG_LABEL_SIZE = 10

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": FIG_FONT_SIZE,
        "axes.titlesize": FIG_TITLE_SIZE,
        "axes.labelsize": FIG_LABEL_SIZE,
        "xtick.labelsize": FIG_FONT_SIZE,
        "ytick.labelsize": FIG_FONT_SIZE,
        "legend.fontsize": FIG_FONT_SIZE - 1,
        "figure.dpi": FIG_DPI,
        "savefig.dpi": FIG_DPI,
        "savefig.format": "png",
        "savefig.bbox": "tight",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
plt.style.use("seaborn-v0_8-white")
sns.set_palette("colorblind")


# FCGR encoders (unchanged)


class FCGREncoder:
    """Frequency Chaos Game Representation encoder"""

    def __init__(self, k=4):
        self.k = k
        self.grid_size = 2**k
        self.vertices = {
            "A": (0.0, 0.0),
            "C": (0.0, 1.0),
            "G": (1.0, 1.0),
            "T": (1.0, 0.0),
        }

    def encode_sequence(self, sequence):
        sequence = sequence.upper()
        counts = np.zeros((self.grid_size, self.grid_size))
        curr_x, curr_y = 0.5, 0.5
        for nucleotide in sequence:
            if nucleotide not in self.vertices:
                continue
            vertex = self.vertices[nucleotide]
            curr_x = 0.5 * (curr_x + vertex[0])
            curr_y = 0.5 * (curr_y + vertex[1])
            row = min(int(curr_x * self.grid_size), self.grid_size - 1)
            col = min(int(curr_y * self.grid_size), self.grid_size - 1)
            counts[row, col] += 1
        if len(sequence) > 0:
            return counts.flatten() / len(sequence)
        return counts.flatten()

    def encode_variants(self, sequences):
        return np.array([self.encode_sequence(seq) for seq in sequences])


class MultiScaleFCGR:
    """Multi-scale FCGR encoder"""

    def __init__(self, k_values=[3, 4]):
        self.k_values = k_values
        self.encoders = [FCGREncoder(k=k) for k in k_values]

    def encode_variants(self, sequences):
        encoded_features = []
        for encoder in self.encoders:
            features = encoder.encode_variants(sequences)
            encoded_features.append(features)
        return np.hstack(encoded_features)




class TopologicalFeatureExtractor:
    """Extract topological features via persistent homology.

    Operates on the FCGR/sequence representation rather than the biological
    feature space: call ``extract_global_tda(X_fcgr)`` where ``X_fcgr`` is the
    PCA-compressed FCGR matrix (n_samples x 20). Produces ~20 topological
    descriptors per sample.
    """

    def __init__(self, homology_dimensions=(0, 1), n_neighbors=75):
        self.homology_dimensions = homology_dimensions
        self.n_neighbors = n_neighbors
        if TDA_AVAILABLE:
            self.vr_persistence = VietorisRipsPersistence(
                homology_dimensions=homology_dimensions, n_jobs=-1
            )
            self.persistence_entropy = PersistenceEntropy()
            self.amplitude = Amplitude(metric="persistence")

    def extract_local_tda(self, X, indices):
        if not TDA_AVAILABLE:
            return self._zero_features()

        X_local = X[indices].reshape(len(indices), -1)
        try:
            diagrams = self.vr_persistence.fit_transform([X_local])
        except Exception:
            return self._zero_features()

        features = {}
        for dim in self.homology_dimensions:
            diagram = diagrams[0][diagrams[0][:, 2] == dim]
            prefix = f"h{dim}"
            if len(diagram) > 0:
                birth = diagram[:, 0]
                death = diagram[:, 1]
                persistence = death - birth

                # ---- original 3 features ----
                total_pers = float(np.sum(persistence))
                features[f"total_persistence_{prefix}"] = total_pers
                if total_pers > 0:
                    p_norm = persistence / total_pers
                    entropy = -np.sum(p_norm * np.log(p_norm + 1e-10))
                    features[f"entropy_{prefix}"] = float(entropy)
                else:
                    features[f"entropy_{prefix}"] = 0.0
                features[f"n_components_{prefix}"] = float(len(diagram))

                # ---- new features ----
                features[f"amplitude_{prefix}"] = float(np.max(persistence))
                features[f"persistence_std_{prefix}"] = float(np.std(persistence))
                features[f"persistence_max_{prefix}"] = float(np.max(persistence))
                features[f"birth_mean_{prefix}"] = float(np.mean(birth))
                features[f"death_mean_{prefix}"] = float(np.mean(death))
                # Betti mean: proportion of filtration steps where feature lives
                # Approximate as ratio of mean persistence to total filtration span
                span = float(np.max(death) - np.min(birth)) if len(death) > 0 else 1.0
                span = max(span, 1e-10)
                features[f"betti_mean_{prefix}"] = float(np.mean(persistence) / span)
                # L1 norm of first persistence landscape (simplified)
                sorted_p = np.sort(persistence)[::-1]
                l1_land = float(np.sum(sorted_p * np.arange(1, len(sorted_p) + 1)))
                features[f"landscape_l1_{prefix}"] = l1_land
            else:
                for key in [
                    f"total_persistence_{prefix}",
                    f"entropy_{prefix}",
                    f"n_components_{prefix}",
                    f"amplitude_{prefix}",
                    f"persistence_std_{prefix}",
                    f"persistence_max_{prefix}",
                    f"birth_mean_{prefix}",
                    f"death_mean_{prefix}",
                    f"betti_mean_{prefix}",
                    f"landscape_l1_{prefix}",
                ]:
                    features[key] = 0.0
        return features

    def _zero_features(self):
        features = {}
        for dim in self.homology_dimensions:
            prefix = f"h{dim}"
            for key in [
                f"total_persistence_{prefix}",
                f"entropy_{prefix}",
                f"n_components_{prefix}",
                f"amplitude_{prefix}",
                f"persistence_std_{prefix}",
                f"persistence_max_{prefix}",
                f"birth_mean_{prefix}",
                f"death_mean_{prefix}",
                f"betti_mean_{prefix}",
                f"landscape_l1_{prefix}",
            ]:
                features[key] = 0.0
        return features

    def _process_single_sample(self, X, nbrs, i, n_neighbors):
        distances, indices = nbrs.kneighbors([X[i]], n_neighbors=n_neighbors)
        return self.extract_local_tda(X, indices[0])

    def extract_global_tda(self, X, n_jobs=-1):
        """Extract TDA features in parallel across samples.

        ``X`` must be the FCGR representation (PCA-compressed, n_samples x 20),
        not the biological/standard features.
        """
        n_samples = X.shape[0]
        n_neighbors = min(self.n_neighbors, n_samples)
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
        nbrs.fit(X)
        if n_jobs == -1:
            n_jobs = multiprocessing.cpu_count()
        elif n_jobs < 0:
            n_jobs = max(1, multiprocessing.cpu_count() + 1 + n_jobs)
        print(f"Extracting TDA features (over FCGR space) using {n_jobs} CPU cores...")
        tda_features_list = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(
            delayed(self._process_single_sample)(X, nbrs, i, n_neighbors)
            for i in range(n_samples)
        )
        tda_df = pd.DataFrame(tda_features_list)
        return tda_df.values


# TFDFEPreprocessor


class TFDFEPreprocessor:
    """End-to-end preprocessing pipeline for the framework.

    The TDA extractor receives ``X_fcgr`` (the PCA-compressed FCGR matrix)
    rather than ``X_standard`` (biological features), so the topological
    features are structurally independent of the biological scalar scores and
    give the ensemble an orthogonal signal axis. FCGR is PCA-compressed from
    320 to 20 features.
    """

    def __init__(self, use_fcgr=True, use_tda=True, genome_path="hg19.fa"):
        self.leakage_cols = Config.LEAKAGE_COLS
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer_num = SimpleImputer(strategy="median")

        self.use_fcgr = use_fcgr
        self.use_tda = use_tda
        self.genome_path = genome_path

        # PCA to compress FCGR features
        self.fcgr_pca = None
        self.fcgr_pca_n_components = Config.FCGR_PCA_COMPONENTS

        self.genome = None
        if use_fcgr and PYFAIDX_AVAILABLE and Path(genome_path).exists():
            print(f"Loading reference genome: {genome_path}")
            self.genome = Fasta(genome_path)
            print("Genome loaded successfully")
        elif use_fcgr:
            print(f"WARNING: Genome file not found at {genome_path}")

        if use_fcgr:
            self.fcgr_encoder = MultiScaleFCGR(k_values=Config.FCGR_K_VALUES)

        if use_tda:
            self.tda_extractor = TopologicalFeatureExtractor(
                homology_dimensions=Config.TDA_HOMOLOGY_DIMS,
                n_neighbors=Config.TDA_N_NEIGHBORS,
            )

        self.feature_names = None
        self.fcgr_feature_names = None
        self.tda_feature_names = None

    def _extract_sequence_context(self, variant_df):
        """Extract genomic sequences around each variant"""
        print(f"\nExtracting genomic sequences (±{Config.SEQUENCE_WINDOW}bp)...")
        sequences = []
        failed_extractions = 0
        for idx, row in tqdm(
            variant_df.iterrows(),
            total=len(variant_df),
            desc="Sequence Extraction",
            ncols=80,
        ):
            try:
                chrom = str(row["chr"])
                if not chrom.startswith("chr"):
                    chrom = f"chr{chrom}"
                pos = int(row["pos"])
                start = max(0, pos - Config.SEQUENCE_WINDOW - 1)
                end = pos + Config.SEQUENCE_WINDOW
                if self.genome is not None and chrom in self.genome:
                    seq = str(self.genome[chrom][start:end]).upper()
                    if len(seq) < (2 * Config.SEQUENCE_WINDOW - 10):
                        needed_length = 2 * Config.SEQUENCE_WINDOW + len(
                            row.get("ref", "A")
                        )
                        seq = seq.ljust(needed_length, "N")
                    sequences.append(seq)
                else:
                    seq_length = 2 * Config.SEQUENCE_WINDOW + len(row.get("ref", "A"))
                    fallback_seq = "".join(
                        np.random.choice(["A", "C", "G", "T"], seq_length)
                    )
                    sequences.append(fallback_seq)
                    failed_extractions += 1
            except Exception:
                seq_length = 2 * Config.SEQUENCE_WINDOW + len(row.get("ref", "A"))
                fallback_seq = "".join(
                    np.random.choice(["A", "C", "G", "T"], seq_length)
                )
                sequences.append(fallback_seq)
                failed_extractions += 1
        print(
            f"Successfully extracted: {len(sequences) - failed_extractions}/{len(sequences)}"
        )
        return sequences

    def fit_transform(self, variant_df, target_col="LABEL_PATHOGENIC"):
        """Fit and transform training data with all feature types"""
        print("\nTF-DFE preprocessing pipeline")

        df_clean = variant_df.drop(columns=self.leakage_cols, errors="ignore")
        y = df_clean[target_col]
        X_tabular = df_clean.drop(target_col, axis=1, errors="ignore")

        num_cols = X_tabular.select_dtypes(include=["number"]).columns
        X_num = X_tabular[num_cols].copy()
        X_num = pd.DataFrame(
            self.imputer_num.fit_transform(X_num),
            columns=num_cols,
            index=X_tabular.index,
        )
        cat_cols = X_tabular.select_dtypes(include=["object"]).columns
        X_cat = X_tabular[cat_cols].copy()
        for col in cat_cols:
            self.label_encoders[col] = LabelEncoder()
            X_cat[col] = self.label_encoders[col].fit_transform(X_cat[col].astype(str))

        X_combined = pd.concat([X_num, X_cat], axis=1)
        self.feature_names = X_combined.columns.tolist()
        X_standard = self.scaler.fit_transform(X_combined)
        print(f"Standard features: {len(self.feature_names)}")

        # FCGR encoding + PCA compression
        if self.use_fcgr:
            print("\nGenerating FCGR fractal features...")
            sequences = self._extract_sequence_context(variant_df)
            print("Encoding sequences to FCGR vectors...")
            X_fcgr_raw = self.fcgr_encoder.encode_variants(sequences)
            print(f"FCGR raw features: {X_fcgr_raw.shape[1]}")

            n_comp = min(
                self.fcgr_pca_n_components, X_fcgr_raw.shape[1], X_fcgr_raw.shape[0] - 1
            )
            print(f"Compressing FCGR {X_fcgr_raw.shape[1]}D -> {n_comp}D via PCA...")
            self.fcgr_pca = PCA(n_components=n_comp, random_state=Config.RANDOM_STATE)
            X_fcgr = self.fcgr_pca.fit_transform(X_fcgr_raw)
            evr = float(np.sum(self.fcgr_pca.explained_variance_ratio_)) * 100
            print(
                f"FCGR PCA features: {X_fcgr.shape[1]}  (explained variance: {evr:.1f}%)"
            )
            self.fcgr_feature_names = [f"fcgr_pca_{i}" for i in range(X_fcgr.shape[1])]
            # Store raw FCGR for TDA
            self._X_fcgr_for_tda = X_fcgr
        else:
            X_fcgr = np.array([]).reshape(X_standard.shape[0], 0)
            self.fcgr_feature_names = []
            self._X_fcgr_for_tda = None

        # TDA now over FCGR representation, not biological space
        if self.use_tda and TDA_AVAILABLE:
            print("\nGenerating TDA topological features (over FCGR space)...")
            # Use PCA-compressed FCGR as input when available; fallback to standard
            X_for_tda = (
                self._X_fcgr_for_tda if self._X_fcgr_for_tda is not None else X_standard
            )
            X_tda = self.tda_extractor.extract_global_tda(X_for_tda)
            n_tda_features = X_tda.shape[1]
            self.tda_feature_names = [f"tda_{i}" for i in range(n_tda_features)]
            print(f"TDA features (expanded): {X_tda.shape[1]}")
        else:
            X_tda = np.array([]).reshape(X_standard.shape[0], 0)
            self.tda_feature_names = []

        all_features = [X_standard, X_fcgr, X_tda]
        X_final = np.hstack([f for f in all_features if f.shape[1] > 0])

        self.all_feature_names = (
            self.feature_names + self.fcgr_feature_names + self.tda_feature_names
        )

        print(f"\nFinal feature space: {X_final.shape[1]} features")
        print(f"  Standard: {len(self.feature_names)}")
        print(f"  FCGR PCA: {len(self.fcgr_feature_names)}")
        print(f"  TDA:      {len(self.tda_feature_names)}")
        return X_final, y

    def transform(self, variant_df, target_col="LABEL_PATHOGENIC"):
        """Transform new data with the fitted preprocessor."""
        df_clean = variant_df.drop(columns=self.leakage_cols, errors="ignore")
        y = df_clean[target_col] if target_col in df_clean.columns else None
        X_tabular = (
            df_clean.drop(target_col, axis=1, errors="ignore")
            if target_col in df_clean.columns
            else df_clean
        )

        num_cols = [
            c
            for c in self.feature_names
            if c in X_tabular.columns and X_tabular[c].dtype != object
        ]
        cat_cols = [
            c
            for c in self.feature_names
            if c in X_tabular.columns and X_tabular[c].dtype == object
        ]

        X_num = (
            X_tabular[num_cols].copy()
            if num_cols
            else pd.DataFrame(index=X_tabular.index)
        )
        X_num = pd.DataFrame(
            self.imputer_num.transform(X_num), columns=num_cols, index=X_tabular.index
        )
        X_cat = (
            X_tabular[cat_cols].copy()
            if cat_cols
            else pd.DataFrame(index=X_tabular.index)
        )
        for col in cat_cols:
            enc = self.label_encoders[col]
            known = set(enc.classes_)
            X_cat[col] = (
                X_cat[col]
                .astype(str)
                .apply(lambda v: v if v in known else enc.classes_[0])
            )
            X_cat[col] = enc.transform(X_cat[col])

        X_combined = pd.concat([X_num, X_cat], axis=1)
        X_standard = self.scaler.transform(X_combined)

        if self.use_fcgr:
            sequences = self._extract_sequence_context(variant_df)
            X_fcgr_raw = self.fcgr_encoder.encode_variants(sequences)
            X_fcgr = (
                self.fcgr_pca.transform(X_fcgr_raw)
                if self.fcgr_pca is not None
                else X_fcgr_raw
            )
        else:
            X_fcgr = np.array([]).reshape(X_standard.shape[0], 0)

        # TDA over FCGR at inference too
        if self.use_tda and TDA_AVAILABLE:
            X_for_tda = X_fcgr if X_fcgr.shape[1] > 0 else X_standard
            X_tda = self.tda_extractor.extract_global_tda(X_for_tda)
        else:
            X_tda = np.array([]).reshape(X_standard.shape[0], 0)

        all_features = [X_standard, X_fcgr, X_tda]
        X_final = np.hstack([f for f in all_features if f.shape[1] > 0])
        return X_final, y


# Supervised metric learner (unchanged)


class SupervisedMetricLearner:
    """Learns discriminative metric space for KNORA neighbor selection"""

    def __init__(self, n_components=15):
        self.n_components = n_components
        self.lda = None
        self.pca = None
        self.mi_weights = None
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X, y):
        print("  Learning supervised metric space...")
        self.mi_weights = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
        self.mi_weights = self.mi_weights / (np.max(self.mi_weights) + 1e-10)
        X_weighted = X * self.mi_weights

        self.lda = LinearDiscriminantAnalysis(n_components=1)
        X_lda = self.lda.fit_transform(X, y)

        n_pca = min(self.n_components - 1, X.shape[1] - 1, X.shape[0] - 1)
        self.pca = PCA(n_components=n_pca, random_state=42)
        X_pca = self.pca.fit_transform(X_weighted)

        X_combined = np.hstack([X_lda, X_pca])
        self.scaler.fit(X_combined)
        self.is_fitted = True
        print(f"  Metric space: {X_combined.shape[1]}D")
        return self

    def transform(self, X):
        X_weighted = X * self.mi_weights
        X_lda = self.lda.transform(X)
        X_pca = self.pca.transform(X_weighted)
        X_combined = np.hstack([X_lda, X_pca])
        return self.scaler.transform(X_combined)


# MultiViewFeatureManager


class MultiViewFeatureManager:
    """Manages multi-view feature representation"""

    def __init__(self, n_standard=18, n_fcgr=20, n_tda=20):
        self.n_standard = n_standard
        self.n_fcgr = n_fcgr
        self.n_tda = n_tda
        self.view_slices = {
            "biological": (0, n_standard),
            "fcgr": (n_standard, n_standard + n_fcgr),
            "tda": (n_standard + n_fcgr, n_standard + n_fcgr + n_tda),
            "combined": (0, n_standard + n_fcgr + n_tda),
        }


# Pairwise Q-statistic diversity analyser


def compute_pairwise_diversity(
    trained_models, feature_indices, X_val, y_val, output_dir
):
    """Compute the pairwise Q-statistic and disagreement matrices.

    Measures inter-learner error correlation on the validation set; a lower
    Q-statistic and higher disagreement indicate a more diverse pool. Results
    are saved as pairwise_q_statistic.csv, pairwise_disagreement.csv, and
    Fig_Diversity_Heatmap.png.

    Returns (q_matrix_df, disagreement_matrix_df).
    """
    names = list(trained_models.keys())
    n = len(names)
    preds = {}
    for name, model in trained_models.items():
        idx = feature_indices[name]
        X_view = X_val[:, idx]
        preds[name] = model.predict(X_view)

    y_arr = np.array(y_val)
    q_matrix = np.zeros((n, n))
    dis_matrix = np.zeros((n, n))

    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            if i == j:
                q_matrix[i, j] = 1.0
                dis_matrix[i, j] = 0.0
                continue
            ci = (preds[ni] == y_arr).astype(int)
            cj = (preds[nj] == y_arr).astype(int)
            N11 = np.sum((ci == 1) & (cj == 1))
            N00 = np.sum((ci == 0) & (cj == 0))
            N10 = np.sum((ci == 1) & (cj == 0))
            N01 = np.sum((ci == 0) & (cj == 1))
            denom = N11 * N00 - N10 * N01
            q_denom = N11 * N00 + N10 * N01
            q_matrix[i, j] = denom / (q_denom + 1e-12)
            dis_matrix[i, j] = (N10 + N01) / (N11 + N00 + N10 + N01 + 1e-12)

    q_df = pd.DataFrame(q_matrix, index=names, columns=names)
    dis_df = pd.DataFrame(dis_matrix, index=names, columns=names)
    q_df.to_csv(output_dir / "pairwise_q_statistic.csv")
    dis_df.to_csv(output_dir / "pairwise_disagreement.csv")

    # Plot heatmap
    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_DOUBLE_COL_WIDTH * 4, FIG_SINGLE_COL_WIDTH * 4)
    )
    sns.heatmap(
        q_df,
        ax=axes[0],
        cmap="RdYlGn_r",
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".1f",
        linewidths=0.1,
        annot_kws={"size": 8},
        cbar_kws={'shrink': 0.8}
    )
    axes[0].set_title(
        "(A) Q-Statistic\n(−1=diverse, +1=correlated)",
        fontweight="bold",
        loc="center",
        fontsize=16,
    )
    plt.setp(
        axes[0].get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=12,
    )
    plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=12)
    sns.heatmap(
        dis_df,
        ax=axes[1],
        cmap="Blues",
        vmin=0,
        vmax=0.3,
        annot=True,
        fmt=".2f",
        linewidths=0.1,
        annot_kws={"size": 8},
        cbar_kws={'shrink': 0.8}
    )
    axes[1].set_title(
        "(B) Pairwise Disagreement Rate\n(higher=more diverse)",
        fontweight="bold",
        loc="center",
        fontsize=16,
    )
    plt.setp(
        axes[1].get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
        fontsize=12,
    )
    plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(
        output_dir / "Fig_Diversity_Heatmap.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.savefig(
        output_dir / "Fig_Diversity_Heatmap.tiff", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close()

    mean_q = np.mean(q_matrix[np.triu_indices(n, k=1)])
    mean_dis = np.mean(dis_matrix[np.triu_indices(n, k=1)])
    print(
        f"\n  [Diversity Analysis] Mean Q-statistic: {mean_q:.4f}  "
        f"(target < 0.50 for diverse pool)"
    )
    print(
        f"  [Diversity Analysis] Mean Disagreement: {mean_dis:.4f}  "
        f"(target > 0.05 for diverse pool)"
    )
    print(
        f"  Saved: pairwise_q_statistic.csv, pairwise_disagreement.csv, "
        f"Fig_Diversity_Heatmap.png"
    )
    return q_df, dis_df


# DiverseFeatureSubspaceFactory


# Random subspace helper (diversity patch)

def _rand_subspace(rng, bio_cols, fcgr_cols, tda_cols,
                   bio_frac, other_frac, force=None):
    """Return a randomized, mostly-disjoint column subset.

    Every member keeps SOME bio (bio carries 91% of signal, so starving it
    entirely just gets the member pruned) but a *different* random subset,
    plus a random subset of FCGR/TDA. `force` builds true single-view
    specialists whose errors are structurally different from the bio pool.
    """
    cols = []
    if force == "fcgr":
        k = max(3, int(round(len(bio_cols) * 0.30)))
        cols += list(rng.choice(bio_cols, size=k, replace=False))
        cols += list(fcgr_cols)
    elif force == "tda":
        k = max(3, int(round(len(bio_cols) * 0.30)))
        cols += list(rng.choice(bio_cols, size=k, replace=False))
        cols += list(tda_cols)
    else:
        k_bio = max(3, int(round(len(bio_cols) * bio_frac)))
        cols += list(rng.choice(bio_cols, size=k_bio, replace=False))
        other = list(fcgr_cols) + list(tda_cols)
        if other:
            k_other = max(1, int(round(len(other) * other_frac)))
            cols += list(rng.choice(other, size=k_other, replace=False))
    return sorted(set(int(c) for c in cols))


class DiverseFeatureSubspaceFactory:

    def __init__(self, n_standard=18, n_fcgr=20, n_tda=20):
        self.view_manager = MultiViewFeatureManager(n_standard, n_fcgr, n_tda)
        self.models = {}
        self.feature_indices = {}
        self._metric_learner = None  # set by train_ensemble() for kNN_SML


    def create_diverse_ensemble(self, random_state=42):
        from sklearn.ensemble import (
            RandomForestClassifier, ExtraTreesClassifier,
            GradientBoostingClassifier, HistGradientBoostingClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.calibration import CalibratedClassifierCV

        n_std = self.view_manager.n_standard
        n_fcgr = self.view_manager.n_fcgr
        n_tda = self.view_manager.n_tda

        bio_cols = list(range(n_std))
        fcgr_cols = list(range(n_std, n_std + n_fcgr))
        tda_cols = list(range(n_std + n_fcgr, n_std + n_fcgr + n_tda))

        bio_frac = getattr(Config, "SUBSPACE_BIO_FRACTION", 0.60)
        other_frac = getattr(Config, "SUBSPACE_OTHER_FRACTION", 0.70)
        replicas = getattr(Config, "POOL_REPLICAS", 2)

        models, feature_indices = {}, {}
        bootstrap_seeds, dirichlet_seeds = {}, {}

        def make_arch(kind, seed):
            if kind == "xgb" and XGBOOST_AVAILABLE:
                return XGBClassifier(n_estimators=180, max_depth=4, learning_rate=0.05,
                                     subsample=0.8, colsample_bytree=0.7,
                                     tree_method='hist', device='cuda',
                                     random_state=seed, n_jobs=-1, verbosity=0)
            if kind == "cat" and CATBOOST_AVAILABLE:
                return CatBoostClassifier(iterations=180, depth=4, learning_rate=0.05,
                                          random_seed=seed, verbose=False)
            if kind == "lgbm" and LGBM_AVAILABLE:
                return LGBMClassifier(n_estimators=160, num_leaves=48, learning_rate=0.05,
                                      feature_fraction=0.7, bagging_fraction=0.8,
                                      device_type='gpu',
                                      bagging_freq=1, random_state=seed, n_jobs=-1,
                                      verbose=-1)
            if kind == "rf":
                base = RandomForestClassifier(n_estimators=180, max_depth=12,
                                              max_features="sqrt", min_samples_leaf=5,
                                              random_state=seed, n_jobs=-1)
                return CalibratedClassifierCV(base, method="isotonic", cv=3)
            if kind == "et":
                base = ExtraTreesClassifier(n_estimators=180, max_depth=12,
                                            max_features="sqrt", random_state=seed,
                                            n_jobs=-1)
                return CalibratedClassifierCV(base, method="isotonic", cv=3)
            if kind == "gb":
                base = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                  learning_rate=0.05, subsample=0.8,
                                                  max_features="sqrt", random_state=seed)
                return CalibratedClassifierCV(base, method="isotonic", cv=3)
            if kind == "hgb":
                return HistGradientBoostingClassifier(max_iter=150, max_depth=6,
                                                      learning_rate=0.05,
                                                      random_state=seed)
            if kind == "mlp":
                base = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                                     solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                                     max_iter=300, early_stopping=True,
                                     validation_fraction=0.1, random_state=seed)
                return CalibratedClassifierCV(base, method="isotonic", cv=3)
            if kind == "enet":
                return LogisticRegression(penalty="elasticnet", solver="saga",
                                          l1_ratio=0.5, C=0.5, max_iter=3000,
                                          random_state=seed, n_jobs=-1)
            return None

        # Recipe = (kind, subspace-mode). replicated `replicas` times with
        # different seeds/subspaces to form a bagging pool.
        recipes = [
            ("xgb",  None), ("cat",  None), ("lgbm", None),
            ("rf",   None), ("et",   None), ("hgb",  None),
            ("mlp",  None), ("gb",   None),
            ("xgb",  "fcgr"), ("hgb", "tda"), ("enet", None),
        ]

        counter = 0
        for rep in range(replicas):
            for kind, force in recipes:
                seed = random_state + 101 * rep + 7 * counter
                est = make_arch(kind, seed)
                if est is None:
                    continue
                rng = np.random.RandomState(seed)
                idx = _rand_subspace(rng, bio_cols, fcgr_cols, tda_cols,
                                     bio_frac, other_frac, force=force)
                name = f"{kind}_{('all' if force is None else force)}_r{rep}_{counter}"
                models[name] = est
                feature_indices[name] = idx
                bootstrap_seeds[name] = seed + 1
                dirichlet_seeds[name] = seed + 2
                counter += 1

        # One instance-based specialist in the SML metric space (kept full-view,
        # no row bootstrap -- kNN is already high-variance / instance-based).
        models["kNN_SML"] = KNeighborsClassifier(n_neighbors=15, metric="euclidean",
                                                 n_jobs=-1)
        feature_indices["kNN_SML"] = list(range(n_std + n_fcgr + n_tda))
        dirichlet_seeds["kNN_SML"] = random_state + 999

        self.models = models
        self.feature_indices = feature_indices
        self.bootstrap_seeds = bootstrap_seeds
        self.dirichlet_seeds = dirichlet_seeds
        self.no_bootstrap = {"kNN_SML"}
        print(f"\nCreated {len(models)} random-subspace/bagging base models "
              f"(replicas={replicas}, +FCGR/TDA specialists, +kNN_SML)")
        return models, feature_indices

    def train_ensemble(self, X_train, y_train, X_val=None, y_val=None,
                       consensus_scores=None):
        from sklearn.model_selection import StratifiedKFold, train_test_split

        trained_models = {}
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.15, stratify=y_train,
                random_state=42)

        # NOTE: consensus-margin DOWN-weighting is intentionally NOT used here.
        # With AMBIGUOUS_SAMPLE_WEIGHT == 1.0 it is a no-op; we instead give each
        # model its own Dirichlet (Bayesian-bootstrap) reweighting below, so
        # members emphasize DIFFERENT samples rather than all ignoring the same
        # ambiguous ones. (Old down-weighting shown to peg Q -> ~1.)
        base_weight = np.ones(len(y_train))
        if consensus_scores is not None and len(np.asarray(consensus_scores)) == len(y_train):
            cs = np.asarray(consensus_scores)
            aw = getattr(Config, "AMBIGUOUS_SAMPLE_WEIGHT", 1.0)
            if aw != 1.0:  # preserved for backward-compat if ever re-enabled
                base_weight = np.where(np.abs(cs - 0.5) < Config.AMBIGUOUS_MARGIN,
                                       aw, 1.0)

        print(f"\nTraining {len(self.models)} random-subspace/bagging models (OOF)...")
        n_classifiers = len(self.models)
        n_train = len(y_train)
        oof_probas = np.zeros((n_train, n_classifiers))
        skf = StratifiedKFold(n_splits=5, shuffle=True,
                              random_state=Config.RANDOM_STATE)

        sml = SupervisedMetricLearner(n_components=15)
        sml.fit(X_val, y_val)
        self._metric_learner = sml
        X_train_sml = sml.transform(X_train)
        X_val_sml = sml.transform(X_val)

        boot_frac = getattr(Config, "BOOTSTRAP_FRACTION", 0.75)
        dir_alpha = getattr(Config, "DIRICHLET_ALPHA", 0.60)



        def boot_rows(name, n):
            if name in getattr(self, "no_bootstrap", set()):
                return np.arange(n)
            seed = self.bootstrap_seeds.get(name, 0)
            m = max(1, int(round(boot_frac * n)))
            return np.random.RandomState(seed).choice(n, size=m, replace=True)

        import time
        for clf_idx, (name, model) in enumerate(self.models.items()):
            start_time = time.time()
            print(f"    -> Training {clf_idx+1}/{n_classifiers}: {name} (OOF 5-fold CV) ...")
            is_knn_sml = (name == "kNN_SML")
            indices = self.feature_indices[name]
            X_view = X_train_sml if is_knn_sml else X_train[:, indices]

            try:
                # ----- OOF: bootstrap + Dirichlet applied WITHIN each fold-train
                fold_oof = np.zeros(n_train)
                for tr_idx, va_idx in skf.split(X_view, y_train):
                    fm = copy.deepcopy(model)
                    # bootstrap the fold-train rows (indices are positions in tr_idx)
                    if name in getattr(self, "no_bootstrap", set()):
                        br = np.arange(len(tr_idx))
                    else:
                        seed = self.bootstrap_seeds.get(name, 0) + 13
                        m = max(1, int(round(boot_frac * len(tr_idx))))
                        br = np.random.RandomState(seed).choice(len(tr_idx), size=m,
                                                                replace=True)
                    rows = tr_idx[br]
                    # Dirichlet drawn on the *resampled* block:
                    w = (np.random.RandomState(self.dirichlet_seeds.get(name, 0) + 1)
                         .dirichlet(np.full(len(rows), dir_alpha)) * len(rows)) \
                        * base_weight[rows]
                    Xf, yf = X_view[rows], y_train[rows]
                    try:
                        fm.fit(Xf, yf, sample_weight=w)
                    except TypeError:
                        fm.fit(Xf, yf)
                    if hasattr(fm, "predict_proba"):
                        p = fm.predict_proba(X_view[va_idx])
                        fold_oof[va_idx] = p[:, 1] if p.shape[1] == 2 else p[:, 0]
                    else:
                        fold_oof[va_idx] = fm.predict(X_view[va_idx]).astype(float)
                oof_probas[:, clf_idx] = fold_oof

                # ----- final fit on bootstrap + Dirichlet of the full train view
                rows = boot_rows(name, n_train)
                w = (np.random.RandomState(self.dirichlet_seeds.get(name, 0))
                     .dirichlet(np.full(len(rows), dir_alpha)) * len(rows)) \
                    * base_weight[rows]
                try:
                    model.fit(X_view[rows], y_train[rows], sample_weight=w)
                except TypeError:
                    model.fit(X_view[rows], y_train[rows])
                trained_models[name] = model
                
                elapsed = time.time() - start_time
                print(f"       [Done] {name} trained in {elapsed:.1f}s.")
            except Exception as e:
                print(f"  Failed: {name} - {e}")
                oof_probas[:, clf_idx] = 0.5

        self._X_val_sml, self._X_train_sml = X_val_sml, X_train_sml
        self.models = trained_models
        self.oof_probas = oof_probas
        self.oof_y_train = y_train

        print("\n[Diversity Analysis] Computing pairwise Q-statistic matrix...")
        try:
            tmp_models, tmp_indices = {}, {}
            for nm, mdl in trained_models.items():
                if nm == "kNN_SML":
                    tmp_models[nm] = _SMLWrappedKNN(mdl, sml, n_total=X_val.shape[1])
                else:
                    tmp_models[nm] = mdl
                tmp_indices[nm] = self.feature_indices[nm]
            self.q_df, self.dis_df = compute_pairwise_diversity(
                tmp_models, tmp_indices, X_val, y_val, Config.OUTPUT_DIR)
        except Exception as e:
            print(f"  Diversity analysis failed: {e}")
            self.q_df = self.dis_df = None

        return trained_models, X_val, y_val



# Helper: SML-wrapped kNN for diversity matrix computation


class _SMLWrappedKNN:
    """Thin wrapper so kNN_SML's predict() can be called with the full X."""

    def __init__(self, knn_model, sml, n_total):
        self.knn = knn_model
        self.sml = sml
        self.n_total = n_total

    def predict(self, X):
        return self.knn.predict(self.sml.transform(X))

    def predict_proba(self, X):
        return self.knn.predict_proba(self.sml.transform(X))


# FeatureSubspaceWrapper


class FeatureSubspaceWrapper:
    """Wrapper for models with feature subspace.
    For kNN_SML, the feature_indices sentinel is the string 'sml'; the wrapper
    uses the stored SML projector instead of column slicing.
    """

    def __init__(self, model, feature_indices, sml=None, is_sml_model=False):
        self.model = model
        self.feature_indices = feature_indices
        self.sml = sml
        self.is_sml_model = is_sml_model

    def _project(self, X):
        if self.is_sml_model and self.sml is not None:
            return self.sml.transform(X)
        return X[:, self.feature_indices]

    def predict(self, X):
        return self.model.predict(self._project(X))

    def predict_proba(self, X):
        X_sub = self._project(X)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_sub)
        pred = self.model.predict(X_sub)
        return np.column_stack([1 - pred, pred])


# EnhancedKNORAEnsemble


class EnhancedKNORAEnsemble:
    def __init__(
        self,
        base_classifiers,
        k=11,
        min_competence=0.7,
        use_metric_learning=True,
        oof_probas=None,
        oof_y_train=None,
    ):
        self.base_classifiers = base_classifiers
        self.k = k
        self.min_competence = min_competence
        self.use_metric_learning = use_metric_learning
        self.X_val = None
        self.y_val = None
        self.X_val_metric = None
        self.oracle_matrix = None
        self.nn_model = None
        self.metric_learner = None
        self.classifier_names = list(base_classifiers.keys())

        self.oof_probas = oof_probas
        self.oof_y_train = oof_y_train

        self.meta_model = None
        self.stacking_weight = 0.7

        self.decision_threshold = 0.5
        self.best_base_mcc = -1.0
        self.best_base_name = None

        self.active_classifier_names = list(base_classifiers.keys())

    def fit(self, X_val, y_val):
        print("\nBuilding Enhanced KNORA-E (All Fixes + Changes Applied)")

        self.X_val = X_val
        self.y_val = np.array(y_val)
        n_samples = len(y_val)
        n_classifiers = len(self.base_classifiers)

        print("\n[Step 1/4] Computing per-classifier validation accuracy...")
        val_accuracies = {}
        for name, clf in self.base_classifiers.items():
            preds = clf.predict(X_val)
            val_accuracies[name] = float(np.mean(preds == self.y_val))
            print(f"  {name:40} - Val Acc: {val_accuracies[name]:.4f}")

        best_val_acc = max(val_accuracies.values())
        prune_threshold = best_val_acc - Config.WEAK_LEARNER_MARGIN
        self.active_classifier_names = [
            name for name, acc in val_accuracies.items() if acc >= prune_threshold
        ]
        pruned_names = [
            name
            for name in self.base_classifiers
            if name not in self.active_classifier_names
        ]
        if pruned_names:
            print(f"\n  Pruned {len(pruned_names)} weak learner(s): {pruned_names}")
        print(f"  Active pool: {self.active_classifier_names}")

        active_classifiers = {
            name: clf
            for name, clf in self.base_classifiers.items()
            if name in self.active_classifier_names
        }
        n_active = len(active_classifiers)

        best_base_mcc_val = -1.0
        for name, clf in active_classifiers.items():
            preds = clf.predict(X_val)
            mcc_val = matthews_corrcoef(self.y_val, preds)
            if mcc_val > best_base_mcc_val:
                best_base_mcc_val = mcc_val
                self.best_base_name = name
        self.best_base_mcc = best_base_mcc_val
        # Store the actual model object, not just its name
        self.best_base_model = active_classifiers[self.best_base_name]
        print(
            f"\n  Best base learner: {self.best_base_name} "
            f"(Val MCC={self.best_base_mcc:.4f})"
        )

        if self.use_metric_learning:
            print("\n[Step 2/4] Supervised Metric Learning...")
            self.metric_learner = SupervisedMetricLearner(n_components=15)
            self.metric_learner.fit(X_val, y_val)
            self.X_val_metric = self.metric_learner.transform(X_val)

            print(f"\n[Step 3/4] Building k-NN index (k={self.k})...")
            self.nn_model = NearestNeighbors(
                n_neighbors=self.k, metric="euclidean", n_jobs=-1
            )
            self.nn_model.fit(self.X_val_metric)

        print(f"\n[Step 4/4] Computing oracle matrix ({n_samples} x {n_active})")
        self.oracle_matrix = np.zeros((n_samples, n_active), dtype=int)
        self._active_clf_list = list(active_classifiers.items())

        for idx, (name, clf) in enumerate(self._active_clf_list):
            predictions = clf.predict(X_val)
            self.oracle_matrix[:, idx] = (predictions == self.y_val).astype(int)

        self._fit_stacking_meta_model(X_val, active_classifiers)

    def _fit_stacking_meta_model(self, X_val, active_classifiers):
        n_active = len(active_classifiers)
        active_names = list(active_classifiers.keys())

        if self.oof_probas is not None and self.oof_y_train is not None:
            print("\nStacking: fitting meta-model on OOF probabilities...")
            all_names = list(self.base_classifiers.keys())
            active_col_indices = [
                all_names.index(name) for name in active_names if name in all_names
            ]
            meta_train_X = self.oof_probas[:, active_col_indices]
            meta_train_y = self.oof_y_train
        else:
            print("\n[Stacking] Fitting meta-model on validation probabilities...")
            meta_train_X = np.zeros((len(self.y_val), n_active))
            for idx, (name, clf) in enumerate(active_classifiers.items()):
                if hasattr(clf, "predict_proba"):
                    try:
                        p = clf.predict_proba(X_val)
                        meta_train_X[:, idx] = p[:, 1] if p.shape[1] == 2 else p[:, 0]
                    except Exception:
                        meta_train_X[:, idx] = clf.predict(X_val).astype(float)
                else:
                    meta_train_X[:, idx] = clf.predict(X_val).astype(float)
            meta_train_y = self.y_val

        self.meta_model = LogisticRegression(
            C=1.0, max_iter=2000, random_state=Config.RANDOM_STATE
        )
        self.meta_model.fit(meta_train_X, meta_train_y)

        val_active_preds = self._compute_active_preds(X_val)
        fused_val_proba = self._fuse_predictions(X_val, val_active_preds)

        # constrained threshold search
        best_mcc = -1.0
        best_thr = 0.5
        thresholds = np.linspace(
            Config.THRESHOLD_SEARCH_LOW, Config.THRESHOLD_SEARCH_HIGH, 41
        )
        for thr in thresholds:
            preds = (fused_val_proba >= thr).astype(int)
            mcc = matthews_corrcoef(self.y_val, preds)
            if mcc > best_mcc:
                best_mcc = mcc
                best_thr = thr

        self.decision_threshold = best_thr

        # compare the true fused ensemble vs best base on validation
        ensemble_val_mcc = best_mcc
        if ensemble_val_mcc < self.best_base_mcc:
            print(
                f"\n  WARNING: Ensemble val MCC ({ensemble_val_mcc:.4f}) < "
                f"best base MCC ({self.best_base_mcc:.4f}).  "
                f"Fallback to '{self.best_base_name}' activated."
            )
            self._use_fallback = True
        else:
            self._use_fallback = False

        print(
            f"  Meta-model trained. Threshold={best_thr:.2f} "
            f"(val MCC={best_mcc:.4f})  |  fallback={self._use_fallback}"
        )

    def _compute_active_preds(self, X):
        """Positive-class probabilities from every active base classifier.

        Returns an (n_samples, n_active) matrix whose columns follow the order
        of ``self._active_clf_list`` (the same order the meta-model was trained
        on).
        """
        n_samples = X.shape[0]
        n_active = len(self._active_clf_list)
        active_preds = np.zeros((n_samples, n_active))
        for idx, (name, clf) in enumerate(self._active_clf_list):
            if hasattr(clf, "predict_proba"):
                try:
                    p = clf.predict_proba(X)
                    active_preds[:, idx] = p[:, 1] if p.shape[1] == 2 else p[:, 0]
                except Exception:
                    active_preds[:, idx] = clf.predict(X).astype(float)
            else:
                active_preds[:, idx] = clf.predict(X).astype(float)
        return active_preds

    def _fuse_predictions(self, X, active_preds):
        n_samples = X.shape[0]
        n_active = len(self._active_clf_list)

        # Meta-model prediction
        if self.meta_model is not None:
            meta_proba = self.meta_model.predict_proba(active_preds)[:, 1]
        else:
            meta_proba = np.mean(active_preds, axis=1)

        # KNORA competence-weighted vote
        if self.use_metric_learning and self.metric_learner is not None:
            X_metric = self.metric_learner.transform(X)
        else:
            X_metric = X[:, :18] if X.shape[1] > 18 else X

        distances, indices = self.nn_model.kneighbors(X_metric)

        knora_predictions = np.zeros(n_samples)
        sigma = np.mean(distances) + 1e-10
        distance_weights = np.exp(-(distances**2) / (2 * sigma**2))
        distance_weights = distance_weights / (
            distance_weights.sum(axis=1, keepdims=True) + 1e-10
        )

        for i in range(n_samples):
            neighbor_indices = indices[i]
            sample_weights = distance_weights[i]
            if np.sum(sample_weights) < 1e-10:
                sample_weights = np.ones_like(sample_weights) / len(sample_weights)

            neighbor_oracles = self.oracle_matrix[neighbor_indices]
            competence_scores = np.average(
                neighbor_oracles, axis=0, weights=sample_weights
            )
            sample_preds = active_preds[i]

            relative_threshold = max(
                self.min_competence, np.percentile(competence_scores, 40)
            )
            selected_mask = competence_scores >= relative_threshold

            if not np.any(selected_mask):
                top_k_idx = np.argsort(competence_scores)[-max(1, n_active // 2) :]
                selected_mask = np.zeros(n_active, dtype=bool)
                selected_mask[top_k_idx] = True

            sel_preds = sample_preds[selected_mask]
            sel_comps = competence_scores[selected_mask]

            temp = 2.0
            exp_comps = np.exp(sel_comps / temp)
            weights = exp_comps / (np.sum(exp_comps) + 1e-10)
            knora_predictions[i] = np.sum(weights * sel_preds)

        return (
            self.stacking_weight * meta_proba
            + (1.0 - self.stacking_weight) * knora_predictions
        )

    def predict_proba(self, X):
        """Return the fused probability for the positive class.

        When the best-base fallback is active this returns
        ``active_preds[:, best_idx]``. ``run_final_model`` passes the model
        object (not its name) to the SHAP explainer.
        """
        n_samples = X.shape[0]
        n_active = len(self._active_clf_list)

        print(
            f"\nKNORA-E Inference ({n_samples} samples, {n_active} active classifiers)..."
        )

        active_preds = self._compute_active_preds(X)

        # fallback to best base model object (not name string)
        if getattr(self, "_use_fallback", False):
            best_idx = next(
                i
                for i, (name, _) in enumerate(self._active_clf_list)
                if name == self.best_base_name
            )
            print(f"  Using fallback: {self.best_base_name}")
            return active_preds[:, best_idx]

        return self._fuse_predictions(X, active_preds)

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self.decision_threshold
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    # Selective prediction
    def selective_predict(self, X, threshold=None, abstain_threshold=None):
        if threshold is None:
            threshold = self.decision_threshold
        if abstain_threshold is None:
            abstain_threshold = Config.SELECTIVE_ABSTAIN_THRESHOLD

        proba = self.predict_proba(X)
        y_hard = (proba >= threshold).astype(int)

        if self.use_metric_learning and self.metric_learner is not None:
            X_metric = self.metric_learner.transform(X)
        else:
            X_metric = X[:, :18] if X.shape[1] > 18 else X

        distances, indices = self.nn_model.kneighbors(X_metric)

        purity_scores = np.zeros(len(X))
        for i in range(len(X)):
            neighbour_labels = self.y_val[indices[i]]
            pred_label = y_hard[i]
            purity_scores[i] = np.mean(neighbour_labels == pred_label)

        y_selective = np.where(purity_scores >= abstain_threshold, y_hard, -1).astype(
            int
        )

        coverage = float(np.mean(y_selective >= 0))
        n_abstain = int(np.sum(y_selective == -1))
        print(
            f"\n  [Selective Prediction] "
            f"Coverage: {coverage:.3%}  Abstentions: {n_abstain:,}"
        )

        return y_selective, purity_scores, coverage


# EnhancedTFDFEEnsemble


class EnhancedTFDFEEnsemble:
    """Top-level TF-DFE ensemble combining all base learners and the stacker."""

    def __init__(self, n_standard=18, n_fcgr=20, n_tda=20):
        self.n_standard = n_standard
        self.n_fcgr = n_fcgr
        self.n_tda = n_tda
        self.diverse_factory = None
        self.enhanced_knora = None
        self.is_fitted = False

    def fit(
        self, X_train, y_train, X_val=None, y_val=None, consensus_scores_train=None
    ):
        """Fit the full ensemble.

        consensus_scores_train : array-like, shape (n_train,), optional
            CONSENSUS_SCORE values for training samples, used for
            consensus-margin sample weighting. Pass None to disable.
        """
        print("\nEnhanced TF-DFE Training")

        n_features = X_train.shape[1]
        expected = self.n_standard + self.n_fcgr + self.n_tda

        if n_features != expected:
            print(f"Adjusting feature counts: {n_features} features")
            self.n_fcgr = max(0, n_features - self.n_standard - self.n_tda)
            if self.n_fcgr < 0:
                self.n_fcgr = 0
                self.n_tda = max(0, n_features - self.n_standard)

        if X_val is None or y_val is None:
            # FIX: split consensus_scores_train ALONGSIDE X_train/y_train so it
            # stays row-aligned with the (reduced) training set. Previously the
            # inner split shrank y_train but left consensus_scores_train at the
            # full length, causing "consensus_scores length mismatch" and the
            # sample weights being silently dropped.
            if (
                consensus_scores_train is not None
                and len(np.asarray(consensus_scores_train)) == len(y_train)
            ):
                cs_arr = np.asarray(consensus_scores_train)
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                    consensus_scores_train,
                    _cs_val,
                ) = train_test_split(
                    X_train,
                    y_train,
                    cs_arr,
                    test_size=0.15,
                    stratify=y_train,
                    random_state=Config.RANDOM_STATE,
                )
            else:
                if consensus_scores_train is not None:
                    print(
                        "  WARNING: consensus_scores_train length "
                        f"({len(np.asarray(consensus_scores_train))}) != "
                        f"X_train length ({len(y_train)}) before inner split; "
                        "disabling sample weights."
                    )
                    consensus_scores_train = None
                X_train, X_val, y_train, y_val = train_test_split(
                    X_train,
                    y_train,
                    test_size=0.15,
                    stratify=y_train,
                    random_state=Config.RANDOM_STATE,
                )

        self.X_val = X_val
        self.y_val = np.array(y_val)
        y_train_arr = np.array(y_train)

        self.diverse_factory = DiverseFeatureSubspaceFactory(
            n_standard=self.n_standard, n_fcgr=self.n_fcgr, n_tda=self.n_tda
        )
        self.diverse_factory.create_diverse_ensemble()

        # pass consensus scores through
        trained_models, _, _ = self.diverse_factory.train_ensemble(
            X_train,
            y_train_arr,
            X_val,
            y_val,
            consensus_scores=consensus_scores_train,
        )

        # Wrap models —: kNN_SML gets the SML projector
        sml = self.diverse_factory._metric_learner
        wrapped_models = {}
        for name, model in trained_models.items():
            is_sml = name == "kNN_SML"
            wrapped_models[name] = FeatureSubspaceWrapper(
                model,
                self.diverse_factory.feature_indices[name],
                sml=sml if is_sml else None,
                is_sml_model=is_sml,
            )

        self.enhanced_knora = EnhancedKNORAEnsemble(
            base_classifiers=wrapped_models,
            k=Config.KNORA_K,
            min_competence=Config.MIN_COMPETENCE_THRESHOLD,
            use_metric_learning=True,
            oof_probas=self.diverse_factory.oof_probas,
            oof_y_train=self.diverse_factory.oof_y_train,
        )
        self.enhanced_knora.fit(X_val, self.y_val)

        self.is_fitted = True
        print("\nTraining complete")

    def predict_proba(self, X):
        return self.enhanced_knora.predict_proba(X)

    def _optimized_threshold(self):
        return getattr(self.enhanced_knora, "decision_threshold", 0.5)

    def predict(self, X, threshold=None):
        if threshold is None:
            threshold = self._optimized_threshold()
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    def selective_predict(self, X, threshold=None, abstain_threshold=None):
        """Selective prediction with abstention."""
        return self.enhanced_knora.selective_predict(
            X, threshold=threshold, abstain_threshold=abstain_threshold
        )


# Plotting helpers (unchanged from )


def plot_class_distribution(variant_df, output_dir):
    print("  Generating class distribution plot...")
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.7))
    class_counts = variant_df["LABEL_PATHOGENIC"].value_counts()
    colors = ["#0077BB", "#CC3311"]
    bars = ax.bar(
        ["Benign", "Pathogenic"],
        [class_counts.get(0, 0), class_counts.get(1, 0)],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_ylabel("Count")
    ax.set_title("(A) Class Distribution", fontweight="bold", loc="left")
    for bar, count in zip(bars, [class_counts.get(0, 0), class_counts.get(1, 0)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=FIG_FONT_SIZE,
        )
    plt.tight_layout()
    plt.savefig(
        output_dir / "Fig1_class_distribution.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.savefig(
        output_dir / "Fig1_class_distribution.tiff", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close()


def plot_roc_pr_curves(y_true, y_pred_proba, model_name, output_dir):
    print("  Generating ROC and PR curves...")
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(FIG_DOUBLE_COL_WIDTH, FIG_DOUBLE_COL_WIDTH / 2.5)
    )
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auroc = roc_auc_score(y_true, y_pred_proba)
    ax1.plot(
        fpr,
        tpr,
        linewidth=1.5,
        color="#0077BB",
        label=f"{model_name} (AUC = {auroc:.3f})",
    )
    ax1.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.7, label="Random")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("(A) ROC Curve", fontweight="bold", loc="left")
    ax1.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    ax1.set_xlim([-0.01, 1.01])
    ax1.set_ylim([-0.01, 1.01])
    ax1.grid(False)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)
    ax2.plot(
        recall,
        precision,
        linewidth=1.5,
        color="#0077BB",
        label=f"{model_name} (AP = {auprc:.3f})",
    )
    baseline = np.mean(y_true)
    ax2.axhline(
        baseline,
        color="gray",
        linestyle="--",
        linewidth=0.8,
        alpha=0.7,
        label=f"Baseline ({baseline:.3f})",
    )
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("(B) Precision-Recall Curve", fontweight="bold", loc="left")
    ax2.legend(loc="lower left", frameon=True, fancybox=False, edgecolor="black")
    ax2.set_xlim([-0.01, 1.01])
    ax2.set_ylim([-0.01, 1.01])
    ax2.grid(False)
    plt.tight_layout()
    plt.savefig(output_dir / "Fig1_ROC_PR_curves.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(
        output_dir / "Fig1_ROC_PR_curves.tiff", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close()


def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    print("  Generating confusion matrices...")
    cm = confusion_matrix(y_true, y_pred)
    classes = ["Benign", "Pathogenic"]
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.9))
    row_sums = cm.sum(axis=1)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_display = cm.astype("float") / row_sums[:, np.newaxis]
    im = ax.imshow(cm_display, interpolation="nearest", cmap="Blues")
    ax.grid(False)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Proportion", rotation=-90, va="bottom")
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    ax.set_title(f"{model_name} Confusion Matrix", fontweight="bold", pad=10)
    thresh = cm_display.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm_display[i, j]:.1%}\n({cm[i, j]:,})"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color="white" if cm_display[i, j] > thresh else "black",
                fontsize=FIG_FONT_SIZE,
            )
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    plt.savefig(
        output_dir / f"Fig_CM_{safe_name}.png", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.savefig(
        output_dir / f"Fig_CM_{safe_name}.tiff", dpi=FIG_DPI, bbox_inches="tight"
    )
    plt.close()


def plot_tsne(X, y, title, output_dir):
    print("  Generating t-SNE visualization...")
    np.random.seed(Config.RANDOM_STATE)
    idx = np.random.choice(len(X), size=min(2000, len(X)), replace=False)
    X_sub = X[idx]
    y_sub = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
    tsne = TSNE(
        n_components=2, random_state=Config.RANDOM_STATE, perplexity=30, n_iter=1000
    )
    X_embedded = tsne.fit_transform(X_sub)
    fig, ax = plt.subplots(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.9))
    colors = {0: "#0077BB", 1: "#CC3311"}
    # Inside plot_tsne function:
    for label, name in [(0, "Benign"), (1, "Pathogenic")]:
        # Ensure y_sub is a numpy array so the boolean mask index matches X_embedded
        y_arr = np.array(y_sub)
        mask = y_arr == label

        ax.scatter(
            X_embedded[mask, 0],
            X_embedded[mask, 1],
            c=colors[label],
            label=name,
            alpha=0.6,
            s=15,
            edgecolors="none",
        )
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title(title, fontweight="bold")
    ax.legend(
        title="Class", loc="best", frameon=True, fancybox=False, edgecolor="black"
    )
    plt.tight_layout()
    plt.savefig(output_dir / "Fig_tSNE_TFDFE.png", dpi=FIG_DPI, bbox_inches="tight")
    plt.savefig(output_dir / "Fig_tSNE_TFDFE.tiff", dpi=FIG_DPI, bbox_inches="tight")
    plt.close()


def _unwrap_calibrated_estimator(m):
    if not isinstance(m, CalibratedClassifierCV):
        return m
    # Prefer a fitted underlying estimator (has estimators_/tree structure).
    fitted = getattr(m, "calibrated_classifiers_", None)
    if fitted:
        for attr in ("estimator", "base_estimator"):
            est = getattr(fitted[0], attr, None)
            if est is not None and not isinstance(est, str):
                return est
    # Fall back to the (possibly unfitted) template estimator parameter.
    for attr in ("estimator", "base_estimator"):
        est = getattr(m, attr, None)
        if est is not None and not isinstance(est, str):
            return est
    return m


def perform_shap_analysis(
    model, X_sample, feature_names, output_dir, model_name="TF-DFE"
):
    if not SHAP_AVAILABLE:
        print("  SHAP analysis skipped (library not installed)")
        return None
    print("  Running SHAP analysis...")
    try:
        if hasattr(model, "estimators_"):
            explainer = shap.TreeExplainer(model)
        else:
            print("    Using KernelExplainer (slower)...")
            background = shap.sample(X_sample, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
        short_names = [
            name[:25] + "..." if len(name) > 28 else name for name in feature_names
        ]
        fig, ax = plt.subplots(
            figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 1.2)
        )
        shap.summary_plot(
            shap_values,
            X_sample,
            feature_names=short_names,
            plot_type="bar",
            show=False,
            max_display=20,
            color="#0077BB",
        )
        plt.title(f"Feature Importance ({model_name})", fontweight="bold", loc="left")
        plt.xlabel("Mean |SHAP Value|")
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_SHAP_summary.png", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_SHAP_summary.tiff", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.close()
        fig, ax = plt.subplots(
            figsize=(FIG_DOUBLE_COL_WIDTH * 0.7, FIG_SINGLE_COL_WIDTH * 1.2)
        )
        shap.summary_plot(
            shap_values, X_sample, feature_names=short_names, show=False, max_display=20
        )
        plt.title("SHAP Value Distribution", fontweight="bold", loc="left")
        plt.xlabel("SHAP Value (Impact on Pathogenicity)")
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_SHAP_beeswarm.png", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_SHAP_beeswarm.tiff", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.close()
        print("  SHAP analysis complete")
        return shap_values
    except Exception as e:
        print(f"  SHAP analysis failed: {e}")
        return None


# TFDFEvaluator


class TFDFEvaluator:
    @staticmethod
    def calculate_ece(y_true, y_pred_proba, n_bins=10):
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        bin_data = []
        for bl, bu in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            in_bin = (y_pred_proba > bl) & (y_pred_proba <= bu)
            prop = np.mean(in_bin)
            if prop > 0:
                avg_conf = np.mean(y_pred_proba[in_bin])
                avg_acc = np.mean(y_true[in_bin])
                ece += np.abs(avg_acc - avg_conf) * prop
                bin_data.append(
                    {
                        "bin_lower": bl,
                        "bin_upper": bu,
                        "avg_confidence": avg_conf,
                        "avg_accuracy": avg_acc,
                        "count": int(np.sum(in_bin)),
                        "proportion": prop,
                        "calibration_error": np.abs(avg_acc - avg_conf),
                    }
                )
            else:
                bin_data.append(
                    {
                        "bin_lower": bl,
                        "bin_upper": bu,
                        "avg_confidence": np.nan,
                        "avg_accuracy": np.nan,
                        "count": 0,
                        "proportion": 0,
                        "calibration_error": np.nan,
                    }
                )
        return ece, pd.DataFrame(bin_data)

    @staticmethod
    def plot_calibration_curve(
        y_true, y_pred_proba, output_dir, model_name="TF-DFE", n_bins=10
    ):
        print("  Generating calibration curve...")
        prob_true, prob_pred = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins, strategy="uniform"
        )
        ece, bin_data = TFDFEvaluator.calculate_ece(
            np.array(y_true), np.array(y_pred_proba), n_bins
        )
        brier = brier_score_loss(y_true, y_pred_proba)
        fig = plt.figure(figsize=(FIG_SINGLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 1.2))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect Calibration")
        ax1.plot(
            prob_pred,
            prob_true,
            "s-",
            color="#0077BB",
            linewidth=1.5,
            markersize=6,
            label=f"{model_name}",
        )
        ax1.fill_between(prob_pred, prob_pred, prob_true, alpha=0.2, color="#CC3311")
        ax1.set_ylabel("Observed Probability")
        ax1.set_xlim([-0.02, 1.02])
        ax1.set_ylim([-0.02, 1.02])
        ax1.set_title("Calibration Plot", fontweight="bold", loc="left")
        ax1.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
        ax1.grid(False)
        textstr = f"ECE = {ece:.4f}\nBrier = {brier:.4f}"
        props = dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9)
        ax1.text(
            0.95,
            0.05,
            textstr,
            transform=ax1.transAxes,
            fontsize=FIG_FONT_SIZE,
            va="bottom",
            ha="right",
            bbox=props,
        )
        ax1.set_xticklabels([])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        ax2.hist(
            y_pred_proba,
            bins=n_bins,
            range=(0, 1),
            color="#0077BB",
            edgecolor="black",
            alpha=0.7,
            linewidth=0.5,
        )
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Count")
        ax2.set_xlim([-0.02, 1.02])
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_Calibration_Curve.png", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_Calibration_Curve.tiff", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.close()
        bin_data.to_csv(output_dir / "calibration_bins.csv", index=False)
        print(f"    ECE: {ece:.6f}")
        print(f"    Brier Score: {brier:.6f}")
        print(f"    Saved: Fig_Calibration_Curve.png, calibration_bins.csv")
        return ece, brier, bin_data

    @staticmethod
    def perform_repeated_cv(
        X,
        y,
        model_factory_fn,
        n_splits=5,
        n_repeats=2,
        n_standard=10,
        n_fcgr=20,
        n_tda=20,
        output_dir=None,
    ):
        print(f"\n  Running {n_repeats}x{n_splits}-fold Repeated Stratified CV...")
        rskf = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=Config.RANDOM_STATE
        )
        cv_results = []
        fold_idx = 0
        y_arr = np.array(y) if hasattr(y, "values") else y
        for train_idx, test_idx in tqdm(
            rskf.split(X, y_arr),
            total=n_splits * n_repeats,
            desc="CV Progress",
            ncols=80,
        ):
            fold_idx += 1
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y_arr[train_idx], y_arr[test_idx]
            model = model_factory_fn(n_standard, n_fcgr, n_tda)
            model.fit(X_tr, y_tr)
            y_pred_cv = model.predict(X_te)
            y_pred_proba_cv = model.predict_proba(X_te)
            fold_metrics = {
                "repeat": (fold_idx - 1) // n_splits + 1,
                "fold": (fold_idx - 1) % n_splits + 1,
                "mcc": matthews_corrcoef(y_te, y_pred_cv),
                "auprc": average_precision_score(y_te, y_pred_proba_cv),
                "auroc": roc_auc_score(y_te, y_pred_proba_cv),
                "f1": f1_score(y_te, y_pred_cv, zero_division=0),
                "precision": precision_score(y_te, y_pred_cv, zero_division=0),
                "recall": recall_score(y_te, y_pred_cv, zero_division=0),
                "accuracy": accuracy_score(y_te, y_pred_cv),
                "brier": brier_score_loss(y_te, y_pred_proba_cv),
            }
            cv_results.append(fold_metrics)
        cv_df = pd.DataFrame(cv_results)
        summary_stats = {}
        for metric in [
            "mcc",
            "auprc",
            "auroc",
            "f1",
            "precision",
            "recall",
            "accuracy",
            "brier",
        ]:
            values = cv_df[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            n = len(values)
            ci_95 = stats.t.ppf(0.975, n - 1) * (std_val / np.sqrt(n))
            summary_stats[metric] = {
                "mean": mean_val,
                "std": std_val,
                "ci_95": ci_95,
                "ci_lower": mean_val - ci_95,
                "ci_upper": mean_val + ci_95,
                "formatted": f"{mean_val:.4f} ± {ci_95:.4f}",
            }
        if output_dir:
            cv_df.to_csv(output_dir / "cv_metrics.csv", index=False)
            summary_df = pd.DataFrame(
                [
                    {
                        "metric": m,
                        "mean": s["mean"],
                        "std": s["std"],
                        "ci_95": s["ci_95"],
                        "ci_lower": s["ci_lower"],
                        "ci_upper": s["ci_upper"],
                        "formatted": s["formatted"],
                    }
                    for m, s in summary_stats.items()
                ]
            )
            summary_df.to_csv(output_dir / "cv_summary_stats.csv", index=False)
        print(f"\n  CV Results Summary (Mean ± 95% CI):")
        for m in ["mcc", "auprc", "auroc", "f1"]:
            print(f"    {m.upper():10}: {summary_stats[m]['formatted']}")
        return cv_df, summary_stats

    @staticmethod
    def plot_cv_distribution(cv_df, output_dir, metrics=["mcc", "auprc"]):
        print("  Generating CV distribution plot...")
        fig, axes = plt.subplots(
            1,
            len(metrics),
            figsize=(FIG_SINGLE_COL_WIDTH * len(metrics), FIG_SINGLE_COL_WIDTH * 0.8),
        )
        if len(metrics) == 1:
            axes = [axes]
        colors = ["#0077BB", "#EE7733", "#009988", "#CC3311"]
        metric_names = {
            "mcc": "MCC",
            "auprc": "AUPRC",
            "auroc": "AUROC",
            "f1": "F1-Score",
        }
        for i, metric in enumerate(metrics):
            ax = axes[i]
            parts = ax.violinplot(
                cv_df[metric], positions=[1], showmeans=True, showextrema=True
            )
            for pc in parts["bodies"]:
                pc.set_facecolor(colors[i % len(colors)])
                pc.set_alpha(0.6)
            bp = ax.boxplot(
                cv_df[metric],
                positions=[1],
                widths=0.15,
                patch_artist=True,
                showfliers=True,
            )
            bp["boxes"][0].set_facecolor("white")
            bp["boxes"][0].set_alpha(0.8)
            jitter = np.random.uniform(-0.05, 0.05, len(cv_df))
            ax.scatter(
                np.ones(len(cv_df)) + jitter,
                cv_df[metric],
                alpha=0.5,
                s=20,
                color=colors[i % len(colors)],
                edgecolor="black",
                linewidth=0.5,
            )
            mean_val = cv_df[metric].mean()
            std_val = cv_df[metric].std()
            n = len(cv_df)
            ci_95 = stats.t.ppf(0.975, n - 1) * (std_val / np.sqrt(n))
            ax.axhline(mean_val, color="red", linestyle="--", linewidth=1, alpha=0.7)
            ax.set_ylabel(metric_names.get(metric, metric.upper()))
            ax.set_title(
                f"{metric_names.get(metric, metric.upper())}\n{mean_val:.4f} ± {ci_95:.4f}",
                fontweight="bold",
            )
            ax.set_xticks([])
            ax.grid(False)
        plt.suptitle("Cross-Validation Metric Distributions", fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_CV_Distribution.png", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_CV_Distribution.tiff", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.close()

    @staticmethod
    def perform_mcnemar_test(
        y_true,
        y_pred_model1,
        y_pred_model2,
        model1_name="Enhanced Model",
        model2_name="Baseline",
        output_dir=None,
    ):
        print(f"\n  Performing McNemar's Test: {model1_name} vs {model2_name}")
        y_true = np.array(y_true)
        y_pred_model1 = np.array(y_pred_model1)
        y_pred_model2 = np.array(y_pred_model2)
        correct_1 = y_pred_model1 == y_true
        correct_2 = y_pred_model2 == y_true
        a = int(np.sum(correct_1 & correct_2))
        b = int(np.sum(correct_1 & ~correct_2))
        c = int(np.sum(~correct_1 & correct_2))
        d = int(np.sum(~correct_1 & ~correct_2))
        if b + c == 0:
            statistic, p_value = np.nan, np.nan
            sig_symbol = "N/A"
            interpretation = "No disagreements"
        else:
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
            if b + c < 25:
                p_value = 2 * min(
                    stats.binom.cdf(min(b, c), b + c, 0.5),
                    1 - stats.binom.cdf(max(b, c) - 1, b + c, 0.5),
                )
            if p_value < 0.001:
                sig_symbol, interpretation = "***", "Highly significant (p<0.001)"
            elif p_value < 0.01:
                sig_symbol, interpretation = "**", "Very significant (p<0.01)"
            elif p_value < 0.05:
                sig_symbol, interpretation = "*", "Significant (p<0.05)"
            else:
                sig_symbol, interpretation = "ns", "No significant difference"
        acc1 = np.mean(correct_1)
        acc2 = np.mean(correct_2)
        result = {
            "model1": model1_name,
            "model2": model2_name,
            "model1_accuracy": acc1,
            "model2_accuracy": acc2,
            "accuracy_difference": acc1 - acc2,
            "both_correct": a,
            "model1_only_correct": b,
            "model2_only_correct": c,
            "both_wrong": d,
            "statistic": statistic,
            "p_value": p_value,
            "significance": sig_symbol,
            "interpretation": interpretation,
        }
        print(f"\n    McNemar Statistic: {statistic:.4f}")
        print(f"    P-value: {p_value:.6f}  ({sig_symbol})")
        print(f"    {interpretation}")
        print(f"    Accuracy improvement: {acc1 - acc2:+.4f}")
        if output_dir:
            pd.DataFrame([result]).to_csv(
                output_dir / "significance_test_mcnemar.csv", index=False
            )
        return result

    @staticmethod
    def generate_misclassification_report(
        variant_df, y_true, y_pred, y_pred_proba, test_indices, output_dir
    ):
        print("  Generating misclassification report...")
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_proba = np.array(y_pred_proba)
        misclass_mask = y_true != y_pred
        misclass_indices = np.where(misclass_mask)[0]
        original_indices = test_indices[misclass_indices]
        coord_cols = [
            c for c in ["chr", "pos", "ref", "alt"] if c in variant_df.columns
        ]
        records = []
        for i, orig_idx in enumerate(original_indices):
            li = misclass_indices[i]
            rec = {
                "test_index": li,
                "original_index": orig_idx,
                "y_true": int(y_true[li]),
                "y_pred": int(y_pred[li]),
                "y_pred_proba": float(y_pred_proba[li]),
                "error_type": "FP" if y_pred[li] == 1 else "FN",
                "confidence": abs(y_pred_proba[li] - 0.5) * 2,
            }
            for col in coord_cols:
                rec[col] = variant_df.iloc[orig_idx][col]
            records.append(rec)
        df = pd.DataFrame(records).sort_values("confidence", ascending=False)
        print(f"    Total misclassifications: {len(df)}")
        print(f"    False Positives (FP): {int(np.sum(df['error_type']=='FP'))}")
        print(f"    False Negatives (FN): {int(np.sum(df['error_type']=='FN'))}")
        df.to_csv(output_dir / "misclassifications.csv", index=False)
        print("    Saved: misclassifications.csv")
        return df

    @staticmethod
    def aggregate_shap_by_category(
        shap_values, feature_names, n_standard, n_fcgr, n_tda, output_dir
    ):
        print("  Aggregating SHAP values by feature category...")
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        category_shap = {
            "Standard/Biological": [],
            "Fractal (FCGR)": [],
            "Topological (TDA)": [],
        }
        feature_category_map = []
        for i, fname in enumerate(feature_names):
            if i < n_standard:
                category = "Standard/Biological"
            elif i < n_standard + n_fcgr:
                category = "Fractal (FCGR)"
            else:
                category = "Topological (TDA)"
            category_shap[category].append(mean_abs_shap[i])
            feature_category_map.append(
                {
                    "feature_name": fname,
                    "category": category,
                    "mean_abs_shap": mean_abs_shap[i],
                }
            )
        aggregated_results = []
        for category, values in category_shap.items():
            if values:
                aggregated_results.append(
                    {
                        "category": category,
                        "n_features": len(values),
                        "mean_abs_shap": np.mean(values),
                        "sum_abs_shap": np.sum(values),
                        "std_abs_shap": np.std(values),
                        "max_abs_shap": np.max(values),
                        "relative_importance": np.sum(values)
                        / np.sum(mean_abs_shap)
                        * 100,
                    }
                )
        aggregated_df = pd.DataFrame(aggregated_results).sort_values(
            "mean_abs_shap", ascending=False
        )
        feature_df = pd.DataFrame(feature_category_map)
        feature_df.to_csv(output_dir / "feature_importance_detailed.csv", index=False)
        aggregated_df.to_csv(
            output_dir / "feature_importance_aggregated.csv", index=False
        )
        print("\n    Category-Level Importance:")
        for _, row in aggregated_df.iterrows():
            print(
                f"    {row['category']:25} - Mean |SHAP|: {row['mean_abs_shap']:.6f} "
                f"({row['relative_importance']:.1f}%)"
            )
        return aggregated_df, feature_df

    @staticmethod
    def plot_feature_importance_stacked(aggregated_df, output_dir):
        print("  Generating feature importance stacked bar chart...")
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(FIG_DOUBLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.7),
            gridspec_kw={'wspace': 0.6}
        )
        colors = {
            "Standard/Biological": "#0077BB",
            "Fractal (FCGR)": "#EE7733",
            "Topological (TDA)": "#009988",
        }
        categories = aggregated_df["category"].tolist()
        bar_colors = [colors.get(c, "#888888") for c in categories]
        bars = ax1.barh(
            categories,
            aggregated_df["mean_abs_shap"],
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax1.set_xlabel("Mean |SHAP Value|")
        ax1.set_title("(A) Feature Category Importance", fontweight="bold", loc="left")
        ax1.grid(False)
        for bar, val in zip(bars, aggregated_df["mean_abs_shap"]):
            ax1.text(
                val + 0.001,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=FIG_FONT_SIZE - 1,
            )
        ax2.set_aspect("equal")
        wedges, texts = ax2.pie(
            aggregated_df["relative_importance"],
            colors=bar_colors,
            startangle=90,
            explode=[0.02] * len(categories),
            wedgeprops=dict(linewidth=1, edgecolor="white"),
        )
        ax2.set_title("(B) Relative Contribution", fontweight="bold")
        
        # Use a legend instead of direct labels to avoid overlapping on small slices
        legend_labels = [
            f"{c} ({v:.1f}%)"
            for c, v in zip(categories, aggregated_df["relative_importance"])
        ]
        ax2.legend(
            wedges,
            legend_labels,
            title="Category",
            loc="center left",
            bbox_to_anchor=(0.95, 0.5),
            frameon=True,
            fontsize=FIG_FONT_SIZE - 1,
            edgecolor="black"
        )
        plt.tight_layout(w_pad=4.0)
        plt.savefig(
            output_dir / "Fig_Feature_Importance_Stacked.png",
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
        plt.savefig(
            output_dir / "Fig_Feature_Importance_Stacked.tiff",
            dpi=FIG_DPI,
            bbox_inches="tight",
        )
        plt.close()

    @staticmethod
    def generate_learning_curve(
        X,
        y,
        model_factory_fn,
        n_standard,
        n_fcgr,
        n_tda,
        output_dir,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9],
    ):
        print("\n  Generating Learning Curve...")
        print(f"    Training sizes: {[f'{s*100:.0f}%' for s in train_sizes]}")
        y_arr = np.array(y) if hasattr(y, "values") else y
        results = []
        for train_size in tqdm(train_sizes, desc="Learning Curve", ncols=80):
            if train_size < 1.0:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X,
                    y_arr,
                    train_size=train_size,
                    stratify=y_arr,
                    random_state=Config.RANDOM_STATE,
                )
            else:
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X,
                    y_arr,
                    test_size=0.2,
                    stratify=y_arr,
                    random_state=Config.RANDOM_STATE,
                )
            model = model_factory_fn(n_standard, n_fcgr, n_tda)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_te)
            y_pred_proba = model.predict_proba(X_te)
            results.append(
                {
                    "train_size_pct": train_size * 100,
                    "n_train_samples": len(X_tr),
                    "n_test_samples": len(X_te),
                    "mcc": matthews_corrcoef(y_te, y_pred),
                    "auprc": average_precision_score(y_te, y_pred_proba),
                    "auroc": roc_auc_score(y_te, y_pred_proba),
                    "f1": f1_score(y_te, y_pred, zero_division=0),
                    "accuracy": accuracy_score(y_te, y_pred),
                }
            )
        results_df = pd.DataFrame(results)
        fig, axes = plt.subplots(
            1, 2, figsize=(FIG_DOUBLE_COL_WIDTH, FIG_SINGLE_COL_WIDTH * 0.7)
        )
        ax1 = axes[0]
        ax1.plot(
            results_df["train_size_pct"],
            results_df["mcc"],
            "o-",
            color="#0077BB",
            linewidth=1.5,
            markersize=6,
            label="MCC",
        )
        ax1.plot(
            results_df["train_size_pct"],
            results_df["f1"],
            "s--",
            color="#EE7733",
            linewidth=1.5,
            markersize=6,
            label="F1-Score",
        )
        ax1.set_xlabel("Training Data Size (%)")
        ax1.set_ylabel("Score")
        ax1.set_title("(A) Learning Curve — MCC & F1", fontweight="bold", loc="left")
        ax1.legend()
        ax1.grid(False)
        ax1.set_xlim([0, 100])
        ax1.set_ylim([0, 1.05])
        ax2 = axes[1]
        ax2.plot(
            results_df["train_size_pct"],
            results_df["auprc"],
            "o-",
            color="#009988",
            linewidth=1.5,
            markersize=6,
            label="AUPRC",
        )
        ax2.plot(
            results_df["train_size_pct"],
            results_df["auroc"],
            "s--",
            color="#CC3311",
            linewidth=1.5,
            markersize=6,
            label="AUROC",
        )
        ax2.set_xlabel("Training Data Size (%)")
        ax2.set_ylabel("Score")
        ax2.set_title(
            "(B) Learning Curve — AUPRC & AUROC", fontweight="bold", loc="left"
        )
        ax2.legend()
        ax2.grid(False)
        ax2.set_xlim([0, 100])
        ax2.set_ylim([0, 1.05])
        plt.tight_layout()
        plt.savefig(
            output_dir / "Fig_Learning_Curve.png", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.savefig(
            output_dir / "Fig_Learning_Curve.tiff", dpi=FIG_DPI, bbox_inches="tight"
        )
        plt.close()
        results_df.to_csv(output_dir / "learning_curve_data.csv", index=False)
        print("\n    Learning Curve Results:")
        for _, row in results_df.iterrows():
            print(
                f"    {row['train_size_pct']:5.0f}% ({row['n_train_samples']:,} samples): "
                f"MCC={row['mcc']:.4f}, AUPRC={row['auprc']:.4f}"
            )
        return results_df


# BaselineEnsemble (unchanged)


class BaselineEnsemble:
    def __init__(self, n_biological_features=18, random_state=42):
        self.n_biological_features = n_biological_features
        self.random_state = random_state
        self.models = {}
        self.is_fitted = False

    def _create_base_models(self):
        rs = self.random_state
        return {
            "RF_1": RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=rs, n_jobs=-1
            ),
            "RF_2": RandomForestClassifier(
                n_estimators=100, max_depth=15, random_state=rs + 1, n_jobs=-1
            ),
            "ET_1": ExtraTreesClassifier(
                n_estimators=100, max_depth=10, random_state=rs + 2, n_jobs=-1
            ),
            "ET_2": ExtraTreesClassifier(
                n_estimators=100, max_depth=15, random_state=rs + 3, n_jobs=-1
            ),
            "GB_1": GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=rs + 4
            ),
            "GB_2": GradientBoostingClassifier(
                n_estimators=100, max_depth=7, random_state=rs + 5
            ),
            "HGB_1": HistGradientBoostingClassifier(
                max_iter=100, max_depth=6, random_state=rs + 6
            ),
            "HGB_2": HistGradientBoostingClassifier(
                max_iter=100, max_depth=8, random_state=rs + 7
            ),
        }

    def fit(self, X, y):
        X_bio = X[:, : self.n_biological_features]
        self.models = self._create_base_models()
        for name, model in tqdm(
            self.models.items(), desc="Training Baseline", ncols=80
        ):
            model.fit(X_bio, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        X_bio = X[:, : self.n_biological_features]
        probas = [model.predict_proba(X_bio)[:, 1] for model in self.models.values()]
        return np.mean(probas, axis=0)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# Metrics helpers (unchanged)


def compute_metrics(y_true, y_pred, y_pred_proba=None):
    metrics = {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_pred_proba is not None:
        metrics["auprc"] = average_precision_score(y_true, y_pred_proba)
        metrics["auroc"] = roc_auc_score(y_true, y_pred_proba)
        metrics["brier"] = brier_score_loss(y_true, y_pred_proba)
    return metrics


def print_metrics_report(metrics, model_name="Model"):
    print(f"\n{model_name} — Performance Metrics")
    print("\nPrimary Metrics:")
    print(f"  Matthews Correlation Coefficient (MCC): {metrics.get('mcc', 0):.6f}")
    print(f"  AUPRC (Precision-Recall AUC):           {metrics.get('auprc', 0):.6f}")
    print(f"  Brier Score (Calibration):              {metrics.get('brier', 0):.6f}")
    print("\nSecondary Metrics:")
    print(f"  Accuracy:   {metrics.get('accuracy', 0):.6f}")
    print(f"  Precision:  {metrics.get('precision', 0):.6f}")
    print(f"  Recall:     {metrics.get('recall', 0):.6f}")
    print(f"  F1-Score:   {metrics.get('f1', 0):.6f}")
    print(f"  AUROC:      {metrics.get('auroc', 0):.6f}")
    print(f"  Cohen's κ:  {metrics.get('kappa', 0):.6f}")


# run_final_model —


def run_final_model():
    """Run the full TF-DFE framework end-to-end on the dataset."""
    start_time = time.time()

    print("\nTF-DFE: Final Framework Run")
    print("Best Model: Enhanced KNORA-E with OOF Stacking")
    print(f"\nOutput Directory: {Config.OUTPUT_DIR}")

    print("\nStep 1: Load Data & EDA")
    if not Path(Config.DATA_PATH).exists():
        print(f"ERROR: Data file not found at {Config.DATA_PATH}")
        return

    print(f"\nLoading: {Config.DATA_PATH}")
    variant_df = pd.read_csv(Config.DATA_PATH)
    print(f"Loaded {len(variant_df):,} variants")

    if "LABEL_PATHOGENIC" in variant_df.columns:
        target_dist = variant_df["LABEL_PATHOGENIC"].value_counts().to_dict()
        print("\nClass Distribution:")
        for label, count in sorted(target_dist.items()):
            label_name = "Pathogenic" if label == 1 else "Benign"
            print(f"  {label_name:12}: {count:6,} ({100*count/len(variant_df):5.2f}%)")

    print("\nGenerating EDA Figures...")
    plot_class_distribution(variant_df, Config.OUTPUT_DIR)

    print("\nStep 2: Feature Engineering")
    preprocessor = TFDFEPreprocessor(
        use_fcgr=True, use_tda=True, genome_path=Config.GENOME_PATH
    )
    X, y = preprocessor.fit_transform(variant_df)

    n_standard = len(preprocessor.feature_names)
    n_fcgr = len(preprocessor.fcgr_feature_names)
    n_tda = len(preprocessor.tda_feature_names)

    print(f"\nFeature engineering complete — Total features: {X.shape[1]}")

    print("\nStep 3: Train Best Model")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE,
    )
    print(f"\nData Split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # Extract CONSENSUS_SCORE for sample weighting
    consensus_scores_train = None
    if "CONSENSUS_SCORE" in variant_df.columns:
        y_arr_full = np.array(y) if hasattr(y, "values") else y
        _, _, _, _, train_idx_raw, _ = train_test_split(
            X,
            y_arr_full,
            np.arange(len(y_arr_full)),
            test_size=Config.TEST_SIZE,
            stratify=y,
            random_state=Config.RANDOM_STATE,
        )
        # Extract training-set CONSENSUS_SCORE (before inner val split)
        # The factory will handle mis-alignment if lengths differ.
        cs_all = variant_df["CONSENSUS_SCORE"].values
        consensus_scores_train = cs_all[train_idx_raw]
        print(
            f"  CONSENSUS_SCORE available for "
            f"{len(consensus_scores_train):,} training variants."
        )
    else:
        print("  CONSENSUS_SCORE not in dataset; " "sample weighting disabled.")

    model = EnhancedTFDFEEnsemble(n_standard=n_standard, n_fcgr=n_fcgr, n_tda=n_tda)
    train_start = time.time()
    model.fit(X_train, y_train, consensus_scores_train=consensus_scores_train)
    train_time = time.time() - train_start
    print(f"\nTraining complete ({train_time/60:.2f} minutes)")

    print("\nGenerating predictions on test set...")
    y_pred_proba = model.predict_proba(X_test)
    opt_threshold = model._optimized_threshold()
    print(f"Using optimized decision threshold: {opt_threshold:.2f}")
    y_pred = model.predict(X_test, threshold=opt_threshold)

    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    print_metrics_report(metrics, model_name="TF-DFE")

    print("\nEvaluating all base models on Unseen Test Set...")
    try:
        active_preds_proba = model.enhanced_knora._compute_active_preds(X_test)
        all_metrics_list = []
        
        # Add main TF-DFE metrics
        tf_dfe_row = {"Model": "TF-DFE (Full Ensemble)"}
        tf_dfe_row.update(metrics)
        all_metrics_list.append(tf_dfe_row)

        # Compute metrics for each active base model
        for idx, (name, clf) in enumerate(model.enhanced_knora._active_clf_list):
            y_base_proba = active_preds_proba[:, idx]
            y_base_pred = (y_base_proba >= 0.5).astype(int)
            base_metrics = compute_metrics(y_test, y_base_pred, y_base_proba)
            row = {"Model": name}
            row.update(base_metrics)
            all_metrics_list.append(row)
            
        all_metrics_df = pd.DataFrame(all_metrics_list)
        out_csv = Config.OUTPUT_DIR / "final_all_models_test_metrics.csv"
        all_metrics_df.to_csv(out_csv, index=False)
        print(f"  Successfully saved test metrics for all models to: {out_csv.name}")
    except Exception as e:
        print(f"  Failed to compute base model test metrics: {e}")

    # Selective prediction metrics
    print("\nStep 3A: Selective Prediction Metrics")
    y_selective, purity_scores, coverage = model.selective_predict(X_test)
    confident_mask = y_selective >= 0
    if confident_mask.sum() > 0:
        y_te_arr = np.array(y_test) if hasattr(y_test, "values") else y_test
        sel_mcc = matthews_corrcoef(
            y_te_arr[confident_mask], y_selective[confident_mask]
        )
        sel_acc = accuracy_score(y_te_arr[confident_mask], y_selective[confident_mask])
        print(f"  Coverage:             {coverage:.3%}")
        print(f"  Selective MCC:        {sel_mcc:.6f}  (vs full {metrics['mcc']:.6f})")
        print(
            f"  Selective Accuracy:   {sel_acc:.6f}  (vs full {metrics['accuracy']:.6f})"
        )
        selective_df = pd.DataFrame(
            {
                "coverage": [coverage],
                "n_predicted": [int(confident_mask.sum())],
                "n_abstained": [int((~confident_mask).sum())],
                "selective_mcc": [sel_mcc],
                "selective_accuracy": [sel_acc],
                "full_mcc": [metrics["mcc"]],
                "full_accuracy": [metrics["accuracy"]],
                "abstain_threshold": [Config.SELECTIVE_ABSTAIN_THRESHOLD],
            }
        )
        selective_df.to_csv(
            Config.OUTPUT_DIR / "selective_prediction_metrics.csv", index=False
        )
        print("  Saved: selective_prediction_metrics.csv")

    print("\nStep 4: Generate Analysis Figures")
    plot_roc_pr_curves(y_test, y_pred_proba, "TF-DFE", Config.OUTPUT_DIR)
    plot_confusion_matrix(y_test, y_pred, "TF-DFE", Config.OUTPUT_DIR)
    plot_tsne(X_test, y_test, "TF-DFE Feature Space Separation", Config.OUTPUT_DIR)

    # SHAP analysis —: use model OBJECT not string name
    shap_model = None
    shap_indices = None
    shap_feature_names = None
    if (
        SHAP_AVAILABLE
        and hasattr(model, "diverse_factory")
        and model.diverse_factory is not None
    ):
        for name, m in model.diverse_factory.models.items():
            # Unwrap CalibratedClassifierCV to get the underlying fitted
            # estimator for SHAP compatibility. Never pass the string name and
            # never the "deprecated" base_estimator sentinel.
            base_m = _unwrap_calibrated_estimator(m)
            if hasattr(base_m, "estimators_") or any(
                k in name for k in ["XGB", "LGB", "CatBoost"]
            ):
                shap_model = base_m  # ← model OBJECT (BUG FIX)
                shap_indices = model.diverse_factory.feature_indices[name]
                shap_feature_names = [
                    preprocessor.all_feature_names[i] for i in shap_indices
                ]
                X_shap = X_test[: min(500, len(X_test)), shap_indices]
                break
        if shap_model:
            perform_shap_analysis(
                shap_model, X_shap, shap_feature_names, Config.OUTPUT_DIR, "TF-DFE"
            )
        else:
            print("  SHAP analysis skipped (no compatible model found)")

    print("\nAll figures generated")

    print("\nStep 4A: Calibration Analysis")
    y_test_arr = np.array(y_test) if hasattr(y_test, "values") else y_test
    ece, brier, calibration_bins = TFDFEvaluator.plot_calibration_curve(
        y_test_arr,
        y_pred_proba,
        output_dir=Config.OUTPUT_DIR,
        model_name="TF-DFE",
        n_bins=10,
    )
    metrics["ece"] = ece

    print("\nStep 4B: Statistical Significance Test (McNemar's)")
    print("  Training Baseline Model for comparison...")
    baseline_model = BaselineEnsemble(
        n_biological_features=n_standard, random_state=Config.RANDOM_STATE
    )
    baseline_model.fit(X_train, y_train)
    y_pred_baseline = baseline_model.predict(X_test)
    y_pred_proba_baseline = baseline_model.predict_proba(X_test)
    baseline_metrics = compute_metrics(
        y_test_arr, y_pred_baseline, y_pred_proba_baseline
    )
    print(f"\n  Baseline Model Metrics:")
    print(f"    MCC:    {baseline_metrics['mcc']:.6f}")
    print(f"    AUPRC:  {baseline_metrics['auprc']:.6f}")
    print(f"    F1:     {baseline_metrics['f1']:.6f}")

    mcnemar_result = TFDFEvaluator.perform_mcnemar_test(
        y_test_arr,
        y_pred,
        y_pred_baseline,
        model1_name="TF-DFE",
        model2_name="Baseline Ensemble",
        output_dir=Config.OUTPUT_DIR,
    )

    print("\nStep 4C: Error Analysis")
    y_array = np.array(y) if hasattr(y, "values") else y
    _, _, _, _, train_indices, test_indices = train_test_split(
        X,
        y_array,
        np.arange(len(y_array)),
        test_size=Config.TEST_SIZE,
        stratify=y,
        random_state=Config.RANDOM_STATE,
    )
    misclass_df = TFDFEvaluator.generate_misclassification_report(
        variant_df, y_test_arr, y_pred, y_pred_proba, test_indices, Config.OUTPUT_DIR
    )

    # Step 4D: Aggregated Feature Importance — applied
    print("\nStep 4D: Aggregated Feature Importance by Category")
    if SHAP_AVAILABLE and shap_model is not None:
        try:
            print("  Computing SHAP values for category aggregation...")
            sample_size = min(1000, len(X_test))
            X_sample_full = X_test[:sample_size]

            # prefer models with full feature view for category aggregation
            agg_model = None
            agg_indices = None
            for cand in ["Full_GradientBoosting", "Full_HistGradient"]:
                if cand in model.diverse_factory.models:
                    raw = model.diverse_factory.models[cand]
                    # Unwrap CalibratedClassifierCV to the fitted underlying
                    # estimator. FIX: raw.base_estimator returns the string
                    # "deprecated" in modern sklearn, which crashed SHAP with
                    # "'str' object has no attribute 'predict_proba'".
                    agg_model = _unwrap_calibrated_estimator(raw)
                    agg_indices = model.diverse_factory.feature_indices[cand]
                    break
            if agg_model is None:
                agg_model = shap_model
                agg_indices = shap_indices

            X_sample_subset = X_sample_full[:, agg_indices]
            agg_feature_names = [preprocessor.all_feature_names[i] for i in agg_indices]

            # Final guard: never let a non-estimator (e.g. a leftover string)
            # reach the explainer.
            if isinstance(agg_model, str) or not hasattr(agg_model, "predict_proba"):
                raise RuntimeError(
                    f"aggregation model is not a usable estimator: {type(agg_model)!r}"
                )

            if hasattr(agg_model, "estimators_"):
                explainer = shap.TreeExplainer(agg_model)
            else:
                background = shap.sample(X_sample_subset, 50)
                explainer = shap.KernelExplainer(agg_model.predict_proba, background)

            agg_shap_values = explainer.shap_values(X_sample_subset)
            if isinstance(agg_shap_values, list) and len(agg_shap_values) == 2:
                agg_shap_values = agg_shap_values[1]
            elif isinstance(agg_shap_values, np.ndarray) and agg_shap_values.ndim == 3:
                agg_shap_values = agg_shap_values[:, :, 1]

            aggregated_df, feature_df = TFDFEvaluator.aggregate_shap_by_category(
                agg_shap_values,
                agg_feature_names,
                n_standard,
                n_fcgr,
                n_tda,
                Config.OUTPUT_DIR,
            )
            TFDFEvaluator.plot_feature_importance_stacked(
                aggregated_df, Config.OUTPUT_DIR
            )
        except Exception as e:
            print(f"  Aggregated SHAP analysis failed: {e}")
    else:
        print("  Skipped (SHAP not available or no compatible model)")

    print("\nStep 4E: Learning Curve Analysis")

    def model_factory_fn(n_std, n_fcg, n_td):
        return EnhancedTFDFEEnsemble(n_standard=n_std, n_fcgr=n_fcg, n_tda=n_td)

    if len(X) > 50000:
        print(
            f"  Dataset size ({len(X):,}) is large. Using subset for learning curve..."
        )
        lc_sample_size = min(50000, len(X))
        lc_indices = np.random.choice(len(X), lc_sample_size, replace=False)
        X_lc = X[lc_indices]
        y_lc = y_array[lc_indices]
    else:
        X_lc = X
        y_lc = y_array

    learning_curve_df = TFDFEvaluator.generate_learning_curve(
        X_lc,
        y_lc,
        model_factory_fn,
        n_standard,
        n_fcgr,
        n_tda,
        Config.OUTPUT_DIR,
        train_sizes=[0.1, 0.3, 0.5, 0.7, 0.9],
    )

    print("\nStep 4F: Repeated Stratified K-Fold Cross-Validation")
    if len(X) > 100000:
        print(f"  Dataset size ({len(X):,}) is large. Using subset for CV analysis...")
        cv_sample_size = min(50000, len(X))
        cv_indices = np.random.choice(len(X), cv_sample_size, replace=False)
        X_cv = X[cv_indices]
        y_cv = y_array[cv_indices]
    else:
        X_cv = X
        y_cv = y_array

    cv_df, cv_summary = TFDFEvaluator.perform_repeated_cv(
        X_cv,
        y_cv,
        model_factory_fn,
        n_splits=5,
        n_repeats=2,
        n_standard=n_standard,
        n_fcgr=n_fcgr,
        n_tda=n_tda,
        output_dir=Config.OUTPUT_DIR,
    )
    TFDFEvaluator.plot_cv_distribution(
        cv_df, Config.OUTPUT_DIR, metrics=["mcc", "auprc"]
    )

    print("\nStep 5: Save Reports")
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(Config.OUTPUT_DIR / "final_metrics.csv", index=False)

    predictions_df = pd.DataFrame(
        {
            "y_true": y_test.values if hasattr(y_test, "values") else y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
        }
    )
    predictions_df.to_csv(Config.OUTPUT_DIR / "final_predictions.csv", index=False)

    with open(Config.OUTPUT_DIR / "final_report.txt", "w", encoding="utf-8") as rf:
        rf.write("TF-DFE Final Analysis Report\n")
        rf.write("All 7 Fixes + Supervisor Changes 1, 2, 3 Applied\n\n")
        rf.write("Change Summary:\n")
        rf.write("  FIX 1–7 (Pre-processing and data structure fixes retained)\n")
        rf.write(
            "  CHANGE 1: +MLP (BioTDA), +kNN_SML, +ElasticNet LR; Q-statistic diversity\n"
        )
        rf.write(
            "  CHANGE 2: TDA over FCGR/sequence space; ~20 topological features (was 6)\n"
        )
        rf.write(
            f"  CHANGE 3: Consensus-margin weighting + selective prediction "
            f"(θ={Config.SELECTIVE_ABSTAIN_THRESHOLD})\n"
        )
        rf.write("  BUG FIX: SHAP fallback uses model object, not string name\n\n")
        rf.write("Dataset Information\n")
        rf.write(f"  Total variants: {len(variant_df):,}\n")
        rf.write(f"  Training samples: {len(X_train):,}\n")
        rf.write(f"  Test samples: {len(X_test):,}\n")
        rf.write(f"  Total features: {X.shape[1]:,}\n")
        rf.write(f"    Standard/Biological: {n_standard}\n")
        rf.write(f"    FCGR PCA (FIX 3):    {n_fcgr}\n")
        rf.write(f"    TDA (CHANGE 2):      {n_tda}\n\n")
        rf.write("Performance Metrics (Test Set)\n")
        for metric, value in metrics.items():
            rf.write(f"  {metric.upper():<30}: {value:.6f}\n")
        rf.write(f"\nCalibration Analysis\n")
        rf.write(f"  ECE: {ece:.6f}\n  Brier: {brier:.6f}\n")
        rf.write("\nCross-Validation (Mean ± 95% CI)\n")
        for m, s in cv_summary.items():
            rf.write(f"  {m.upper():<30}: {s['formatted']}\n")
        rf.write("\nMcNemar Test\n")
        rf.write(f"  Statistic: {mcnemar_result['statistic']:.4f}\n")
        rf.write(f"  P-value: {mcnemar_result['p_value']:.6f}\n")
        rf.write(f"  Significance: {mcnemar_result['significance']}\n")
        rf.write(f"  {mcnemar_result['interpretation']}\n\n")
        rf.write("Classification Report\n")
        rf.write(
            classification_report(
                y_test, y_pred, target_names=["Benign", "Pathogenic"], digits=4
            )
        )

    total_time = time.time() - start_time
    print(f"\nFINAL ANALYSIS RUN COMPLETE")
    print(f"\nFinal Performance (Test Set):")
    print(f"  MCC (Primary):      {metrics['mcc']:.6f}")
    print(f"  AUPRC:              {metrics['auprc']:.6f}")
    print(f"  AUROC:              {metrics['auroc']:.6f}")
    print(f"  F1-Score:           {metrics['f1']:.6f}")
    print(f"  ECE (Calibration):  {metrics.get('ece', ece):.6f}")
    print(f"\nCross-Validation (Mean ± 95% CI):")
    print(f"  MCC:   {cv_summary['mcc']['formatted']}")
    print(f"  AUPRC: {cv_summary['auprc']['formatted']}")
    print(f"\nStatistical Significance:")
    print(
        f"  McNemar p-value: {mcnemar_result['p_value']:.6f} "
        f"({mcnemar_result['significance']})"
    )
    print(f"  {mcnemar_result['interpretation']}")
    print(f"\nOutput Directory: {Config.OUTPUT_DIR}")
    print(f"\nTotal Runtime: {total_time/60:.2f} minutes")


import sys

class TerminalLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.log.write("Terminal Output - Stage 08\n==========================\n\n")
        self.flush()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

if __name__ == "__main__":
    log_file = Config.OUTPUT_DIR / "code_8_terminal_output.txt"
    # Ensure directory exists before creating the logger file
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    sys.stdout = TerminalLogger(log_file)
    try:
        run_final_model()
    finally:
        sys.stdout.log.close()
        sys.stdout = sys.stdout.terminal
