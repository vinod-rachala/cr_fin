# pip install dask[complete] recordlinkage pandas pyarrow loguru tqdm scikit-learn

import dask
import dask.dataframe as dd
import pandas as pd
import recordlinkage
from loguru import logger
from tqdm import tqdm
from dask import delayed

# ----------------------------------------------------------------------
# 1. Configuration
# ----------------------------------------------------------------------
AUTH_PATH = "auth.csv"
SETTLE_PATH = "settlement.csv"
TRUE_PAIRS_PATH = "true_pairs.csv"

PROB_THRESHOLD = 0.8

JOIN_CONDITIONS = {
    "S1": {"amount_tolerance": 2, "date_offset": 1, "features": ["merchant_id", "amount", "date"]},
    "S2": {"amount_tolerance": 5, "date_offset": 3, "features": ["merchant_id", "amount", "date", "currency"]},
    "S3": {"amount_tolerance": 10, "date_offset": 7, "features": ["merchant_id", "amount", "region", "txn_type"]},
}

OUTPUT_MATCHES = "all_matches.parquet"
OUTPUT_METRICS = "merchant_metrics.parquet"
LOG_FILE = "join_pipeline.log"

logger.add(LOG_FILE, rotation="5 MB")
logger.info("Pipeline started")

# ----------------------------------------------------------------------
# 2. Load all available data
# ----------------------------------------------------------------------
auth_dd = dd.read_csv(AUTH_PATH, dtype=str, parse_dates=["date"])
settle_dd = dd.read_csv(SETTLE_PATH, dtype=str, parse_dates=["date"])

auth = auth_dd.compute()
settle = settle_dd.compute()

merchant_ids = sorted(auth["merchant_id"].unique().tolist())
logger.info(f"Loaded {len(auth)} auth, {len(settle)} settlement records across {len(merchant_ids)} merchants.")

# ----------------------------------------------------------------------
# 3. EM-based linkage per condition
# ----------------------------------------------------------------------
def run_linkage(condition_name, cond, auth_df, settle_df, threshold):
    try:
        indexer = recordlinkage.Index()
        indexer.block("merchant_id")
        candidate_links = indexer.index(auth_df, settle_df)

        compare = recordlinkage.Compare()
        compare.numeric("amount", "amount", method="gauss",
                        offset=cond["amount_tolerance"], scale=20, label="amount")
        compare.date("date", "date", offset=cond["date_offset"], label="date")
        for opt in ["currency","region","txn_type"]:
            if opt in cond["features"]:
                compare.exact(opt, opt, label=opt)

        features = compare.compute(candidate_links, auth_df, settle_df)
        if len(features) == 0:
            return pd.DataFrame()

        em = recordlinkage.ECMClassifier()
        em.fit(features)
        probs = pd.Series(em.probabilities_, index=features.index, name="prob").reset_index()
        probs = probs[probs["prob"] >= threshold]
        if len(probs) == 0:
            return pd.DataFrame()

        probs["auth_id"] = auth_df.loc[probs["level_0"], "auth_id"].values
        probs["settle_id"] = settle_df.loc[probs["level_1"], "settle_id"].values
        probs["condition_set"] = condition_name
        return probs[["auth_id", "settle_id", "prob", "condition_set"]]

    except Exception as e:
        logger.error(f"Error in {condition_name}: {e}")
        return pd.DataFrame()

# ----------------------------------------------------------------------
# 4. Evaluation metrics
# ----------------------------------------------------------------------
true_pairs = pd.read_csv(TRUE_PAIRS_PATH)
true_set = set(map(tuple, true_pairs[["auth_id", "settle_id"]].values))

def evaluate(pred_df):
    pred_set = set(map(tuple, pred_df[["auth_id","settle_id"]].values))
    tp = len(true_set & pred_set)
