#!/usr/bin/env python3
# ==============================================================================================
#  GUIDE SOC FALSE-POSITIVE REDUCTION — FINAL, REPRODUCIBLE EXPERIMENT SCRIPT
# ----------------------------------------------------------------------------------------------
#  Paper : "An ensemble machine learning framework for reducing false positives in security
#           operations centers using big data analytics" (PeerJ Computer Science, revision)
#  Data  : Microsoft GUIDE (Kaggle "microsoft-security-incident-prediction"):
#           GUIDE_Train.csv (training / validation)   GUIDE_Test.csv (held-out evaluation, used once)
#
#  Usage :  python3 guide_soc_experiment.py --data-dir /path/to/GUIDE            (all stages)
#           python3 guide_soc_experiment.py --data-dir /path/to/GUIDE --stages audit,main
#           python3 guide_soc_experiment.py --data-dir /path/to/GUIDE --stages report
#  Output:  ./guide_results/  (tables/*.csv, figures/*.png, stage_*.json, results_all.json,
#           RESULTS_SUMMARY.md, all_tables.xlsx, run.log, environment.json)
#
#  Reviewer comments addressed (labels used in the response letter):
#   R1.2/R2.3  fully specified fusion: weighted soft voting, weights + thresholds selected on validation
#   R1.3       every hyper-parameter in CONFIG below (exported to tables/table_config.csv)
#   R1.4/R2.5  leakage-safe pipeline: (OrgId, IncidentId)-disjoint splits, train-only fitting,
#              causal label-free context features, label-shuffle test, row-split comparison
#   R1.5/R2.2  ablation (RF, XGB, RF+XGB, +IF, +Apriori, full, w/o context features, meta-learner)
#   R1.6/R1.7/R2.1 exact counts, metrics from confusion matrices, absolute vs relative reductions
#   R2.4       cluster-bootstrap 95% CIs, McNemar tests, grouped 5-fold CV, seed variation
#   R2.6       three-tier routing with explicit operating constraints
#   R2.7       latency, throughput, scalability, robustness to missing fields, temporal drift
#   R2.8       independent held-out GUIDE_Test, unseen-organisation and temporal generalisation
# ==============================================================================================
import argparse, gc, json, os, platform, sys, time, warnings
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import binomtest, chi2, ttest_rel

warnings.filterwarnings("ignore")
SEED = 42

# ----------------------------------------------------------------------------------------------
# CONFIGURATION (Table 10 of the paper is generated from this dictionary)
# ----------------------------------------------------------------------------------------------
CONFIG = dict(
    data=dict(train_file="GUIDE_Train.csv", test_file="GUIDE_Test.csv",
              val_share=0.15,                 # share of GUIDE_Train incidents used for validation
              train_frac=1.0,                 # share of GUIDE_Train incidents used (memory control)
              aux_frac=0.25,                  # share of GUIDE_Train incidents for auxiliary analyses
              window_minutes=60, chunksize=1_000_000),
    rf=dict(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=5,
            max_features="sqrt", max_samples=0.5, bootstrap=True, class_weight="balanced_subsample",
            train_rows=1_000_000, random_state=SEED, n_jobs=-1),
    xgb=dict(n_estimators=1500, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
             reg_lambda=1.0, reg_alpha=0.0, min_child_weight=1, tree_method="hist",
             eval_metric="logloss", early_stopping_rounds=50, random_state=SEED, n_jobs=-1),
    iforest=dict(n_estimators=200, max_samples=256, max_features=1.0, contamination="auto",
                 train_rows=1_000_000, random_state=SEED, n_jobs=-1),
    apriori=dict(min_support=0.01, min_confidence=0.60, min_lift=1.10, max_len=3,
                 top_detectors=200, max_rules=500, mining_rows=300_000),
    baselines=dict(lr_rows=500_000, dt_min_samples_leaf=5),
    fusion=dict(grid_step=0.05, target_recall=0.95),
    routing=dict(max_threat_leak=0.01, escalate_precision=0.95),
    stats=dict(n_boot=1000, n_boot_auc=100),
    aux=dict(cv_folds=5, seeds=[1, 2, 3], robustness_rows=1_000_000, latency_reps=200,
             scalability_fracs=[0.1, 0.25, 0.5]),
)
GUIDE_NUMERIC = ["OrgId", "DetectorId", "AlertTitle", "DeviceId", "IpAddress", "AccountName", "DeviceName",
                 "ApplicationId", "ApplicationName", "OAuthApplicationId", "FileName", "ResourceIdName",
                 "OSFamily", "OSVersion", "CountryCode", "State", "City", "EmailClusterId"]
GUIDE_CATEG = ["Category", "EntityType", "EvidenceRole", "ThreatFamily", "ResourceType", "Roles",
               "AntispamDirection", "SuspicionLevel"]
LEAKAGE = ["IncidentGrade", "LastVerdict", "ActionGrouped", "ActionGranular"]          # removed
IDENTIFIERS = ["Id", "Sha256", "Url", "RegistryKey", "RegistryValueName", "RegistryValueData",
               "FolderPath", "AccountSid", "AccountUpn", "AccountObjectId", "NetworkMessageId", "Usage"]
READ_COLS = ["OrgId", "IncidentId", "AlertId", "Timestamp", "MitreTechniques", "IncidentGrade"] + \
            [c for c in GUIDE_NUMERIC if c != "OrgId"] + GUIDE_CATEG
CONTEXT_FEATS = ["hour", "dayofweek", "MitreCount", "inc_evidence_sofar", "inc_alerts_sofar", "inc_age_min",
                 "alert_evidence_sofar", "org_alerts_prev60m", "ctx_categories_prev60m"]
FREQ_COLS = ["DetectorId", "AlertTitle", "OrgId", "DeviceId", "IpAddress", "AccountName", "DeviceName",
             "ApplicationId", "FileName", "City", "CountryCode"]
TENANT_ENTITY_IDS = ["OrgId", "DeviceId", "IpAddress", "AccountName", "DeviceName", "ApplicationId",
                     "ApplicationName", "FileName", "City", "State", "CountryCode", "OAuthApplicationId",
                     "ResourceIdName", "EmailClusterId"]
OUT = None
LOGF = None


# ==============================================================================================
# Utilities
# ==============================================================================================
def log(*a):
    msg = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(msg, flush=True)
    if LOGF:
        LOGF.write(msg + "\n"); LOGF.flush()


def jdump(obj, name):
    with open(os.path.join(OUT, name), "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


def jload(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def mem_gb():
    try:
        with open("/proc/meminfo") as f:
            d = {l.split(":")[0]: int(l.split()[1]) for l in f}
        return d["MemTotal"] / 1e6, d["MemAvailable"] / 1e6
    except Exception:
        return None, None


def log_mem(where):
    """log the process' current and peak memory (helps users with little RAM)"""
    try:
        import resource
        cur = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
        pk = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
        log(f"memory [{where}]: process {cur:.1f} GB (peak {pk:.1f} GB), system available {mem_gb()[1]:.1f} GB")
    except Exception:
        pass


def hash01(keys, salt=0):
    """deterministic, well-mixed pseudo-random number in [0,1) per key (splitmix64) -> reproducible
    incident-level sampling that is independent of OrgId / IncidentId structure"""
    with np.errstate(over="ignore"):
        z = np.asarray(keys).astype(np.uint64) + np.uint64((salt + 1) * 0x9E3779B97F4A7C15 % 2**64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) / float(2 ** 53)


def epoch_s(ts):
    """seconds since 1970 for a datetime Series, independent of the pandas time unit (ns in pandas 2, us in 3)"""
    v = pd.to_datetime(ts, errors="coerce", utc=True)
    return v.dt.tz_convert(None).values.astype("datetime64[s]").astype(np.int64)


def inc_key(org, inc):
    return org.astype(np.int64) * 10_000_000 + inc.astype(np.int64)


# ==============================================================================================
# 1. Loading (chunked, incident-level sampling, compact dtypes)
# ==============================================================================================
CAT_LIST = None   # global, ordered list of alert categories (both files) -> consistent codes


def load_guide(path, frac=1.0, salt=0, keep_usage=False):
    cols = READ_COLS + (["Usage"] if keep_usage else [])
    parts, n_read = [], 0
    header = pd.read_csv(path, nrows=0).columns.str.strip().tolist()
    usecols = [c for c in cols if c in header]
    cat_cols = [c for c in GUIDE_CATEG + ["IncidentGrade", "MitreTechniques"] + (["Usage"] if keep_usage else []) if c in usecols]
    for ch in pd.read_csv(path, usecols=usecols, chunksize=CONFIG["data"]["chunksize"], low_memory=False):
        n_read += len(ch)
        ch = ch[ch["IncidentGrade"].notna()]
        k = inc_key(ch["OrgId"].values, ch["IncidentId"].values)
        if frac < 1.0:
            m = hash01(k, salt) < frac
            ch = ch[m]; k = k[m]
        ch = ch.assign(inc_key=k)
        for c in GUIDE_NUMERIC + ["IncidentId", "AlertId"]:
            if c in ch:
                v = pd.to_numeric(ch[c], errors="coerce")
                ch[c] = v.astype("float32") if c == "EmailClusterId" else v.fillna(-1).astype("int64" if c in ("IncidentId", "AlertId") else "int32")
        for c in cat_cols:                       # compact strings chunk by chunk (keeps peak memory low)
            ch[c] = ch[c].astype("category")
        ch["Timestamp"] = pd.to_datetime(ch["Timestamp"], errors="coerce", utc=True)
        parts.append(ch)
    from pandas.api.types import union_categoricals
    cats = {c: union_categoricals([p[c] for p in parts], ignore_order=True) for c in cat_cols}
    for p_ in parts:
        p_.drop(columns=cat_cols, inplace=True)
    df = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    for c in cat_cols:
        df[c] = pd.Categorical(cats[c])
    del cats
    if CAT_LIST is not None:
        df["Category"] = df["Category"].cat.set_categories(CAT_LIST)
    df["label"] = (df["IncidentGrade"] == "TruePositive").astype(np.int8)
    df["alert_key"] = inc_key(df["OrgId"].values, df["AlertId"].values)
    log(f"loaded {os.path.basename(path)} frac={frac}: read {n_read:,}, kept {len(df):,} rows, "
        f"{df.inc_key.nunique():,} incidents, TP share {df.label.mean():.4f}")
    return df, n_read


# ==============================================================================================
# 2. Causal, label-free context features
# ==============================================================================================
def org_burst_table(paths, window_min):
    """distinct alerts of the same organisation in the previous window, computed on the complete
    alert stream (both files, label-free, past-only) -> Series indexed by alert_key"""
    firsts = []; cats = set()
    for p in paths:
        for ch in pd.read_csv(p, usecols=["OrgId", "AlertId", "Timestamp", "Category"], chunksize=2_000_000):
            cats.update(ch["Category"].dropna().astype(str).unique().tolist())
            ch["t"] = epoch_s(ch["Timestamp"])
            ch["ak"] = inc_key(ch["OrgId"].values, ch["AlertId"].values)
            firsts.append(ch.groupby("ak").agg(org=("OrgId", "first"), t=("t", "min")))
    f = pd.concat(firsts); f = f.groupby(level=0).agg(org=("org", "first"), t=("t", "min"))
    f = f.sort_values(["org", "t"])
    out = np.zeros(len(f), dtype=np.float32); w = window_min * 60
    orgs = f["org"].values; t = f["t"].values
    starts = np.r_[0, np.flatnonzero(np.diff(orgs)) + 1, len(f)]
    for a, b in zip(starts[:-1], starts[1:]):
        tt = t[a:b]
        out[a:b] = np.arange(b - a) - np.searchsorted(tt, tt - w, side="left")
    log(f"org burst table: {len(f):,} alerts, {len(cats)} alert categories")
    return pd.Series(out, index=f.index), sorted(cats)


def add_context_features(df, burst, window_min):
    df.sort_values(["Timestamp", "alert_key"], kind="mergesort", inplace=True, ignore_index=True)
    ts = df["Timestamp"]; t_sec = epoch_s(ts)
    df["hour"] = ts.dt.hour.astype("float32"); df["dayofweek"] = ts.dt.dayofweek.astype("float32")
    mt = df["MitreTechniques"]
    if not isinstance(mt.dtype, pd.CategoricalDtype):
        mt = mt.astype("category")
    per_cat = np.array([0 if str(x) == "" else len(str(x).split(";")) for x in mt.cat.categories] + [0], dtype=np.float32)
    df["MitreCount"] = per_cat[mt.cat.codes.values]          # code -1 (missing) -> last entry (0)
    g = df.groupby("inc_key", sort=False)
    df["inc_evidence_sofar"] = (g.cumcount() + 1).astype("float32")
    first_alert = ~df.duplicated(["inc_key", "alert_key"])
    df["inc_alerts_sofar"] = first_alert.groupby(df["inc_key"]).cumsum().astype("float32")
    tmin = pd.Series(t_sec).groupby(df["inc_key"].values).cummin().values
    df["inc_age_min"] = ((t_sec - tmin) / 60).astype("float32")
    df["alert_evidence_sofar"] = (df.groupby("alert_key").cumcount() + 1).astype("float32")
    df["org_alerts_prev60m"] = df["alert_key"].map(burst).fillna(0).astype("float32").values
    # other alert categories seen in the same incident in the previous window (bitmask, past-only)
    cat_codes = np.minimum(df["Category"].cat.codes.values.astype(np.int64), 62)   # -1 = missing
    order = np.lexsort((t_sec, df["inc_key"].values))
    ik = df["inc_key"].values[order]; tt = t_sec[order]; cc = cat_codes[order].tolist(); ttl = tt.tolist()
    mask = np.zeros(len(df), dtype=np.int64); w = window_min * 60
    starts = np.r_[0, np.flatnonzero(np.diff(ik)) + 1, len(ik)].tolist()
    out = [0] * len(df)
    for a, b in zip(starts[:-1], starts[1:]):
        if b - a == 1:
            continue
        cnt = {}; cur = 0; lo = a
        for j in range(a, b):
            tj = ttl[j]
            while ttl[lo] < tj - w:
                c0 = cc[lo]
                if c0 >= 0:
                    cnt[c0] -= 1
                    if cnt[c0] == 0: cur &= ~(1 << c0)
                lo += 1
            cj = cc[j]
            out[j] = cur & ~(1 << cj) if cj >= 0 else cur
            if cj >= 0:
                cnt[cj] = cnt.get(cj, 0) + 1
                cur |= (1 << cj)
    mask[:] = out
    ctx = np.zeros(len(df), dtype=np.int64); ctx[order] = mask
    df["_ctx_mask"] = ctx
    df["ctx_categories_prev60m"] = np.unpackbits(ctx.view(np.uint8).reshape(-1, 8), axis=1).sum(1).astype(np.float32)
    return df


# ==============================================================================================
# 3. Encoding (fitted on training data only)
# ==============================================================================================
class Encoder:
    """Memory-efficient encoder fitted on TRAINING data only:
       numeric fields (median imputation) + training-set frequency encodings + one-hot encoding of
       categorical fields (the TOP_K most frequent training categories with >= MIN_FREQ occurrences;
       rarer -> '<col>_infrequent', missing -> '<col>_NA'; unseen categories at inference ->
       '<col>_infrequent'). TOP_K bounds the matrix width (and RAM) on the complete GUIDE data."""
    MIN_FREQ = 50
    TOP_K = 30

    def fit(self, tr, drop=()):
        self.num_cols = [c for c in GUIDE_NUMERIC + CONTEXT_FEATS if c in tr.columns and c not in drop]
        self.cat_cols = [c for c in GUIDE_CATEG if c in tr.columns and c not in drop]
        self.freq_cols = [c for c in FREQ_COLS if c in tr.columns and c not in drop and "freq_" + c not in drop]
        self.freq = {c: tr[c].value_counts() for c in self.freq_cols}
        self.medians = tr[self.num_cols].median().astype("float32")
        self.keep, names_cat = {}, []
        for c in self.cat_cols:
            vc = tr[c].value_counts(dropna=True)
            keep = [str(v) for v in vc[vc >= self.MIN_FREQ].index[:self.TOP_K]]
            self.keep[c] = {v: i for i, v in enumerate(keep)}
            names_cat += [f"{c}_{v}" for v in keep] + [f"{c}_infrequent", f"{c}_NA"]
        self.names = self.num_cols + ["freq_" + c for c in self.freq_cols] + names_cat
        return self

    def transform(self, d, batch=500_000):
        n_num, n_frq = len(self.num_cols), len(self.freq_cols)
        out = np.zeros((len(d), len(self.names)), dtype=np.float32)
        for a in range(0, len(d), batch):
            s = d.iloc[a:a + batch]; m = len(s)
            out[a:a + m, :n_num] = s[self.num_cols].astype("float32").fillna(self.medians).values
            for j, c in enumerate(self.freq_cols):
                out[a:a + m, n_num + j] = s[c].map(self.freq[c]).fillna(0).astype("float32").values
            off = n_num + n_frq; rows = np.arange(a, a + m)
            for c in self.cat_cols:
                col = s[c] if isinstance(s[c].dtype, pd.CategoricalDtype) else s[c].astype("category")
                k = len(self.keep[c]); lut = np.array([self.keep[c].get(str(v), k) for v in col.cat.categories] + [k + 1], dtype=np.int64)
                codes = col.cat.codes.values.astype(np.int64)            # -1 = missing -> last entry (NA)
                out[rows, off + lut[codes]] = 1.0
                off += k + 2
        return out


class LazyX:
    """Encoded matrix that is never held in memory as a whole: slicing X[a:b] encodes only those
    rows. Used for GUIDE_Test (4.1 M rows) so that peak RAM stays bounded."""
    def __init__(self, enc, d):
        self.enc, self.d = enc, d

    def __len__(self):
        return len(self.d)

    def __getitem__(self, sl):
        return self.enc.transform(self.d.iloc[sl])


# ==============================================================================================
# 4. Apriori correlation layer (class-association rules, training labels only)
# ==============================================================================================
class CorrelationLayer:
    FAMILIES = [("CAT", "Category"), ("ENT", "EntityType"), ("ROLE", "EvidenceRole"), ("SUSP", "SuspicionLevel")]

    def _items(self, d):
        """sparse boolean item matrix (rows x vocabulary); vocabulary grows only while fitting"""
        n = len(d); R_, C_ = [], []
        def add(rows, names):
            u, inv = np.unique(np.asarray(names, dtype=object).astype(str), return_inverse=True)
            if self._growing:
                for x in u: self.vocab.setdefault(x, len(self.vocab))
            m = np.array([self.vocab.get(x, -1) for x in u], dtype=np.int64)[inv]
            ok = m >= 0; R_.append(np.asarray(rows)[ok]); C_.append(m[ok])
        allrows = np.arange(n)
        for pre, col in self.FAMILIES:
            v = d[col].astype(object); v = v.where(v.notna(), "NA").astype(str).values.astype(object)
            add(allrows, pre + "=" + v)
        det = d["DetectorId"].values
        add(allrows, np.where(np.isin(det, self.top_det), "DET=" + det.astype(str).astype(object), "DET=OTHER"))
        mt = d["MitreTechniques"].astype("string").fillna("").values
        nz = np.flatnonzero(mt != "")
        if len(nz):
            lists = [x.split(";") for x in mt[nz]]
            lens = np.fromiter((len(x) for x in lists), dtype=np.int64, count=len(lists))
            toks = np.array([t for x in lists for t in x], dtype=object); rws = np.repeat(nz, lens)
            keep = toks != ""
            add(rws[keep], "T=" + toks[keep])
        masks = d["_ctx_mask"].values
        for bit, name in enumerate(self.cat_names[:63]):
            r = np.flatnonzero((masks >> bit) & 1)
            if len(r):
                add(r, np.full(len(r), "CTX=" + name, dtype=object))
        R = np.concatenate(R_) if R_ else np.zeros(0, np.int64); C = np.concatenate(C_) if C_ else np.zeros(0, np.int64)
        return sparse.csr_matrix((np.ones(len(R), dtype=bool), (R, C)), shape=(n, max(len(self.vocab), 1)))

    def fit(self, tr, cfg):
        from mlxtend.frequent_patterns import apriori, association_rules
        self.cfg = cfg
        self.cat_names = list(map(str, tr["Category"].cat.categories))
        self.top_det = tr["DetectorId"].value_counts().index[:cfg["top_detectors"]].values
        s = tr.sample(n=min(cfg["mining_rows"], len(tr)), random_state=SEED)
        self.vocab = {"Y=TP": 0, "Y=BEN": 1}; self._growing = True
        M = self._items(s).tocoo()
        lab = s["label"].values
        R = np.r_[M.row, np.arange(len(lab))]; C = np.r_[M.col, np.where(lab == 1, 0, 1)]
        self._growing = False
        inv_vocab = {v: k for k, v in self.vocab.items()}
        M = sparse.csr_matrix((np.ones(len(R), dtype=bool), (R, C)), shape=(len(lab), len(self.vocab)))
        onehot = pd.DataFrame.sparse.from_spmatrix(M, columns=range(M.shape[1]))
        t0 = time.time()
        fi = apriori(onehot, min_support=cfg["min_support"], use_colnames=False, max_len=cfg["max_len"], low_memory=True)
        try:
            rules = association_rules(fi, num_itemsets=len(onehot), metric="confidence", min_threshold=cfg["min_confidence"])
        except TypeError:
            rules = association_rules(fi, metric="confidence", min_threshold=cfg["min_confidence"])
        for col in ["antecedents", "consequents"]:
            rules[col] = rules[col].apply(lambda z: frozenset(inv_vocab[int(i)] for i in z))
        rules = rules[rules["consequents"].apply(lambda z: z in (frozenset({"Y=TP"}), frozenset({"Y=BEN"})))]
        rules = rules[~rules["antecedents"].apply(lambda z: any(x.startswith("Y=") for x in z))]
        rules = rules[rules["lift"] >= cfg["min_lift"]].copy()
        rules["cons"] = rules["consequents"].apply(lambda z: next(iter(z)))
        rules["alen"] = rules["antecedents"].apply(len)
        rules = rules.sort_values(["alen", "confidence"], ascending=[True, False])
        kept = []
        for r in rules.itertuples():
            if any(k.cons == r.cons and k.antecedents < r.antecedents and k.confidence >= r.confidence for k in kept):
                continue
            kept.append(r)
        rules = pd.DataFrame(kept) if kept else pd.DataFrame(columns=rules.columns)
        if len(rules):
            rules["score"] = rules["confidence"] * rules["support"]
            rules = rules.sort_values("score", ascending=False).head(cfg["max_rules"]).reset_index(drop=True)
        self.rules = rules; self.n_itemsets = len(fi); self.prior = float(tr["label"].mean())
        self.mining_seconds = time.time() - t0
        log(f"Apriori: {len(fi)} itemsets -> {len(rules)} rules ({(rules.cons == 'Y=TP').sum() if len(rules) else 0} TP)")
        return self

    def score(self, d, batch=1_000_000):
        out = np.empty(len(d), dtype=np.float32); covered = 0
        for a in range(0, len(d), batch):
            s = d.iloc[a:a + batch]
            X = self._items(s).tocsc()
            n = len(s); tp = np.full(n, np.nan); ben = np.full(n, np.nan)
            colrows = {}
            def rows_of(item):
                j = self.vocab.get(item, -1)
                if j < 0 or j >= X.shape[1]:
                    return np.array([], dtype=np.int64)
                if j not in colrows:
                    colrows[j] = X.indices[X.indptr[j]:X.indptr[j + 1]]
                return colrows[j]
            for r in self.rules.itertuples():
                ants = list(r.antecedents); m = rows_of(ants[0])
                for x in ants[1:]:
                    if len(m) == 0: break
                    m = np.intersect1d(m, rows_of(x), assume_unique=True)
                if len(m):
                    arr = tp if r.cons == "Y=TP" else ben
                    arr[m] = np.fmax(arr[m], r.confidence)
            covered += int((~np.isnan(tp) | ~np.isnan(ben)).sum())
            tpe = np.where(np.isnan(tp), self.prior, tp); bee = np.where(np.isnan(ben), 1 - self.prior, ben)
            out[a:a + n] = tpe / (tpe + bee)
        self.coverage_ = covered / max(len(d), 1)
        return out


# ==============================================================================================
# 5. Metrics, thresholds, statistics
# ==============================================================================================
def cm_counts(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return (int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum()),
            int(((p == 0) & (y == 1)).sum()), int(((p == 0) & (y == 0)).sum()))


def metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(precision=prec, recall=rec, f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                fpr=fp / (fp + tn) if fp + tn else 0.0, accuracy=(tp + tn) / max(tp + fp + fn + tn, 1),
                tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn))


def thr_for_recall(y, s, target):
    """largest threshold whose recall >= target"""
    pos = np.asarray(s)[np.asarray(y) == 1]
    k = max(int(np.ceil(target * len(pos))) - 1, 0)          # k-th largest (0-based)
    return float(np.partition(pos, len(pos) - 1 - k)[len(pos) - 1 - k])


def _group_index(groups):
    _, gi = np.unique(groups, return_inverse=True)
    return gi, gi.max() + 1


def cluster_boot(y, pred, groups, n, rng):
    gi, G = _group_index(groups)
    agg = np.zeros((G, 4))
    for k, m in enumerate([(pred == 1) & (y == 1), (pred == 1) & (y == 0), (pred == 0) & (y == 1), (pred == 0) & (y == 0)]):
        agg[:, k] = np.bincount(gi, weights=m.astype(float), minlength=G)
    res = {k: [] for k in ["precision", "recall", "f1", "fpr"]}
    for _ in range(n):
        w = np.bincount(rng.integers(0, G, G), minlength=G)
        m = metrics(*(w @ agg))
        for k in res: res[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in res.items()}


def paired_boot(y, pa, pb, groups, n, rng):
    gi, G = _group_index(groups)
    cols = [(pa == 1) & (y == 0), (pb == 1) & (y == 0), (pa == 1) & (y == 1), (pb == 1) & (y == 1), y == 0, y == 1]
    agg = np.column_stack([np.bincount(gi, weights=c.astype(float), minlength=G) for c in cols])
    dfpr, drec = [], []
    for _ in range(n):
        w = np.bincount(rng.integers(0, G, G), minlength=G); s = w @ agg
        dfpr.append(s[0] / s[4] - s[1] / s[4]); drec.append(s[2] / s[5] - s[3] / s[5])
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return pc(dfpr), pc(drec)


def mcnemar(y, pa, pb):
    ca, cb = pa == y, pb == y
    b, c = int((ca & ~cb).sum()), int((~ca & cb).sum())
    if b + c == 0: return dict(b=b, c=c, p=1.0)
    if b + c < 50: return dict(b=b, c=c, p=float(binomtest(b, b + c, 0.5).pvalue))
    st = (abs(b - c) - 1) ** 2 / (b + c)
    return dict(b=b, c=c, stat=float(st), p=float(chi2.sf(st, 1)))


def auc_ci(y, s, groups, n, rng):
    from sklearn.metrics import roc_auc_score, average_precision_score
    gi, G = _group_index(groups)
    order = np.argsort(gi, kind="stable"); bounds = np.r_[0, np.cumsum(np.bincount(gi, minlength=G))]
    A, P = [], []
    for _ in range(n):
        pick = rng.integers(0, G, G)
        idx = np.concatenate([order[bounds[k]:bounds[k + 1]] for k in pick])
        if len(np.unique(y[idx])) < 2: continue
        A.append(roc_auc_score(y[idx], s[idx])); P.append(average_precision_score(y[idx], s[idx]))
    pc = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return pc(A), pc(P)


# ==============================================================================================
# 6. Layers, fusion and routing
# ==============================================================================================
COMPONENTS = ["rf", "xgb", "iso", "corr"]


def subsample_idx(n, k, seed=SEED):
    return np.sort(np.random.default_rng(seed).choice(n, min(n, k), replace=False))


def train_layers(Xtr, ytr, Xva, yva, tr_df, cfg=CONFIG, seed=SEED, with_corr=True, tag=""):
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    import xgboost as xgb
    L, T = {}, {}
    rfc = {k: v for k, v in cfg["rf"].items() if k != "train_rows"}; rfc["random_state"] = seed
    ii = subsample_idx(len(ytr), cfg["rf"]["train_rows"], seed)
    t0 = time.time(); L["rf"] = RandomForestClassifier(**rfc).fit(Xtr[ii], ytr[ii]); T["rf"] = time.time() - t0
    xc = dict(cfg["xgb"]); xc["random_state"] = seed
    xc["scale_pos_weight"] = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    t0 = time.time(); L["xgb"] = xgb.XGBClassifier(**xc).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    T["xgb"] = time.time() - t0; L["xgb_rounds"] = int(L["xgb"].best_iteration) + 1; L["spw"] = xc["scale_pos_weight"]
    ic = {k: v for k, v in cfg["iforest"].items() if k != "train_rows"}; ic["random_state"] = seed
    jj = subsample_idx(len(ytr), cfg["iforest"]["train_rows"], seed + 1)
    t0 = time.time(); L["iso"] = IsolationForest(**ic).fit(Xtr[jj]); T["iforest"] = time.time() - t0
    if with_corr:
        t0 = time.time(); L["corr"] = CorrelationLayer().fit(tr_df, cfg["apriori"]); T["apriori"] = time.time() - t0
    L["rf_rows"] = len(ii); L["if_rows"] = len(jj)
    log(f"{tag} layers trained: " + ", ".join(f"{k}={v:.0f}s" for k, v in T.items()) + f", xgb rounds={L['xgb_rounds']}")
    return L, T


def predict_batched(fn, X, batch=250_000):
    return np.concatenate([fn(X[a:a + batch]) for a in range(0, len(X), batch)]).astype(np.float32)


def layer_scores(L, X, d=None, batch=250_000):
    S = {"rf": [], "xgb": [], "iso": []}
    for a in range(0, len(X), batch):                 # each batch is encoded once (LazyX) and scored by all layers
        z = X[a:a + batch]
        S["rf"].append(L["rf"].predict_proba(z)[:, 1]); S["xgb"].append(L["xgb"].predict_proba(z)[:, 1])
        S["iso"].append(-L["iso"].score_samples(z))  # Eq.11, higher = anomalous
    S = {k: np.concatenate(v).astype(np.float32) for k, v in S.items()}
    if "corr" in L and d is not None:
        S["corr"] = L["corr"].score(d)
    return S


def simplex(k, step):
    n = int(round(1 / step))
    def rec(k, n):
        if k == 1: yield (n,); return
        for i in range(n + 1):
            for r in rec(k - 1, n - i): yield (i,) + r
    for w in rec(k, n): yield np.array(w) / n


def fit_fusion(Sv, yv, comps, cfg):
    """weights on a simplex grid + threshold chosen on VALIDATION: minimise FPR s.t. recall >= target"""
    Mfull = np.column_stack([Sv[c] for c in comps]).astype(np.float32)
    sub = subsample_idx(len(yv), 400_000, 99) if len(yv) > 400_000 else np.arange(len(yv))
    M = Mfull[sub]; ys = np.asarray(yv)[sub]; neg = ys == 0; pos = ys == 1; best = None
    for w in (simplex(len(comps), cfg["grid_step"]) if len(comps) > 1 else [np.array([1.0])]):
        if len(comps) > 1 and not any(w[comps.index(c)] > 0 for c in ("rf", "xgb") if c in comps) and \
                any(c in comps for c in ("rf", "xgb")):
            continue
        r = M @ w
        thr = thr_for_recall(ys, r, cfg["target_recall"])
        fpr = float((r[neg] >= thr).mean()); rec = float((r[pos] >= thr).mean())
        key = (round(fpr, 7), -rec)
        if best is None or key < best[0]:
            best = (key, w, thr)
    thr_full = thr_for_recall(yv, Mfull @ best[1], cfg["target_recall"])
    return dict(weights=dict(zip(comps, map(float, best[1]))), threshold=float(thr_full),
                grid_rows=int(len(sub)))


def apply_fusion(S, fus):
    r = sum(w * S[c] for c, w in fus["weights"].items())
    return r.astype(np.float32), (r >= fus["threshold"]).astype(np.int8)


def routing(yv, rv, yt, rt, cfg):
    t_low = float(np.quantile(rv[yv == 1], cfg["max_threat_leak"]))
    o = np.argsort(-rv); cp = np.cumsum(yv[o]) / np.arange(1, len(o) + 1)
    ok = np.flatnonzero(cp >= cfg["escalate_precision"])
    t_high = max(float(rv[o][ok.max()]) if len(ok) else float(rv.max()), t_low)
    def band(y, r):
        lo, hi = r < t_low, r >= t_high; mid = ~lo & ~hi
        return dict(auto_dismiss_share=float(lo.mean()), review_share=float(mid.mean()), escalate_share=float(hi.mean()),
                    threats_auto_dismissed=int((lo & (y == 1)).sum()), threats_total=int((y == 1).sum()),
                    benign_auto_dismissed=int((lo & (y == 0)).sum()), benign_total=int((y == 0).sum()),
                    escalate_precision=float(y[hi].mean()) if hi.any() else None,
                    review_tp_rate=float(y[mid].mean()) if mid.any() else None)
    return dict(t_low=t_low, t_high=t_high, val=band(yv, rv), test=band(yt, rt))


def evaluate_variant(name, y, s_test, thr, groups, rng, n_boot):
    from sklearn.metrics import roc_auc_score, average_precision_score
    p = (s_test >= thr).astype(np.int8); m = metrics(*cm_counts(y, p))
    m["threshold"] = float(thr); m["ci95"] = cluster_boot(y, p, groups, n_boot, rng)
    if len(np.unique(s_test)) > 2:
        m["roc_auc"] = float(roc_auc_score(y, s_test)); m["pr_auc"] = float(average_precision_score(y, s_test))
    return m, p


def strat_group_folds(y, groups, n_splits, seed=SEED):
    """fold id per row such that no group spans two folds and each fold has ~equal numbers of
    positive and negative rows (greedy bin packing of groups, largest first; deterministic).
    Equivalent in purpose to sklearn's StratifiedGroupKFold, but fast for ~10^6 groups."""
    y = np.asarray(y); g_u, g_inv = np.unique(groups, return_inverse=True)
    pos = np.bincount(g_inv, weights=(y == 1).astype(float), minlength=len(g_u))
    neg = np.bincount(g_inv, weights=(y == 0).astype(float), minlength=len(g_u))
    tie = np.random.default_rng(seed).random(len(g_u))
    order = np.lexsort((tie, -(pos + neg)))
    tp, tn = max(pos.sum() / n_splits, 1.0), max(neg.sum() / n_splits, 1.0)
    fp, fn = [0.0] * n_splits, [0.0] * n_splits; gfold = np.zeros(len(g_u), dtype=np.int16)
    pos_l, neg_l = pos.tolist(), neg.tolist(); rng_k = range(n_splits)
    for gi in order.tolist():
        pg, ng = pos_l[gi], neg_l[gi]
        k = min(rng_k, key=lambda j: max((fp[j] + pg) / tp, (fn[j] + ng) / tn))
        gfold[gi] = k; fp[k] += pg; fn[k] += ng
    return gfold[g_inv]


def strat_group_parts(y, groups, shares, seed=SEED, n_splits=20):
    """incident/org-disjoint parts with given shares (multiples of 1/n_splits), class-stratified"""
    fold = strat_group_folds(y, groups, n_splits, seed)
    edges = np.round(np.cumsum(shares) * n_splits).astype(int)
    return np.searchsorted(edges, fold, side="right").clip(0, len(shares) - 1)


def split_by_hash(keys, shares, salt):
    """assign groups to consecutive parts by a deterministic hash: shares e.g. [0.70,0.15,0.15]"""
    u = hash01(keys, salt); edges = np.cumsum(shares)
    return np.searchsorted(edges, u, side="right").clip(0, len(shares) - 1)


# ==============================================================================================
# 7. Stages
# ==============================================================================================
def prepare(path, frac, salt, burst, keep_usage=False):
    df, n_read = load_guide(path, frac, salt, keep_usage)
    t0 = time.time(); df = add_context_features(df, burst, CONFIG["data"]["window_minutes"])
    log(f"context features: {time.time() - t0:.0f}s"); return df, n_read


def get_burst(args):
    global CAT_LIST
    p = os.path.join(OUT, "cache_org_burst.pkl")
    if os.path.exists(p):
        b, CAT_LIST = pd.read_pickle(p)
        return b
    b, CAT_LIST = org_burst_table([os.path.join(args.data_dir, CONFIG["data"][f]) for f in ("train_file", "test_file")],
                                  CONFIG["data"]["window_minutes"])
    pd.to_pickle((b, CAT_LIST), p); return b


# ---------------------------------------------------------------- stage: audit (R1.6, R2.1, R2.8)
def stage_audit(args):
    res = {}
    keys = {}
    for f in ("train_file", "test_file"):
        path = os.path.join(args.data_dir, CONFIG["data"][f]); n = 0; grades = Counter(); orgs = set(); incs = set()
        dets = set(); tmin = tmax = None; usage = Counter(); cols = None
        for ch in pd.read_csv(path, chunksize=2_000_000, low_memory=False,
                              usecols=lambda c: c.strip() in ("OrgId", "IncidentId", "AlertId", "Timestamp", "IncidentGrade", "DetectorId", "Usage")):
            ch.columns = ch.columns.str.strip(); n += len(ch)
            grades.update(ch["IncidentGrade"].fillna("MISSING").value_counts().to_dict())
            orgs.update(ch["OrgId"].unique()); dets.update(ch["DetectorId"].unique())
            incs.update(inc_key(ch["OrgId"].values, ch["IncidentId"].values).tolist())
            if "Usage" in ch: usage.update(ch["Usage"].value_counts().to_dict())
            ts = ch["Timestamp"].astype(str); tmin = min(tmin or ts.min(), ts.min()); tmax = max(tmax or ts.max(), ts.max())
        keys[f] = (incs, orgs, dets)
        labelled = n - grades.get("MISSING", 0)
        res[f] = dict(rows=n, rows_labelled=labelled, grade_counts=dict(grades), incidents=len(incs), orgs=len(orgs),
                      detectors=len(dets), time_min=tmin, time_max=tmax, usage=dict(usage),
                      tp_share=grades.get("TruePositive", 0) / max(labelled, 1))
        log(f"audit {CONFIG['data'][f]}: {n:,} rows, {len(incs):,} incidents, TP share {res[f]['tp_share']:.4f}")
    (ti, to, td), (ei, eo, ed) = keys["train_file"], keys["test_file"]
    res["overlap"] = dict(incident_pairs=len(ti & ei), test_orgs_seen_in_train=len(eo & to), test_orgs=len(eo),
                          test_detectors_seen_in_train=len(ed & td), test_detectors=len(ed))
    jdump(res, "stage_audit.json"); return res


# ---------------------------------------------------------------- stage: main (most reviewer points)
def stage_main(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.preprocessing import MaxAbsScaler
    from sklearn.metrics import roc_auc_score, brier_score_loss
    rng = np.random.default_rng(SEED); R = {}
    burst = get_burst(args)
    full, n_read = prepare(os.path.join(args.data_dir, CONFIG["data"]["train_file"]), CONFIG["data"]["train_frac"], 0, burst)
    part = strat_group_parts(full["label"].values, full["inc_key"].values, [1 - CONFIG["data"]["val_share"], CONFIG["data"]["val_share"]])
    tr = full[part == 0].reset_index(drop=True); va = full[part == 1].reset_index(drop=True); del full; gc.collect()
    te, _ = prepare(os.path.join(args.data_dir, CONFIG["data"]["test_file"]), 1.0, 0, burst)
    R["split"] = {n: dict(rows=len(d), tp=int(d.label.sum()), non_tp=int((d.label == 0).sum()), tp_share=float(d.label.mean()),
                          incidents=int(d.inc_key.nunique()), orgs=int(d.OrgId.nunique()))
                  for n, d in [("train", tr), ("validation", va), ("test", te)]}
    R["split"]["incident_overlap_train_val"] = int(len(set(tr.inc_key) & set(va.inc_key)))
    R["split"]["incident_overlap_train_test"] = int(len(set(tr.inc_key) & set(te.inc_key)))
    R["split"]["guide_train_rows_read"] = n_read
    log("split:", R["split"])
    enc = Encoder().fit(tr)
    log_mem("before encoding")
    est = (len(tr) + len(va)) * len(enc.names) * 4 / 1e9
    log(f"{len(enc.names)} features; the encoded training + validation matrices need about {est:.1f} GB")
    if mem_gb()[1] is not None and est > 0.7 * mem_gb()[1]:
        log("WARNING: little free memory for this training size. If the run is killed (exit code 137), "
            "add swap space or rerun with --train-frac 0.5")
    Xtr = enc.transform(tr); log_mem("train matrix encoded")
    Xva = enc.transform(va); Xte = LazyX(enc, te)
    ytr, yva, yte = tr.label.values, va.label.values, te.label.values
    gte = te.inc_key.values
    R["n_features"] = len(enc.names); R["feature_names"] = enc.names
    log(f"features: {Xtr.shape}, memory available {mem_gb()[1]:.1f} GB")
    for c in ("IncidentId", "AlertId"):                       # keys already combined into inc_key / alert_key
        for d_ in (tr, va, te):
            if c in d_: d_.drop(columns=c, inplace=True)
    gc.collect()

    # ---- main layers
    L, T = train_layers(Xtr, ytr, Xva, yva, tr, tag="main"); log_mem("main layers trained")
    R["train_time_s"] = T; R["xgb_rounds"] = L["xgb_rounds"]; R["scale_pos_weight"] = L["spw"]
    R["rf_rows"] = L["rf_rows"]; R["if_rows"] = L["if_rows"]
    t0 = time.time(); Sva = layer_scores(L, Xva, va); Ste = layer_scores(L, Xte, te); R["score_time_s"] = time.time() - t0
    rules = L["corr"].rules
    R["apriori"] = dict(n_itemsets=L["corr"].n_itemsets, n_rules=len(rules), n_tp_rules=int((rules.cons == "Y=TP").sum()),
                        n_ben_rules=int((rules.cons == "Y=BEN").sum()), test_coverage=L["corr"].coverage_,
                        top_rules=[dict(antecedent=" & ".join(sorted(r.antecedents)), consequent=r.cons,
                                        support=float(r.support), confidence=float(r.confidence), lift=float(r.lift))
                                   for r in rules.head(20).itertuples()])
    R["layer_auc_test"] = {k: float(roc_auc_score(yte, v)) for k, v in Ste.items()}
    R["brier_test"] = {k: float(brier_score_loss(yte, np.clip(v, 0, 1))) for k, v in Ste.items()}
    R["feature_importance"] = dict(
        rf=pd.Series(L["rf"].feature_importances_, index=enc.names).sort_values(ascending=False).head(20).to_dict(),
        xgb=pd.Series(L["xgb"].feature_importances_, index=enc.names).sort_values(ascending=False).head(20).to_dict())

    # ---- baselines
    V = {}
    det_rate = tr.groupby("DetectorId")["label"].mean(); prior = float(ytr.mean())
    V["Forward-all (SIEM as-is)"] = dict(val=np.ones(len(yva), np.float32), test=np.ones(len(yte), np.float32), fixed=0.5)
    V["Detector-reputation rule"] = dict(val=va.DetectorId.map(det_rate).fillna(prior).values.astype(np.float32),
                                         test=te.DetectorId.map(det_rate).fillna(prior).values.astype(np.float32))
    ii = subsample_idx(len(ytr), CONFIG["baselines"]["lr_rows"])
    sc = MaxAbsScaler().fit(Xtr[ii]); t0 = time.time()
    lr = LogisticRegression(max_iter=300, class_weight="balanced", random_state=SEED).fit(sc.transform(Xtr[ii]), ytr[ii])
    T["logreg"] = time.time() - t0
    V["Logistic regression"] = dict(val=predict_batched(lambda z: lr.predict_proba(sc.transform(z))[:, 1], Xva),
                                    test=predict_batched(lambda z: lr.predict_proba(sc.transform(z))[:, 1], Xte))
    jj = subsample_idx(len(ytr), CONFIG["rf"]["train_rows"]); t0 = time.time()
    dt = DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=CONFIG["baselines"]["dt_min_samples_leaf"],
                                random_state=SEED).fit(Xtr[jj], ytr[jj]); T["dtree"] = time.time() - t0
    V["Decision tree (threshold 0.5)"] = dict(val=dt.predict_proba(Xva)[:, 1], test=predict_batched(lambda z: dt.predict_proba(z)[:, 1], Xte), fixed=0.5)
    V["Isolation Forest only"] = dict(val=Sva["iso"], test=Ste["iso"])
    V["Apriori correlation only"] = dict(val=Sva["corr"], test=Ste["corr"])
    del lr, dt; gc.collect()

    # ---- ablation via fusion of layer scores
    ABL = {"Random Forest only": ["rf"], "XGBoost only": ["xgb"], "RF + XGBoost": ["rf", "xgb"],
           "RF + XGBoost + Isolation Forest": ["rf", "xgb", "iso"], "RF + XGBoost + Apriori": ["rf", "xgb", "corr"],
           "Full layered model": COMPONENTS}
    FUS = {}
    for n, comps in ABL.items():
        FUS[n] = fit_fusion(Sva, yva, comps, CONFIG["fusion"])
        V[n] = dict(val=apply_fusion(Sva, FUS[n])[0], test=apply_fusion(Ste, FUS[n])[0], thr=FUS[n]["threshold"],
                    weights=FUS[n]["weights"])
        log(f"fusion {n}: {FUS[n]}")
    meta = LogisticRegression(max_iter=1000).fit(np.column_stack([Sva[c] for c in COMPONENTS]), yva)
    V["Full model, stacked LR meta-learner"] = dict(val=meta.predict_proba(np.column_stack([Sva[c] for c in COMPONENTS]))[:, 1],
                                                    test=meta.predict_proba(np.column_stack([Ste[c] for c in COMPONENTS]))[:, 1],
                                                    meta_coef=dict(zip(COMPONENTS, map(float, meta.coef_[0]))))
    iso_v = -Sva["iso"]; iso_t = -Ste["iso"]   # v1-style fixed fusion (decision-function scale, min-max on own data)
    mm = lambda z: (z - z.min()) / (z.max() - z.min() + 1e-12)
    V["Fixed-weight fusion 0.5/0.3/0.2 (threshold 0.5)"] = dict(val=0.5 * Sva["xgb"] + 0.3 * mm(iso_v) + 0.2 * Sva["rf"],
                                                               test=0.5 * Ste["xgb"] + 0.3 * mm(iso_t) + 0.2 * Ste["rf"], fixed=0.5)

    # ---- latency (full pipeline incl. feature encoding, Apriori + fusion), robustness, shuffle test, IF flags: need models
    lat = {}
    for bs, reps in [(1, CONFIG["aux"]["latency_reps"]), (1000, 20), (10000, 5)]:
        tt = []
        for r_ in range(reps):
            a = (r_ * 7919 * bs) % max(len(te) - bs, 1); sl = slice(a, a + bs)
            t0 = time.perf_counter(); S_ = layer_scores(L, Xte[sl], te.iloc[sl]); apply_fusion(S_, FUS["Full layered model"])
            tt.append(time.perf_counter() - t0)
        tt = np.array(tt); lat[f"batch_{bs}"] = dict(median_ms=float(np.median(tt) * 1e3), p95_ms=float(np.percentile(tt, 95) * 1e3),
                                                     alerts_per_s=float(bs / np.median(tt)))
    R["latency"] = lat; log("latency", lat); log_mem("baselines, ablation, latency done")
    flag = predict_batched(lambda z: L["iso"].predict(z) == -1, Xte).astype(bool); _, pf = apply_fusion(Ste, FUS["Full layered model"])
    R["anomaly_flags"] = dict(flagged_share=float(flag.mean()), threat_share_flagged=float(yte[flag].mean()) if flag.any() else None,
                              flagged_benign=int((flag & (yte == 0)).sum()), flagged_benign_escalated=int((flag & (yte == 0) & (pf == 1)).sum()),
                              flagged_threats=int((flag & (yte == 1)).sum()), flagged_threats_caught=int((flag & (yte == 1) & (pf == 1)).sum()))
    # robustness: blank a share of categorical / detector / MITRE fields on an incident sample of the test set
    rsel = np.flatnonzero(hash01(te.inc_key.values, 11) < CONFIG["aux"]["robustness_rows"] / max(len(te), 1))
    rob = {}
    for frac in [0.0, 0.1, 0.2, 0.3]:
        tc = te.iloc[rsel].copy(); r2 = np.random.default_rng(7)
        for c in GUIDE_CATEG + ["DetectorId", "MitreTechniques"]:
            msk = r2.random(len(tc)) < frac
            if c in GUIDE_CATEG:
                tc[c] = tc[c].astype(object); tc.loc[msk, c] = np.nan; tc[c] = tc[c].astype("category")
            elif c == "DetectorId":
                tc.loc[msk, c] = -1
            else:
                tc.loc[msk, c] = np.nan
        tc["Category"] = tc["Category"].cat.set_categories(te["Category"].cat.categories)
        S_ = layer_scores(L, enc.transform(tc), tc); _, p = apply_fusion(S_, FUS["Full layered model"])
        rob[str(frac)] = metrics(*cm_counts(tc.label.values, p))
    R["robustness_missing"] = dict(rows=int(len(rsel)), results=rob); log("robustness", {k: round(v["fpr"], 4) for k, v in rob.items()})
    # label-shuffle sanity test
    kk = subsample_idx(len(ytr), 300_000, 5)
    rfs = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, n_jobs=-1, random_state=SEED).fit(
        Xtr[kk], np.random.default_rng(3).permutation(ytr[kk]))
    ss = predict_batched(lambda z: rfs.predict_proba(z)[:, 1], Xte)
    R["shuffle_test"] = dict(roc_auc=float(roc_auc_score(yte, ss)), accuracy=float(((ss >= 0.5) == yte).mean()),
                             majority_rate=float(max(yte.mean(), 1 - yte.mean())))
    del rfs; gc.collect()

    # ---- sensitivity: Apriori thresholds, IF parameters, operating points
    s1 = []
    for ms in [0.005, 0.01, 0.02]:
        for mc in [0.5, 0.6, 0.7, 0.8]:
            cl = CorrelationLayer().fit(tr, dict(CONFIG["apriori"], min_support=ms, min_confidence=mc))
            sv = dict(Sva, corr=cl.score(va)); st = dict(Ste, corr=cl.score(te)); cov = cl.coverage_
            f = fit_fusion(sv, yva, COMPONENTS, CONFIG["fusion"]); _, p = apply_fusion(st, f)
            s1.append(dict(min_support=ms, min_confidence=mc, n_rules=len(cl.rules), coverage=cov,
                           val_auc=float(roc_auc_score(yva, sv["corr"])), weight_corr=f["weights"]["corr"],
                           **{k: v for k, v in metrics(*cm_counts(yte, p)).items() if k in ("fpr", "recall", "precision")}))
    R["sens_apriori"] = s1
    s2 = []
    for msamp in [256, 1024, 4096]:
        for ne in [100, 200]:
            iso = IsolationForest(**dict({k: v for k, v in CONFIG["iforest"].items() if k != "train_rows"},
                                         max_samples=msamp, n_estimators=ne)).fit(Xtr[subsample_idx(len(ytr), CONFIG["iforest"]["train_rows"], SEED + 1)])
            sv = dict(Sva, iso=predict_batched(lambda z: -iso.score_samples(z), Xva))
            st = dict(Ste, iso=predict_batched(lambda z: -iso.score_samples(z), Xte))
            f = fit_fusion(sv, yva, COMPONENTS, CONFIG["fusion"]); _, p = apply_fusion(st, f)
            s2.append(dict(max_samples=msamp, n_estimators=ne, val_auc=float(roc_auc_score(yva, sv["iso"])),
                           weight_iso=f["weights"]["iso"],
                           **{k: v for k, v in metrics(*cm_counts(yte, p)).items() if k in ("fpr", "recall", "precision")}))
    R["sens_iforest"] = s2
    s3 = []
    for tgt in [0.90, 0.93, 0.95, 0.97, 0.99]:
        for nm, comps in [("Full layered model", COMPONENTS), ("XGBoost only", ["xgb"]), ("Random Forest only", ["rf"])]:
            f = fit_fusion(Sva, yva, comps, dict(CONFIG["fusion"], target_recall=tgt)); _, p = apply_fusion(Ste, f)
            s3.append(dict(target_recall=tgt, model=nm, threshold=f["threshold"], **metrics(*cm_counts(yte, p))))
    R["operating_points"] = s3

    # ---- scalability: training time vs training size (main training set)
    scal = {"1.0": dict(rows=len(ytr), train_time_s=T, total_s=float(sum(T[k] for k in ("rf", "xgb", "iforest", "apriori"))),
                        **{k: v for k, v in metrics(*cm_counts(yte, (V["Full layered model"]["test"] >= FUS["Full layered model"]["threshold"]).astype(int))).items()
                           if k in ("fpr", "recall", "precision")})}
    del L; gc.collect()
    for fr in CONFIG["aux"]["scalability_fracs"]:
        ii = np.flatnonzero(hash01(tr.inc_key.values, 13) < fr)      # incident-level subsample of the training set
        Ls, Ts = train_layers(Xtr[ii], ytr[ii], Xva, yva, tr.iloc[ii], tag=f"scal {fr}")
        sv = layer_scores(Ls, Xva, va); st = layer_scores(Ls, Xte, te)
        f = fit_fusion(sv, yva, COMPONENTS, CONFIG["fusion"]); _, p = apply_fusion(st, f)
        scal[str(fr)] = dict(rows=len(ii), train_time_s=Ts, total_s=float(sum(Ts.values())), xgb_rounds=Ls["xgb_rounds"],
                             **{k: v for k, v in metrics(*cm_counts(yte, p)).items() if k in ("fpr", "recall", "precision")})
        del Ls; gc.collect()
    R["scalability"] = scal; log_mem("sensitivity and scalability done")

    # ---- ablation: complete model without engineered temporal / correlation features (retrained)
    del Xtr, Xva, Xte; gc.collect()
    enc_n = Encoder().fit(tr, drop=CONTEXT_FEATS)
    Xtr_n, Xva_n, Xte_n = enc_n.transform(tr), enc_n.transform(va), LazyX(enc_n, te)
    Ln, _ = train_layers(Xtr_n, ytr, Xva_n, yva, tr, with_corr=False, tag="no-context")
    sv = layer_scores(Ln, Xva_n); st = layer_scores(Ln, Xte_n)
    f = fit_fusion(sv, yva, ["rf", "xgb", "iso"], CONFIG["fusion"])
    V["Full model without temporal & correlation features"] = dict(val=apply_fusion(sv, f)[0], test=apply_fusion(st, f)[0],
                                                                   thr=f["threshold"], weights=f["weights"])
    del Ln, Xtr_n, Xva_n, Xte_n; gc.collect()

    # ---- evaluate every variant once on GUIDE_Test
    res, preds = {}, {}
    for n, v in V.items():
        thr = v.get("fixed", v.get("thr"))
        if thr is None:
            thr = thr_for_recall(yva, np.asarray(v["val"]), CONFIG["fusion"]["target_recall"])
        m, p = evaluate_variant(n, yte, np.asarray(v["test"], np.float32), thr, gte, rng, CONFIG["stats"]["n_boot"])
        for k in ("weights", "meta_coef"):
            if k in v: m[k] = v[k]
        res[n] = m; preds[n] = p
        log(f"TEST {n}: prec {m['precision']:.4f} rec {m['recall']:.4f} FPR {m['fpr']:.4f}")
    ref = "Full layered model"
    for n in res:
        if n != ref:
            res[n]["mcnemar_vs_full"] = mcnemar(yte, preds[ref], preds[n])
            res[n]["delta_fpr_ci95"], res[n]["delta_recall_ci95"] = paired_boot(yte, preds[ref], preds[n], gte, CONFIG["stats"]["n_boot"], rng)
    for n in [ref, "XGBoost only", "Random Forest only", "RF + XGBoost"]:
        res[n]["roc_auc_ci95"], res[n]["pr_auc_ci95"] = auc_ci(yte, np.asarray(V[n]["test"], np.float32), gte, CONFIG["stats"]["n_boot_auc"], rng)
    R["test_results"] = res
    R["routing"] = routing(yva, V[ref]["val"], yte, V[ref]["test"], CONFIG["routing"])
    seen = te.OrgId.isin(set(tr.OrgId)).values
    R["test_by_org_seen"] = dict(seen_share=float(seen.mean()),
                                 seen=metrics(*cm_counts(yte[seen], preds[ref][seen])),
                                 unseen=metrics(*cm_counts(yte[~seen], preds[ref][~seen])) if (~seen).sum() > 0 else None)
    np.savez_compressed(os.path.join(OUT, "scores_test.npz"), y=yte, groups=gte, risk_full=V[ref]["test"],
                        xgb=V["XGBoost only"]["test"], rf=V["Random Forest only"]["test"], rfxgb=V["RF + XGBoost"]["test"],
                        rule=V["Detector-reputation rule"]["test"], lr=V["Logistic regression"]["test"],
                        thr_full=FUS[ref]["threshold"], t_low=R["routing"]["t_low"], t_high=R["routing"]["t_high"])
    R["train_time_s"] = T
    jdump(R, "stage_main.json"); log("stage main done"); return R


# ---------------------------------------------------------------- auxiliary helpers
def aux_data(args, burst):
    return prepare(os.path.join(args.data_dir, CONFIG["data"]["train_file"]), CONFIG["data"]["aux_frac"], 101, burst)[0]


def run_split(tr, va, te, tag, drop=(), comps_list=None, seed=SEED):
    from sklearn.metrics import roc_auc_score
    enc = Encoder().fit(tr, drop=drop)
    Xtr, Xva, Xte = enc.transform(tr), enc.transform(va), LazyX(enc, te)
    L, T = train_layers(Xtr, tr.label.values, Xva, va.label.values, tr, seed=seed, tag=tag)
    Sv, St = layer_scores(L, Xva, va), layer_scores(L, Xte, te); del L, Xtr, Xva, Xte; gc.collect()
    out = {}
    for n, comps in (comps_list or [("Full layered model", COMPONENTS), ("RF + XGBoost", ["rf", "xgb"]),
                                    ("XGBoost only", ["xgb"]), ("Random Forest only", ["rf"])]):
        f = fit_fusion(Sv, va.label.values, comps, CONFIG["fusion"]); r, p = apply_fusion(St, f)
        out[n] = dict(**metrics(*cm_counts(te.label.values, p)), roc_auc=float(roc_auc_score(te.label.values, r)),
                      weights=f["weights"], _pred=p)
    return out


def strip_preds(d):
    return {k: {a: b for a, b in v.items() if a != "_pred"} for k, v in d.items()}


# ---------------------------------------------------------------- stage: temporal (R2.7, R2.8)
def stage_temporal(args):
    rng = np.random.default_rng(SEED); burst = get_burst(args); d = aux_data(args, burst)
    start = d.groupby("inc_key")["Timestamp"].transform("min")
    q70, q85 = start.quantile([0.70, 0.85])
    tr, va, te = d[start < q70].reset_index(drop=True), d[(start >= q70) & (start < q85)].reset_index(drop=True), d[start >= q85].reset_index(drop=True)
    del d; gc.collect()
    R = dict(train_rows=len(tr), val_rows=len(va), test_rows=len(te), train_tp_share=float(tr.label.mean()),
             test_tp_share=float(te.label.mean()), boundaries=[str(q70), str(q85)],
             unseen_org_share=float((~te.OrgId.isin(set(tr.OrgId))).mean()),
             unseen_detector_share=float((~te.DetectorId.isin(set(tr.DetectorId))).mean()))
    static = run_split(tr, va, te, "temporal")
    R["static"] = strip_preds(static)
    fus_cfg = static["Full layered model"]["weights"]
    # periodic refresh: retrain on train + validation period
    trv = pd.concat([tr, va], ignore_index=True)
    enc = Encoder().fit(trv); X1, Xv, Xt = enc.transform(trv), enc.transform(va), enc.transform(te)
    L, _ = train_layers(X1, trv.label.values, Xv, va.label.values, trv, tag="temporal-refresh")
    Sv, St = layer_scores(L, Xv, va), layer_scores(L, Xt, te)
    f = fit_fusion(Sv, va.label.values, COMPONENTS, CONFIG["fusion"]); _, p = apply_fusion(St, f)
    R["refreshed"] = metrics(*cm_counts(te.label.values, p))
    R["refreshed"]["ci95"] = cluster_boot(te.label.values, p, te.inc_key.values, 500, rng)
    R["static"]["Full layered model"]["ci95"] = cluster_boot(te.label.values, static["Full layered model"]["_pred"], te.inc_key.values, 500, rng)
    R["mcnemar_refresh_vs_static"] = mcnemar(te.label.values, p, static["Full layered model"]["_pred"])
    jdump(R, "stage_temporal.json"); log("temporal", {k: R[k] for k in ("train_tp_share", "test_tp_share")}); return R


# ---------------------------------------------------------------- stage: unseen organisations (R2.8)
def stage_orgshift(args):
    rng = np.random.default_rng(SEED); burst = get_burst(args); d = aux_data(args, burst)
    part = strat_group_parts(d["label"].values, d["OrgId"].values, [0.70, 0.15, 0.15], seed=17)
    tr, va, te = (d[part == k].reset_index(drop=True) for k in range(3)); del d; gc.collect()
    R = dict(train_orgs=int(tr.OrgId.nunique()), test_orgs=int(te.OrgId.nunique()),
             org_overlap=int(len(set(tr.OrgId) & set(te.OrgId))), test_rows=len(te), test_tp_share=float(te.label.mean()))
    a = run_split(tr, va, te, "org-out all features")
    a["Full layered model"]["ci95"] = cluster_boot(te.label.values, a["Full layered model"]["_pred"], te.inc_key.values, 500, rng)
    R["all_features"] = strip_preds(a)
    R["without_org_id"] = strip_preds(run_split(tr, va, te, "org-out w/o OrgId", drop=("OrgId", "freq_OrgId"),
                                                comps_list=[("Full layered model", COMPONENTS), ("XGBoost only", ["xgb"])]))
    R["without_tenant_entity_ids"] = strip_preds(run_split(tr, va, te, "org-out w/o entity IDs",
                                                           drop=tuple(TENANT_ENTITY_IDS + ["freq_" + c for c in TENANT_ENTITY_IDS]),
                                                           comps_list=[("Full layered model", COMPONENTS), ("XGBoost only", ["xgb"])]))
    jdump(R, "stage_orgshift.json"); return R


# ---------------------------------------------------------------- stage: leakage check (R2.5)
def stage_leakage(args):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    burst = get_burst(args); d = aux_data(args, burst); R = {}
    for name, part in [("incident_disjoint", strat_group_parts(d.label.values, d.inc_key.values, [0.85, 0.15], seed=23)),
                       ("row_level_random", (np.random.default_rng(SEED).random(len(d)) < 0.15).astype(int))]:
        tr, te = d[part == 0], d[part == 1]
        enc = Encoder().fit(tr); Xtr, Xte = enc.transform(tr), LazyX(enc, te)
        ii = subsample_idx(len(tr), CONFIG["rf"]["train_rows"])
        rf = RandomForestClassifier(**{k: v for k, v in CONFIG["rf"].items() if k != "train_rows"}).fit(Xtr[ii], tr.label.values[ii])
        s = predict_batched(lambda z: rf.predict_proba(z)[:, 1], Xte)
        R[name] = dict(roc_auc=float(roc_auc_score(te.label.values, s)), **metrics(*cm_counts(te.label.values, (s >= 0.5).astype(int))),
                       test_incidents_seen_in_train=float(te.inc_key.isin(set(tr.inc_key)).mean()))
        log("leakage", name, R[name]["roc_auc"]); del rf, Xtr, Xte; gc.collect()
    jdump(R, "stage_leakage.json"); return R


# ---------------------------------------------------------------- stage: grouped CV (R2.4)
def stage_cv(args):
    burst = get_burst(args); d = aux_data(args, burst); k = CONFIG["aux"]["cv_folds"]
    fold = strat_group_folds(d.label.values, d.inc_key.values, k, seed=31); rows = []
    for f in range(k):
        rest = d[fold != f].reset_index(drop=True); te = d[fold == f].reset_index(drop=True)
        inner = strat_group_parts(rest.label.values, rest.inc_key.values, [0.85, 0.15], seed=37) == 1
        tr, va = rest[~inner].reset_index(drop=True), rest[inner].reset_index(drop=True); del rest
        out = run_split(tr, va, te, f"cv{f}", comps_list=[("Full layered model", COMPONENTS), ("RF + XGBoost", ["rf", "xgb"]),
                                                          ("XGBoost only", ["xgb"]), ("Random Forest only", ["rf"]),
                                                          ("RF + XGBoost + Isolation Forest", ["rf", "xgb", "iso"]),
                                                          ("RF + XGBoost + Apriori", ["rf", "xgb", "corr"])])
        rows.append(strip_preds(out)); log(f"cv fold {f}", {n: round(v["fpr"], 4) for n, v in out.items()})
        jdump(dict(folds=rows), "stage_cv.json"); gc.collect()
    summ = {}
    for n in rows[0]:
        for m in ("precision", "recall", "f1", "fpr", "roc_auc"):
            v = [r[n][m] for r in rows]; summ[f"{n}|{m}"] = dict(mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)))
        if n != "Full layered model":
            t = ttest_rel([r["Full layered model"]["fpr"] for r in rows], [r[n]["fpr"] for r in rows])
            summ[f"{n}|paired_t_fpr_vs_full"] = dict(t=float(t.statistic), p=float(t.pvalue))
    jdump(dict(folds=rows, summary=summ), "stage_cv.json"); return summ


# ---------------------------------------------------------------- stage: seed variation (R2.4)
def stage_seeds(args):
    burst = get_burst(args); d = aux_data(args, burst)
    part = strat_group_parts(d.label.values, d.inc_key.values, [0.70, 0.15, 0.15], seed=41)
    tr, va, te = (d[part == k].reset_index(drop=True) for k in range(3)); del d
    rows = {}
    for s in CONFIG["aux"]["seeds"]:
        rows[str(s)] = strip_preds(run_split(tr, va, te, f"seed{s}", seed=s,
                                             comps_list=[("Full layered model", COMPONENTS), ("RF + XGBoost", ["rf", "xgb"]), ("XGBoost only", ["xgb"])]))
        jdump(rows, "stage_seeds.json")
    return rows


# ==============================================================================================
# 8. Report: tables (CSV + Excel), figures, summary markdown
# ==============================================================================================
def stage_report(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    tdir, fdir = os.path.join(OUT, "tables"), os.path.join(OUT, "figures")
    os.makedirs(tdir, exist_ok=True); os.makedirs(fdir, exist_ok=True)
    A, M, Tm, O, Lk, C, Sd = (jload(f"stage_{s}.json") for s in ("audit", "main", "temporal", "orgshift", "leakage", "cv", "seeds"))
    tables = {}
    pct = lambda x: None if x is None else round(100 * x, 3)
    ci = lambda c: f"{100*c[0]:.2f}–{100*c[1]:.2f}"
    # config
    tables["table_config"] = pd.DataFrame([(k, p, str(v)) for k, d in CONFIG.items() for p, v in d.items()],
                                          columns=["component", "parameter", "value"])
    if A:
        tables["table_dataset_audit"] = pd.DataFrame([dict(file=CONFIG["data"][f], rows=A[f]["rows"], rows_labelled=A[f]["rows_labelled"],
                                                           **{f"grade_{g}": c for g, c in A[f]["grade_counts"].items()},
                                                           tp_share_pct=pct(A[f]["tp_share"]), incidents=A[f]["incidents"],
                                                           organisations=A[f]["orgs"], detectors=A[f]["detectors"],
                                                           time_min=A[f]["time_min"], time_max=A[f]["time_max"])
                                                      for f in ("train_file", "test_file")])
        tables["table_train_test_overlap"] = pd.DataFrame([A["overlap"]])
    if M:
        tables["table_split"] = pd.DataFrame([dict(split=k, **v) for k, v in M["split"].items() if isinstance(v, dict)])
        rows = []
        for n, r in M["test_results"].items():
            rows.append(dict(model=n, precision=round(r["precision"], 4), recall=round(r["recall"], 4), f1=round(r["f1"], 4),
                             fpr_pct=pct(r["fpr"]), fpr_ci95_pct=ci(r["ci95"]["fpr"]), recall_ci95=f"{r['ci95']['recall'][0]:.3f}–{r['ci95']['recall'][1]:.3f}",
                             roc_auc=round(r.get("roc_auc", np.nan), 4), pr_auc=round(r.get("pr_auc", np.nan), 4),
                             tp=r["tp"], fp=r["fp"], fn=r["fn"], tn=r["tn"], threshold=round(r["threshold"], 4),
                             weights=json.dumps(r.get("weights", r.get("meta_coef", ""))),
                             delta_fpr_pp_ci95=(f"{100*r['delta_fpr_ci95'][0]:+.2f} to {100*r['delta_fpr_ci95'][1]:+.2f}" if "delta_fpr_ci95" in r else "reference"),
                             mcnemar_p=(r["mcnemar_vs_full"]["p"] if "mcnemar_vs_full" in r else None)))
        tables["table_test_results_all_models"] = pd.DataFrame(rows)
        full = M["test_results"]["Full layered model"]; rule = M["test_results"]["Detector-reputation rule"]
        red = []
        for n in ["Detector-reputation rule", "Forward-all (SIEM as-is)", "Random Forest only", "Logistic regression"]:
            b = M["test_results"][n]["fpr"]
            red.append(dict(baseline=n, baseline_fpr_pct=pct(b), full_fpr_pct=pct(full["fpr"]),
                            absolute_reduction_pp=round(100 * (b - full["fpr"]), 3),
                            relative_reduction_pct=round(100 * (b - full["fpr"]) / b, 2) if b else None,
                            baseline_recall=round(M["test_results"][n]["recall"], 4), full_recall=round(full["recall"], 4)))
        tables["table_fp_reduction"] = pd.DataFrame(red)
        rt = M["routing"]
        tables["table_routing"] = pd.DataFrame([dict(t_low=rt["t_low"], t_high=rt["t_high"], **{f"test_{k}": v for k, v in rt["test"].items()},
                                                     **{f"val_{k}": v for k, v in rt["val"].items()})])
        tables["table_layers"] = pd.DataFrame([dict(layer=k, test_roc_auc=M["layer_auc_test"][k], test_brier=M["brier_test"][k]) for k in M["layer_auc_test"]])
        tables["table_latency"] = pd.DataFrame([dict(batch=k, **v) for k, v in M["latency"].items()])
        tables["table_training_time"] = pd.DataFrame([dict(component=k, seconds=v) for k, v in M["train_time_s"].items()] +
                                                     [dict(component="xgb_rounds", seconds=M["xgb_rounds"]), dict(component="rf_rows", seconds=M["rf_rows"])])
        tables["table_scalability"] = pd.DataFrame([dict(train_frac=k, rows=v["rows"], total_s=v["total_s"], **{f"t_{a}": b for a, b in v["train_time_s"].items()},
                                                         fpr_pct=pct(v["fpr"]), recall=v["recall"]) for k, v in sorted(M["scalability"].items(), key=lambda z: float(z[0]))])
        tables["table_robustness"] = pd.DataFrame([dict(blanked_share=k, **v) for k, v in M["robustness_missing"]["results"].items()])
        tables["table_sens_apriori"] = pd.DataFrame(M["sens_apriori"]); tables["table_sens_iforest"] = pd.DataFrame(M["sens_iforest"])
        tables["table_operating_points"] = pd.DataFrame(M["operating_points"])
        tables["table_top_rules"] = pd.DataFrame(M["apriori"]["top_rules"])
        tables["table_feature_importance"] = pd.DataFrame([dict(model=m, feature=f, importance=v) for m in ("rf", "xgb") for f, v in M["feature_importance"][m].items()])
        tables["table_checks"] = pd.DataFrame([dict(check="label-shuffle test", **M["shuffle_test"]),
                                               dict(check="IsolationForest anomaly flags", **M["anomaly_flags"]),
                                               dict(check="test rows from orgs seen in training", seen_share=M["test_by_org_seen"]["seen_share"],
                                                    fpr_seen=M["test_by_org_seen"]["seen"]["fpr"],
                                                    fpr_unseen=(M["test_by_org_seen"]["unseen"] or {}).get("fpr"))])
    if Tm:
        tables["table_temporal"] = pd.DataFrame([dict(setting="static", model=n, **{k: v for k, v in r.items() if k not in ("weights", "ci95")}) for n, r in Tm["static"].items()] +
                                                [dict(setting="refreshed (train+val)", model="Full layered model", **{k: v for k, v in Tm["refreshed"].items() if k != "ci95"})])
    if O:
        rows = []
        for s in ("all_features", "without_org_id", "without_tenant_entity_ids"):
            for n, r in O[s].items():
                rows.append(dict(features=s, model=n, **{k: v for k, v in r.items() if k not in ("weights", "ci95")}))
        tables["table_unseen_orgs"] = pd.DataFrame(rows)
    if Lk:
        tables["table_leakage_split"] = pd.DataFrame([dict(split=k, **v) for k, v in Lk.items()])
    if C and "summary" in C:
        tables["table_cv"] = pd.DataFrame([dict(key=k, **v) for k, v in C["summary"].items()])
    if Sd:
        tables["table_seeds"] = pd.DataFrame([dict(seed=s, model=n, **{k: v for k, v in r.items() if k != "weights"}) for s, d in Sd.items() for n, r in d.items()])
    for n, t in tables.items():
        t.to_csv(os.path.join(tdir, n + ".csv"), index=False)
    try:
        with pd.ExcelWriter(os.path.join(OUT, "all_tables.xlsx")) as xw:
            for n, t in tables.items():
                t.to_excel(xw, sheet_name=n[:31], index=False)
    except Exception as e:
        log("Excel export skipped:", e)

    # ---------------- figures
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"; BLUE, ORANGE, AQUA, YEL, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED})
    if M:
        r = M["test_results"]["Full layered model"]; cm = np.array([[r["tp"], r["fn"]], [r["fp"], r["tn"]]]); nm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(4.4, 3.8)); ax.imshow(nm, cmap="Blues", vmin=0, vmax=1.15)
        lab = [["TP", "FN"], ["FP", "TN"]]
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{lab[i][j]}\n{cm[i,j]:,}\n({100*nm[i,j]:.1f}% of row)", ha="center", va="center", color="white" if nm[i, j] > .6 else INK)
        ax.set_xticks([0, 1], ["Predicted threat", "Predicted non-threat"]); ax.set_yticks([0, 1], ["Actual threat", "Actual non-threat"])
        ax.spines[:].set_visible(False); ax.tick_params(length=0); ax.set_title(f"Full layered model, GUIDE_Test (n = {cm.sum():,})")
        fig.savefig(os.path.join(fdir, "fig_confusion_matrix.png"), dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
        order = ["Random Forest only", "XGBoost only", "RF + XGBoost", "RF + XGBoost + Isolation Forest", "RF + XGBoost + Apriori",
                 "Full layered model", "Full model without temporal & correlation features", "Full model, stacked LR meta-learner", "Detector-reputation rule"]
        order = [o for o in order if o in M["test_results"]]
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True); ys = np.arange(len(order))[::-1]
        for ax, met, sc_, ttl in [(axs[0], "fpr", 100, "False-positive rate (%)"), (axs[1], "recall", 1, "Recall")]:
            for y, n in zip(ys, order):
                rr = M["test_results"][n]; v = rr[met] * sc_; lo, hi = np.array(rr["ci95"][met]) * sc_
                c = VIOLET if n == "Full layered model" else (MUTED if "rule" in n else BLUE)
                ax.plot([lo, hi], [y, y], color=c, lw=2); ax.plot(v, y, "o", color=c, ms=7, mec="white", mew=1.5)
                ax.annotate(f"{v:.2f}" if met == "fpr" else f"{v:.3f}", (hi, y), textcoords="offset points", xytext=(5, -3), fontsize=7.5)
            ax.set_title(ttl, loc="left"); ax.grid(axis="x", color=GRID)
        axs[0].set_yticks(ys, order); axs[0].set_xscale("log"); axs[0].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); axs[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        fig.tight_layout(); fig.savefig(os.path.join(fdir, "fig_ablation.png"), dpi=300, facecolor="white"); plt.close(fig)
        from sklearn.metrics import roc_curve, precision_recall_curve
        z = np.load(os.path.join(OUT, "scores_test.npz"))
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for key, c, nm_ in [("risk_full", VIOLET, "Full layered"), ("xgb", ORANGE, "XGBoost"), ("rf", BLUE, "Random Forest"),
                            ("lr", AQUA, "Logistic regression"), ("rule", MUTED, "Detector rule")]:
            f_, t_, _ = roc_curve(z["y"], z[key]); axs[0].plot(f_, t_, color=c, lw=1.8, label=nm_)
            p_, r_, _ = precision_recall_curve(z["y"], z[key]); axs[1].plot(r_, p_, color=c, lw=1.8, label=nm_)
        axs[0].set_xscale("log"); axs[0].set_xlim(1e-3, 1); axs[0].set_xlabel("False-positive rate"); axs[0].set_ylabel("True-positive rate")
        axs[1].set_xlabel("Recall"); axs[1].set_ylabel("Precision"); axs[0].legend(frameon=False, fontsize=8)
        for ax, t_ in zip(axs, ["(a) ROC (log FPR axis)", "(b) Precision–recall"]): ax.set_title(t_, loc="left"); ax.grid(color=GRID)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, "fig_roc_pr.png"), dpi=300, facecolor="white"); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        bins = np.linspace(0, 1, 81)
        ax.hist(z["risk_full"][z["y"] == 0], bins=bins, color=AQUA, alpha=.75, label="non-threat", density=True)
        ax.hist(z["risk_full"][z["y"] == 1], bins=bins, color=ORANGE, alpha=.75, label="threat", density=True)
        ax.set_yscale("log")
        for v_, n_ in [(float(z["t_low"]), r"$\tau_{low}$"), (float(z["thr_full"]), r"$\tau$"), (float(z["t_high"]), r"$\tau_{high}$")]:
            ax.axvline(v_, color=INK, lw=1, ls=":"); ax.annotate(n_, (v_, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 3),
                                                                 textcoords="offset points", ha="center", fontsize=9)
        ax.set_xlabel("Fused risk score"); ax.set_ylabel("Density (log scale)"); ax.legend(frameon=False, loc="lower center", ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, "fig_risk_distribution.png"), dpi=300, facecolor="white"); plt.close(fig)
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        sc = M["scalability"]; ks = sorted(sc, key=float); rws = [sc[k]["rows"] / 1e6 for k in ks]
        for l, c in [("rf", BLUE), ("xgb", ORANGE), ("apriori", AQUA), ("iforest", YEL)]:
            axs[0].plot(rws, [sc[k]["train_time_s"].get(l, np.nan) for k in ks], "-o", color=c, label=l)
        axs[0].plot(rws, [sc[k]["total_s"] for k in ks], "-o", color=INK, label="total")
        for x_, k in zip(rws, ks):
            axs[0].annotate(f"FPR {100*sc[k]['fpr']:.1f}%", (x_, sc[k]["total_s"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=7)
        axs[0].set_xlabel("Training records (millions)"); axs[0].set_ylabel("Training time (s)"); axs[0].legend(frameon=False, fontsize=8)
        axs[0].set_title("(a) Scalability", loc="left")
        for m_, c in [("Random Forest only", BLUE), ("XGBoost only", ORANGE), ("Full layered model", VIOLET)]:
            pts = [p for p in M["operating_points"] if p["model"] == m_]
            axs[1].plot([p["target_recall"] for p in pts], [100 * p["fpr"] for p in pts], "-o", color=c, label=m_)
        axs[1].set_yscale("log"); axs[1].yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter()); axs[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axs[1].set_xlabel("Validation recall target R*"); axs[1].set_ylabel("Test FPR (%)"); axs[1].legend(frameon=False, fontsize=8)
        axs[1].set_title("(b) Operating points", loc="left")
        for ax in axs: ax.grid(color=GRID)
        fig.tight_layout(); fig.savefig(os.path.join(fdir, "fig_scalability_operating_points.png"), dpi=300, facecolor="white"); plt.close(fig)
        fi = pd.Series(M["feature_importance"]["xgb"]).head(15)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4.2)); ax.barh(fi.index, fi.values, color=BLUE)
        ax.set_xlabel("XGBoost importance (gain share)"); fig.tight_layout()
        fig.savefig(os.path.join(fdir, "fig_feature_importance.png"), dpi=300, facecolor="white"); plt.close(fig)

    # ---------------- summary markdown
    L_ = ["# GUIDE experiment — results summary", "", f"Generated: {time.strftime('%Y-%m-%d %H:%M')}", ""]
    if A:
        L_ += ["## Data", f"- GUIDE_Train: {A['train_file']['rows']:,} rows ({A['train_file']['rows_labelled']:,} labelled), "
               f"{A['train_file']['incidents']:,} incidents, {A['train_file']['orgs']:,} orgs, TP share {100*A['train_file']['tp_share']:.2f}%",
               f"- GUIDE_Test: {A['test_file']['rows']:,} rows, {A['test_file']['incidents']:,} incidents, {A['test_file']['orgs']:,} orgs, "
               f"TP share {100*A['test_file']['tp_share']:.2f}%", f"- Overlap: {A['overlap']}", ""]
    if M:
        f_ = M["test_results"]["Full layered model"]
        L_ += ["## Main result (GUIDE_Test, used once)",
               f"- Full layered model: precision {f_['precision']:.4f}, recall {f_['recall']:.4f}, F1 {f_['f1']:.4f}, "
               f"FPR {100*f_['fpr']:.2f}% (95% CI {ci(f_['ci95']['fpr'])}%), ROC-AUC {f_.get('roc_auc', float('nan')):.4f}",
               f"- Fusion weights {f_.get('weights')}, threshold {f_['threshold']:.4f}", "", "## All models", "",
               tables["table_test_results_all_models"][["model", "precision", "recall", "fpr_pct", "fpr_ci95_pct", "roc_auc", "mcnemar_p"]].to_string(index=False), "",
               "## FP reduction", "", tables["table_fp_reduction"].to_string(index=False), ""]
    open(os.path.join(OUT, "RESULTS_SUMMARY.md"), "w").write("\n".join(L_))
    allr = {s: jload(f"stage_{s}.json") for s in ("audit", "main", "temporal", "orgshift", "leakage", "cv", "seeds")}
    jdump(allr, "results_all.json")
    log(f"report written: {len(tables)} tables in {tdir}, figures in {fdir}")


# ==============================================================================================
# 9. Main
# ==============================================================================================
STAGES = dict(audit=stage_audit, main=stage_main, temporal=stage_temporal, orgshift=stage_orgshift,
              leakage=stage_leakage, cv=stage_cv, seeds=stage_seeds, report=stage_report)


def main():
    global OUT, LOGF
    ap = argparse.ArgumentParser(description="GUIDE SOC false-positive reduction experiments (PeerJ revision)")
    ap.add_argument("--data-dir", default=".", help="folder containing GUIDE_Train.csv and GUIDE_Test.csv")
    ap.add_argument("--out", default="guide_results", help="output folder (created next to where you run the script)")
    ap.add_argument("--stages", default="audit,main,temporal,orgshift,leakage,cv,seeds,report")
    ap.add_argument("--train-frac", type=float, default=None, help="share of GUIDE_Train incidents to use (default 1.0)")
    ap.add_argument("--aux-frac", type=float, default=None, help="share of GUIDE_Train incidents for auxiliary analyses (default 0.25)")
    ap.add_argument("--resume", action="store_true", help="skip stages whose stage_*.json already exists")
    ap.add_argument("--quick", action="store_true", help="tiny smoke test (for checking the installation only)")
    a = ap.parse_args()
    if a.train_frac is not None: CONFIG["data"]["train_frac"] = a.train_frac
    if a.aux_frac is not None: CONFIG["data"]["aux_frac"] = a.aux_frac
    if a.quick:
        CONFIG["rf"]["n_estimators"] = 20; CONFIG["xgb"]["n_estimators"] = 50; CONFIG["stats"].update(n_boot=50, n_boot_auc=10)
        CONFIG["aux"].update(cv_folds=2, seeds=[1], latency_reps=10)
    OUT = os.path.abspath(a.out); os.makedirs(OUT, exist_ok=True)
    LOGF = open(os.path.join(OUT, "run.log"), "a")
    import sklearn, xgboost, mlxtend, scipy
    env = dict(python=sys.version, platform=platform.platform(), cpu_count=os.cpu_count(), ram_total_gb=mem_gb()[0],
               numpy=np.__version__, pandas=pd.__version__, scikit_learn=sklearn.__version__, xgboost=xgboost.__version__,
               mlxtend=mlxtend.__version__, scipy=scipy.__version__, args=vars(a), config=CONFIG)
    jdump(env, "environment.json")
    log("environment:", {k: v for k, v in env.items() if k not in ("config",)})
    for s in [x.strip() for x in a.stages.split(",") if x.strip()]:
        if a.resume and s != "report" and os.path.exists(os.path.join(OUT, f"stage_{s}.json")):
            log(f"skip {s} (done)"); continue
        t0 = time.time(); log(f"===== stage {s} =====")
        STAGES[s](a); gc.collect()
        log(f"===== stage {s} finished in {(time.time() - t0) / 60:.1f} min =====")
    log("ALL DONE. Results in", OUT)


if __name__ == "__main__":
    main()
