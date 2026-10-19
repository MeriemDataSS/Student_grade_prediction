# -*- coding: utf-8 -*-
# %%
Intro="""
Aix-Marseille University - Machine Learning - Project 
Predicting student exam scores (regression)

Structure of this script:
    0. Setup and data loading
    1. Exploratory Data Analysis (EDA)
    2. Section 1.1 - Implementing and comparing the algorithms
    3. Section 1.2.1 - Feature engineering (Part A) and variable selection (Part B)
    4. Section 1.2.2 - Hyper-parameter selection (two models)
    5. Section 1.3 - Selection and evaluation of the best model
    6. Section 1.4 - Ethical considerations

Protocol used everywhere
    * One single train/test split (75% / 25%, random_state=42).
    * Every choice (hyper-parameters, variables) is made with 5-fold cross-validation
      on the TRAIN set only. The test set is only used to report performance
      and to draw the "test error vs. choice" .
    * Metrics: R2 and MSE on the test set, training time, main hyper-parameters.

Run:  python student_score_project.py     (figures are saved in ./figures)
"""
print(Intro)

# %% ---------------------------------------------------------------------------
# 0. SETUP AND DATA LOADING
# ------------------------------------------------------------------------------
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (KFold, cross_val_score, learning_curve,
                                     train_test_split)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (OneHotEncoder, PolynomialFeatures,
                                   SplineTransformer)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# >>> Change this path if needed (the file has a title row, hence skiprows=1)
DATA_PATH = os.environ.get("DATA_PATH", "data_exam_score.xlsx")
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid", context="notebook", palette="deep")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


def save_fig(name):
    """Save the current figure in ./figures and display it."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.show()


df = pd.read_excel(DATA_PATH, skiprows=1)
df = df.drop("student_id", axis=1)          # identifier: no predictive meaning

TARGET = "exam_score"
CAT_RAW = ["sleep_quality", "facility_rating", "exam_difficulty",
           "gender", "study_method", "course", "internet_access"]
NUM_RAW = ["age", "study_hours", "class_attendance", "sleep_hours"]
ALL_RAW = CAT_RAW + NUM_RAW

# %% ---------------------------------------------------------------------------
# 1. EXPLORATORY DATA ANALYSIS
# ------------------------------------------------------------------------------
print("=" * 90)
print("1. EXPLORATORY DATA ANALYSIS")
print("=" * 90)

# ---- 1.1 Structure and quality of the data -----------------------------------
print(f"\nShape: {df.shape[0]} students, {df.shape[1] - 1} explanatory variables + target")
print("\nTypes:\n", df.dtypes.to_string())
print("\nMissing values per column:\n", df.isna().sum().to_string())
print("\nDuplicated rows:", df.duplicated().sum())
print("\nNumerical summary:\n", df[NUM_RAW + [TARGET]].describe().T.round(2))
print("\nCategorical summary:")
for c in CAT_RAW:
    print(f"  {c:16s}", (df[c].value_counts(normalize=True) * 100).round(1).to_dict())

n_cap = int((df[TARGET] == 100).sum())
n_floor = int((df[TARGET] == df[TARGET].min()).sum())
print(f"\nStudents with exam_score = 100 (ceiling): {n_cap} ({n_cap / len(df):.1%})")
print(f"Students at the minimum score {df[TARGET].min():.1f} (floor): {n_floor} ({n_floor / len(df):.1%})")
print(f"Skewness of exam_score: {df[TARGET].skew():.3f}")
print(f"kurtosis: {df[TARGET].kurt():.3f}")

# RESULTS - data quality
comment1="""
Comment 1 : 
--------

* 20,000 students, 11 explanatory variables, no missing values and no duplicated.
* rows: no imputation or cleaning is needed (student_id is dropped: identifier).
* Categorical variables are almost perfectly balanced (each gender ~33%, each facility rating ~33%,
    each course ~14%, each study method ~20%).
* Only exam_difficulty (49% moderate / 31% easy / 20% hard) and internet_access (85% yes) are unbalanced. 
* No resampling or stratified split is needed.
* Exam_score is CENSORED on both sides: 
    200 students (1.0%) sit exactly on the minimum 19.6 and 493 students (2.5%) exactly on the maximum 100. 
"""
print(comment1)

# ---- 1.2 Target distribution ---------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df[TARGET], bins=40, kde=True, ax=ax[0], color="#4C72B0")
mu, sd = df[TARGET].mean(), df[TARGET].std()
xs = np.linspace(df[TARGET].min(), df[TARGET].max(), 200)
ax[0].plot(xs, stats.norm.pdf(xs, mu, sd) * len(df) * (df[TARGET].max() - df[TARGET].min()) / 40,
           "r--", lw=1.5, label="Normal fit")
ax[0].axvline(mu, color="k", ls=":", label=f"mean = {mu:.1f}")
ax[0].set_title("Distribution of exam_score")
ax[0].legend()

sns.boxplot(x=df[TARGET], ax=ax[1], color="#55A868")
ax[1].set_title("Boxplot of exam_score")
stats.probplot(df[TARGET], dist="norm", plot=ax[2])
ax[2].set_title("Normal Q-Q plot of exam_score")
save_fig("eda01_target_distribution.png")



comment2="""
Comment 2 : 
--------
# RESULTS - target distribution (figure eda01)

* Mean 62.5, standard deviation 18.9; almost symmetric (skewness = -0.05) and 
    relatively flatter than the normal distribution (kurtosis = -0.62).
* The histogram has two spikes at 19.6 and at 100,this is probably due to the way the data were 
    generated or recorded rather than reflecting a natural phenomenon. 
* The Q-Q plot follows the normal line in the middle and leaves it only in the
    two tails for the same reason.
* No transformation of the target is needed; there is no outlier in the usualsense 
    (the box plot has no point beyond the whiskers).
"""
print(comment2)


# ---- 1.3 Explanatory variables: distributions ---------------------------------
# Numerical variables 
fig, ax = plt.subplots(1, 4, figsize=(17, 3.6))
for a, c in zip(ax, NUM_RAW):
    sns.histplot(df[c], bins=25, ax=a, color="#4C72B0")
    a.set_title(f"{c}  (mean={df[c].mean():.1f})")
save_fig("eda02_numeric_distributions.png")

# categorical variables 
fig, ax = plt.subplots(2, 4, figsize=(17, 7))
for a, c in zip(ax.ravel(), CAT_RAW):
    order = df[c].value_counts().index
    sns.countplot(x=df[c], order=order, ax=a, color="#8172B3")
    a.set_title(f"{c}")
    a.set_xlabel("")
    a.tick_params(axis="x", rotation=25)
ax.ravel()[-1].axis("off")
save_fig("eda03_categorical_counts.png")

comment3="""
Comment 3 : 

# RESULTS - explanatory variables (figures eda02, eda03)

* The numerical variables are close to UNIFORM (age 17-24, study_hours 0-8 h,
    class_attendance 41-99%, sleep_hours 4-10 h), not bell-shaped as in real surveys. 
    Together with the perfect balance of categories this suggests a
    clean (probably simulated) because real dataset would be messier.
* No scaling is required for OLS / splines / forests; no extreme values to treat.
"""
print(comment3)


# ---- 1.4 Target vs categorical variables --------------------------------------
fig, ax = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
for a, c in zip(ax.ravel(), CAT_RAW):
    order = df.groupby(c)[TARGET].mean().sort_values().index
    sns.boxplot(x=c, y=TARGET, data=df, order=order, ax=a, palette="Set2", showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="red", markeredgecolor="k"))
    a.set_title(f"exam_score by {c}")
    a.set_xlabel("")
    a.tick_params(axis="x", rotation=25)
ax.ravel()[-1].axis("off")
fig.suptitle("Exam score by categorical variable (red diamond = mean)", y=1.01, fontweight="bold")
save_fig("eda04_target_by_categories.png")

comment4="""
Comment 4 :
# RESULTS - target vs explanatory variables (figures eda04)

#  Categorical variables (mean exam_score by group):
*   study_method: coaching 68.6 | mixed 64.2 | group study 61.0 | online videos 60.2 |self-study 58.7 
        => 10-point gap between the best and the worst method.
*   sleep_quality: good 67.1 | average 62.5 | poor 57.9 -> 9-point gap.
*   facility_rating: high 66.2 | medium 62.8 | low 58.6 -> 7.6-point gap.
*   gender:, course, internet_access, exam_difficulty: group means differ by less than 1 point -> no visible effect. 
      (e.g. gender: 62.4 / 62.6 / 62.6) 
*   It is surprising that exam_difficulty has no effect at all.
"""
print(comment4)






# ---- 1.5 Target vs numerical variables (scatter + binned mean) -----------------
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for a, c in zip(ax, NUM_RAW):
    a.scatter(df[c], df[TARGET], s=4, alpha=0.15, color="grey")
    bins = pd.cut(df[c], 12)
    m = df.groupby(bins, observed=True).agg(x=(c, "mean"), y=(TARGET, "mean"))
    a.plot(m.x, m.y, "o-", color="crimson", label="binned mean")
    coef = np.polyfit(df[c], df[TARGET], 1)
    a.plot(m.x, np.polyval(coef, m.x), "b--", label=f"linear fit (slope={coef[0]:.2f})")
    a.set_xlabel(c)
    a.set_ylabel(TARGET)
    a.set_title(f"exam_score vs {c}")
    a.legend(fontsize=8)
save_fig("eda05_target_vs_numeric.png")



# ---- 1.6 Correlation matrix ----------------------------------------------------
corr = df[NUM_RAW + [TARGET]].corr()
plt.figure(figsize=(6.5, 5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, square=True)
plt.title("Pearson correlation (numerical variables)")
save_fig("eda06_correlation_heatmap.png")
print("\nCorrelation with exam_score:\n", corr[TARGET].drop(TARGET).round(3).to_string())


comment5="""
Comment 5 :
# RESULTS - target vs explanatory variables (figures eda05, eda06)
#  Numerical variables:

*   study_hours: the binned means form an almost perfect STRAIGHT LINE
        (about +5.9 points per extra hour, correlation 0.72) -> the strongest driver.
*   class_attendance: linear too, about +0.34 point per attendance point
        (+3.4 points for 10 more points), correlation 0.31.
*   sleep_hours: weak positive linear effect (+1.4 point per hour, correlation 0.13).
*   age: flat line (correlation 0.007) -> useless.

*   The four numerical variables are NOT correlated with each other (all |r| < 0.02),
*   so there is no multicollinearity and each variable brings its own information.
"""
print(comment5)

# ---- 1.7 Strength of association of EVERY variable with the target -------------
def eta_squared(frame, cat, y=TARGET):
    """Share of the variance of y explained by a categorical variable (ANOVA eta^2)."""
    grand = frame[y].mean()
    ssb = frame.groupby(cat)[y].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    return ssb / ((frame[y] - grand) ** 2).sum()


assoc = []
for c in NUM_RAW:
    r, p = stats.pearsonr(df[c], df[TARGET])
    assoc.append((c, "numeric", r ** 2, p))
    
for c in CAT_RAW:
    groups = [g[TARGET].values for _, g in df.groupby(c)]
    _, p = stats.f_oneway(*groups)
    assoc.append((c, "categorical", eta_squared(df, c), p))
    
assoc = pd.DataFrame(assoc, columns=["variable", "type", "var_explained", "p_value"])
assoc = assoc.sort_values("var_explained", ascending=False).reset_index(drop=True)
assoc["significant_5pct"] = assoc.p_value < 0.05
print("\nUnivariate association with exam_score (share of variance explained):\n",
      assoc.round(4).to_string(index=False))

plt.figure(figsize=(8, 4.8))
colors = ["#C44E52" if not s else "#4C72B0" for s in assoc.significant_5pct]
plt.barh(assoc.variable[::-1], assoc.var_explained[::-1] * 100, color=colors[::-1])
plt.xlabel("% of exam_score variance explained (univariate)")
plt.title("Which variables matter? (blue = significant at 5%, red = not)")
save_fig("eda07_variable_association.png")

comment6="""
Comment  :
# RESULTS - strength of the association with exam_score (figure eda07)

*  Share of the variance of exam_score explained by each variable alone:
*   study_hours 51.5% | class_attendance 9.5% | sleep_quality 4.0% |
*   study_method 3.5% | facility_rating 2.7% | sleep_hours 1.8%.

#  These six variables are significant (p < 0.001). Their shares add up to about
     73%, which is exactly the R2 reached later by the regression models: because the
*    predictors are independent, their contributions simply add up.
*   course, internet_access, age, exam_difficulty and gender explain 0.0% - 0.02% of the variance 
    and are not significant (p-values between 0.27 and 0.83).
*    They will be tested as candidates for removal in section 1.2.1 (Part B).
"""
print(comment6)


# ---- 1.8 Interactions: are the slopes different between groups? ---------------
# This plot justifies (or not) the interaction features built in section 1.2.1.
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
pairs = [("study_hours", "study_method"), ("class_attendance", "facility_rating"),
         ("sleep_hours", "sleep_quality"), ("class_attendance", "gender")]
for a, (xv, hv) in zip(ax, pairs):
    for lvl, g in df.groupby(hv):
        b = pd.cut(g[xv], 8)
        m = g.groupby(b, observed=True).agg(x=(xv, "mean"), y=(TARGET, "mean"))
        a.plot(m.x, m.y, "o-", label=str(lvl), ms=4)
    a.set_xlabel(xv)
    a.set_ylabel("mean exam_score")
    a.set_title(f"{xv} x {hv}")
    a.legend(fontsize=8, title=hv)
save_fig("eda08_interactions.png")

slopes = []
for xv, hv in pairs:
    for lvl, g in df.groupby(hv):
        slopes.append({"x": xv, "group_var": hv, "group": lvl,
                       "slope": np.polyfit(g[xv], g[TARGET], 1)[0]})
slopes = pd.DataFrame(slopes)
print("\nSlope of exam_score on x within each group (equal slopes = no interaction):\n",
      slopes.round(3).to_string(index=False))

comment7="""
Comment  :
# RESULTS - interactions (figure eda08)

*  The curves of the different groups are parallel, and the slopes are practically
   identical: study_hours slope = 5.8-5.9 whatever the study method, attendance slope
      = 0.33-0.34 whatever the facility rating, sleep_hours slope = 1.39-1.48 whatever
    the sleep quality, attendance slope = 0.32-0.35 for the three genders.
  -> The effects are ADDITIVE: no interaction between these variables. We expect the

* interaction features of section 1.2.1 not to improve the predictions. 
"""
print(comment7)


# ---- 1.9 Combination of the two categorical drivers ---------------------------
pivot = df.pivot_table(index="study_method", columns="facility_rating",
                       values=TARGET, aggfunc="mean")[["low", "medium", "high"]]
pivot2 = df.pivot_table(index="study_method", columns="sleep_quality",
                        values=TARGET, aggfunc="mean")[["poor", "average", "good"]]
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[0])
ax[0].set_title("Mean score: study_method x facility_rating")
sns.heatmap(pivot2, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax[1])
ax[1].set_title("Mean score: study_method x sleep_quality")
save_fig("eda09_category_heatmaps.png")


comment8="""
Comment  :
# RESULTS - what the EDA tells us (figure eda09)

*  The heat-maps confirm that categorical effects simply ADD UP: the best cell is
       coaching + high facilities (72.0) ,coaching + good sleep_quality (73.8)
       and the worst is self-study + low facilities (54.1),self-study + poor sleep_quality (54.4).
       
       and every row/column shows the same ordering.
       
#  EDA conclusions:
1. The score is, to a very good approximation, a LINEAR and ADDITIVE function of
     6 variables (study_hours >> class_attendance > sleep_quality, study_method,
     facility_rating > sleep_hours).
2. Five variables (age, gender, course, internet_access, exam_difficulty) carry
      no information. 
3. No curvature, no interaction visible.
4. The scores are censored at 19.6 and 100.

*  Expectation: a simple model (OLS) should be very hard to beat, while flexible
      models (high-degree polynomial, random forest) should overfit the noise.

"""
print(comment8)

# %% ---------------------------------------------------------------------------
# COMMON TOOLS: split, pipelines, evaluation
# -------------------------------------------------------------------------------
# ---- Feature engineering functions (used from section 1.2.1 on) ---------------
ORD_QUALITY = {"poor": 0, "average": 1, "good": 2}
ORD_FACILITY = {"low": 0, "medium": 1, "high": 2}

X = df.drop(TARGET, axis=1).copy()
y = df[TARGET]

# New features (Part A of section 1.2.1). Each one is justified by the EDA above.
X["study_hours_sq"] = X.study_hours ** 2                        # curvature of study effect
X["study_x_attendance"] = X.study_hours * X.class_attendance / 100       # "effective" study
X["sleep_quality_score"] = X.sleep_quality.map(ORD_QUALITY)              # ordinal 0-1-2
X["facility_score"] = X.facility_rating.map(ORD_FACILITY)        # ordinal 0-1-2
X["is_coaching"] = (X.study_method == "coaching").astype(int)    # best method vs all others
X["sleep_x_quality"] = X.sleep_hours * X.sleep_quality_score  # hours slept x how well
X["attendance_x_facility"] = X.class_attendance * X_all.facility_score
X["study_x_coaching"] = X.study_hours * X_all.is_coaching

# Interactions of the first version of the project (kept to check whether they help)
X["att_x_male"] = X.class_attendance * (X.gender == "male")
X["att_x_other"] = X.class_attendance * (X.gender == "other")
X["att_x_fac_low"] = X.class_attendance * (X.facility_rating == "low")
X["att_x_fac_medium"] = X.class_attendance * (X.facility_rating == "medium")
X["study_x_mixed"] = X.study_hours * (X.study_method == "mixed")
X["study_x_online"] = X.study_hours * (X.study_method == "online videos")

FE_FAMILIES = {
    "study_hours_sq": ["study_hours_sq"],
    "study_x_attendance": ["study_x_attendance"],
    "ordinal scores (sleep_quality, facility)": ["sleep_quality_score", "facility_score"],
    "is_coaching": ["is_coaching"],
    "sleep_x_quality": ["sleep_x_quality"],
    "attendance_x_facility": ["attendance_x_facility"],
    "study_x_coaching": ["study_x_coaching"],
    "attendance_x_gender (v1)": ["att_x_male", "att_x_other"],
    "attendance_x_facility dummies (v1)": ["att_x_fac_low", "att_x_fac_medium"],
    "study_x_method dummies (v1)": ["study_x_mixed", "study_x_online"],
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE)

kf5 = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")


def build_pipeline(model, cols, degree=None, n_knots=None, **rf_params):
    """
    Build a full pipeline (preprocessing + estimator) for the chosen columns.
      - categorical columns -> OneHotEncoder(drop='first')
      - 'spline' model: continuous numeric columns -> SplineTransformer (piecewise
        polynomial basis), binary/dummy columns pass through unchanged
      - other models: numeric columns pass through
    model in {'ols', 'poly', 'spline', 'rf'}
    """
    cat = [c for c in cols if X[c].dtype == object or str(X[c].dtype) in ("str", "string")]
    num = [c for c in cols if c not in cat]
    steps = []
    if cat:
        steps.append(("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat))
    if model == "spline":
        cont = [c for c in num if X_all[c].nunique() > 2]
        binary = [c for c in num if c not in cont]
        if cont:
            steps.append(("spl", SplineTransformer(degree=degree, n_knots=n_knots), cont))
        if binary:
            steps.append(("bin", "passthrough", binary))
    elif num:
        steps.append(("num", "passthrough", num))
    pre = ColumnTransformer(steps)

    if model == "ols":
        return make_pipeline(pre, LinearRegression())
    if model == "poly":
        return make_pipeline(pre, PolynomialFeatures(degree=degree), LinearRegression())
    if model == "spline":
        return make_pipeline(pre, LinearRegression())
    if model == "rf":
        params = dict(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        params.update(rf_params)
        return make_pipeline(pre, RandomForestRegressor(**params))
    raise ValueError(model)


def evaluate(pipe, cols, cv=True):
    """Fit on train, return train/test metrics, fit time and 5-fold CV MSE (train only)."""
    Xtr, Xte = X_train[cols], X_test[cols]
    t0 = time.perf_counter()
    pipe.fit(Xtr, y_train)
    fit_time = time.perf_counter() - t0
    p_tr, p_te = pipe.predict(Xtr), pipe.predict(Xte)
    out = {
        "R2 Train": r2_score(y_train, p_tr),
        "R2 Test": r2_score(y_test, p_te),
        "MSE Train": mean_squared_error(y_train, p_tr),
        "MSE Test": mean_squared_error(y_test, p_te),
        "Time (s)": fit_time,
    }
    if cv:
        out["CV MSE"] = -cross_val_score(clone(pipe), Xtr, y_train, cv=kf5,
                                         scoring="neg_mean_squared_error").mean()
    return out


# %% ---------------------------------------------------------------------------
# 2. SECTION 1.1 - IMPLEMENTING THE ALGORITHMS (all 11 original variables)
# ------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("2. SECTION 1.1 - MODEL COMPARISON (same 11 explanatory variables for every model)")
print("=" * 90)

rows = []


def add_row(name, params, res):
    rows.append({"Model": name, "Hyper-parameters": params, **res})


# ---- 2.1 OLS -------------------------------------------------------------------
add_row("OLS", "none", evaluate(build_pipeline("ols", ALL_RAW), ALL_RAW))

# ---- 2.2 Polynomial regression (degree 1, 2, 3) --------------------------------
for d in [1, 2, 3]:
    add_row(f"Polynomial (d={d})", f"degree={d}",
            evaluate(build_pipeline("poly", ALL_RAW, degree=d), ALL_RAW))

# ---- 2.3 Piecewise / spline regression: grid degree x number of knots ----------
for deg in [1, 2, 3]:
    for k in [2, 3, 4, 5, 6]:
        add_row(f"Spline (d={deg}, knots={k})", f"degree={deg}, n_knots={k}",
                evaluate(build_pipeline("spline", ALL_RAW, degree=deg, n_knots=k), ALL_RAW))

# ---- 2.4 Random forest (default hyper-parameters, 100 trees) -------------------
add_row("Random Forest", "n_estimators=100, max_features=1.0, min_samples_leaf=1",
        evaluate(build_pipeline("rf", ALL_RAW), ALL_RAW, cv=False))

res11 = pd.DataFrame(rows)
res11.to_csv("results_1_1_all_models.csv", index=False)

# Summary table: one line per family (best spline chosen by CV MSE on the train set)
best_spline = res11[res11.Model.str.startswith("Spline")].sort_values("CV MSE").iloc[0]
summary11 = pd.concat([
    res11[res11.Model.isin(["OLS", "Polynomial (d=1)", "Polynomial (d=2)", "Polynomial (d=3)"])],
    best_spline.to_frame().T,
    res11[res11.Model == "Random Forest"]
]).reset_index(drop=True)
cols_show = ["Model", "R2 Test", "MSE Test", "Time (s)", "Hyper-parameters"]
print("\n--- COMPARISON TABLE (Section 1.1) ---")
print(summary11[cols_show].round({"R2 Test": 4, "MSE Test": 3, "Time (s)": 3}).to_string(index=False))
print("\nFull spline grid sorted by CV MSE (top 5):")
print(res11[res11.Model.str.startswith("Spline")].sort_values("CV MSE")
      [["Model", "R2 Test", "MSE Test", "CV MSE", "Time (s)"]].head(5).round(3).to_string(index=False))

# Figure: test MSE of all models
plot11 = summary11.copy()
plot11["MSE Test"] = plot11["MSE Test"].astype(float)
plt.figure(figsize=(9, 4.2))
bars = plt.barh(plot11.Model[::-1], plot11["MSE Test"][::-1],
                color=sns.color_palette("deep", len(plot11))[::-1])
for b, v in zip(bars, plot11["MSE Test"][::-1]):
    plt.text(v + 0.3, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
plt.xlim(plot11["MSE Test"].min() - 5, plot11["MSE Test"].max() + 8)
plt.xlabel("Test MSE (lower is better)")
plt.title("Section 1.1 - Test MSE of all models (11 original variables)")
save_fig("m01_model_comparison.png")

# Figure: spline heat-map (test MSE) and polynomial degree (train vs test)
sp = res11[res11.Model.str.startswith("Spline")].copy()
sp["degree"] = sp["Hyper-parameters"].str.extract(r"degree=(\d)").astype(int)
sp["knots"] = sp["Hyper-parameters"].str.extract(r"n_knots=(\d)").astype(int)
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.heatmap(sp.pivot(index="degree", columns="knots", values="MSE Test").astype(float),
            annot=True, fmt=".2f", cmap="viridis_r", ax=ax[0])
ax[0].set_title("Spline: test MSE (degree x n_knots)")
po = res11[res11.Model.str.startswith("Polynomial")]
ax[1].plot([1, 2, 3], po["MSE Train"].astype(float), "o-", label="Train MSE")
ax[1].plot([1, 2, 3], po["MSE Test"].astype(float), "s-", label="Test MSE")
ax[1].set_xticks([1, 2, 3])
ax[1].set_xlabel("Polynomial degree")
ax[1].set_ylabel("MSE")
ax[1].set_title("Polynomial regression: train vs test MSE")
ax[1].legend()
save_fig("m02_spline_poly_curves.png")

comment9="""
Comment  :

*     OLS is the best or tied for best. The best spline selected by cross-validation is
         degree 1 with 2 knots, i.e. a straight line: it is the same model as OLS.
*     Making the polynomial more flexible makes the test error WORSE (95.9 -> 97.0 -> 106.4), so:
         there is no curvature to capture, so the extra columns only fit noise.
*     All 15 spline configurations lie within 0.2 MSE points of each other
         (95.76-95.90): adding knots or degrees is useless here.
*     The random forest is the worst model (MSE 112.3) and 40 times slower than OLS: 
          a forest approximates a straight line with steps and needs a lot of data to do it;
*     it also wastes splits on the five useless variables.
*     Ranking: OLS = Poly1 = Spline(1,2) < other splines < Poly2 < Poly3 < Random forest.
"""
print(comment9)

# %% ---------------------------------------------------------------------------
# 3. SECTION 1.2.1 - FEATURE ENGINEERING AND VARIABLE SELECTION
# ------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("3. SECTION 1.2.1 - FEATURE ENGINEERING AND VARIABLE SELECTION")
print("=" * 90)

# Model of our choice for this section: SPLINE regression with the hyper-parameters
# selected by CV in section 1.1. OLS is reported next to it as a reference, because
# the spline family contains OLS as a special case (degree=1, 2 knots).

SP_DEG = int(best_spline["Hyper-parameters"].split("degree=")[1].split(",")[0])
SP_KNOTS = int(best_spline["Hyper-parameters"].split("n_knots=")[1])
print(f"Model of choice: Spline regression (degree={SP_DEG}, n_knots={SP_KNOTS})")


def eval_both(cols):
    """Evaluate the model of choice (spline) and OLS on a set of columns."""
    r_sp = evaluate(build_pipeline("spline", cols, degree=SP_DEG, n_knots=SP_KNOTS), cols)
    r_ols = evaluate(build_pipeline("ols", cols), cols)
    return r_sp, r_ols


# ---- 3.A  Part B first: variable selection among the ORIGINAL variables --------
RELEVANT = list(assoc[assoc.significant_5pct].variable)
print("\nVariables significant at 5% in the EDA:", RELEVANT)

subsets = {
    "All 11 original variables": ALL_RAW,
    "Drop age": [c for c in ALL_RAW if c != "age"],
    "Drop age, gender": [c for c in ALL_RAW if c not in ("age", "gender")],
    "Drop age, gender, course": [c for c in ALL_RAW if c not in ("age", "gender", "course")],
    "Drop age, gender, course, exam_difficulty":
        [c for c in ALL_RAW if c not in ("age", "gender", "course", "exam_difficulty")],
    "Relevant variables only (EDA)": RELEVANT,
    "Quantitative habits only": ["study_hours", "class_attendance", "sleep_hours"],
    "study_hours only": ["study_hours"],
}
rows = []
for name, cols in subsets.items():
    r_sp, r_ols = eval_both(cols)
    rows.append({"Variable set": name, "n_vars": len(cols),
                 "Spline test MSE": r_sp["MSE Test"], "Spline CV MSE": r_sp["CV MSE"],
                 "OLS test MSE": r_ols["MSE Test"], "OLS CV MSE": r_ols["CV MSE"]})
sel_table = pd.DataFrame(rows)
print("\n--- Variable selection: combinations of original variables ---")
print(sel_table.round(3).to_string(index=False))

# Leave-one-out from the relevant set: what is the cost of removing each variable?
rows = []
ref = evaluate(build_pipeline("ols", RELEVANT), RELEVANT)["MSE Test"]
for v in RELEVANT:
    cols = [c for c in RELEVANT if c != v]
    r = evaluate(build_pipeline("ols", cols), cols)
    rows.append({"Removed variable": v, "OLS test MSE": r["MSE Test"],
                 "Increase vs relevant set": r["MSE Test"] - ref})
loo = pd.DataFrame(rows).sort_values("Increase vs relevant set", ascending=False)
print("\n--- Cost of removing each relevant variable (OLS) ---")
print(loo.round(3).to_string(index=False))

# Greedy forward selection (criterion: CV MSE on train; we also track the test MSE)
remaining, chosen, path = list(ALL_RAW), [], []
while remaining:
    cand = []
    for v in remaining:
        cols = chosen + [v]
        cvm = -cross_val_score(build_pipeline("ols", cols), X_train[cols], y_train, cv=kf5,
                               scoring="neg_mean_squared_error").mean()
        cand.append((cvm, v))
    cvm, v = min(cand)
    chosen.append(v)
    remaining.remove(v)
    r = evaluate(build_pipeline("ols", chosen), chosen, cv=False)
    path.append({"step": len(chosen), "added": v, "CV MSE": cvm, "Test MSE": r["MSE Test"],
                 "Train MSE": r["MSE Train"]})
path = pd.DataFrame(path)
print("\n--- Forward selection path (OLS) ---")
print(path.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
ax[0].plot(path.step, path["Train MSE"], "o--", label="Train MSE", alpha=0.7)
ax[0].plot(path.step, path["CV MSE"], "^-", label="CV MSE (train)")
ax[0].plot(path.step, path["Test MSE"], "s-", label="Test MSE")
ax[0].set_xticks(path.step)
ax[0].set_xticklabels([f"{s}\n+{a}" for s, a in zip(path.step, path.added)],
                      rotation=60, ha="right", fontsize=8)
ax[0].set_ylabel("MSE")
ax[0].set_title("Forward selection: error vs number of variables")
ax[0].legend()
zoom = path[path.step >= 4]
ax[1].plot(zoom.step, zoom["CV MSE"], "^-", label="CV MSE (train)")
ax[1].plot(zoom.step, zoom["Test MSE"], "s-", label="Test MSE")
ax[1].set_xticks(zoom.step)
ax[1].set_title("Zoom on steps 4 to 11 (the curve is flat after 6 variables)")
ax[1].legend()
save_fig("m03_forward_selection.png")

fig, ax = plt.subplots(figsize=(9, 4.2))
y_pos = np.arange(len(sel_table))
ax.barh(y_pos - 0.2, sel_table["OLS test MSE"], 0.4, label="OLS")
ax.barh(y_pos + 0.2, sel_table["Spline test MSE"], 0.4, label="Spline")
ax.set_yticks(y_pos)
ax.set_yticklabels(sel_table["Variable set"])
ax.invert_yaxis()
ax.set_xlim(90, sel_table[["OLS test MSE", "Spline test MSE"]].values.max() + 5)
ax.set_xlabel("Test MSE")
ax.set_title("Test MSE for different combinations of original variables")
ax.legend()
save_fig("m04_variable_sets.png")

comment10="""
Comments : 
# RESULTS - Part B: variable selection (figures m03, m04)
*    Removing the 5 useless variables lowers both the CV MSE (96.05 -> 95.95) and the
         test MSE (95.90 -> 95.75). The gain is small, but it is consistent and gives a
         simpler model (6 variables instead of 11).
         
*    Removing an important variable is very costly (increase of test MSE):
         study_hours +191.7 | class_attendance +32.9 | sleep_quality +15.4 |
         study_method +11.8 | facility_rating +10.2 | sleep_hours +6.3.
         
*    Forward selection: test MSE goes 172.6 -> 139.9 -> 123.9 -> 112.3 -> 102.1 -> 95.75
         with the first six variables, then the curve is FLAT. Adding exam_difficulty, age,
         gender, internet_access and course makes the test MSE rise from 95.75 to 95.90
         while the train MSE only falls from 95.76 to 95.71: the beginning of overfitting.
         
*    The "elbow" is clearly at 6 variables.
*    Habits alone (study_hours, attendance, sleep_hours) are not enough (MSE 132.8):
         the three categorical variables explain ~37 MSE points.
         
*    The spline column is identical to the OLS column because the spline selected in

*    section 1.1 (degree 1, 2 knots) is a linear model.
"""
print(comment10)

# ---- 3.B  Part A: feature engineering on top of the relevant variables ---------
base_cols = list(RELEVANT)
rows = []
r_sp, r_ols = eval_both(base_cols)
rows.append({"Added features": "(none) relevant variables", "n_new": 0,
             "Spline test MSE": r_sp["MSE Test"], "Spline CV MSE": r_sp["CV MSE"],
             "OLS test MSE": r_ols["MSE Test"], "OLS CV MSE": r_ols["CV MSE"]})
for fam, newcols in FE_FAMILIES.items():
    cols = base_cols + newcols
    r_sp, r_ols = eval_both(cols)
    rows.append({"Added features": fam, "n_new": len(newcols),
                 "Spline test MSE": r_sp["MSE Test"], "Spline CV MSE": r_sp["CV MSE"],
                 "OLS test MSE": r_ols["MSE Test"], "OLS CV MSE": r_ols["CV MSE"]})
    
# every engineered feature together
all_new = [c for f in FE_FAMILIES.values() for c in f]
r_sp, r_ols = eval_both(base_cols + all_new)
rows.append({"Added features": "ALL engineered features", "n_new": len(all_new),
             "Spline test MSE": r_sp["MSE Test"], "Spline CV MSE": r_sp["CV MSE"],
             "OLS test MSE": r_ols["MSE Test"], "OLS CV MSE": r_ols["CV MSE"]})
fe_table = pd.DataFrame(rows)
print("\n--- Part A: effect of each engineered feature (added to the relevant variables) ---")
print(fe_table.round(3).to_string(index=False))

ref_sp = fe_table.loc[0, "Spline CV MSE"]
plt.figure(figsize=(9, 5))
d_cv = (fe_table["Spline CV MSE"] - ref_sp)[1:]
d_te = (fe_table["Spline test MSE"] - fe_table.loc[0, "Spline test MSE"])[1:]
yy = np.arange(len(d_cv))
plt.barh(yy - 0.2, d_cv, 0.4, label="CV MSE change")
plt.barh(yy + 0.2, d_te, 0.4, label="Test MSE change")
plt.axvline(0, color="k", lw=1)
plt.yticks(yy, fe_table["Added features"][1:])
plt.gca().invert_yaxis()
plt.xlabel("Change in MSE vs. relevant variables only (negative = improvement)")
plt.title("Part A - Does each engineered feature improve the spline model?")
plt.legend()
save_fig("m05_feature_engineering.png")

# Parsimony rule: among all candidate sets, keep the SIMPLEST one whose CV MSE is
# within 0.1 MSE points of the best CV MSE (differences below that are pure noise).
candidates = {"relevant variables": base_cols}
for fam, newcols in FE_FAMILIES.items():
    candidates[f"relevant + {fam}"] = base_cols + newcols
candidates["relevant + ALL engineered"] = base_cols + all_new
cv_scores = {}
for name, cols in candidates.items():
    cv_scores[name] = -cross_val_score(
        build_pipeline("spline", cols, degree=SP_DEG, n_knots=SP_KNOTS),
        X_train[cols], y_train, cv=kf5, scoring="neg_mean_squared_error").mean()
best_cv = min(cv_scores.values())
ok = [n for n, s in cv_scores.items() if s <= best_cv + 0.1]
chosen_name = min(ok, key=lambda n: len(candidates[n]))
FINAL_COLS = candidates[chosen_name]
print(f"\nBest CV MSE = {best_cv:.3f}. Simplest set within 0.1 of it: '{chosen_name}'")
print("FINAL VARIABLES:", FINAL_COLS)

comment11="""
Comment : 
# RESULTS - Part B: feature engineering (figure m05)

*     14 new columns in 10 families were tested (7 new ideas + the 3 interaction families
      of our first version). None of them improves the model:

        - Changes of the test MSE are between -0.035 and +0.022 (best: study_x_coaching,
              95.715 vs 95.750) and the CV MSE is unchanged or slightly worse (95.945 -
              95.986 vs 95.948). These differences are far below the noise level.
              
*     All 14 features together: CV MSE 96.03 (worse than the 6 original variables).
         study_hours_sq gives 95.746 vs 95.750: no curvature, as seen in the EDA.
         study_x_attendance, sleep_x_quality, attendance_x_facility, study_x_coaching:
         no interaction, as seen in the parallel slopes of figure eda08.
         
*     The interactions with gender of our first version (attendance x gender) are
         useless too, which is logical since gender has no effect on the score.
*     The ordinal scores (sleep_quality_score, facility_score) and is_coaching are
         linear combinations of the dummy variables, so they give exactly the same MSE.
         They are still useful to INTERPRET the model: the coefficients are
         poor/average/good = -4.7/0/+4.6 and low/medium/high = -7.8/-3.9/0, i.e. almost
         equally spaced, so the effect is nearly linear in the level. They could replace
         4 dummies by 2 numeric columns with the same accuracy.
         
*     Decision (parsimony rule: simplest set within 0.1 of the best CV MSE):
         FINAL VARIABLES = the 6 relevant original variables, without engineered features.
         This is a legitimate and honest result: when the data-generating process is linear
         and additive, adding features can only add noise.
"""
print(comment11)

# %% ---------------------------------------------------------------------------
# 4. SECTION 1.2.2 - HYPER-PARAMETER SELECTION (two models)
# ------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("4. SECTION 1.2.2 - HYPER-PARAMETER OPTIMISATION")
print("=" * 90)

# ---- 4.1 Model 1: Spline regression -------------------------------------------
# Key hyper-parameters: degree of the polynomial pieces and number of knots.
print("\nModel 1: Spline regression - key hyper-parameters: degree, n_knots")
rows = []
for deg in [1, 2, 3, 4]:
    for k in [2, 3, 4, 5, 6, 8, 10, 15]:
        r = evaluate(build_pipeline("spline", FINAL_COLS, degree=deg, n_knots=k), FINAL_COLS)
        rows.append({"degree": deg, "n_knots": k, "Train MSE": r["MSE Train"],
                     "CV MSE": r["CV MSE"], "Test MSE": r["MSE Test"], "R2 Test": r["R2 Test"],
                     "Time (s)": r["Time (s)"]})
sp_grid = pd.DataFrame(rows)
sp_best = sp_grid.sort_values("CV MSE").iloc[0]
sp_grid.to_csv("results_1_2_2_spline_grid.csv", index=False)
print(sp_grid.sort_values("CV MSE").head(8).round(3).to_string(index=False))
print(f"\nBest spline (by CV): degree={int(sp_best.degree)}, n_knots={int(sp_best.n_knots)} "
      f"-> CV MSE={sp_best['CV MSE']:.3f}, test MSE={sp_best['Test MSE']:.3f}")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
sns.heatmap(sp_grid.pivot(index="degree", columns="n_knots", values="CV MSE"),
            annot=True, fmt=".2f", cmap="viridis_r", ax=ax[0])
ax[0].set_title("Spline: CV MSE (train set)")
sns.heatmap(sp_grid.pivot(index="degree", columns="n_knots", values="Test MSE"),
            annot=True, fmt=".2f", cmap="viridis_r", ax=ax[1])
ax[1].set_title("Spline: test MSE")
for deg, g in sp_grid.groupby("degree"):
    ax[2].plot(g.n_knots, g["Test MSE"], "o-", label=f"degree {deg}")
ax[2].set_xlabel("n_knots")
ax[2].set_ylabel("Test MSE")
ax[2].set_title("Spline: test MSE vs number of knots")
ax[2].legend()
save_fig("m06_spline_hyperparameters.png")

# ---- 4.2 Model 2: Random forest -------------------------------------------------
# Key hyper-parameters: max_features (decorrelation of trees), min_samples_leaf
# (smoothness / regularisation) and n_estimators (variance reduction).
print("\nModel 2: Random forest - key hyper-parameters: max_features, min_samples_leaf, n_estimators")
rf_kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
rows = []
for mf in [2, 4, 6, 8, 11]:
    for leaf in [1, 5, 10, 20, 50]:
        pipe = build_pipeline("rf", FINAL_COLS, max_features=mf, min_samples_leaf=leaf)
        cvm = -cross_val_score(pipe, X_train[FINAL_COLS], y_train, cv=rf_kf,
                               scoring="neg_mean_squared_error").mean()
        r = evaluate(pipe, FINAL_COLS, cv=False)
        rows.append({"max_features": mf, "min_samples_leaf": leaf, "CV MSE": cvm,
                     "Test MSE": r["MSE Test"], "R2 Test": r["R2 Test"],
                     "Train MSE": r["MSE Train"], "Time (s)": r["Time (s)"]})
rf_grid = pd.DataFrame(rows)
rf_best = rf_grid.sort_values("CV MSE").iloc[0]
rf_grid.to_csv("results_1_2_2_rf_grid.csv", index=False)
print(rf_grid.sort_values("CV MSE").head(8).round(3).to_string(index=False))
print(f"\nBest RF (by CV): max_features={int(rf_best.max_features)}, "
      f"min_samples_leaf={int(rf_best.min_samples_leaf)} -> CV MSE={rf_best['CV MSE']:.3f}, "
      f"test MSE={rf_best['Test MSE']:.3f}")

# number of trees, other parameters fixed at their best values
rows = []
for n_est in [10, 25, 50, 100, 200, 300]:
    pipe = build_pipeline("rf", FINAL_COLS, n_estimators=n_est,
                          max_features=int(rf_best.max_features),
                          min_samples_leaf=int(rf_best.min_samples_leaf))
    r = evaluate(pipe, FINAL_COLS, cv=False)
    rows.append({"n_estimators": n_est, "Test MSE": r["MSE Test"], "Train MSE": r["MSE Train"],
                 "Time (s)": r["Time (s)"]})
rf_trees = pd.DataFrame(rows)
print("\nEffect of the number of trees:\n", rf_trees.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 4, figsize=(21, 4.2))
sns.heatmap(rf_grid.pivot(index="max_features", columns="min_samples_leaf", values="CV MSE"),
            annot=True, fmt=".1f", cmap="viridis_r", ax=ax[0])
ax[0].set_title("Random forest: CV MSE")
sns.heatmap(rf_grid.pivot(index="max_features", columns="min_samples_leaf", values="Test MSE"),
            annot=True, fmt=".1f", cmap="viridis_r", ax=ax[1])
ax[1].set_title("Random forest: test MSE")
for mf, g in rf_grid.groupby("max_features"):
    ax[2].plot(g.min_samples_leaf, g["Test MSE"], "o-", label=f"max_features={mf}")
ax[2].set_xlabel("min_samples_leaf")
ax[2].set_ylabel("Test MSE")
ax[2].set_title("RF: test MSE vs min_samples_leaf")
ax[2].legend(fontsize=7)
ax[3].plot(rf_trees.n_estimators, rf_trees["Test MSE"], "s-", label="Test MSE")
ax[3].plot(rf_trees.n_estimators, rf_trees["Train MSE"], "o--", label="Train MSE")
ax[3].set_xlabel("n_estimators")
ax[3].set_title("RF: error vs number of trees")
ax[3].legend()
save_fig("m07_rf_hyperparameters.png")

comment12="""
Comments :

# RESULTS - Section 1.2.2: hyper-parameters (figures m06, m07)

#  MODEL 1 - SPLINE (degree in 1-4, n_knots in 2-15, 32 combinations)

   * The minimum of the CV MSE (95.95) is reached for degree=1, n_knots=2 = a straight
     line. More knots / higher degrees decrease the TRAIN MSE (95.76 -> 95.4) and give
     nearly identical TEST MSE (95.56-95.75 for the best ones), but the CV MSE
    increases. These differences are about 0.2 MSE points (0.2%): noise.
    
  * Very flexible splines (15 knots) lose a little accuracy: test MSE 96.0-96.1 and CV
    MSE 96.2-96.3, with the lowest train MSE (95.4). This is the typical start of
     overfitting. The hyper-parameter selected by CV is the simplest one, as the
     principle of parsimony recommends. Over the whole grid the test MSE only moves
     between 95.56 and 96.13 (R2 = 0.734-0.735).
     
#  MODEL 2 - RANDOM FOREST (max_features, min_samples_leaf, n_estimators)

   * Best by CV: max_features=4, min_samples_leaf=5 -> CV MSE 107.5, test MSE 105.9
     (R2 = 0.707), versus 112.3 with the default values: tuning gains 6.4 MSE points.
   * min_samples_leaf is the key parameter: with fully grown trees (leaf = 1) the train
     MSE is 15 while the test MSE is 109-114 (the forest memorises the noise); leaves of
     5-10 observations regularise (train MSE about 60-83) and give the best results;
     with leaf = 50 the trees become too coarse (underfitting: train MSE 109, test 116).
   * max_features = 4 is the best value (mean test MSE over the grid: 108.5), while 2 is
     too restrictive (113.0) and 8-11 (almost all columns) make the trees too similar
     (111-112): a moderate value decorrelates the trees without starving them.
   * More trees help with diminishing returns: test MSE 110.6 (10 trees), 105.9 (100),
     105.1 (300), while the training time grows linearly (0.13 s -> 3.8 s).
   * Even tuned, the forest remains about 10 MSE points WORSE than the linear model:
     the problem is the model class (steps vs a straight line), not the tuning.
"""
print(comment12)

# %% ---------------------------------------------------------------------------
# 5. SECTION 1.3 - SELECTION AND EVALUATION OF THE BEST MODEL
# ------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("5. SECTION 1.3 - BEST MODEL")
print("=" * 90)

# Final comparison of all the tuned models on the final variable set
tuned = {
    "OLS": (build_pipeline("ols", FINAL_COLS), "none"),
    "Polynomial (d=2)": (build_pipeline("poly", FINAL_COLS, degree=2), "degree=2"),
    f"Spline (tuned)": (build_pipeline("spline", FINAL_COLS, degree=int(sp_best.degree),
                                       n_knots=int(sp_best.n_knots)),
                        f"degree={int(sp_best.degree)}, n_knots={int(sp_best.n_knots)}"),
    "Random Forest (tuned)": (build_pipeline("rf", FINAL_COLS, max_features=int(rf_best.max_features),
                                             min_samples_leaf=int(rf_best.min_samples_leaf)),
                              f"max_features={int(rf_best.max_features)}, "
                              f"min_samples_leaf={int(rf_best.min_samples_leaf)}, n_estimators=100"),
}
rows, fitted = [], {}
for name, (pipe, hp) in tuned.items():
    r = evaluate(pipe, FINAL_COLS, cv=(name != "Random Forest (tuned)"))
    fitted[name] = pipe
    rows.append({"Model": name, "Hyper-parameters": hp, **r})
final_table = pd.DataFrame(rows)
# For the random forest the CV MSE comes from the 3-fold grid search of section 4.2
final_table.loc[final_table.Model == "Random Forest (tuned)", "CV MSE"] = rf_best["CV MSE"]
final_table.to_csv("results_final_models.csv", index=False)
print("\n--- Final comparison (tuned models, final variables) ---")
print(final_table[["Model", "R2 Train", "R2 Test", "MSE Test", "CV MSE", "Time (s)",
                   "Hyper-parameters"]].round(4).to_string(index=False))

# The best model is chosen on CV MSE (computed on the train set only)
best_name = final_table.sort_values("CV MSE").iloc[0]["Model"]
best_pipe = fitted[best_name]
print(f"\nBEST MODEL (lowest CV MSE): {best_name}")
print(f"Variables: {FINAL_COLS}")

p_tr, p_te = best_pipe.predict(X_train[FINAL_COLS]), best_pipe.predict(X_test[FINAL_COLS])
print(f"R2  train = {r2_score(y_train, p_tr):.4f} | R2  test = {r2_score(y_test, p_te):.4f}")
print(f"MSE train = {mean_squared_error(y_train, p_tr):.3f} | MSE test = {mean_squared_error(y_test, p_te):.3f}")
print(f"RMSE test = {np.sqrt(mean_squared_error(y_test, p_te)):.2f} points | "
      f"MAE test = {np.mean(np.abs(y_test - p_te)):.2f} points")

# ---- 5.1 Predicted vs actual -----------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5.2), sharex=True, sharey=True)
for a, (yt, yp, lab) in zip(ax, [(y_train, p_tr, "Train"), (y_test, p_te, "Test")]):
    a.scatter(yt, yp, s=6, alpha=0.25, color="#4C72B0")
    a.plot([20, 100], [20, 100], "r--", label="perfect prediction")
    a.set_title(f"{lab}: predicted vs actual  (R2={r2_score(yt, yp):.3f}, "
                f"MSE={mean_squared_error(yt, yp):.1f})")
    a.set_xlabel("Actual exam_score")
    a.set_ylabel("Predicted exam_score")
    a.legend()
save_fig("b01_predicted_vs_actual.png")

# ---- 5.2 Residual diagnostics ------------------------------------------------------
res_te = y_test - p_te
fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
ax[0].scatter(p_te, res_te, s=6, alpha=0.25, color="#55A868")
ax[0].axhline(0, color="r", ls="--")
ax[0].set_xlabel("Predicted score")
ax[0].set_ylabel("Residual (actual - predicted)")
ax[0].set_title("Residuals vs predicted (test)")
sns.histplot(res_te, bins=40, kde=True, ax=ax[1], color="#C44E52")
ax[1].set_title(f"Residual distribution (std={res_te.std():.2f})")
stats.probplot(res_te, dist="norm", plot=ax[2])
ax[2].set_title("Normal Q-Q plot of residuals")
save_fig("b02_residual_diagnostics.png")

# ---- 5.3 Which variables drive the prediction? (permutation importance, test set)
pi = permutation_importance(best_pipe, X_test[FINAL_COLS], y_test, n_repeats=10,
                            random_state=RANDOM_STATE, scoring="neg_mean_squared_error")
imp = pd.DataFrame({"variable": FINAL_COLS, "MSE increase": pi.importances_mean,
                    "std": pi.importances_std}).sort_values("MSE increase", ascending=False)
print("\nPermutation importance (increase of test MSE when the variable is shuffled):\n",
      imp.round(2).to_string(index=False))
plt.figure(figsize=(8, 4.2))
plt.barh(imp.variable[::-1], imp["MSE increase"][::-1], xerr=imp["std"][::-1], color="#4C72B0")
plt.xlabel("Increase of test MSE when the variable is shuffled")
plt.title(f"Permutation importance - {best_name}")
save_fig("b03_permutation_importance.png")

# Average effect of each variable (only meaningful for the linear-type models)
ols_final = fitted["OLS"]
coefs = pd.Series(ols_final[-1].coef_, index=ols_final[0].get_feature_names_out()).sort_values()
print("\nOLS coefficients (points of exam_score), final variables:\n", coefs.round(2).to_string())

# ---- 5.4 WEAKNESSES of the model ----------------------------------------------------
print("\n--- Weaknesses analysis ---")
# (a) error by predicted level
bins = pd.cut(p_te, [0, 40, 50, 60, 70, 80, 90, 100, 130])
wk = pd.DataFrame({"res": res_te.values, "bin": bins}).groupby("bin", observed=True)["res"].agg(
    mean_residual="mean", rmse=lambda s: np.sqrt((s ** 2).mean()), n="count")
print("Residual by predicted-score range:\n", wk.round(2).to_string())

# (b) censored scores: the target is cut at the floor (19.6) and at the ceiling (100)
n_out = int((p_te > 100).sum())
print(f"\nPredictions above 100 on the test set: {n_out}")
lo_s, hi_s = y.min(), 100.0
p_clip = np.clip(p_te, lo_s, hi_s)
print(f"Test MSE without clipping: {mean_squared_error(y_test, p_te):.3f} | "
      f"with predictions clipped to [{lo_s:.1f}, 100]: {mean_squared_error(y_test, p_clip):.3f}")
is_floor, is_ceil = (y_test <= lo_s).values, (y_test >= hi_s).values
print(f"Students at the floor (n={is_floor.sum()}): mean prediction = {p_te[is_floor].mean():.1f} | "
      f"students at the ceiling (n={is_ceil.sum()}): mean prediction = {p_te[is_ceil].mean():.1f}")
unc = ~(is_floor | is_ceil)
print(f"Test MSE on uncensored students only (n={unc.sum()}): "
      f"{mean_squared_error(y_test[unc], p_te[unc]):.3f}")
# 95% prediction interval for one student (residuals are ~ normal, homoscedastic)
print(f"Approx. 95% prediction interval for a student: +/- {1.96 * res_te.std():.1f} points")

# (c) irreducible-noise ceiling: the residual variance is not reduced by any model
print(f"\nResidual std (test) = {res_te.std():.2f} points -> a model can hardly do better "
      f"than R2 = {r2_score(y_test, p_te):.3f} with the available variables")

# (d) error by subgroup (also used in the ethics section)
grp_rows = []
for c in ["gender", "course", "internet_access", "facility_rating", "study_method"]:
    tmp = pd.DataFrame({"g": X_test[c].values, "res": res_te.values})
    for lvl, g in tmp.groupby("g"):
        grp_rows.append({"variable": c, "group": lvl, "n": len(g), "mean_residual": g.res.mean(),
                         "RMSE": np.sqrt((g.res ** 2).mean())})
grp = pd.DataFrame(grp_rows)
print("\nResiduals by subgroup (test set):\n", grp.round(2).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(17, 4.4))
ax[0].bar(range(len(wk)), wk.mean_residual, color="#4C72B0")
ax[0].set_xticks(range(len(wk)))
ax[0].set_xticklabels([str(i) for i in wk.index], rotation=45, ha="right")
ax[0].axhline(0, color="k")
ax[0].set_ylabel("mean residual")
ax[0].set_title("Bias by predicted-score range")
ax[1].bar(range(len(wk)), wk.rmse, color="#DD8452")
ax[1].set_xticks(range(len(wk)))
ax[1].set_xticklabels([str(i) for i in wk.index], rotation=45, ha="right")
ax[1].set_ylabel("RMSE")
ax[1].set_title("Error size by predicted-score range")
g_plot = grp[grp.variable.isin(["gender", "course", "internet_access"])]
ax[2].barh(g_plot.variable + " = " + g_plot.group.astype(str), g_plot.mean_residual, color="#8172B3")
ax[2].axvline(0, color="k")
ax[2].set_xlabel("mean residual (test)")
ax[2].set_title("Bias by subgroup (fairness check)")
save_fig("b04_weaknesses.png")

# (e) learning curve: do we need more data or a more complex model?
sizes, tr_sc, va_sc = learning_curve(clone(best_pipe), X_train[FINAL_COLS], y_train, cv=kf5,
                                     train_sizes=np.linspace(0.05, 1.0, 8),
                                     scoring="neg_mean_squared_error")
plt.figure(figsize=(7.5, 4.3))
plt.plot(sizes, -tr_sc.mean(1), "o-", label="Train MSE")
plt.plot(sizes, -va_sc.mean(1), "s-", label="CV MSE")
plt.xlabel("Number of training students")
plt.ylabel("MSE")
plt.title(f"Learning curve - {best_name}")
plt.legend()
save_fig("b05_learning_curve.png")

# Final overview figure of all tuned models
plt.figure(figsize=(8.5, 3.8))
ft = final_table.sort_values("MSE Test", ascending=False)
plt.barh(ft.Model, ft["MSE Test"], color=sns.color_palette("deep", len(ft)))
for i, v in enumerate(ft["MSE Test"]):
    plt.text(v + 0.2, i, f"{v:.2f}", va="center")
plt.xlim(ft["MSE Test"].min() - 3, ft["MSE Test"].max() + 5)
plt.xlabel("Test MSE")
plt.title("Final comparison of tuned models (final variable set)")
save_fig("b06_final_comparison.png")

comment13="""
Comments :
# RESULTS - Section 1.3: best model

#  BEST MODEL: linear regression (OLS; the tuned spline is the same model)
     on 6 variables: study_hours, class_attendance, sleep_hours, sleep_quality,study_method, facility_rating.

#  Why this model?

   * Algorithm: lowest CV MSE (95.95) and lowest test MSE (95.75). All richer models
     (polynomial d=2: 95.81, random forest: 105.9) are equal or worse. The EDA showed
     an additive linear relationship, which is exactly the OLS hypothesis.
   * Hyper-parameters: none for OLS; the spline hyper-parameters selected by CV collapse
     to the linear case. Training time: 0.02 s.
   * Variables: the 6 variables with a significant effect; the 5 others only add noise.

#  Performance:
   * R2 train = 0.7313 | R2 test = 0.7347 -> no overfitting (the test value is even
     slightly higher by chance).  MSE train = 95.76 | MSE test = 95.75.
   * RMSE = 9.79 points and MAE = 7.87 points on a 0-100 scale.
   * For comparison the tuned random forest has R2 train = 0.814 but R2 test = 0.707:
     a 0.107 overfitting gap.
   * Predicted vs actual (figure b01): the points are spread symmetrically around the
     diagonal in train and test, with the same shape: a constant band, not a curve.

#  Interpretation (OLS coefficients and permutation importance, figure b03):

   * +5.9 points per extra study hour; +0.34 per attendance point; +1.45 per hour of
     sleep; good vs poor sleep quality: +9.3; coaching vs self-study: +9.5; high vs
     low facilities: +7.8.
   * Increase of the test MSE when a variable is shuffled: study_hours 376 |
     class_attendance 69 | sleep_quality 30 | study_method 24 | facility_rating 19 |
     sleep_hours 12.5. Study time alone dominates the prediction.

#  WEAKNESSES of the model:

   1. LARGE IRREDUCIBLE ERROR. 27% of the variance is not explained (RMSE 9.8 points).
      The residuals are symmetric, nearly normal and have the same spread (RMSE 9.4-10.1
      for predicted scores below 90; smaller above because of the ceiling) in every range
      of predicted score (figures b02, b04): they look like pure noise
      (motivation, prior level, luck...). The learning curve (b05) is already flat:
      more data or a more complex model will NOT help; only NEW variables (previous
      grades, for instance) could.
   2. CENSORED TARGET. Scores are cut at 19.6 and 100, but a linear model does not know
      it: 18 test students get a predicted score above 100 (mean residual -5.7 in that
      range), the 53 students at the floor are predicted around 31 on average and the
      107 students at the ceiling around 91. Clipping the predictions to [19.6, 100]
      lowers the test MSE from 95.75 to 95.64 at no cost. On the 4,840 uncensored test
      students the MSE is 93.7, which shows that censoring inflates the error a little.
      The principled solution is a censored (Tobit) regression.
   3. WIDE INDIVIDUAL UNCERTAINTY. A 95% prediction interval is +/- 19 points: a student
      predicted at 62 could realistically score anywhere between 43 and 81. The model is
      good for describing groups and trends but not for judging one person.
   4. FAIRNESS. The mean residual is between -0.17 and +0.47 point and the RMSE between
      9.5 and 10.2 for all genders, courses, internet-access and study-method groups: no
      visible systematic bias. The model does not use gender, age or course.
   5. ASSOCIATION IS NOT CAUSATION. The model describes correlations in this sample
      (study_hours may reflect motivation, for example). The very regular data also
      suggest a simulated dataset: on real data, performance would probably be lower and
      non-linearities more likely.
"""
print(comment13)

# %% ---------------------------------------------------------------------------
# 6. SECTION 1.4 - ETHICAL CONSIDERATIONS
# ------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("6. SECTION 1.4 - ETHICAL CONSIDERATIONS (see comments below for the discussion)")
print("=" * 90)

ETHICS = """
SECTION 1.4 - ETHICAL CONSIDERATIONS

1) Should universities use such a model to recruit students?
   NO, not as a decision tool.
   - The model is built on variables measured AFTER enrolment (class attendance, study
     hours, study method, facility rating). They do not exist for an applicant, so the
     model cannot even be applied to admissions without being rebuilt on other data.
   - Individual predictions are imprecise (95% interval of about +/- 19 points): rejecting
     or accepting someone on such a score would be unfair to many people.
   - Risk of discrimination: our model does not use gender, age or course (they carry no
     information, and the residuals show no visible bias between these groups), but an
     admission model would be tempted to use proxies (internet access, school, area)
     that encode social disadvantage and would reproduce existing inequalities. A model
     that selects "students who will succeed" can also create a self-fulfilling prophecy.
   - Legal framework: in the EU, GDPR (article 22) restricts decisions based solely on
     automated processing, and the AI Act classifies AI systems used for access or
     admission to education as HIGH-RISK systems (transparency, human oversight, audits).
   - If used at all: only as a decision-support tool, with human review, published
     criteria, an appeal procedure and regular fairness audits.

2) Should students know their predicted scores?
   YES, under conditions.
   - Advantages: transparency (students have a right to understand how data about them
     are used), early warning, and the possibility to act (the model shows that study
     time, attendance, sleep and method matter).
   - Risks: anxiety or discouragement for students predicted low, stereotype threat and
     self-fulfilling prophecy, and false precision (R2 = 0.73 does not mean that
     the score of one student is known: +/- 19 points).
   - Recommended practice: voluntary access, a RANGE rather than one number, an explanation
     of the factors and of what the student can change, support and guidance attached
     to the prediction, and no sharing of individual predictions with third parties.

3) Should universities use such a model to identify key variables for success and
   plan interventions?
   YES, this is the most defensible use, with precautions.
   - The model points to levers on which the university can act: study time (+5.9 points
     per hour), attendance (+0.34 per point), sleep quality (good vs poor: +9.3),
     coaching (+9.5 vs self-study) and facilities (+7.8 between low and high).
     Support can be targeted at at-risk students (tutoring, coaching, attendance follow-up).
   - Caution 1 - causality: the model shows association, not causation. Interventions must be
     piloted and evaluated (randomised or A/B tests) before being generalised.
   - Caution 2 - equity: a low facility_rating is a resource problem of the institution, not
     a student's fault: the answer is to invest, not to blame.
   - Caution 3 - privacy and surveillance: collecting sleep or study-time data is intrusive.
     Use consent, data minimisation (the five useless variables do not need to be collected)
     and keep the purpose limited to support, never to sanction.
"""
print(ETHICS)
