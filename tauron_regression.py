"""
TAURON (TPE) — analiza regresji wzgledem pozostalych spolek z GPW.

Skrypt:
  1. Wczytuje wszystkie dzienne notowania ze stooq (format .txt) z katalogu data/.
  2. Liczy logarytmiczne stopy zwrotu.
  3. Wybiera 4 spolki najsilniej skorelowane z TAURON-em.
  4. Testuje stacjonarnosc (ADF), usuwa outliery |r| > 5 %.
  5. Diagnozuje wspolliniowosc (Farrar-Glauber, Haitovsky, VIF).
  6. Estymuje OLS i raportuje testy resztowe (DW, Breusch-Pagan, ACF, Q-Q, ...).
  7. Prognozuje ostatnie 10 % obserwacji (MAE / RMSE / sMAPE).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


DATA_DIR = Path("data")
MAIN_TICKER_FILE = DATA_DIR / "tpe.txt"
TOP_N = 4
OUTLIER_THRESHOLD = 0.05


# 1) LOAD DATA AND COMPUTE LOG RETURNS -------------------------------------
def load_log_returns(path: Path) -> pd.Series | None:
    """Zwraca logarytmiczne stopy zwrotu z pliku stooq (.txt). None gdy pusty/uszkodzony."""
    if path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "<DATE>" not in df.columns or "<CLOSE>" not in df.columns or len(df) < 2:
        return None
    df["<DATE>"] = pd.to_datetime(df["<DATE>"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["<DATE>"]).set_index("<DATE>").sort_index()
    price = df["<CLOSE>"].astype(float)
    return np.log(price).diff().rename(path.stem.upper())


tauron_returns = load_log_returns(MAIN_TICKER_FILE).dropna().rename("TAURON")

other_returns = []
for f in sorted(DATA_DIR.iterdir()):
    if f.suffix.lower() != ".txt" or f.name == MAIN_TICKER_FILE.name:
        continue
    s = load_log_returns(f)
    if s is not None:
        other_returns.append(s.dropna())

returns_df = pd.concat([tauron_returns] + other_returns, axis=1)
# Wymagamy minimalnej liczby wspolnych obserwacji, by korelacja byla sensowna.
min_obs = 250
returns_df = returns_df.dropna(axis=1, thresh=min_obs).dropna()
print("returns_df shape:", returns_df.shape)


# 2) SELECT VARIABLES BY CORRELATION ---------------------------------------
corr_with_tauron = returns_df.corr()["TAURON"].abs().sort_values(ascending=False)
print("\nRanking korelacji z TAURON-em (top 8):")
print(corr_with_tauron.head(8))

selected_vars = corr_with_tauron.drop("TAURON").index[:TOP_N].tolist()
print("\nWybrane zmienne objasniajace:", selected_vars)


# 3) AUGMENTED DICKEY-FULLER (ADF) TEST ------------------------------------
def ensure_stationary(series: pd.Series, name: str, alpha: float = 0.05) -> pd.Series:
    pvalue = adfuller(series, autolag="AIC")[1]
    if pvalue > alpha:
        print(f"{name}: niestacjonarny (p={pvalue:.3f}) -> roznicowanie")
        return series.diff().dropna().rename(name)
    print(f"{name}: stacjonarny (p={pvalue:.3f})")
    return series.rename(name)


print("\n=== Test ADF ===")
stationary = {"TAURON": ensure_stationary(returns_df["TAURON"], "TAURON")}
for var in selected_vars:
    stationary[var] = ensure_stationary(returns_df[var], var)

stationary_df = pd.concat(stationary.values(), axis=1).dropna()
print("stationary_df shape:", stationary_df.shape)
display(stationary_df.head())


# 3a) REMOVE OUTLIERS BEYOND +/- 5% ----------------------------------------
mask = (stationary_df.abs() <= OUTLIER_THRESHOLD).all(axis=1)
stationary_df = stationary_df.loc[mask]
print(f"Po usunieciu obserwacji |r| > {OUTLIER_THRESHOLD * 100:.1f}%: {stationary_df.shape[0]} wierszy")


# 4) PLOT STATIONARY SERIES ------------------------------------------------
stationary_df[selected_vars].plot(subplots=True, figsize=(8, 6), marker="o", linestyle="-")
plt.suptitle("Stacjonarne serie: top 4 zmienne", y=1.02)
plt.tight_layout()
plt.show()

stationary_df.plot(subplots=True, figsize=(10, 8), marker="o", linestyle="-")
plt.suptitle("Stacjonarne serie: TAURON + top 4", y=1.02)
plt.tight_layout()
plt.show()

corr_stat = stationary_df.corr()
print("\nMacierz korelacji (po stacjonaryzacji):")
print(corr_stat)

plt.figure(figsize=(6, 5))
sns.heatmap(corr_stat, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Macierz korelacji - dane stacjonarne")
plt.show()


# 4a) LINEARITY CHECK: TAURON vs Each Explanatory Variable -----------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, var in zip(axes.ravel(), selected_vars):
    sns.regplot(
        x=stationary_df[var],
        y=stationary_df["TAURON"],
        ax=ax,
        scatter_kws=dict(alpha=0.4),
        line_kws=dict(color="red"),
    )
    ax.set_xlabel(f"{var} (log returns)")
    ax.set_ylabel("TAURON (log returns)")
    ax.set_title(f"TAURON vs {var}")
plt.tight_layout()
plt.suptitle("Sprawdzenie liniowosci: TAURON vs zmienne objasniajace", y=1.02)
plt.show()


# 5a) MULTICOLLINEARITY TEST: Farrar-Glauber & Haitovsky -------------------
R = stationary_df[selected_vars].corr()
det_R = np.linalg.det(R)
n_obs = stationary_df.shape[0]
k_vars = len(selected_vars)
df_chi = k_vars * (k_vars - 1) / 2

FG_stat = -(n_obs - 1 - (2 * k_vars + 5) / 6) * np.log(det_R)
FG_pvalue = 1 - stats.chi2.cdf(FG_stat, df_chi)

H_stat = -(n_obs - (2 * k_vars + 11) / 6) * np.log(det_R)
H_pvalue = 1 - stats.chi2.cdf(H_stat, df_chi)

multi_df = pd.DataFrame({
    "Determinant":            [det_R],
    "Log Determinant":        [np.log(det_R)],
    "Log (1 - Determinant)":  [np.log(1 - det_R)],
    "N":                      [n_obs],
    "k":                      [k_vars],
    "Degrees of Freedom":     [df_chi],
    "Farrar-Glauber stat":    [FG_stat],
    "Haitovsky stat":         [H_stat],
    "Farrar-Glauber p-value": [FG_pvalue],
    "Haitovsky p-value":      [H_pvalue],
})
print("\n=== Farrar-Glauber & Haitovsky ===")
display(multi_df)


# 6) OLS REGRESSION --------------------------------------------------------
Y = stationary_df["TAURON"]
X = sm.add_constant(stationary_df[selected_vars])
model = sm.OLS(Y, X).fit()
print("\n=== OLS Regression Results ===")
print(model.summary())

params = model.params
formula = f"TAURON_t = {params['const']:.6f}"
for var in selected_vars:
    formula += f" + ({params[var]:.6f})*{var}_t"
print("\nRownanie regresji:\n", formula)


# 7) DIAGNOSTIC TESTS ------------------------------------------------------
resid = model.resid
fitted = model.fittedvalues
exog = model.model.exog
names = model.model.exog_names

vif = pd.Series(
    [variance_inflation_factor(exog, i) for i in range(exog.shape[1])],
    index=names,
)
print("\nVariance Inflation Factors:")
display(vif)

dw = sm.stats.stattools.durbin_watson(resid)
print("Durbin-Watson:", dw)

bp_test = sms.het_breuschpagan(resid, exog)
print("Breusch-Pagan p-value:", bp_test[1])

fig = plt.figure(constrained_layout=True, figsize=(14, 10))
gs = fig.add_gridspec(3, 2)

ax1 = fig.add_subplot(gs[0, 0])
plot_acf(resid, ax=ax1, lags=40, title="ACF reszt")

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist(resid, bins=30, alpha=0.6)
ax2.axvline(0, linestyle="--", color="grey")
ax2.set_title("Histogram reszt (zerowa srednia)")

ax3 = fig.add_subplot(gs[1, :])
ax3.scatter(fitted, resid, alpha=0.4)
ax3.axhline(0, linestyle="--", color="grey")
ax3.set_xlabel("Wartosci dopasowane")
ax3.set_ylabel("Reszty")
ax3.set_title("Reszty vs dopasowane (homoskedastycznosc)")

ax4 = fig.add_subplot(gs[2, 0])
sm.qqplot(resid, line="45", ax=ax4, fit=True)
ax4.set_title("QQ plot reszt (normalnosc)")

ax5 = fig.add_subplot(gs[2, 1])
colors = plt.cm.viridis(np.linspace(0, 1, k_vars))
for c, var in zip(colors, selected_vars):
    ax5.scatter(stationary_df[var], resid, alpha=0.4, label=var, color=c)
ax5.axhline(0, linestyle="--", color="grey")
ax5.set_title("Reszty vs zmienne objasniajace (brak endogenicznosci)")
ax5.legend(frameon=False, fontsize=8)

plt.show()


# 8) FORECASTING - LAST 10% OF DATA ----------------------------------------
n_forecast = int(0.1 * len(stationary_df))
train, test = stationary_df.iloc[:-n_forecast], stationary_df.iloc[-n_forecast:]
model_train = sm.OLS(train["TAURON"], sm.add_constant(train[selected_vars])).fit()

y_pred = model_train.predict(sm.add_constant(test[selected_vars]))
y_true = test["TAURON"]

MAE = np.mean(np.abs(y_pred - y_true))
RMSE = np.sqrt(np.mean((y_pred - y_true) ** 2))
SMAPE = np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true))) * 100

print("\n=== Prognoza (ostatnie 10 %) ===")
print(f"MAE   = {MAE:.6f}")
print(f"RMSE  = {RMSE:.6f}")
print(f"sMAPE = {SMAPE:.2f}%")

forecast_df = pd.DataFrame({"Actual": y_true, "Forecast": y_pred})
display(forecast_df.head())

plt.figure(figsize=(10, 4))
plt.plot(test.index, y_true, label="Rzeczywiste")
plt.plot(test.index, y_pred, label="Prognoza", alpha=0.7)
plt.legend()
plt.title("TAURON: rzeczywiste vs prognoza (dziennie)")
plt.show()
