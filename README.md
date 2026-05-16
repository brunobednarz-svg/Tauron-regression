# TAURON (TPE) - analiza regresji wzgledem spolek GPW

Projekt zaliczeniowy z ekonometrii. Buduje wielowymiarowy model OLS dla
logarytmicznych stop zwrotu **TAURON-a (ticker TPE)** w oparciu o 4 spolki
z Warszawskiej Gieldy Papierow Wartosciowych najsilniej z nim skorelowane.

Dane zrodlowe: notowania dzienne ze [stooq.pl](https://stooq.pl) (format
`<TICKER>,<PER>,<DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<VOL>,<OPENINT>`),
ponad 420 tickerow w katalogu `data/`.

## Zakres analizy

1. **Wczytanie danych** - log-stopy zwrotu z `data/*.txt`, odsiew spolek
   o zbyt krotkiej historii (< 250 wspolnych obserwacji).
2. **Wybor zmiennych** - ranking wartosci bezwzglednej korelacji z TAURON-em,
   top 4 jako regresory.
3. **ADF** - test stacjonarnosci, automatyczne roznicowanie w razie potrzeby.
4. **Czyszczenie outlierow** - obciecie |r| > 5 %.
5. **Wspolliniowosc** - Farrar-Glauber, Haitovsky, VIF.
6. **OLS** - pelne podsumowanie z `statsmodels` + zapisane rownanie regresji.
7. **Diagnostyka reszt** - Durbin-Watson, Breusch-Pagan, ACF, Q-Q, scatter
   reszt vs zmienne (sprawdzenie endogenicznosci).
8. **Prognoza out-of-sample** - ostatnie 10 % obserwacji, metryki MAE / RMSE /
   sMAPE oraz wykres porownawczy.

## Struktura repo

```
.
├── data/                       # ~426 plikow .txt ze stooq (notowania GPW)
├── tauron_regression.py        # skrypt uruchamialny z linii polecen
├── tauron_regression.ipynb     # ta sama logika w Jupyterze
├── requirements.txt
└── README.md
```

## Uruchomienie

```bash
cd ~/Downloads/tauron-wse-regression

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python3 tauron_regression.py
# albo:
jupyter notebook tauron_regression.ipynb
```

## Wymagania

- Python 3.10+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `statsmodels`,
  opcjonalnie `ipython` / `jupyter`

## Uwagi

- TAURON jako glowna serie wskazuje stala `MAIN_TICKER_FILE` w skrypcie -
  jesli chcesz przebadac inna spolke, podmien plik (np. `pkn.txt`, `kgh.txt`).
- Liczbe zmiennych objasniajacych reguluje `TOP_N` (domyslnie 4).
- Prog outlierow to `OUTLIER_THRESHOLD` (domyslnie 5 % dziennej stopy zwrotu).
