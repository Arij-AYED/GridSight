import numpy as np
import lightgbm as lgb
from src.features import load, build_features, FEATURES, TARGET


def metrics(y, pred):
    return {"MAE": np.mean(np.abs(y - pred)),
            "RMSE": np.sqrt(np.mean((y - pred) ** 2)),
            "MAPE %": np.mean(np.abs((y - pred) / y)) * 100}


df = build_features(load())
train = df[df.index < "2024-01-01"]
val = df[(df.index >= "2024-01-01") & (df.index < "2025-01-01")]
test = df[df.index >= "2025-01-01"]
print(len(train), len(val), len(test))

# Baselines : "demain = aujourd'hui" et "demain = même heure il y a 7 jours"
print("Naif 24h :", metrics(test[TARGET], test["lag_24"]))
print("Naif 168h:", metrics(test[TARGET], test["lag_168"]))

# LightGBM
model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          random_state=42, verbose=-1)
model.fit(train[FEATURES], train[TARGET],
          eval_set=[(val[FEATURES], val[TARGET])],
          callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

pred = model.predict(test[FEATURES])
print("LightGBM :", metrics(test[TARGET], pred))

imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: -x[1])
print(imp)
import matplotlib.pyplot as plt

test = test.copy()
test["pred"] = pred
test["err"] = test[TARGET] - test["pred"]
test["ape"] = (test["err"].abs() / test[TARGET]) * 100

print("Biais moyen (MW) :", round(test["err"].mean(), 1))
print("\nMAPE par heure :\n", test.groupby("hour")["ape"].mean().round(2))
print("\nMAPE par mois :\n", test.groupby("month")["ape"].mean().round(2))
print("\nMAPE jours fériés :", round(test[test.is_holiday == 1]["ape"].mean(), 2))
print("MAPE hors fériés  :", round(test[test.is_holiday == 0]["ape"].mean(), 2))

# 5 % des jours les plus froids
froid = test[test["temp_24h"] < test["temp_24h"].quantile(0.05)]
print("MAPE jours très froids :", round(froid["ape"].mean(), 2))

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
test["err"].plot.hist(bins=60, ax=ax[0], title="Distribution des erreurs (MW)")
test["err"].resample("D").mean().plot(ax=ax[1], title="Erreur moyenne par jour")
plt.show()