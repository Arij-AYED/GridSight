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