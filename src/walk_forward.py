import numpy as np
import lightgbm as lgb
from src.features import load, build_features, FEATURES, TARGET

df = build_features(load())

# Train on everything before the test year, test on that year
folds = [("2023-01-01", "2024-01-01"),
         ("2024-01-01", "2025-01-01"),
         ("2025-01-01", "2026-01-01")]

for start, end in folds:
    train = df[df.index < start]
    test = df[(df.index >= start) & (df.index < end)]
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              random_state=42, verbose=-1)
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])
    mape = np.mean(np.abs((test[TARGET] - pred) / test[TARGET])) * 100
    print(f"Test {start[:4]} : MAPE = {mape:.2f}%  (train = {len(train)} rows)")