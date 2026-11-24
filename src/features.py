import duckdb
import holidays
import pandas as pd

DB = "data/gridsight.duckdb"
def load() -> pd.DataFrame:
    con = duckdb.connect(DB, read_only=True)
    con.execute("SET TimeZone='UTC'")
    df = con.execute("SELECT * FROM dataset ORDER BY ts").df()
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.set_index("ts")

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 1. combler les 5 trous
    df["consommation"] = df["consommation"].interpolate(limit=6)

    # 2. calendrier (heure de Paris)
    loc = df.index.tz_convert("Europe/Paris")
    df["hour"] = loc.hour
    df["dow"] = loc.dayofweek
    df["month"] = loc.month
    df["doy"] = loc.dayofyear
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    fr = holidays.France(years=range(2021, 2028))
    df["is_holiday"] = [int(d in fr) for d in loc.date]

    # 3. météo
    df["temp_cold"] = (15 - df["temp"]).clip(lower=0)   # chauffage
    df["temp_hot"] = (df["temp"] - 22).clip(lower=0)    # clim
    df["temp_24h"] = df["temp"].rolling(24).mean()

    # 4. lags (>= 24h : on prévoit J+1, donc on ne regarde jamais le futur)
    for lag in (24, 48, 168):
        df[f"lag_{lag}"] = df["consommation"].shift(lag)
    df["roll_168"] = df["consommation"].shift(24).rolling(168).mean()

    return df.dropna(subset=["consommation", "lag_168", "roll_168", "temp_24h"])


FEATURES = ["hour", "dow", "month", "doy", "is_weekend", "is_holiday",
            "temp", "wind", "temp_cold", "temp_hot", "temp_24h",
            "lag_24", "lag_48", "lag_168", "roll_168"]
TARGET = "consommation"