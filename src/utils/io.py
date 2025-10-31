def save_model(model, path: str) -> None:
    import joblib
    joblib.dump(model, path)

def load_model(path: str):
    import joblib
    return joblib.load(path)

def save_data(data, path: str) -> None:
    import pandas as pd
    pd.DataFrame(data).to_csv(path, index=False)

def load_data(path: str):
    import pandas as pd
    return pd.read_csv(path)

def save_json(data, path: str) -> None:
    import json
    with open(path, 'w') as f:
        json.dump(data, f)

def load_json(path: str):
    import json
    with open(path, 'r') as f:
        return json.load(f)