import pandas as pd
from src.pipeline import XAIIDSPipeline

def test_prepare_binary_labels():
    df = pd.DataFrame({
        "dur":[0.1,0.2,0.3,0.4],
        "proto":["tcp","tcp","udp","udp"],
        "label":[0,1,0,1]
    })
    p = XAIIDSPipeline()
    X,y,t = p.prepare(df)
    assert t == "label"
    assert list(y) == [0,1,0,1]
    assert "dur" in X.columns
