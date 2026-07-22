from app import predict_from_text


def test_predict_from_text_returns_disease():
    result = predict_from_text("I have fever, headache, and skin rash")
    assert "disease" in result["response"].lower()
    assert result["predicted_disease"] is not None
    assert "fever" in result["symptoms"] or "headache" in result["symptoms"] or "skin_rash" in result["symptoms"]
    assert len(result["top_predictions"]) >= 3
    assert result["top_predictions"][0]["disease"] == result["predicted_disease"]
    assert result["top_predictions"][0]["confidence"] >= 0
