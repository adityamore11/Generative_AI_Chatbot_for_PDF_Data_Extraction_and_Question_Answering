from app.ingestion import split_text, table_to_summary


def test_table_summary_preserves_header_value_pairs():
    result = table_to_summary([["Penalty", "Yards"], ["Holding", "10"]], 1)
    assert "Penalty: Holding" in result
    assert "Yards: 10" in result


def test_split_text_keeps_all_content():
    text = "First rule applies. Second rule applies. Third rule applies."
    chunks = list(split_text(text, max_chars=35, overlap=5))
    assert len(chunks) > 1
    assert "Third rule applies." in chunks[-1]
