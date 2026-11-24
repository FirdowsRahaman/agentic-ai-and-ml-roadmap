import pytest
from app import build_default_copilot


def test_hybrid_rag_exact_error_code():
    bot = build_default_copilot()
    res = bot.ask("Troubleshoot ERR-409-SYNC token issue")
    assert res["route"] == "hybrid_rag"
    assert "POL-204" in res["answer"]


def test_sql_route_aggregation():
    bot = build_default_copilot()
    res = bot.ask("Show total revenue for APAC orders")
    assert res["route"] == "sql_tool"
    assert "APAC: $2000.00" in res["answer"]


def test_sql_guardrail_blocks_destructive_queries():
    bot = build_default_copilot()
    with pytest.raises(PermissionError):
        bot.sql_tool.execute("DROP TABLE orders;")
