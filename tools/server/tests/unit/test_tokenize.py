import pytest
from utils import *

server = ServerPreset.tinyllama2()


@pytest.fixture(autouse=True)
def create_server():
    global server
    server = ServerPreset.tinyllama2()


def test_tokenize_detokenize():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content
    })
    assert res_tok.status_code == 200
    assert len(res_tok.body["tokens"]) > 5
    # detokenize
    res_detok = server.make_request("POST", "/detokenize", data={
        "tokens": res_tok.body["tokens"],
    })
    assert res_detok.status_code == 200
    assert res_detok.body["content"].strip() == content


def test_tokenize_with_bos():
    global server
    server.start()
    # tokenize
    content = "What is the capital of France ?"
    bosId = 1
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "add_special": True,
    })
    assert res_tok.status_code == 200
    assert res_tok.body["tokens"][0] == bosId


def test_tokenize_with_pieces():
    global server
    server.start()
    # tokenize
    content = "This is a test string with unicode 媽 and emoji 🤗"
    res_tok = server.make_request("POST", "/tokenize", data={
        "content": content,
        "with_pieces": True,
    })
    assert res_tok.status_code == 200
    for token in res_tok.body["tokens"]:
        assert "id" in token
        assert token["id"] > 0
        assert "piece" in token
        assert len(token["piece"]) > 0


@pytest.mark.parametrize("content", [
    [1, "x", 2.5],
    [1, True],
    [False],
    [1, "x", None],
    [1, 4294967297],
    [18446744073709551615],  # uint64 max - must not overflow to -1 (valid)
])
def test_tokenize_invalid_token_ids(content):
    """Invalid token IDs in mixed arrays should return 400."""
    global server
    server.start()
    res = server.make_request("POST", "/tokenize", data={"content": content})
    assert res.status_code == 400
    assert res.body["error"]["type"] == "invalid_request_error"
    # uint64 max: verify error message to confirm proper bounds checking
    # vs. silent wrap-around to -1 which would return 200 success with [-1] tokens
    if 18446744073709551615 in content:
        assert "Token ID out of int32 range: 18446744073709551615" in res.body["error"]["message"]
