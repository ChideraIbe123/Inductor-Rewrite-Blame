from rewrite_blame.store import Store, measurement_key


def test_store_roundtrip_and_key_sensitivity(tmp_path):
    with Store(tmp_path / "m.sqlite") as st:
        assert st.get("m", "env", "h1") is None
        st.put("m", "env", "h1", {"model": "m", "ms": 1.5})
        assert st.get("m", "env", "h1") == {"model": "m", "ms": 1.5}
        assert st.get("m", "env", "h2") is None
        assert st.get("m", "env2", "h1") is None
        assert st.get("m2", "env", "h1") is None
        assert st.count() == 1
        st.put("m", "env", "h1", {"model": "m", "ms": 2.0})  # replace
        assert st.count() == 1 and st.get("m", "env", "h1")["ms"] == 2.0


def test_store_persists_across_connections_and_lists(tmp_path):
    p = tmp_path / "m.sqlite"
    with Store(p) as st:
        st.put("a", "env", "h", {"model": "a"})
        st.put("b", "env", "h", {"model": "b"})
    with Store(p) as st:
        assert st.models() == ["a", "b"]
        assert len(st.all(model="a")) == 1 and len(st.all(env="env")) == 2
        out = tmp_path / "x.json"
        assert st.export_json(out, models=["b"]) == 1


def test_measurement_key_includes_protocol():
    assert measurement_key("m", "e", "h", "p1") != measurement_key("m", "e", "h", "p2")
    assert len(measurement_key("m", "e", "h")) == 24
