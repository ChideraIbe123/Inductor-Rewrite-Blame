from rewrite_blame.repeat import repeatability, render_repeat, session_of


def _row(model, env, session, sh, med, ts=0):
    return {"model": model, "env": env, "session": session, "state_hash": sh, "timing": {"median": med}, "timestamp": ts}


def test_session_of_falls_back_to_protocol_key_or_timestamp():
    assert session_of({"session": "s1"}) == "s1"
    assert session_of({"protocol_key": "w10r5i10@2026-10-08/rep2"}) == "2026-10-08"
    import datetime, time
    ts = time.mktime(datetime.datetime(2026, 10, 7, 12).timetuple())
    assert session_of({"protocol_key": "w10r5i10", "timestamp": ts}) == "2026-10-07"
    assert session_of({}) == "unknown"


def test_repeatability_agreement_counts():
    rows = []
    for sess, base in [("d1", 10.0), ("d2", 10.5)]:            # machine A, two sessions
        rows += [_row("m", "A", sess, "base", base + x) for x in (0.0, 0.02, -0.02)]
        rows.append(_row("m", "A", sess, "sw1", base * 1.10))   # slower in both sessions
        rows.append(_row("m", "A", sess, "sw2", base * (1.10 if sess == "d1" else 1.0)))  # disagrees
    rows += [_row("m", "B", "d1", "base", 20.0 + x) for x in (0.0, 0.05, -0.05)]
    rows.append(_row("m", "B", "d1", "sw1", 20.0 * 0.9))       # faster on machine B -> cross-machine disagreement
    rows.append(_row("m", "B", "d1", "sw2", 20.0))
    rows.append(_row("other", "A", "d1", "base", 1.0))        # ignored
    rep = repeatability(rows, "m", "base", {"sw1": "sw1", "sw2": "sw2"})
    assert len(rep.baselines) == 3
    sw1 = next(s for s in rep.switches if s["switch"] == "sw1")
    assert sw1["same_machine_agree"] is True and sw1["cross_machine_agree"] is False
    sw2 = next(s for s in rep.switches if s["switch"] == "sw2")
    assert sw2["same_machine_agree"] is False
    assert rep.agreement["same_machine"] == {"agree": 3, "disagree": 1}
    assert rep.agreement["cross_machine"]["disagree"] >= 1
    text = render_repeat(rep)
    assert "Repeatability: m" in text and "sw1" in text


def test_single_run_baseline_uses_floor_tau():
    rows = [_row("m", "A", "d1", "base", 10.0), _row("m", "A", "d1", "sw", 10.05)]
    rep = repeatability(rows, "m", "base", {"sw": "sw"})
    assert rep.baselines[0]["tau_ms"] == 0.1
    assert rep.switches[0]["cells"][0]["verdict"] == "same"
