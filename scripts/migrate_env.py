"""Re-key a measurements store to this machine's current fingerprint (after the fingerprint format changed)."""
import json, sqlite3, sys
from rewrite_blame import env
from rewrite_blame.store import measurement_key
path = sys.argv[1] if len(sys.argv) > 1 else "results/measurements.sqlite"
threads = int(sys.argv[2]) if len(sys.argv) > 2 else env.default_threads()
new = env.fingerprint(threads)
con = sqlite3.connect(path)
rows = con.execute("SELECT key, model, env, state_hash, protocol, payload FROM measurements").fetchall()
n = 0
for key, model, old_env, sh, proto, payload in rows:
    if old_env == new:
        continue
    d = json.loads(payload); d["env"] = new
    con.execute("DELETE FROM measurements WHERE key=?", (key,))
    con.execute("INSERT OR REPLACE INTO measurements (key, model, env, state_hash, protocol, created, payload) VALUES (?,?,?,?,?,?,?)",
                (measurement_key(model, new, sh, proto), model, new, sh, proto, d.get("timestamp", 0), json.dumps(d)))
    n += 1
con.commit()
print(f"migrated {n} rows to {new!r}")
