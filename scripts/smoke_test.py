#!/usr/bin/env python3
"""
FitScan smoke test — runs against live local API.
Usage: cd fitscan && python scripts/smoke_test.py
Requires: make dev running (http://localhost:8000)
"""
import sys, json
from pathlib import Path
import httpx

BASE    = "http://localhost:8000"
FIXTURE = Path("tests/fixtures/sample_front.jpg")
GREEN   = "\033[0;32m"; RED = "\033[0;31m"; NC = "\033[0m"
results = []

def ok(name):   results.append((name, True));  print(f"  {GREEN}PASS{NC}  {name}")
def fail(name, detail=""): results.append((name, False)); print(f"  {RED}FAIL{NC}  {name}" + (f": {detail}" if detail else ""))

client = httpx.Client(timeout=90)

# 1. Health
try:
    r = client.get(f"{BASE}/api/v1/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    ok("Health check")
except Exception as e:
    fail("Health check", str(e))
    print(f"\n{RED}FitScan not running. Start with: make dev{NC}")
    sys.exit(1)

# 2. Register key
api_key = None
try:
    r = client.post(f"{BASE}/api/v1/keys/register", data={
        "owner_name": "Smoke Test", "owner_email": "smoke@test.local", "tier": "free"
    })
    assert r.status_code == 200
    api_key = r.json()["api_key"]
    assert api_key.startswith("fsk_")
    ok(f"Register API key ({api_key[:12]}...)")
except Exception as e:
    fail("Register API key", str(e))

# 3. Reject unauthenticated
# FS-NIDLO-TEST-10 — assertion expects 401, which the verify_api_key
# dependency raises when X-API-Key is missing. Requires
# FITSCAN_REQUIRE_API_KEY=true (the prod default) — when the local
# `make dev` env has it set false, this assertion legitimately fails
# and that's a hint to the operator: smoke tests should run against
# a prod-shaped env.
try:
    r = client.post(f"{BASE}/api/v1/measure",
        files={"image": ("x.jpg", b"fake", "image/jpeg")}, data={"height_cm": "165"})
    assert r.status_code == 401, (
        f"expected 401 (verify_api_key gate), got {r.status_code}. "
        "If you're hitting this in dev, ensure FITSCAN_REQUIRE_API_KEY=true."
    )
    ok("Reject unauthenticated request")
except Exception as e:
    fail("Reject unauthenticated request", str(e))

# 4. OpenAPI schema accessible
try:
    r = client.get(f"{BASE}/openapi.json")
    assert r.status_code == 200
    assert "/api/v1/measure" in json.dumps(r.json())
    ok("OpenAPI schema accessible at /openapi.json")
except Exception as e:
    fail("OpenAPI schema", str(e))

# 5. Swagger UI accessible
try:
    r = client.get(f"{BASE}/docs")
    assert r.status_code == 200
    ok("Swagger UI accessible at /docs")
except Exception as e:
    fail("Swagger UI", str(e))

# 6. Real measure (if fixture exists)
avatar_id = None
if FIXTURE.exists():
    try:
        with open(FIXTURE, "rb") as f:
            r = client.post(f"{BASE}/api/v1/measure",
                headers={"X-API-Key": api_key or "fsk_test"},
                files={"image": ("sample.jpg", f, "image/jpeg")},
                data={"height_cm": "165", "gender_profile": "ladies",
                      "enable_smpl": "false", "enable_claude": "false"})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "success"
        assert "waist_cm" in data and data["waist_cm"]["confidence"] > 0.3
        avatar_id = data.get("avatar_id")
        ok(f"Measure (waist={data['waist_cm']['value_cm']}cm, conf={data['waist_cm']['confidence']:.2f})")
    except Exception as e:
        fail("Measure with fixture", str(e))
else:
    print(f"  SKIP  Real measure (add a photo at {FIXTURE} to enable)")

# 7. Avatar thumbnail
if avatar_id:
    try:
        r = client.get(f"{BASE}/api/v1/avatar/{avatar_id}/thumbnail",
            headers={"X-API-Key": api_key or "fsk_test"})
        assert r.status_code == 200
        ok("Avatar thumbnail download")
    except Exception as e:
        fail("Avatar thumbnail", str(e))

# Summary
passed = sum(1 for _, p in results if p)
total  = len(results)
print()
if passed == total:
    print(f"{GREEN}All {total} smoke tests passed.{NC}")
else:
    print(f"{RED}{total - passed}/{total} tests FAILED.{NC}")
    sys.exit(1)
