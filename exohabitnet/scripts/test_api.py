"""Diagnose the NASA Exoplanet Archive TAP endpoint."""
import requests

url = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

# Test 1: Check available tables first
print("=== Test 1: List tables ===")
r = requests.get(url, params={
    "query": "SELECT table_name FROM TAP_SCHEMA.tables",
    "format": "json", "lang": "ADQL"
}, timeout=30)
print(f"Status: {r.status_code}")
if r.ok:
    print(r.text[:600])
else:
    print(r.text[:400])

# Test 2: Minimal select with TOP (SQL Server style) 
print("\n=== Test 2: TOP syntax ===")
r2 = requests.get(url, params={
    "query": "SELECT TOP 5 kepid, kepoi_name, koi_disposition FROM cumulative",
    "format": "json", "lang": "ADQL"
}, timeout=30)
print(f"Status: {r2.status_code}")
print(r2.text[:400])

# Test 3: Try without any row limit
print("\n=== Test 3: No LIMIT/TOP ===")
r3 = requests.get(url, params={
    "query": "SELECT kepid, kepoi_name FROM cumulative WHERE ROWNUM <= 5",
    "format": "json", "lang": "ADQL"
}, timeout=30)
print(f"Status: {r3.status_code}")
print(r3.text[:400])
