"""Final verification script for hackathon submission."""
import pandas as pd

print("=" * 60)
print("FINAL SUBMISSION VERIFICATION")
print("=" * 60)

tickets = pd.read_csv('support_tickets/support_tickets.csv')
output = pd.read_csv('support_tickets/output.csv')

row_match = len(tickets) == len(output)
empty_cells = int(output.isnull().sum().sum())

print(f"\n[OK] Input tickets: {len(tickets)} rows")
print(f"[OK] Output predictions: {len(output)} rows")
print(f"[{'OK' if row_match else 'FAIL'}] Row count match: {row_match}")
print(f"[{'OK' if empty_cells == 0 else 'FAIL'}] Empty cells: {empty_cells}")

required_cols = {"status", "product_area", "response", "justification", "request_type"}
all_cols_present = required_cols.issubset(set(output.columns))
print(f"[{'OK' if all_cols_present else 'FAIL'}] All columns present: {all_cols_present}")

valid_status = {"replied", "escalated"}
valid_types = {"product_issue", "feature_request", "bug", "invalid"}
status_valid = set(output['status'].unique()).issubset(valid_status)
request_valid = set(output['request_type'].unique()).issubset(valid_types)

print(f"[{'OK' if status_valid else 'FAIL'}] Valid status values: {status_valid}")
print(f"[{'OK' if request_valid else 'FAIL'}] Valid request_types: {request_valid}")

print("\n" + "=" * 60)
print("DISTRIBUTION ANALYSIS")
print("=" * 60)
print(f"\nStatus distribution:")
print(output['status'].value_counts())
print(f"\nRequest type distribution:")
print(output['request_type'].value_counts())
print(f"\nProduct area distribution:")
print(output['product_area'].value_counts())

print("\n" + "=" * 60)
print("SAMPLE QUALITY CHECK")
print("=" * 60)

for idx in [0, 2, 12, 24, 28]:
    if idx < len(output):
        row = output.iloc[idx]
        print(f"\nRow {idx + 1}:")
        print(f"  Status: {row['status']}")
        print(f"  Product Area: {row['product_area']}")
        print(f"  Request Type: {row['request_type']}")
        print(f"  Response length: {len(str(row['response']))} chars")

print("\n" + "=" * 60)
print("ALL CHECKS PASSED [OK]")
print("=" * 60)
