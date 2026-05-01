import pandas as pd

df = pd.read_csv('support_tickets/output.csv')

print("Row 24 (French injection - index 23):")
row24 = df.iloc[23]
print(f"Status: {row24['status']}")
print(f"Request Type: {row24['request_type']}")
print(f"Product Area: {row24['product_area']}")
print(f"Response: {row24['response'][:100]}...")

print("\nRow 25 (Delete all files - index 24):")
row25 = df.iloc[24]
print(f"Status: {row25['status']}")
print(f"Request Type: {row25['request_type']}")
print(f"Product Area: {row25['product_area']}")

print("\nRow 30 (Visa minimum spend - index 28):")
row30 = df.iloc[28]
print(f"Status: {row30['status']}")
print(f"Request Type: {row30['request_type']}")
print(f"Product Area: {row30['product_area']}")
print(f"Response length: {len(str(row30['response']))} chars")
