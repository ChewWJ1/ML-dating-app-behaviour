import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/docx_tables.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Let's search for "TABLE" and print the structure of each table
tables = text.split("=== TABLE ")
print(f"Total tables found: {len(tables) - 1}")

for i, t in enumerate(tables[1:]):
    lines = t.strip().split("\n")
    print(f"\nTable {i}:")
    # Print the first 5 rows to see structure
    for line in lines[:8]:
        print("  " + line[:120])
    if len(lines) > 8:
        print("  ... and more rows")
