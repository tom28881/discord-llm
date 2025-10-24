#!/usr/bin/env python3
import sqlite3

# Test the column detection logic
conn = sqlite3.connect('data/db.sqlite')
cursor = conn.cursor()

print("Testing column detection logic...")

cursor.execute("PRAGMA table_info(messages)")
columns_info = cursor.fetchall()
print(f"Raw PRAGMA output: {columns_info}")

columns = [col[1] for col in columns_info]
print(f"Column names: {columns}")

has_author_id = 'author_id' in columns
has_author_name = 'author_name' in columns

print(f"has_author_id: {has_author_id}")
print(f"has_author_name: {has_author_name}")

if has_author_id and has_author_name:
    print("Would use query WITH author columns")
else:
    print("Would use fallback query WITHOUT author columns")

conn.close()