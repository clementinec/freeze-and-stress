import sqlite3

conn = sqlite3.connect(r"E:\00_research\03_freezethenstress\freeze-and-stress-main\epw_pipeline\energyplus_runner\results\office\Los_Angeles\2025\eplusout.sql")
cur = conn.cursor()

# 看所有表
tables = cur.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
print("表：", tables)

# 看 ReportData 列
cols = cur.execute("PRAGMA table_info('ReportData');").fetchall()
print("列数：", len(cols))
print("列名：")
for c in cols:
    print(c[1])

conn.close()
