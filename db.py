import sqlite3

def init_db():
    """Initialize the SQLite database."""
    conn = sqlite3.connect("chat_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_analysis (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name_of_employee TEXT,
            satisfaction TEXT
        )
    """)
    conn.commit()
    conn.close()

def insert_analysis(name, satisfaction):
    """Insert analyzed chat data into the database."""
    conn = sqlite3.connect("chat_data.db")
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO chat_analysis (name_of_employee, satisfaction)
        VALUES (?, ?)
    """, (name, satisfaction))
    conn.commit()
    conn.close()

def fetch_analysis():
    """Fetch all chat analysis data."""
    conn = sqlite3.connect("chat_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_analysis")
    rows = cursor.fetchall()
    conn.close()
    return rows
