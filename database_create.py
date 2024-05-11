import sqlite3
import os

def create_database(db_name="portfolio.db"):
    if os.path.exists(db_name):
        print(f"The database '{db_name}' already exists.")
        return False
    conn = sqlite3.connect(db_name)
    conn.close()
    return True


def create_table(db_name="portfolio.db", table_name="portfolio"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    create_table_query = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date_created TEXT DEFAULT CURRENT_TIMESTAMP,
        n_assets INTEGER,
        l1_opts TEXT,
        l2_opts TEXT,
        n_tscv INTEGER,
        date_training_start TEXT,
        date_training_end TEXT,
        n_days INTEGER,
        best_l TEXT,
        tscv_metric TEXT,
        tscv_size INTEGER,
        testing_window INTEGER,
        training_window INTEGER,
        training_weights TEXT,
        portfolio_testing_returns TEXT,
        testing_metric TEXT,
        testing_performance TEXT,
        testing_optimal_weights_performance TEXT,
        exception TEXT
    );
    """
    try:
        cursor.execute(create_table_query)
        conn.commit()
    except sqlite3.OperationalError as e:
        if "already exists" in str(e):
            print(f"Table '{table_name}' already exists.")
        else:
            raise e
    conn.close()

if __name__ == "__main__":
    create_database()
    create_table()
