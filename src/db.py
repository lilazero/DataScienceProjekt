"""
Wiederverwendbare Funktionen für die MySQL-Datenbankverbindung.
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config import DB_CONFIG


def get_engine() -> Engine:
    """
    Erstellt eine SQLAlchemy-Engine für die Verbindung zur MySQL-Datenbank.
    Nutzt den PyMySQL-Treiber (reine Python-Implementierung, keine
    zusätzlichen Systemabhängigkeiten nötig).
    """
    url = (
        f"mysql+pymysql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
        f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
    )
    engine = create_engine(url, pool_pre_ping=True)
    return engine


def test_connection() -> bool:
    """Einfacher Verbindungstest. Gibt True zurück, wenn die Verbindung klappt."""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn.exec_driver_sql("SELECT 1")
        print("✅ Verbindung zur MySQL-Datenbank erfolgreich.")
        return True
    except Exception as e:
        print("❌ Verbindung fehlgeschlagen:")
        print(f"   {e}")
        return False


if __name__ == "__main__":
    test_connection()
