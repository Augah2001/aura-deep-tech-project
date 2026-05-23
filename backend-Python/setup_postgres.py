from __future__ import annotations

import argparse
import os
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
PSQL = Path(r"C:\Program Files\PostgreSQL\18\bin\psql.exe")


def create_database(user: str, password: str, host: str, port: int, database: str) -> None:
    try:
        import psycopg2  # type: ignore
    except ImportError as exc:
        raise SystemExit("Install psycopg2-binary before running this script.") from exc

    conn = psycopg2.connect(dbname="postgres", user=user, password=password, host=host, port=port)
    conn.autocommit = True
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (database,))
            exists = cursor.fetchone() is not None
            if not exists:
                cursor.execute(f'CREATE DATABASE "{database}"')
                print(f"Created database {database}")
            else:
                print(f"Database {database} already exists")
    finally:
        conn.close()


def write_env(user: str, password: str, host: str, port: int, database: str) -> None:
    env_path = APP_DIR / ".env"
    url = f"postgresql://{user}:{password}@{host}:{port}/{database}"
    env_path.write_text(f"AURA_DATABASE_URL={url}\n", encoding="utf-8")
    print(f"Wrote {env_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and configure the AURA PostgreSQL database.")
    parser.add_argument("--user", default="postgres")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--database", default="aura_gateway")
    parser.add_argument("--password", default=os.getenv("PGPASSWORD") or os.getenv("POSTGRES_PASSWORD"))
    args = parser.parse_args()

    if not args.password:
        raise SystemExit("Set PGPASSWORD or pass --password before running this script.")

    create_database(args.user, args.password, args.host, args.port, args.database)
    write_env(args.user, args.password, args.host, args.port, args.database)

    os.environ["AURA_DATABASE_URL"] = f"postgresql://{args.user}:{args.password}@{args.host}:{args.port}/{args.database}"
    from app import storage

    storage.init_storage()
    print(storage.storage_backend())


if __name__ == "__main__":
    main()
