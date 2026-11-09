-- Runs once, when the local database volume is first created.
-- The API's role: not the table owner and no BYPASSRLS, so row-level security applies to it.
\getenv app_password APP_DB_PASSWORD
CREATE ROLE marketing_app WITH LOGIN PASSWORD :'app_password'
    NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
CREATE DATABASE marketing_test;
