#!/bin/bash


DB_USER="mark"
DB_HOST="localhost"
DB_PORT="5432"
NEW_DB_NAME="persp"

echo "Checking/Creating database: ${NEW_DB_NAME}..."

# 2. Connect to the default 'postgres' database to create the new database
# The '|| true' ensures the script doesn't crash if the database already exists
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d postgres -c "CREATE DATABASE $NEW_DB_NAME;" || echo "Database may already exist, proceeding..."


echo "Now attempting to add postgis extension"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$NEW_DB_NAME" -c "CREATE EXTENSION IF NOT EXISTS postgis;"

echo "Database setup completed successfully!"



