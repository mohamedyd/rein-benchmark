#!/bin/bash

sudo -u postgres psql <<PGSCRIPT

DROP database rein;
CREATE database rein;
CREATE user reinuser;
ALTER USER reinuser WITH PASSWORD 'abcd1234';
GRANT ALL PRIVILEGES on database rein to reinuser ;
PGSCRIPT

echo "PG database and user has been created."
