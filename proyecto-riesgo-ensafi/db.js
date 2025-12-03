const { Pool } = require('pg');

// Configura con tus credenciales reales de PostgreSQL
const pool = new Pool({
  user: 'postgres',
  host: 'localhost',
  database: 'dw_ensafi',
  password: '2801',
  port: 5432,
});

module.exports = pool;