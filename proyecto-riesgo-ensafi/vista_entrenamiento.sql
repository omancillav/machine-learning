CREATE OR REPLACE VIEW vista_datos_ml AS
SELECT 
    d.edad_jefe_hogar,
    d.nivel_educativo_jefe,
    v.tiene_internet,
    e.fk_id_region AS id_region,
    f.nivel_riesgo
FROM fact_endeudamiento f
JOIN dim_demografia d ON f.id_demografia = d.id_demografia
JOIN dim_vivienda v ON f.id_vivienda = v.id_vivienda
JOIN dim_geografia g ON f.id_geografia = g.id_geografia
JOIN dim_estado e ON g.fk_id_estado = e.id_estado;