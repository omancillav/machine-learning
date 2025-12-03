CREATE OR REPLACE VIEW vista_hipoteca AS
SELECT 
    f.monto_ingreso_mensual,
    d.edad_jefe_hogar,
    d.numero_personas_hogar,
    d.nivel_educativo_jefe,
    v.tiene_internet,
    -- Convertimos el texto a número para que la red neuronal lo entienda
    CASE 
        WHEN f.tiene_credito_hipotecario = 'Sí' THEN 1 
        ELSE 0 
    END as tiene_hipoteca
FROM fact_endeudamiento f
JOIN dim_demografia d ON f.id_demografia = d.id_demografia
JOIN dim_vivienda v ON f.id_vivienda = v.id_vivienda -- Usamos la unión numérica correcta
WHERE f.monto_ingreso_mensual > 0; -- Ignoramos errores de captura con ingresos en 0