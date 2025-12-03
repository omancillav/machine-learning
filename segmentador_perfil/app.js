const tf = require("@tensorflow/tfjs");
const express = require("express");
const bodyParser = require("body-parser");
const { Pool } = require("pg");
const app = express();
const port = 3000;

// Configuración de PostgreSQL
const pool = new Pool({
  user: "postgres",
  host: "localhost",
  database: "dw_ensafi",
  password: "2801",
  port: 5432,
});

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static("public"));

let model;
let categoriasEducacion = [];
let categoriasLocalidad = [];
let categoriasPerfil = [];
let isModelTrained = false;

// --- DEFINICIÓN DEL MODELO ---
function createModel(inputShape, outputUnits) {
  const model = tf.sequential();

  // Capa de entrada
  model.add(
    tf.layers.dense({
      inputShape: [inputShape],
      units: 64,
      activation: "relu",
      kernelInitializer: "heNormal", // Ayuda a evitar NaN al inicio
    })
  );

  model.add(tf.layers.dropout({ rate: 0.2 }));

  model.add(tf.layers.dense({ units: 32, activation: "relu" }));

  model.add(
    tf.layers.dense({
      units: outputUnits,
      activation: "softmax", // Clasificación multiclase
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.03), // Learning rate más bajo y seguro
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
}

// --- CARGA DE DATOS Y ENTRENAMIENTO ---
async function trainModelFromDB() {
  console.log("🔄 Extrayendo datos y limpiando nulos...");

  /* SOLUCIÓN SQL: Usamos COALESCE para evitar NULLs que causan NaN.
       Si un dato es nulo, le ponemos un valor por defecto.
    */
  const query = `
        SELECT 
            COALESCE(f.ingreso_mensual_hogar, 0) as ingreso,
            COALESCE(d.edad_jefe_hogar, 30) as edad,
            COALESCE(d.numero_personas_hogar, 1) as personas,
            COALESCE(d.nivel_educativo_jefe, 'Sin estudios') as educacion,
            COALESCE(g.tipo_localidad, 'Urbana') as localidad,
            (CASE WHEN v.tiene_internet THEN 1 ELSE 0 END + 
             CASE WHEN v.tiene_agua THEN 1 ELSE 0 END + 
             CASE WHEN v.tiene_drenaje THEN 1 ELSE 0 END) as score_servicios,
            CASE 
                WHEN f.ratio_deuda_ingreso < 0.3 THEN 'Ahorrador Cauteloso'
                WHEN f.ratio_deuda_ingreso >= 0.3 AND f.monto_deuda_total > 100000 THEN 'Deudor Inversor'
                ELSE 'Deudor de Riesgo'
            END as etiqueta_perfil
        FROM fact_endeudamiento f
        LEFT JOIN dim_demografia d ON f.id_demografia = d.id_demografia
        LEFT JOIN dim_geografia g ON f.id_geografia = g.id_geografia
        LEFT JOIN dim_vivienda v ON f.id_vivienda = v.id_vivienda
        WHERE f.ingreso_mensual_hogar > 0
        ORDER BY RANDOM() 
        LIMIT 6000; 
    `;

  try {
    const res = await pool.query(query);
    let data = res.rows;

    if (data.length === 0) throw new Error("⚠️ La base de datos no devolvió registros.");

    // 1. Extraer categorías únicas
    categoriasEducacion = [...new Set(data.map((d) => d.educacion))];
    categoriasLocalidad = [...new Set(data.map((d) => d.localidad))];
    categoriasPerfil = [...new Set(data.map((d) => d.etiqueta_perfil))];

    console.log(`✅ Registros crudos: ${data.length}`);
    console.log("📍 Localidades (Raw):", categoriasLocalidad);

    // 2. Procesamiento y Limpieza (JS)
    let inputs = [];
    let labels = [];

    data.forEach((d) => {
      // Normalización Segura
      // Math.max(1, ...) evita log(0) o log(negativo)
      const valIngreso = Math.log1p(Math.max(1, parseFloat(d.ingreso)));
      const valEdad = parseFloat(d.edad) / 100;
      const valPersonas = parseFloat(d.personas) / 20;

      // Si indexOf no encuentra la categoría, devuelve -1. Lo convertimos a 0 para no romper la red.
      const idxEducacion = Math.max(0, categoriasEducacion.indexOf(d.educacion));
      const idxLocalidad = Math.max(0, categoriasLocalidad.indexOf(d.localidad));

      const valServicios = parseFloat(d.score_servicios) / 3;

      const vectorInput = [valIngreso, valEdad, valPersonas, idxEducacion, idxLocalidad, valServicios];

      // VERIFICACIÓN ANTI-NaN
      // Si algún valor en el vector es NaN, ignoramos toda la fila
      if (vectorInput.some((v) => isNaN(v) || v === null || v === undefined)) {
        return; // Saltar registro corrupto
      }

      inputs.push(vectorInput);

      // One-Hot Encoding para la salida
      const rowLabel = new Array(categoriasPerfil.length).fill(0);
      const idxPerfil = categoriasPerfil.indexOf(d.etiqueta_perfil);
      if (idxPerfil !== -1) {
        rowLabel[idxPerfil] = 1;
        labels.push(rowLabel);
      } else {
        // Si falla el label, sacamos el input que acabamos de meter para mantener sincronía
        inputs.pop();
      }
    });

    console.log(`✨ Registros limpios listos para entrenar: ${inputs.length}`);

    // DEBUG: Mostrar el primer registro para ver qué entra
    if (inputs.length > 0) {
      console.log("🔍 Ejemplo de Input [IngresoLog, Edad, Personas, EduIdx, LocIdx, Serv]:", inputs[0]);
    }

    const xs = tf.tensor2d(inputs);
    const ys = tf.tensor2d(labels);

    // 3. Entrenar
    model = createModel(6, categoriasPerfil.length);

    console.log("🚀 Iniciando entrenamiento...");
    await model.fit(xs, ys, {
      epochs: 50,
      batchSize: 32,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if ((epoch + 1) % 10 === 0) {
            // Si logs.loss sigue siendo NaN, detenemos para no perder tiempo
            if (isNaN(logs.loss)) {
              console.log(`❌ ERROR: Loss NaN en época ${epoch + 1}. Revisa tus datos.`);
              model.stopTraining = true;
            } else {
              console.log(`Época ${epoch + 1}: Loss: ${logs.loss.toFixed(4)} - Acc: ${(logs.acc * 100).toFixed(2)}%`);
            }
          }
        },
      },
    });

    isModelTrained = true;
    console.log("✅ Modelo listo.");

    xs.dispose();
    ys.dispose();
  } catch (err) {
    console.error("Error crítico:", err);
  }
}

trainModelFromDB();

// --- API ---

app.get("/opciones", (req, res) => {
  if (!isModelTrained) return res.status(503).json({ error: "Entrenando modelo..." });
  res.json({
    educacion: categoriasEducacion,
    localidad: categoriasLocalidad,
  });
});

app.post("/predict", (req, res) => {
  if (!isModelTrained) return res.status(503).json({ error: "Modelo no listo." });

  const { ingreso, edad, personas, educacion, localidad, servicios } = req.body;

  if (!ingreso || !edad) return res.status(400).json({ error: "Faltan datos" });

  try {
    const idxEdu = Math.max(0, categoriasEducacion.indexOf(educacion));
    const idxLoc = Math.max(0, categoriasLocalidad.indexOf(localidad));

    const inputVector = [
      Math.log1p(Math.max(1, parseFloat(ingreso))),
      parseFloat(edad) / 100,
      parseFloat(personas) / 20,
      idxEdu,
      idxLoc,
      parseInt(servicios) / 3,
    ];

    const inputTensor = tf.tensor2d([inputVector]);
    const prediction = model.predict(inputTensor);
    const pValues = prediction.dataSync();

    const maxProbIndex = pValues.indexOf(Math.max(...pValues));
    const perfilPredicho = categoriasPerfil[maxProbIndex];
    const confianza = (pValues[maxProbIndex] * 100).toFixed(2);

    res.json({
      perfil: perfilPredicho,
      confianza: confianza,
    });

    inputTensor.dispose();
  } catch (error) {
    console.error(error);
    res.status(500).json({ error: "Error en predicción" });
  }
});

app.listen(port, () => {
  console.log(`Servidor en http://localhost:${port}`);
});
