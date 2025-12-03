const tf = require("@tensorflow/tfjs");
const express = require("express");
const bodyParser = require("body-parser");
const { Client } = require("pg");

// --- CONFIGURACIÓN ---
const app = express();
const port = 3001;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static("public"));

// Configuración de BD (Ajusta con tus credenciales)
const dbConfig = {
  user: "postgres",
  host: "localhost",
  database: "dw_ensafi",
  password: "2801",
  port: 5432,
};

// Variables globales para normalización
let model;
let maxIngreso = 1;
let maxDeuda = 1;

// --- FUNCIÓN PRINCIPAL DE ENTRENAMIENTO ---
async function entrenarModelo() {
  console.log("--- 1. Conectando a PostgreSQL ---");
  const client = new Client(dbConfig);

  try {
    await client.connect();

    // Extraemos los 4 datos clave
    const query = `
            SELECT 
                f.ingreso_mensual_hogar, 
                f.monto_deuda_total,
                d.edad_jefe_hogar,
                d.numero_dependientes
            FROM fact_endeudamiento f
            JOIN dim_demografia d ON f.id_demografia = d.id_demografia
            LIMIT 3000; 
        `;

    const res = await client.query(query);
    let data = res.rows;

    // Limpieza de datos (Evitamos nulos que causan NaN)
    data = data
      .filter((row) => row.ingreso_mensual_hogar != null && row.monto_deuda_total != null)
      .map((row) => ({
        ingreso: parseFloat(row.ingreso_mensual_hogar) || 0,
        deuda: parseFloat(row.monto_deuda_total) || 0,
        edad: parseInt(row.edad_jefe_hogar) || 30,
        dependientes: parseInt(row.numero_dependientes) || 0,
      }));

    if (data.length === 0) return;

    // Calculamos máximos para normalizar (Escalar datos entre 0 y 1)
    maxIngreso = Math.max(...data.map((d) => d.ingreso)) || 10000;
    maxDeuda = Math.max(...data.map((d) => d.deuda)) || 10000;

    console.log(`--- 2. Generando Etiquetas usando las 4 Variables ---`);

    const inputs = [];
    const labels = [];

    data.forEach((row) => {
      // === LÓGICA DE NEGOCIO EXPERTA (Aquí usamos todos los datos) ===

      // A. IMPACTO DE DEPENDIENTES
      // Asumimos costo de vida básico de $1,500 por persona dependiente + $2,000 base
      const gastoVida = 2000 + row.dependientes * 1500;

      // Ingreso Real = Lo que gana - Lo que cuesta vivir
      let ingresoDisponible = row.ingreso - gastoVida;

      // B. CÁLCULO DE RATIO BASE
      let scoreRiesgo = 0;

      if (ingresoDisponible <= 0) {
        // Si no le alcanza para comer, el riesgo es altísimo sin importar la deuda
        scoreRiesgo = 5.0; // Valor alto arbitrario
      } else {
        // Qué porcentaje de su dinero libre compromete la deuda
        scoreRiesgo = row.deuda / ingresoDisponible;
      }

      // C. IMPACTO DE LA EDAD (Penalización)
      // Si es mayor de 60 años, es más riesgoso tener deuda (menos futuro laboral)
      if (row.edad > 60) {
        scoreRiesgo = scoreRiesgo * 1.3; // Aumentamos el riesgo un 30%
      }
      // Si es muy joven (<21), riesgo levemente mayor por inexperiencia
      if (row.edad < 21) {
        scoreRiesgo = scoreRiesgo * 1.1;
      }

      // === CLASIFICACIÓN FINAL ===
      // Definimos las etiquetas (0, 1, 2) basadas en el Score calculado con las 4 variables
      let etiqueta;
      if (scoreRiesgo < 0.4) etiqueta = 0; // ESTABLE
      else if (scoreRiesgo < 1.0) etiqueta = 1; // RIESGO MODERADO
      else etiqueta = 2; // CRÍTICO

      // Guardamos Inputs Normalizados (0 a 1)
      inputs.push([row.ingreso / maxIngreso, row.deuda / maxDeuda, row.edad / 100, row.dependientes / 10]);

      labels.push(etiqueta);
    });

    // --- 3. CONSTRUCCIÓN DE LA RED NEURONAL ---
    const xsTensor = tf.tensor2d(inputs);
    const ysTensor = tf.oneHot(tf.tensor1d(labels, "int32"), 3);

    model = tf.sequential();

    // Capa de entrada + Oculta 1
    model.add(
      tf.layers.dense({
        units: 32,
        activation: "relu",
        inputShape: [4],
      })
    );

    // Capa Oculta 2 (Para entender relaciones complejas como Edad vs Deuda)
    model.add(
      tf.layers.dense({
        units: 16,
        activation: "relu",
      })
    );

    // Capa de Salida (3 Clases)
    model.add(
      tf.layers.dense({
        units: 3,
        activation: "softmax",
      })
    );

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    console.log("--- 4. Entrenando Modelo... ---");
    await model.fit(xsTensor, ysTensor, {
      epochs: 50,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if ((epoch + 1) % 10 === 0) {
            console.log(`Época ${epoch + 1}: Pérdida=${logs.loss.toFixed(4)}, Precisión=${logs.acc.toFixed(4)}`);
          }
        },
      },
    });

    xsTensor.dispose();
    ysTensor.dispose();
    console.log("--- LISTO: Servidor esperando peticiones ---");
  } catch (err) {
    console.error("ERROR:", err);
  } finally {
    await client.end();
  }
}

// --- ENDPOINT PREDICTIVO ---
app.post("/predict", (req, res) => {
  if (!model) return res.status(503).json({ error: "Modelo cargando..." });

  const { ingreso, deuda, edad, dependientes } = req.body;

  // Normalizamos IGUAL que en el entrenamiento
  const inputData = [
    parseFloat(ingreso) / maxIngreso,
    parseFloat(deuda) / maxDeuda,
    parseFloat(edad) / 100,
    parseFloat(dependientes) / 10,
  ];

  const tensor = tf.tensor2d([inputData]);
  const prediction = model.predict(tensor);
  const resultData = prediction.dataSync(); // Probabilidades

  // Obtenemos la clase ganadora
  const maxProb = Math.max(...resultData);
  const classIndex = resultData.indexOf(maxProb);
  const niveles = ["ESTABLE", "EN RIESGO", "CRÍTICO"];

  res.json({
    nivel: niveles[classIndex],
    probabilidad: (maxProb * 100).toFixed(2),
  });

  tensor.dispose();
});

// Inicializar
app.listen(port, () => {
  console.log(`Servidor iniciado en http://localhost:${port}`);
  entrenarModelo();
});
