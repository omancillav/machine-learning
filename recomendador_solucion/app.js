const tf = require("@tensorflow/tfjs");
const express = require("express");
const bodyParser = require("body-parser");
const { Client } = require("pg");

const app = express();
const port = 3000;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static("public"));

// --- CREDENCIALES ---
const dbConfig = {
  user: "postgres",
  host: "localhost",
  database: "dw_ensafi",
  password: "2801",
  port: 5432,
};

let model;

// Variables para normalización
let maxIngreso = 1;
let maxDeuda = 1;

// Mapeo para convertir texto de base de datos a números
// Revisa en tu BD cómo están escritos exactamente. Asumimos estos estándar:
const mapaTenencia = {
  Propia: 1,
  Pagándola: 1, // Se considera propia para efectos de patrimonio
  Rentada: 0,
  Prestada: 0.5,
  Intestada: 0.5,
  Otra: 0,
};

async function entrenarModelo() {
  console.log("--- 1. Conectando a DW Ensafi ---");
  const client = new Client(dbConfig);

  try {
    await client.connect();

    // QUERY MULTI-DIMENSIONAL
    // Unimos Hechos + Demografía + Vivienda
    const query = `
            SELECT 
                f.ingreso_mensual_hogar, 
                f.monto_deuda_total,
                d.numero_dependientes,
                v.tenencia_vivienda
            FROM fact_endeudamiento f
            JOIN dim_demografia d ON f.id_demografia = d.id_demografia
            JOIN dim_vivienda v ON f.id_vivienda = v.id_vivienda
            LIMIT 4000;
        `;

    const res = await client.query(query);
    let data = res.rows;

    // Limpieza
    data = data
      .filter((row) => row.ingreso_mensual_hogar != null)
      .map((row) => ({
        ingreso: parseFloat(row.ingreso_mensual_hogar) || 0,
        deuda: parseFloat(row.monto_deuda_total) || 0,
        dependientes: parseInt(row.numero_dependientes) || 0,
        // Limpiamos espacios y mapeamos. Si no hace match, asumimos 'Rentada' (0)
        tenencia: mapaTenencia[row.tenencia_vivienda ? row.tenencia_vivienda.trim() : ""] || 0,
      }));

    if (data.length === 0) return;

    // Máximos para normalización
    maxIngreso = Math.max(...data.map((d) => d.ingreso)) || 10000;
    maxDeuda = Math.max(...data.map((d) => d.deuda)) || 10000;

    console.log(`--- 2. Generando Estrategias (Ground Truth) ---`);

    const inputs = [];
    const labels = [];

    data.forEach((row) => {
      // === LÓGICA DE SOLUCIÓN FINANCIERA ===

      // Calculamos Ratio de Endeudamiento
      // Evitamos división por cero
      let ratio = row.ingreso > 0 ? row.deuda / row.ingreso : 100;

      let planRecomendado = 0; // Por defecto Plan A

      if (ratio < 0.35) {
        // CASO 1: Deuda controlada (<35%)
        // No importa si renta o tiene casa, su problema no es grave.
        planRecomendado = 0; // PLAN A: Optimización
      } else if (ratio >= 0.35 && ratio < 0.65) {
        // CASO 2: Endeudamiento Medio/Alto
        // Aquí la VIVIENDA hace la diferencia
        if (row.tenencia === 1) {
          // Tiene casa propia -> Puede hipotecar/apalancar para pagar barato
          planRecomendado = 1; // PLAN B: Reestructuración Patrimonial
        } else {
          // Paga renta -> No tiene activos -> Debe recortar gastos ya
          planRecomendado = 2; // PLAN C: Austeridad de Emergencia
        }
      } else {
        // CASO 3: Sobreendeudamiento (>65%)
        // Si tiene dependientes, es una emergencia social
        if (row.dependientes > 2 || row.tenencia === 0) {
          planRecomendado = 2; // PLAN C: Emergencia/Legal (Riesgo de quiebra)
        } else {
          // Si está solo y tiene casa, tal vez pueda salvarse vendiendo algo
          planRecomendado = 1; // PLAN B
        }
      }

      // Inputs (4 Neuronas de entrada)
      inputs.push([
        row.ingreso / maxIngreso,
        row.deuda / maxDeuda,
        row.dependientes / 10,
        row.tenencia, // 0, 0.5 o 1
      ]);

      labels.push(planRecomendado);
    });

    // --- 3. MODELO MULTICLASE (3 PLANES) ---
    const xsTensor = tf.tensor2d(inputs);
    // One-Hot Encoding para 3 clases: [1,0,0], [0,1,0], [0,0,1]
    const ysTensor = tf.oneHot(tf.tensor1d(labels, "int32"), 3);

    model = tf.sequential();

    // Capas Ocultas
    model.add(tf.layers.dense({ units: 32, activation: "relu", inputShape: [4] }));
    model.add(tf.layers.dense({ units: 16, activation: "relu" }));

    // Salida Softmax (Probabilidad repartida entre los 3 planes)
    model.add(
      tf.layers.dense({
        units: 3,
        activation: "softmax",
      })
    );

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: "categoricalCrossentropy", // Pérdida Multiclase
      metrics: ["accuracy"],
    });

    console.log("--- 4. Entrenando Asesor Financiero... ---");
    await model.fit(xsTensor, ysTensor, {
      epochs: 50,
      shuffle: true,
    });

    xsTensor.dispose();
    ysTensor.dispose();
    console.log("--- ASESOR LISTO ---");
  } catch (err) {
    console.error("ERROR:", err);
  } finally {
    await client.end();
  }
}

// --- API ---
app.post("/predict", (req, res) => {
  if (!model) return res.status(503).json({ error: "Asesor pensando..." });

  const { ingreso, deuda, dependientes, tenencia } = req.body;

  // Convertir texto del select a número
  const valTenencia = parseInt(tenencia); // 0 o 1

  const inputData = [
    parseFloat(ingreso) / maxIngreso,
    parseFloat(deuda) / maxDeuda,
    parseFloat(dependientes) / 10,
    valTenencia,
  ];

  const tensor = tf.tensor2d([inputData]);
  const prediction = model.predict(tensor);
  const resultData = prediction.dataSync(); // Array de 3 probabilidades

  // Encontrar el plan ganador
  const maxProb = Math.max(...resultData);
  const planIndex = resultData.indexOf(maxProb);

  const planes = [
    {
      titulo: "PLAN A: Optimización y Ahorro",
      descripcion:
        "Tus finanzas son estables. Te recomendamos crear un fondo de inversión y automatizar tu ahorro del 10%.",
      color: "success",
    },
    {
      titulo: "PLAN B: Reestructuración Patrimonial",
      descripcion:
        "Tienes deuda considerable pero cuentas con respaldo. Te sugerimos consolidar tus deudas usando tu vivienda como garantía para bajar la tasa de interés.",
      color: "warning",
    },
    {
      titulo: "PLAN C: Austeridad de Emergencia",
      descripcion:
        "¡Alerta Roja! Tu nivel de deuda pone en riesgo el bienestar de tus dependientes. Necesitas asesoría legal para renegociar plazos y detener gastos no esenciales inmediatamente.",
      color: "danger",
    },
  ];

  res.json({
    plan: planes[planIndex],
    certeza: (maxProb * 100).toFixed(2),
  });

  tensor.dispose();
});

app.listen(port, () => {
  console.log(`Asesoría iniciada en http://localhost:${port}`);
  entrenarModelo();
});
