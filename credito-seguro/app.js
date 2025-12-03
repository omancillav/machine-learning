const tf = require("@tensorflow/tfjs");
const express = require("express");
const bodyParser = require("body-parser");
const { Client } = require("pg");

const app = express();
const port = 3000;

app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.use(express.static("public"));

const dbConfig = {
  user: "postgres",
  host: "localhost",
  database: "dw_ensafi",
  password: "2801",
  port: 5432,
};

let model;
let maxIngreso = 1;
let maxDeuda = 1;
let maxLimiteCalculado = 1;

async function entrenarModelo() {
  console.log("--- 1. Conectando a PostgreSQL ---");
  const client = new Client(dbConfig);

  try {
    await client.connect();

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

    // Limpieza
    data = data
      .filter((row) => row.ingreso_mensual_hogar != null)
      .map((row) => ({
        ingreso: parseFloat(row.ingreso_mensual_hogar) || 0,
        deuda: parseFloat(row.monto_deuda_total) || 0,
        edad: parseInt(row.edad_jefe_hogar) || 30,
        dependientes: parseInt(row.numero_dependientes) || 0,
      }));

    if (data.length === 0) return;

    console.log(`--- 2. Calculando Límites (Ground Truth Mejorado) ---`);

    const inputs = [];
    const labels = [];
    let limitesReales = [];

    data.forEach((row) => {
      // A. Cálculo Base (Ingreso Disponible)
      const gastoBasico = 2000;
      const costoHijo = 1500;
      const gastosTotales = gastoBasico + row.dependientes * costoHijo;

      // Asumimos que debe pagar al menos el 10% de su deuda actual al mes
      let ingresoDisponible = row.ingreso - gastosTotales - row.deuda * 0.1;

      // Base inicial: prestamos el 50% de lo que le sobra
      let limiteSeguro = ingresoDisponible * 0.5;

      // ==========================================
      // B. CORRECCIÓN DE LÓGICA DE EDAD (CURVA)
      // ==========================================

      if (row.edad < 21) {
        // MUY JOVEN (18-20): Alto riesgo por inexperiencia/inestabilidad.
        // Penalizamos severamente (reducimos 40%)
        limiteSeguro = limiteSeguro * 0.6;
      } else if (row.edad >= 21 && row.edad < 25) {
        // JOVEN ADULTO (21-24): Riesgo moderado.
        // Penalizamos levemente (reducimos 10%)
        limiteSeguro = limiteSeguro * 0.9;
      } else if (row.edad >= 25 && row.edad <= 55) {
        // EDAD DE ORO (25-55): Mayor estabilidad laboral y financiera.
        // ¡Bonificamos! Les damos un 10% extra de confianza.
        limiteSeguro = limiteSeguro * 1.1;
      } else {
        // TERCERA EDAD (>55): Riesgo de retiro/salud.
        // Penalizamos (reducimos 30%)
        limiteSeguro = limiteSeguro * 0.7;
      }

      // Nadie puede tener límite negativo
      if (limiteSeguro < 0) limiteSeguro = 0;

      limitesReales.push(limiteSeguro);
    });

    // CALCULAR MÁXIMOS PARA NORMALIZAR
    maxIngreso = Math.max(...data.map((d) => d.ingreso)) || 10000;
    maxDeuda = Math.max(...data.map((d) => d.deuda)) || 10000;
    maxLimiteCalculado = Math.max(...limitesReales) || 10000;

    // LLENAR TENSORES
    data.forEach((row, index) => {
      inputs.push([
        row.ingreso / maxIngreso,
        row.deuda / maxDeuda,
        row.edad / 100, // Edad normalizada sobre 100 años
        row.dependientes / 10,
      ]);

      labels.push(limitesReales[index] / maxLimiteCalculado);
    });

    // --- 3. MODELO DE REGRESIÓN ---
    const xsTensor = tf.tensor2d(inputs);
    const ysTensor = tf.tensor2d(labels, [labels.length, 1]);

    model = tf.sequential();

    // Capas densas (Deep Learning)
    model.add(tf.layers.dense({ units: 32, activation: "relu", inputShape: [4] }));
    model.add(tf.layers.dense({ units: 16, activation: "relu" }));

    // Salida Lineal (Monto)
    model.add(tf.layers.dense({ units: 1, activation: "linear" }));

    model.compile({
      optimizer: tf.train.adam(0.01),
      loss: "meanSquaredError",
    });

    console.log("--- 4. Entrenando... ---");
    await model.fit(xsTensor, ysTensor, {
      epochs: 50,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if ((epoch + 1) % 10 === 0) {
            console.log(`Época ${epoch + 1}: Error=${logs.loss.toFixed(6)}`);
          }
        },
      },
    });

    xsTensor.dispose();
    ysTensor.dispose();
    console.log("--- LISTO ---");
  } catch (err) {
    console.error("ERROR:", err);
  } finally {
    await client.end();
  }
}

app.post("/predict", (req, res) => {
  if (!model) return res.status(503).json({ error: "Modelo entrenando..." });

  const { ingreso, deuda, edad, dependientes } = req.body;

  const edadNum = parseFloat(edad);
  if (edadNum < 18 || edadNum > 99) {
    return res.status(400).json({ error: "La edad debe estar entre 18 y 99 años." });
  }

  const inputData = [
    parseFloat(ingreso) / maxIngreso,
    parseFloat(deuda) / maxDeuda,
    parseFloat(edad) / 100,
    parseFloat(dependientes) / 10,
  ];

  const tensor = tf.tensor2d([inputData]);
  const prediction = model.predict(tensor);
  const valorNormalizado = prediction.dataSync()[0];

  let limitePredicho = valorNormalizado * maxLimiteCalculado;
  if (limitePredicho < 0) limitePredicho = 0;

  res.json({
    limite_sugerido: limitePredicho.toFixed(2),
    moneda: "MXN",
  });

  tensor.dispose();
});

app.listen(port, () => {
  console.log(`Servidor iniciado en http://localhost:${port}`);
  entrenarModelo();
});
