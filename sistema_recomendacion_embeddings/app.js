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
  database: "dw_ensafi", // Base de datos correcta
  password: "2801", // Contraseña correcta
  port: 5432,
};

let model;

// --- DICCIONARIOS PARA MAPEO (Categoría -> Índice Entero) ---
const mapEducacion = {
  "Sin instrucción": 0,
  Preescolar: 1,
  "Primaria incompleta": 2,
  "Primaria completa": 3,
  "Secundaria incompleta": 4,
  "Secundaria completa": 5,
  Preparatoria: 6,
  Licenciatura: 7,
  Posgrado: 8,
};

// Productos Financieros (Salidas posibles)
const PRODUCTOS = [
  "Crédito Hipotecario", // 0
  "Tarjeta de Crédito Premium", // 1
  "Tarjeta de Crédito Básica", // 2
  "Crédito Personal", // 3
  "Cuenta de Ahorro Plus", // 4
  "Seguro de Vida", // 5
  "Crédito Automotriz", // 6
  "Microcrédito", // 7
];

// Variables de normalización
let maxIngreso = 1;

async function entrenarModelo() {
  console.log("--- 1. Conectando a DW Ensafi ---");
  const client = new Client(dbConfig);

  try {
    await client.connect();

    // CORRECCIÓN: Eliminamos 'f.capacidad_ahorro' que no existe en tu tabla
    const query = `
            SELECT 
                f.ingreso_mensual_hogar, 
                f.monto_deuda_total,
                d.edad_jefe_hogar, 
                d.numero_dependientes, 
                d.nivel_educativo_jefe,
                d.sexo_jefe_hogar,
                g.nombre_municipio, 
                v.tenencia_vivienda
            FROM fact_endeudamiento f
            JOIN dim_demografia d ON f.id_demografia = d.id_demografia
            JOIN dim_geografia g ON f.id_geografia = g.id_geografia
            JOIN dim_vivienda v ON f.id_vivienda = v.id_vivienda
            LIMIT 5000;
        `;

    const res = await client.query(query);
    let data = res.rows;

    // Limpieza
    data = data.filter((r) => r.ingreso_mensual_hogar != null);
    if (data.length === 0) return;

    // Normalizadores
    maxIngreso = Math.max(...data.map((d) => parseFloat(d.ingreso_mensual_hogar))) || 10000;

    console.log('--- 2. Generando Etiquetas ("Ground Truth") ---');

    const inputNumericos = [];
    const inputRegion = [];
    const inputEdu = [];
    const inputDeuda = [];
    const labels = [];

    data.forEach((row) => {
      const ingreso = parseFloat(row.ingreso_mensual_hogar) || 0;
      const deuda = parseFloat(row.monto_deuda_total) || 0;
      const edad = parseInt(row.edad_jefe_hogar) || 30;
      const dependientes = parseInt(row.numero_dependientes) || 0;

      // CORRECCIÓN: CÁLCULO MANUAL DEL AHORRO (Ya que no está en BD)
      // Estimamos gasto de vida: 2000 base + 1500 por hijo
      const gastoVida = 2000 + dependientes * 1500;
      // Estimamos que paga el 5% de su deuda al mes
      const pagoDeuda = deuda * 0.05;
      // Ahorro es lo que sobra
      let ahorro = ingreso - gastoVida - pagoDeuda;
      if (ahorro < 0) ahorro = 0; // No puede haber ahorro negativo para este input

      // --- PREPROCESAMIENTO CATEGÓRICO ---
      // 1. Región (Simulada basada en el nombre)
      let idRegion = 2;
      if (row.nombre_municipio && row.nombre_municipio.toLowerCase().includes("monterrey")) idRegion = 0;

      // 2. Educación
      let idEdu = mapEducacion[row.nivel_educativo_jefe ? row.nivel_educativo_jefe.trim() : ""] || 0;

      // 3. Categoría Deuda (Calculada)
      let ratio = ingreso > 0 ? deuda / ingreso : 0;
      let idCatDeuda = 0;
      if (ratio > 0.6) idCatDeuda = 4;
      else if (ratio > 0.4) idCatDeuda = 3;
      else if (ratio > 0.2) idCatDeuda = 2;
      else if (ratio > 0) idCatDeuda = 1;

      // --- LÓGICA EXPERTA: ¿CUÁL ES EL PRODUCTO IDEAL? ---
      let productoIdeal = 2; // Tarjeta Básica

      if (row.tenencia_vivienda === "Rentada" && ingreso > 25000 && edad > 28 && edad < 50) {
        productoIdeal = 0; // Hipotecario
      } else if (ingreso > 40000 && ratio < 0.3) {
        productoIdeal = 1; // Tarjeta Premium
      } else if (deuda > ingreso * 0.6) {
        productoIdeal = 3; // Crédito Personal (Consolidación)
      } else if (ahorro > ingreso * 0.2 && deuda === 0) {
        productoIdeal = 4; // Cuenta Ahorro Plus
      } else if (dependientes > 2 && edad > 35) {
        productoIdeal = 5; // Seguro de Vida
      } else if (ingreso > 15000 && ingreso < 30000 && ratio < 0.4) {
        productoIdeal = 6; // Automotriz
      } else if (ingreso < 8000) {
        productoIdeal = 7; // Microcrédito
      }

      // LLENAR INPUTS
      inputNumericos.push([
        ingreso / maxIngreso,
        ahorro / maxIngreso, // Usamos el ahorro calculado
        deuda / maxIngreso,
        ratio,
        edad / 100,
        dependientes / 10,
        row.sexo_jefe_hogar === "Hombre" ? 1 : 0,
        row.tenencia_vivienda === "Propia" ? 1 : 0,
      ]);

      inputRegion.push(idRegion);
      inputEdu.push(idEdu);
      inputDeuda.push(idCatDeuda);

      labels.push(productoIdeal);
    });

    // --- 3. DEFINICIÓN DEL MODELO CON EMBEDDINGS ---

    const tNumericos = tf.tensor2d(inputNumericos);
    const tRegion = tf.tensor2d(inputRegion, [inputRegion.length, 1]);
    const tEdu = tf.tensor2d(inputEdu, [inputEdu.length, 1]);
    const tDeuda = tf.tensor2d(inputDeuda, [inputDeuda.length, 1]);
    const tLabels = tf.oneHot(tf.tensor1d(labels, "int32"), 8);

    const inNum = tf.input({ shape: [8] });
    const inReg = tf.input({ shape: [1] });
    const inEdu = tf.input({ shape: [1] });
    const inDeu = tf.input({ shape: [1] });

    // Embeddings
    const embReg = tf.layers.embedding({ inputDim: 5, outputDim: 8 }).apply(inReg);
    const embEdu = tf.layers.embedding({ inputDim: 10, outputDim: 8 }).apply(inEdu);
    const embDeu = tf.layers.embedding({ inputDim: 6, outputDim: 8 }).apply(inDeu);

    const flatReg = tf.layers.flatten().apply(embReg);
    const flatEdu = tf.layers.flatten().apply(embEdu);
    const flatDeu = tf.layers.flatten().apply(embDeu);

    const concatenated = tf.layers.concatenate().apply([inNum, flatReg, flatEdu, flatDeu]);

    const dense1 = tf.layers.dense({ units: 64, activation: "relu" }).apply(concatenated);
    const dropout = tf.layers.dropout({ rate: 0.2 }).apply(dense1);
    const dense2 = tf.layers.dense({ units: 32, activation: "relu" }).apply(dropout);

    const output = tf.layers.dense({ units: 8, activation: "softmax" }).apply(dense2);

    model = tf.model({ inputs: [inNum, inReg, inEdu, inDeu], outputs: output });

    model.compile({
      optimizer: tf.train.adam(0.005),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    console.log("--- 4. Entrenando Sistema de Recomendación... ---");
    await model.fit([tNumericos, tRegion, tEdu, tDeuda], tLabels, {
      epochs: 40,
      shuffle: true,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          if ((epoch + 1) % 10 === 0) console.log(`Epoca ${epoch + 1}: Acc=${logs.acc.toFixed(4)}`);
        },
      },
    });

    tNumericos.dispose();
    tRegion.dispose();
    tEdu.dispose();
    tDeuda.dispose();
    tLabels.dispose();
    console.log("--- MODELO LISTO ---");
  } catch (err) {
    console.error("Error:", err);
  } finally {
    await client.end();
  }
}

// --- ENDPOINT API ---
app.post("/recommend", (req, res) => {
  if (!model) return res.status(503).json({ error: "Cargando IA..." });

  const { ingreso, deuda, ahorro, edad, dependientes, sexo, vivienda, regionId, eduId } = req.body;

  const numIngreso = parseFloat(ingreso) || 0;
  const numDeuda = parseFloat(deuda) || 0;
  const ratio = numIngreso > 0 ? numDeuda / numIngreso : 0;

  // Categoría deuda
  let idCatDeuda = 0;
  if (ratio > 0.6) idCatDeuda = 4;
  else if (ratio > 0.4) idCatDeuda = 3;
  else if (ratio > 0.2) idCatDeuda = 2;
  else if (ratio > 0) idCatDeuda = 1;

  // Tensores
  const tNum = tf.tensor2d([
    [
      numIngreso / maxIngreso,
      (parseFloat(ahorro) || 0) / maxIngreso,
      numDeuda / maxIngreso,
      ratio,
      parseFloat(edad) / 100,
      parseInt(dependientes) / 10,
      parseInt(sexo),
      parseInt(vivienda),
    ],
  ]);

  const tReg = tf.tensor2d([[parseInt(regionId)]]);
  const tEdu = tf.tensor2d([[parseInt(eduId)]]);
  const tDeu = tf.tensor2d([[idCatDeuda]]);

  const prediction = model.predict([tNum, tReg, tEdu, tDeu]);
  const probs = prediction.dataSync();

  const maxProb = Math.max(...probs);
  const index = probs.indexOf(maxProb);

  res.json({
    producto_recomendado: PRODUCTOS[index],
    confianza: (maxProb * 100).toFixed(2),
    todas_opciones: PRODUCTOS.map((p, i) => ({ producto: p, score: probs[i].toFixed(2) })),
  });

  tNum.dispose();
  tReg.dispose();
  tEdu.dispose();
  tDeu.dispose();
});

app.listen(port, () => {
  console.log(`Recomendador con Embeddings iniciado en http://localhost:${port}`);
  entrenarModelo();
});
