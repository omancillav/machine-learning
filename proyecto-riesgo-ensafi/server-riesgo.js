const express = require("express");
const tf = require("@tensorflow/tfjs");
const cors = require("cors");
const bodyParser = require("body-parser");
const fs = require("fs");

const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(express.static("public")); // Sirve el HTML

// --- CONFIGURACIÓN ---
const PORT = 3000;
const DIR_MODELO = "./modelo_riesgo";
const NIVELES_EDUCATIVOS = ["Sin estudios", "Primaria", "Secundaria", "Preparatoria", "Licenciatura", "Posgrado"];
const NIVELES_RIESGO = ["Bajo", "Medio", "Alto"];

let model;

// --- FUNCIÓN PARA CARGAR MODELO MANUALMENTE ---
async function cargarModeloManual() {
  try {
    if (!fs.existsSync(`${DIR_MODELO}/model.json`)) {
      console.error("No se encuentra el modelo. Ejecuta 'node train_model.js' primero.");
      return;
    }

    // 1. Leer archivos del disco
    const modelJson = JSON.parse(fs.readFileSync(`${DIR_MODELO}/model.json`, "utf8"));
    const weightsBuffer = fs.readFileSync(`${DIR_MODELO}/weights.bin`);

    // 2. Convertir Buffer de Node a ArrayBuffer compatible con TFJS
    const weightsArrayBuffer = weightsBuffer.buffer.slice(
      weightsBuffer.byteOffset,
      weightsBuffer.byteOffset + weightsBuffer.byteLength
    );

    // 3. Cargar usando IOHandler de memoria
    const handler = tf.io.fromMemory(modelJson, weightsArrayBuffer);
    model = await tf.loadLayersModel(handler);

    console.log("✅ Modelo de Riesgo cargado en memoria exitosamente.");
  } catch (error) {
    console.error("❌ Error cargando modelo:", error.message);
  }
}

// Iniciar carga del modelo al arrancar
cargarModeloManual();

// --- ENDPOINT DE PREDICCIÓN ---
app.post("/predict", async (req, res) => {
  if (!model) return res.status(500).json({ error: "El modelo no está listo." });

  try {
    const { edad, educacion, internet, region } = req.body;

    // --- PREPROCESAMIENTO (Debe ser IDÉNTICO a train_model.js) ---

    // 1. Edad / 100
    const tEdad = tf.tensor2d([parseInt(edad) / 100], [1, 1]);

    // 2. Educación (One Hot)
    const idxEduc = NIVELES_EDUCATIVOS.indexOf(educacion);
    if (idxEduc === -1) throw new Error("Nivel educativo no válido");
    const tEduc = tf.oneHot(tf.tensor1d([idxEduc], "int32"), NIVELES_EDUCATIVOS.length);

    // 3. Internet (0 o 1)
    const tInter = tf.tensor2d([internet ? 1 : 0], [1, 1]);

    // 4. Región (One Hot) - Restar 1 para índice 0-7
    const idxRegion = parseInt(region) - 1;
    if (idxRegion < 0 || idxRegion > 7) throw new Error("ID de región inválido (debe ser 1-8)");
    const tRegion = tf.oneHot(tf.tensor1d([idxRegion], "int32"), 8);

    // Concatenar Inputs: Edad + Educ + Inter + Region
    const inputTensor = tf.concat([tEdad, tEduc, tInter, tRegion], 1);

    // --- PREDICCIÓN ---
    const prediction = model.predict(inputTensor);
    const data = await prediction.data(); // Devuelve array [prob_bajo, prob_medio, prob_alto]

    // Obtener el índice mayor
    const maxIndex = data.indexOf(Math.max(...data));
    const resultado = NIVELES_RIESGO[maxIndex];

    // Respuesta al cliente
    res.json({
      riesgo_predicho: resultado,
      confianza: (data[maxIndex] * 100).toFixed(2) + "%",
      probabilidades: {
        Bajo: (data[0] * 100).toFixed(1) + "%",
        Medio: (data[1] * 100).toFixed(1) + "%",
        Alto: (data[2] * 100).toFixed(1) + "%",
      },
    });

    // Limpieza de memoria
    inputTensor.dispose();
    tEdad.dispose();
    tEduc.dispose();
    tInter.dispose();
    tRegion.dispose();
  } catch (err) {
    console.error(err);
    res.status(400).json({ error: err.message });
  }
});

app.listen(PORT, () => console.log(`🚀 Servidor corriendo en http://localhost:${PORT}`));
