const express = require("express");
const tf = require("@tensorflow/tfjs");
const cors = require("cors");
const bodyParser = require("body-parser");
const fs = require("fs");

const app = express();
app.use(cors());
app.use(bodyParser.json());
app.use(express.static("public"));

const PORT = 3000;
const DIR_MODELO = "./modelo_hipoteca";
const NIVELES_EDUCATIVOS = ["Sin estudios", "Primaria", "Secundaria", "Preparatoria", "Licenciatura", "Posgrado"];

let model;

async function cargarModelo() {
  try {
    if (!fs.existsSync(`${DIR_MODELO}/model.json`)) {
      console.error("⚠️ Modelo no encontrado. Ejecuta 'node train_mortgage.js'");
      return;
    }
    const modelJson = JSON.parse(fs.readFileSync(`${DIR_MODELO}/model.json`, "utf8"));
    const weightsBuffer = fs.readFileSync(`${DIR_MODELO}/weights.bin`);
    const weightsArrayBuffer = weightsBuffer.buffer.slice(
      weightsBuffer.byteOffset,
      weightsBuffer.byteOffset + weightsBuffer.byteLength
    );

    const handler = tf.io.fromMemory(modelJson, weightsArrayBuffer);
    model = await tf.loadLayersModel(handler);
    console.log("✅ Modelo de Hipoteca cargado.");
  } catch (error) {
    console.error("❌ Error cargando modelo:", error.message);
  }
}
cargarModelo();

app.post("/predict", async (req, res) => {
  if (!model) return res.status(500).json({ error: "Modelo no listo." });

  try {
    const { ingreso, edad, personas, educacion, internet } = req.body;

    const tIngreso = tf.tensor2d([parseFloat(ingreso) / 50000], [1, 1]);
    const tEdad = tf.tensor2d([parseInt(edad) / 100], [1, 1]);
    const tPers = tf.tensor2d([parseInt(personas) / 10], [1, 1]);

    const idxEduc = NIVELES_EDUCATIVOS.indexOf(educacion);
    const tEduc = tf.oneHot(tf.tensor1d([idxEduc], "int32"), 6);
    const tInter = tf.tensor2d([internet ? 1 : 0], [1, 1]);

    const inputTensor = tf.concat([tIngreso, tEdad, tPers, tEduc, tInter], 1);

    const prediction = model.predict(inputTensor);
    const score = (await prediction.data())[0];

    const porcentaje = (score * 100).toFixed(1);
    let mensaje = score > 0.7 ? "ALTA VIABILIDAD" : score > 0.4 ? "VIABILIDAD MEDIA" : "BAJA VIABILIDAD";

    res.json({ probabilidad: porcentaje + "%", mensaje: mensaje, score_raw: score });

    inputTensor.dispose();
    tIngreso.dispose();
    tEdad.dispose();
    tPers.dispose();
    tEduc.dispose();
    tInter.dispose();
  } catch (err) {
    console.error(err);
    res.status(400).json({ error: err.message });
  }
});

app.listen(PORT, () => console.log(`🚀 Servidor listo en http://localhost:${PORT}`));
