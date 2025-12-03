const express = require("express");
const bodyParser = require("body-parser");
const advisor = require("./advisor_model");

const app = express();
const port = 3003;

app.use(bodyParser.json());
app.use(express.static("public"));

app.post("/predict", (req, res) => {
  try {
    const data = req.body;

    // Validación estricta: Si falta algo, la IA no puede trabajar bien
    if (!data.edad || !data.ingreso || !data.deuda || !data.region) {
      return res.status(400).json({ error: "Todos los campos son obligatorios para el análisis." });
    }

    // Cálculos previos
    const ingreso = parseFloat(data.ingreso);
    const deuda = parseFloat(data.deuda);
    const ratio = ingreso > 0 ? deuda / ingreso : 1;
    // Estimamos ahorro disponible real
    const ahorro = ingreso - deuda * 0.15 - ingreso * 0.5;

    const resultado = advisor.predict({
      edad: parseFloat(data.edad),
      ratio_deuda: ratio,
      id_region: parseInt(data.region),
      seguridad_social: parseInt(data.seguridad_social),
      ahorro: ahorro,
    });

    res.json(resultado);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(port, async () => {
  console.log(`\nServidor web en http://localhost:${port}`);
  await advisor.train(); // Entrenamiento automático al inicio
});
