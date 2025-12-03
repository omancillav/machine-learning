const tf = require("@tensorflow/tfjs");
const pool = require("./db");
const fs = require("fs");
const path = require("path");

const DIR_MODELO = "./modelo_hipoteca";
const NIVELES_EDUCATIVOS = ["Sin estudios", "Primaria", "Secundaria", "Preparatoria", "Licenciatura", "Posgrado"];

// Función para mezclar datos (Shuffle)
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

// Función para guardar modelo manualmente
async function guardarModeloManual(model, dirPath) {
  if (!fs.existsSync(dirPath)) fs.mkdirSync(dirPath);
  await model.save(
    tf.io.withSaveHandler(async (artifacts) => {
      fs.writeFileSync(path.join(dirPath, "model.json"), JSON.stringify(artifacts.modelTopology));
      if (artifacts.weightData) {
        fs.writeFileSync(path.join(dirPath, "weights.bin"), Buffer.from(artifacts.weightData));
      }
      return { modelArtifactsInfo: { dateSaved: new Date(), modelTopologyType: "JSON" } };
    })
  );
  console.log(`✅ Modelo guardado en: ${dirPath}`);
}

async function run() {
  console.log("--- Entrenando Modelo: Propensión a Hipoteca ---");
  console.log("Conectando a PostgreSQL...");

  // OBTENER DATOS BALANCEADOS (50/50)
  const query = `
        (SELECT * FROM vista_hipoteca WHERE tiene_hipoteca = 1 LIMIT 2000)
        UNION ALL
        (SELECT * FROM vista_hipoteca WHERE tiene_hipoteca = 0 LIMIT 2000)
    `;

  const res = await pool.query(query);
  let data = res.rows;

  if (data.length === 0) {
    console.error("❌ Error: No se encontraron datos. Verifica que la vista SQL devuelva registros.");
    pool.end();
    return;
  }

  console.log(`Registros obtenidos: ${data.length}. Mezclando...`);
  data = shuffleArray(data);

  // PREPROCESAMIENTO
  const ingresos = data.map((d) => parseFloat(d.monto_ingreso_mensual) / 50000);
  const edades = data.map((d) => d.edad_jefe_hogar / 100);
  const personas = data.map((d) => d.numero_personas_hogar / 10);
  const educIdx = data.map((d) => NIVELES_EDUCATIVOS.indexOf(d.nivel_educativo_jefe));
  const internet = data.map((d) => (d.tiene_internet ? 1 : 0));

  const tIngreso = tf.tensor2d(ingresos, [ingresos.length, 1]);
  const tEdad = tf.tensor2d(edades, [edades.length, 1]);
  const tPers = tf.tensor2d(personas, [personas.length, 1]);
  const tEduc = tf.oneHot(tf.tensor1d(educIdx, "int32"), 6);
  const tInter = tf.tensor2d(internet, [internet.length, 1]);

  const xs = tf.concat([tIngreso, tEdad, tPers, tEduc, tInter], 1);
  const ys = tf.tensor2d(
    data.map((d) => d.tiene_hipoteca),
    [data.length, 1]
  );

  // ARQUITECTURA
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [xs.shape[1]], units: 32, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: 16, activation: "relu" }));
  model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: "binaryCrossentropy",
    metrics: ["accuracy"],
  });

  // ENTRENAMIENTO
  await model.fit(xs, ys, {
    epochs: 60,
    batchSize: 32,
    validationSplit: 0.2,
    callbacks: {
      onEpochEnd: (e, l) => {
        if ((e + 1) % 10 === 0)
          console.log(
            `Epoca ${e + 1}: Loss=${l.loss.toFixed(4)} Acc=${l.acc.toFixed(4)} Val_Acc=${l.val_acc.toFixed(4)}`
          );
      },
    },
  });

  await guardarModeloManual(model, DIR_MODELO);

  xs.dispose();
  ys.dispose();
  tIngreso.dispose();
  tEdad.dispose();
  tPers.dispose();
  tEduc.dispose();
  tInter.dispose();
  pool.end();
}

run().catch((err) => console.error(err));
