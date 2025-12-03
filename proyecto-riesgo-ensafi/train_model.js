const tf = require("@tensorflow/tfjs");
const pool = require("./db");
const fs = require("fs");
const path = require("path");

const NIVELES_EDUCATIVOS = ["Sin estudios", "Primaria", "Secundaria", "Preparatoria", "Licenciatura", "Posgrado"];
const NIVELES_RIESGO = ["Bajo", "Medio", "Alto"];
const DIR_MODELO = "./modelo_riesgo";

// --- FUNCIÓN DE BARAJAR (Fisher-Yates Shuffle) ---
// Esto es vital para romper el orden del SQL
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
}

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
  console.log(`Modelo guardado en: ${dirPath}`);
}

async function run() {
  console.log("--- Iniciando Entrenamiento (Corregido) ---");

  // 1. Obtener datos (Vienen ordenados por bloques: Alto -> Medio -> Bajo)
  const queryBalanceada = `
        (SELECT * FROM vista_datos_ml WHERE nivel_riesgo = 'Alto' LIMIT 1500)
        UNION ALL
        (SELECT * FROM vista_datos_ml WHERE nivel_riesgo = 'Medio' LIMIT 1500)
        UNION ALL
        (SELECT * FROM vista_datos_ml WHERE nivel_riesgo = 'Bajo' LIMIT 1500)
    `;

  const res = await pool.query(queryBalanceada);
  let data = res.rows;

  // === CORRECCIÓN CRÍTICA: BARAJAR DATOS ===
  console.log("Barajando datos para evitar Val_Acc=0...");
  data = shuffleArray(data);
  // Ahora 'data' tiene los riesgos mezclados aleatoriamente

  // 2. Preprocesamiento
  const edades = data.map((d) => d.edad_jefe_hogar / 100);
  const educacionIdx = data.map((d) => NIVELES_EDUCATIVOS.indexOf(d.nivel_educativo_jefe));
  const internet = data.map((d) => (d.tiene_internet ? 1 : 0));
  const regionesIdx = data.map((d) => parseInt(d.id_region) - 1);

  const tEdad = tf.tensor2d(edades, [edades.length, 1]);
  const tEduc = tf.oneHot(tf.tensor1d(educacionIdx, "int32"), NIVELES_EDUCATIVOS.length);
  const tInter = tf.tensor2d(internet, [internet.length, 1]);
  const tRegion = tf.oneHot(tf.tensor1d(regionesIdx, "int32"), 8);

  const xs = tf.concat([tEdad, tEduc, tInter, tRegion], 1);

  const outputs = data.map((d) => NIVELES_RIESGO.indexOf(d.nivel_riesgo));
  const ys = tf.oneHot(tf.tensor1d(outputs, "int32"), 3);

  // 3. Modelo (Ligeramente ajustado para más potencia)
  const model = tf.sequential();

  model.add(
    tf.layers.dense({
      inputShape: [xs.shape[1]],
      units: 128, // Aumentamos neuronas para captar patrones sutiles
      activation: "relu",
    })
  );

  model.add(tf.layers.dropout({ rate: 0.3 })); // Dropout un poco más alto

  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dense({ units: 32, activation: "relu" }));

  model.add(
    tf.layers.dense({
      units: 3,
      activation: "softmax",
    })
  );

  model.compile({
    optimizer: tf.train.adam(0.0005), // Learning rate más fino
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  // 4. Entrenamiento
  await model.fit(xs, ys, {
    epochs: 100, // 100 suelen ser suficientes si los datos están bien mezclados
    batchSize: 32,
    validationSplit: 0.2, // Ahora sí funcionará bien
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if ((epoch + 1) % 10 === 0) {
          console.log(
            `Epoca ${epoch + 1}: Loss=${logs.loss.toFixed(4)} Acc=${logs.acc.toFixed(4)} Val_Acc=${logs.val_acc.toFixed(
              4
            )}`
          );
        }
      },
    },
  });

  await guardarModeloManual(model, DIR_MODELO);

  xs.dispose();
  ys.dispose();
  tEdad.dispose();
  tEduc.dispose();
  tInter.dispose();
  tRegion.dispose();
  pool.end();
}

run().catch((err) => console.error(err));
