const tf = require("@tensorflow/tfjs");
const { Pool } = require("pg");

const pool = new Pool({
  user: "postgres",
  host: "localhost",
  database: "dw_ensafi",
  password: "2801",
  port: 5432,
});

class FinancialAdvisor {
  constructor() {
    this.model = this.createModel();
    this.isTrained = false;

    // DICCIONARIO DE CONOCIMIENTO EXPERTO
    this.planes_info = {
      0: {
        titulo: "Plan de Consolidación y Rescate",
        descripcion:
          "Tu nivel de endeudamiento es crítico. Necesitas reestructurar tus pasivos inmediatamente para evitar la quiebra técnica.",
        acciones: [
          'Acude a tu banco y solicita una "Compra de Cartera" para unificar deudas.',
          "Destruye o congela tus tarjetas de crédito temporalmente.",
          'Aplica el método "Bola de Nieve": paga primero la deuda más pequeña.',
          "No adquieras nuevos créditos en los próximos 12 meses.",
        ],
      },
      1: {
        titulo: "Plan de Austeridad y Control",
        descripcion:
          "Tus ingresos no cubren tus gastos reales o tu ahorro es negativo. Estás viviendo por encima de tus posibilidades.",
        acciones: [
          "Descarga una app de control de gastos y registra cada peso durante 30 días.",
          'Elimina "gastos hormiga" (cafés, suscripciones no usadas, comidas fuera).',
          "Reducir servicios fijos (plan de celular, internet, cable).",
          "Buscar una fuente de ingreso adicional temporal.",
        ],
      },
      2: {
        titulo: "Plan de Inversión y Crecimiento",
        descripcion:
          "Tienes salud financiera y excedente de capital. Es un error mantener tu dinero estático perdiendo valor contra la inflación.",
        acciones: [
          "Abre una cuenta en CETES Directo para tu fondo de emergencia.",
          "Diversifica: 60% Renta Fija (Bonos) y 40% Renta Variable (ETFs/Acciones).",
          "Considera aportaciones voluntarias a tu AFORE (deducibles de impuestos).",
          "Evalúa bienes raíces si tu capital lo permite.",
        ],
      },
      3: {
        titulo: "Plan de Blindaje y Protección",
        descripcion:
          "Aunque tus finanzas parecen estables, estás en alto riesgo por falta de cobertura médica o social.",
        acciones: [
          "Prioridad #1: Contrata un Seguro de Gastos Médicos Mayores (SGMM).",
          "Crea un fondo de emergencia equivalente a 3 meses de tus gastos.",
          "Si eres freelance, inscríbete al régimen voluntario del IMSS.",
          "Revisa seguros de vida si tienes dependientes económicos.",
        ],
      },
    };
  }

  createModel() {
    const model = tf.sequential();
    // Aumentamos neuronas para captar mejor la relación Edad/Región
    model.add(tf.layers.dense({ inputShape: [5], units: 64, activation: "relu" }));
    model.add(tf.layers.dense({ units: 32, activation: "relu" }));
    model.add(tf.layers.dense({ units: 4, activation: "softmax" })); // 4 Planes

    model.compile({
      optimizer: tf.train.adam(0.002), // Learning rate un poco más agresivo
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
    return model;
  }

  async fetchData() {
    // Consulta exacta a tu DW ENSAFI
    const query = `
            SELECT 
                d.edad_jefe_hogar,
                f.ratio_deuda_ingreso,
                r.id_region,
                CASE WHEN d.tiene_seguridad_social THEN 1 ELSE 0 END as tiene_seguridad_social,
                f.ingreso_mensual_hogar,
                f.monto_deuda_total
            FROM fact_endeudamiento f
            JOIN dim_demografia d ON f.id_demografia = d.id_demografia
            JOIN dim_geografia g ON f.id_geografia = g.id_geografia
            JOIN dim_estado e ON g.fk_id_estado = e.id_estado
            JOIN dim_region r ON e.fk_id_region = r.id_region
            LIMIT 3000;
        `;
    const res = await pool.query(query);
    return res.rows;
  }

  determineBestPlan(row) {
    // Lógica Experta para Entrenar (Etiquetado)
    const ahorro_estimado = row.ingreso_mensual_hogar - row.monto_deuda_total * 0.15 - row.ingreso_mensual_hogar * 0.5; // Asumiendo 50% gasto vital y 15% pago deuda

    // 1. Prioridad: Deuda impagable
    if (row.ratio_deuda_ingreso > 0.4) return 0;
    // 2. Prioridad: Riesgo de salud (sin seguridad social), especialmente si es mayor o tiene ingresos medios
    if (row.tiene_seguridad_social === 0 && row.edad_jefe_hogar > 30) return 3;
    // 3. Prioridad: Déficit mensual
    if (ahorro_estimado < 0) return 1;
    // 4. Prioridad: Excedente
    if (ahorro_estimado > 0) return 2;

    return 1; // Default Austeridad
  }

  async train() {
    console.log("--> [ETL] Extrayendo datos de PostgreSQL...");
    const data = await this.fetchData();

    if (data.length === 0) return console.error("Error: DW vacío.");

    console.log(`--> [ML] Entrenando con ${data.length} perfiles históricos...`);

    const xs = data.map((row) => {
      const ahorro_proxy = row.ingreso_mensual_hogar - row.monto_deuda_total * 0.1;
      return [
        row.edad_jefe_hogar / 100,
        parseFloat(row.ratio_deuda_ingreso),
        parseInt(row.id_region) / 10,
        parseInt(row.tiene_seguridad_social),
        ahorro_proxy / 50000, // Normalización ajustada
      ];
    });

    const ysIndices = data.map((row) => this.determineBestPlan(row));
    const ys = tf.oneHot(tf.tensor1d(ysIndices, "int32"), 4);
    const xsTensor = tf.tensor2d(xs);

    await this.model.fit(xsTensor, ys, {
      epochs: 60,
      batchSize: 64,
      shuffle: true,
      verbose: 0,
    });

    this.isTrained = true;
    console.log("--> [READY] Modelo de Inteligencia Artificial cargado.");

    xsTensor.dispose();
    ys.dispose();
  }

  predict(inputData) {
    if (!this.isTrained) return { error: "Modelo entrenando... espera 10 segundos." };

    // 1. Inferencia de la Red Neuronal
    const inputTensor = tf.tensor2d([
      [
        inputData.edad / 100,
        inputData.ratio_deuda,
        inputData.id_region / 10,
        inputData.seguridad_social,
        inputData.ahorro / 50000,
      ],
    ]);

    const prediction = this.model.predict(inputTensor);
    const bestClassIndex = prediction.argMax(1).dataSync()[0];
    const confidence = prediction.dataSync()[bestClassIndex];

    inputTensor.dispose();

    // 2. Recuperar información experta
    const planInfo = this.planes_info[bestClassIndex];

    // 3. Generar "Por qué" personalizado (Explicabilidad)
    let diagnostico = "";

    // La IA usó todos los datos, aquí explicamos cuáles pesaron más para ESTA clase
    if (bestClassIndex === 0) {
      diagnostico = `Detectamos que destinas el ${(inputData.ratio_deuda * 100).toFixed(
        0
      )}% de tu ingreso a deuda. Esto es insostenible a largo plazo.`;
    } else if (bestClassIndex === 3) {
      diagnostico = `A tus ${inputData.edad} años, no contar con seguridad social representa el mayor riesgo para tu patrimonio, más allá de tus ingresos.`;
    } else if (bestClassIndex === 1) {
      diagnostico = `Tu capacidad de ahorro calculada es baja o negativa. Tus ingresos actuales en la región (zona ${inputData.id_region}) deberían rendir más.`;
    } else {
      diagnostico = `Tienes un perfil financiero sólido con capacidad de ahorro. Tu perfil de edad (${inputData.edad} años) es ideal para interés compuesto.`;
    }

    return {
      titulo: planInfo.titulo,
      descripcion: planInfo.descripcion,
      diagnostico_personalizado: diagnostico,
      acciones: planInfo.acciones,
      confianza: (confidence * 100).toFixed(1),
    };
  }
}

module.exports = new FinancialAdvisor();
