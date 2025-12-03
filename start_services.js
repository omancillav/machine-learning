const { spawn } = require("child_process");
const path = require("path");

const services = [
  { name: "VULNERABILIDAD", dir: "clasificador_vulnerabilidad", script: "app.js", color: "\x1b[36m" }, // Cyan
  { name: "CREDITO", dir: "credito-seguro", script: "app.js", color: "\x1b[32m" }, // Green
  { name: "ADVISOR", dir: "smart-advisor", script: "app.js", color: "\x1b[33m" }, // Yellow
];

console.log("\x1b[1m\x1b[37m--- INICIANDO SUITE FINANCIERA ---\x1b[0m\n");

services.forEach((service) => {
  const servicePath = path.join(__dirname, service.dir);

  console.log(`${service.color}[${service.name}] Iniciando en ${servicePath}...\x1b[0m`);

  const child = spawn("node", [service.script], {
    cwd: servicePath,
    shell: true,
    stdio: "pipe",
  });

  child.stdout.on("data", (data) => {
    const lines = data.toString().trim().split("\n");
    lines.forEach((line) => {
      console.log(`${service.color}[${service.name}] ${line}\x1b[0m`);
    });
  });

  child.stderr.on("data", (data) => {
    console.error(`\x1b[31m[${service.name} ERROR] ${data.toString().trim()}\x1b[0m`);
  });

  child.on("close", (code) => {
    console.log(`\x1b[31m[${service.name}] Proceso terminado con código ${code}\x1b[0m`);
  });
});
