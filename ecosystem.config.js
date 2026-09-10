module.exports = {
  apps: [{
    name: "marketing-tool",
    script: "app.py",
    interpreter: "./venv/bin/python",
    cwd: "/root/Marketing_tool",
    env: {
      GEMINI_API_KEY: process.env.GEMINI_API_KEY
    }
  }, {
    // Vision Lab's worker - the CPU-bound half of an analysis (ffmpeg, the
    // saliency model, OCR), kept out of the API process so it cannot stall it.
    // Run as a module: vision_lab/worker.py uses package imports. It reads its
    // own .env from cwd.
    //
    // kill_timeout: on a restart - that is, on every deploy - the worker stops
    // claiming and finishes the job in flight. Two minutes covers the longest
    // analysis; pm2's default of 1.6 s would kill it mid-job instead.
    name: "vision-worker",
    script: "./venv/bin/python",
    args: "-m vision_lab.worker",
    interpreter: "none",
    cwd: "/root/Marketing_tool",
    kill_timeout: 120000,
    autorestart: true
  }]
}
