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
  }, {
    // AI Briefings' daily generator. Every few minutes it checks each brand's
    // local clock and, once it passes the generation hour, briefs yesterday -
    // once per brand and day. Network-bound (scrumdb + Gemini), so it is light.
    //
    // kill_timeout: on a deploy it finishes the brand in flight (about a
    // minute: a day's queries plus five Gemini calls) instead of being killed
    // halfway; a run cut off anyway is retried on the next pass.
    name: "briefing-worker",
    script: "./venv/bin/python",
    args: "-m ai_briefings.worker",
    interpreter: "none",
    cwd: "/root/Marketing_tool",
    kill_timeout: 90000,
    autorestart: true
  }]
}
