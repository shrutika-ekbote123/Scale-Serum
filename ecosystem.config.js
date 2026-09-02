module.exports = {
  apps: [{
    name: "marketing-tool",
    script: "app.py",
    interpreter: "./venv/bin/python",
    cwd: "/root/Marketing_tool",
    env: {
      GEMINI_API_KEY: process.env.GEMINI_API_KEY
    }
  }]
}
