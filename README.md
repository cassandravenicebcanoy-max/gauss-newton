# PIT Project — Topic #23: Nonlinear Least Squares (Gauss–Newton)

## Project Structure

```
gauss-newton/
├── app.py               # Flask app + Gauss-Newton algorithm
├── requirements.txt     # Python dependencies
├── vercel.json          # Vercel deployment config
├── templates/
│   └── index.html       # Single-page HTML (MathJax + Chart.js)
└── static/
    ├── css/style.css    # Full stylesheet
    └── js/main.js       # Calculator logic + charting
```

## Running Locally

```bash
pip install -r requirements.txt
python app.py
# Open http://localhost:5000
```

## Deploying to Vercel

1. Install Vercel CLI: `npm i -g vercel`
2. From the project folder: `vercel`
3. Follow prompts — select Python framework
4. Your app will be live at `https://your-project.vercel.app`

## Features

- Full mathematical discussion with MathJax-rendered equations
- Two complete step-by-step worked examples
- Interactive calculator with:
  - Custom model expressions (a0, a1, … and x)
  - 4 built-in presets (exponential, Michaelis–Menten, power law, logistic)
  - Chart.js fitted curve visualization
  - Per-iteration table showing convergence
  - Residuals breakdown table
- Safe expression parsing (no `eval` with builtins)
- Pure Python implementation — no NumPy/SciPy
