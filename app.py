from flask import Flask, render_template, request, jsonify
import math
import json

app = Flask(__name__)


# ─────────────────────────────────────────────
#  Safe expression parser
# ─────────────────────────────────────────────
SAFE_NAMES = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "exp": math.exp, "log": math.log, "log10": math.log10,
    "sqrt": math.sqrt, "pi": math.pi, "e": math.e,
    "abs": abs,
}


def safe_eval(expr: str, x: float) -> float:
    """Evaluate a mathematical expression safely with a given x value."""
    allowed = {**SAFE_NAMES, "x": x}
    try:
        return float(eval(expr, {"__builtins__": {}}, allowed))
    except Exception as exc:
        raise ValueError(f"Cannot evaluate expression '{expr}': {exc}")


def numerical_jacobian(model_expr: str, params: list[float],
                        x_data: list[float], h: float = 1e-6) -> list[list[float]]:
    """
    Compute the Jacobian matrix J[i][j] = d f(x_i; params) / d params[j]
    using central finite differences.
    """
    m = len(x_data)
    n = len(params)
    J = [[0.0] * n for _ in range(m)]

    for j in range(n):
        p_fwd = params[:]
        p_bwd = params[:]
        p_fwd[j] += h
        p_bwd[j] -= h
        for i, xi in enumerate(x_data):
            # Inject param values into expression by substitution tokens a0..a9
            f_fwd = eval_model(model_expr, p_fwd, xi)
            f_bwd = eval_model(model_expr, p_bwd, xi)
            J[i][j] = (f_fwd - f_bwd) / (2 * h)
    return J


def eval_model(expr: str, params: list[float], x: float) -> float:
    """
    Evaluate model expression with param tokens a0, a1, … and variable x.
    Supports up to 10 parameters.
    """
    env = {**SAFE_NAMES, "x": x}
    for k, v in enumerate(params):
        env[f"a{k}"] = v
    try:
        return float(eval(expr, {"__builtins__": {}}, env))
    except Exception as exc:
        raise ValueError(f"Model evaluation error: {exc}")


def mat_mul_AtA(A: list[list[float]]) -> list[list[float]]:
    """Compute A^T A."""
    n = len(A[0])
    AtA = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            AtA[i][j] = sum(A[k][i] * A[k][j] for k in range(len(A)))
    return AtA


def mat_mul_Atv(A: list[list[float]], v: list[float]) -> list[float]:
    """Compute A^T v."""
    n = len(A[0])
    result = [0.0] * n
    for j in range(n):
        result[j] = sum(A[i][j] * v[i] for i in range(len(A)))
    return result


def solve_linear(A: list[list[float]], b: list[float]) -> list[float]:
    """Solve Ax = b using Gaussian elimination with partial pivoting."""
    n = len(b)
    # Augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]

    for col in range(n):
        # Partial pivot
        max_row = max(range(col, n), key=lambda r: abs(M[r][col]))
        M[col], M[max_row] = M[max_row], M[col]
        if abs(M[col][col]) < 1e-14:
            raise ValueError("Singular matrix encountered in linear solve.")
        pivot = M[col][col]
        for row in range(col + 1, n):
            factor = M[row][col] / pivot
            for k in range(col, n + 1):
                M[row][k] -= factor * M[col][k]

    # Back substitution
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = M[i][n]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x


def gauss_newton(model_expr: str, x_data: list[float], y_data: list[float],
                 init_params: list[float], max_iter: int = 100,
                 tol: float = 1e-8) -> dict:
    """
    Gauss-Newton nonlinear least squares algorithm.

    Returns a dict with:
        params      – final parameter estimates
        iterations  – list of per-iteration details
        residuals   – final residuals
        rss         – residual sum of squares
        converged   – bool
    """
    params = init_params[:]
    iterations = []

    for it in range(max_iter):
        # Residual vector r_i = y_i - f(x_i; params)
        residuals = [y_data[i] - eval_model(model_expr, params, x_data[i])
                     for i in range(len(x_data))]
        rss = sum(r ** 2 for r in residuals)

        # Jacobian
        J = numerical_jacobian(model_expr, params, x_data)

        # Normal equations: (J^T J) delta = J^T r
        JtJ = mat_mul_AtA(J)
        Jtr = mat_mul_Atv(J, residuals)

        try:
            delta = solve_linear(JtJ, Jtr)
        except ValueError:
            break

        norm_delta = math.sqrt(sum(d ** 2 for d in delta))

        iterations.append({
            "iteration": it + 1,
            "params": [round(p, 8) for p in params],
            "rss": round(rss, 10),
            "delta_norm": round(norm_delta, 10),
            "residuals": [round(r, 6) for r in residuals],
        })

        params = [params[k] + delta[k] for k in range(len(params))]

        if norm_delta < tol:
            # Final residuals after update
            residuals = [y_data[i] - eval_model(model_expr, params, x_data[i])
                         for i in range(len(x_data))]
            rss = sum(r ** 2 for r in residuals)
            iterations.append({
                "iteration": it + 2,
                "params": [round(p, 8) for p in params],
                "rss": round(rss, 10),
                "delta_norm": 0.0,
                "residuals": [round(r, 6) for r in residuals],
            })
            return {
                "params": params,
                "iterations": iterations,
                "residuals": residuals,
                "rss": rss,
                "converged": True,
                "num_iterations": it + 1,
            }

    # Did not converge within max_iter
    residuals = [y_data[i] - eval_model(model_expr, params, x_data[i])
                 for i in range(len(x_data))]
    rss = sum(r ** 2 for r in residuals)
    return {
        "params": params,
        "iterations": iterations,
        "residuals": residuals,
        "rss": rss,
        "converged": False,
        "num_iterations": max_iter,
    }


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json(force=True)

    # Parse inputs
    try:
        model_expr = data.get("model", "").strip()
        if not model_expr:
            return jsonify({"error": "Model expression is required."}), 400

        x_raw = data.get("x_data", "")
        y_raw = data.get("y_data", "")
        init_raw = data.get("init_params", "")

        x_data = [float(v.strip()) for v in x_raw.split(",") if v.strip()]
        y_data = [float(v.strip()) for v in y_raw.split(",") if v.strip()]
        init_params = [float(v.strip()) for v in init_raw.split(",") if v.strip()]

        if len(x_data) != len(y_data):
            return jsonify({"error": "x_data and y_data must have the same length."}), 400
        if len(x_data) < 2:
            return jsonify({"error": "At least 2 data points are required."}), 400
        if len(init_params) < 1:
            return jsonify({"error": "At least one initial parameter is required."}), 400

        max_iter = int(data.get("max_iter", 100))
        tol = float(data.get("tol", 1e-8))

        # Validate model expression
        eval_model(model_expr, init_params, x_data[0])

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        result = gauss_newton(model_expr, x_data, y_data, init_params,
                              max_iter=max_iter, tol=tol)
    except Exception as exc:
        return jsonify({"error": f"Computation error: {exc}"}), 500

    # Build fitted curve points for chart
    x_min, x_max = min(x_data), max(x_data)
    span = x_max - x_min or 1.0
    curve_x = [x_min + span * i / 99 for i in range(100)]
    curve_y = []
    for cx in curve_x:
        try:
            curve_y.append(round(eval_model(model_expr, result["params"], cx), 6))
        except Exception:
            curve_y.append(None)

    result["curve_x"] = [round(v, 6) for v in curve_x]
    result["curve_y"] = curve_y
    result["x_data"] = x_data
    result["y_data"] = y_data
    result["params"] = [round(p, 8) for p in result["params"]]

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
