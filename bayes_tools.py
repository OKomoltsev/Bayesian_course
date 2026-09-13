"""Helpers for the Bayesian inference lectures.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML



FONT_SIZE = "1.25em"    # table text size; raise for projecting a lecture
PAD = "8px 20px"        # cell padding: vertical horizontal


def html_table(headers, rows, rule_before_last=True):
    """Render a bordered table; a rule is drawn above the last (totals) row."""
    th = f"border:1px solid #444;padding:{PAD};text-align:center;font-weight:600;"
    td = f"border:1px solid #444;padding:{PAD};text-align:center;"
    out = [f'<table style="border-collapse:collapse;margin:12px 0;font-size:{FONT_SIZE};">']
    out.append("<tr>" + "".join(f'<th style="{th}">{h}</th>' for h in headers) + "</tr>")
    for i, r in enumerate(rows):
        top = "border-top:2px solid #000;" if (rule_before_last and i == len(rows) - 1) else ""
        out.append("<tr>" + "".join(f'<td style="{td}{top}">{c}</td>' for c in r) + "</tr>")
    return HTML("".join(out) + "</table>")


def bayes_table(hypotheses, prior, likelihood, fmt="{:.4g}"):
    """Display the Bayes table and return the posterior.

    One row per Model.  prior * likelihood is the unnormalised posterior;
    the total of that column is the evidence P(Data).

    `likelihood` may be a single array, or a dict {column name: array} for
    several conditionally independent datasets.  The columns are multiplied
    together, and the prior is applied once.
    """
    prior = np.asarray(prior, dtype=float)
    if not isinstance(likelihood, dict):
        likelihood = {"likelihood": likelihood}
    terms = {k: np.asarray(v, dtype=float) for k, v in likelihood.items()}

    L = np.ones_like(prior)
    for v in terms.values():
        L = L * v
    h = prior * L                          # unnormalised posterior
    evidence = h.sum()                     # P(Data)
    posterior = h / evidence

    f = fmt.format
    rows = [[n, f(p), *[f(v[i]) for v in terms.values()], f(hi), f(po)]
            for i, (n, p, hi, po) in enumerate(zip(hypotheses, prior, h, posterior))]
    rows.append(["Totals:", f(prior.sum()), *[""] * len(terms),
                 f(evidence), f(posterior.sum())])
    display(html_table(["Hypotheses", "prior", *terms,
                        "prior × likelihood", "posterior"], rows))
    return posterior


def posterior_1d(theta, prior, likelihood):
    """Normalised 1D posterior on a grid.  P(Data) is the trapezoid integral."""
    unnorm = np.asarray(prior, float) * np.asarray(likelihood, float)
    evidence = np.trapezoid(unnorm, theta)
    return unnorm / evidence, evidence
