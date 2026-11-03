import numpy as np
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

D = Path("results")
X = np.load(D / "grid_X.npy")
Y = np.load(D / "grid_Y.npy")
Z_exact = np.load(D / "Z_exact.npy")
widths = [int(n) for n in np.load(D / "widths.npy")]
Z_hat = {n: np.load(D / f"Z_hat_n{n}.npy") for n in widths}
max_err = dict(zip(widths, np.load(D / "max_err.npy")))

# --- Figure 1 : surfaces approchées pour quelques largeurs ---
# shown = [2, 4, 8, 32]
shown =  [8, 64]
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}],
           [{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=[f"{n} neurones (erreur abs max = {max_err[n]:.3f})" for n in shown],
)
zmin, zmax = Z_exact.min(), Z_exact.max()
for k, n in enumerate(shown):
    fig.add_trace(
        go.Surface(x=X, y=Y, z=Z_hat[n], colorscale="Viridis",
                   cmin=zmin, cmax=zmax, showscale=(k == 0),
                   colorbar=dict(title="Z")),
        row=k // 2 + 1, col=k % 2 + 1,
    )
fig.update_layout(height=800, margin=dict(l=0, r=0, b=0, t=40))
fig.show()

# # --- Figure 2 : erreur max en fonction de n (échelle log-log) ---
# fig2 = go.Figure(go.Scatter(
#     x=widths, y=[max_err[n] for n in widths], mode="lines+markers"))
# fig2.update_layout(
#     xaxis=dict(title="Nombre de neurones n", type="log"),
#     yaxis=dict(title="Erreur max sur [-5, 5]²", type="log"),
# )
# fig2.show()

# --- Figure 3 : carte de l'erreur pour une largeur donnée ---
# n = 128
# fig3 = go.Figure(go.Surface(x=X, y=Y, z=np.abs(Z_hat[n] - Z_exact),
#                             colorscale="Reds", colorbar=dict(title="|erreur|")))
# fig3.update_layout(title=f"Erreur absolue, n = {n}",
#                    margin=dict(l=0, r=0, b=0, t=40))
# fig3.show()