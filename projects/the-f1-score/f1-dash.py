import dash_ag_grid as dag
import pandas as pd
from dash import Dash, html

data = {"Model": ["Model A", "Model B", "Model C"], "Points": [85, 75, 60]}
df = pd.DataFrame(data)

app = Dash(__name__)

columnDefs = [
    {"field": "Model"},
    {"field": "Points"},
]

grid = dag.AgGrid(
    id="get-started-example-basic",
    rowData=df.to_dict("records"),
    columnDefs=columnDefs,
)

app.layout = html.Div([grid])

if __name__ == "__main__":
    app.run(debug=True)
