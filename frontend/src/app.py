import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import requests

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Welcome to Aleph'),
    html.Div(id='data-display'),
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    )
])

@app.callback(Output('data-display', 'children'),
              Input('interval-component', 'n_intervals'))
def update_data(n):
    response = requests.get('http://backend:8000/data')
    return f"Data from backend: {response.json()['data']}"

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050, debug=True)
