import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html, dash_table, Input, Output
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

# 1. Sample Data
data = {
    'transaction_id': [1, 2, 3, 4, 5],
    'customer_id': ['C1', 'C2', 'C3', 'C1', 'C2'],
    'product_id': ['P1', 'P2', 'P3', 'P2', 'P3'],
    'amount': [100, 200, 150, 300, 120],
    'location': ['NY', 'CA', 'TX', 'NY', 'CA']
}
df = pd.DataFrame(data)

# 2. Generate Graph Image
def create_transaction_graph(df):
    G = nx.Graph()
    G.add_nodes_from(df['customer_id'], bipartite=0)
    G.add_nodes_from(df['product_id'], bipartite=1)
    G.add_edges_from([(row['customer_id'], row['product_id']) for _, row in df.iterrows()])
    
    pos = nx.spring_layout(G)
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, edge_color='gray', ax=ax)
    ax.set_title("Customer-Product Transaction Graph")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{encoded}"

graph_img = create_transaction_graph(df)

# 3. Create App
app = dash.Dash(__name__)

# 4. App Layout
app.layout = html.Div([
    html.H1("📊 Transaction Data Analysis Dashboard", style={'textAlign': 'center'}),

    # Raw Data Table
    html.H2("📋 Data Table"),
    dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{"name": i, "id": i} for i in df.columns],
        page_size=5,
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center'},
        style_data_conditional=[
            {
                'if': {'filter_query': '{amount} > 200', 'column_id': 'amount'},
                'backgroundColor': 'tomato',
                'color': 'white'
            }
        ]
    ),

    # Crosstab
    html.H2("📊 Crosstab (Location vs Product)"),
    dcc.Graph(figure=px.histogram(df, x="product_id", color="location", barmode="group", title="Product Sales by Location")),

    # Filtering
    html.H2("🔍 Filter by Location"),
    dcc.Dropdown(
        options=[{'label': loc, 'value': loc} for loc in df['location'].unique()],
        value='NY',
        id='location-filter'
    ),
    dcc.Graph(id='filtered-chart'),

    # Chart
    html.H2("📈 Total Amount by Customer"),
    dcc.Graph(figure=px.bar(df.groupby("customer_id", as_index=False).sum(), x="customer_id", y="amount", title="Total Amount by Customer")),

    # Graph Analytics (Customer-Product Graph)
    html.H2("📎 Graph Analytics"),
    html.Img(src=graph_img, style={'width': '60%', 'display': 'block', 'margin': 'auto'}),

    # Download buttons (simulated via DataFrame export options)
    html.H2("📁 Export Options"),
    html.Button("Export CSV", id="btn_csv"),
    html.Button("Export Excel", id="btn_excel"),
    dcc.Download(id="download-dataframe-csv"),
    dcc.Download(id="download-dataframe-xlsx"),
])

# 5. Callbacks
@app.callback(
    Output("filtered-chart", "figure"),
    Input("location-filter", "value")
)
def update_chart(location):
    filtered = df[df["location"] == location]
    return px.bar(filtered, x="customer_id", y="amount", color="product_id", title=f"Amount by Customer in {location}")

@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True
)
def download_csv(n_clicks):
    return dcc.send_data_frame(df.to_csv, "transactions.csv")

@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("btn_excel", "n_clicks"),
    prevent_initial_call=True
)
def download_excel(n_clicks):
    return dcc.send_data_frame(df.to_excel, "transactions.xlsx", sheet_name="Data")

# 6. Run App
if __name__ == "__main__":
    app.run(debug=True)

data = pd.read_csv("sample_transactions_100.csv")
