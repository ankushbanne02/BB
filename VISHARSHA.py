import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import os

# 1. Skip CSV creation, since data_analytics.csv is already present

# 2. Graph analytics with NetworkX
def graph_analytics(df):
    G = nx.from_pandas_edgelist(df, 'Customer ID', 'StockCode', ['Quantity'])
    centrality = nx.degree_centrality(G)
    return G, centrality

# 3. Charts & summary statistics
def generate_charts(df):
    os.makedirs("charts", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df['Total'] = df['Quantity'] * df['Price']
    customer_group = df.groupby('Customer ID')['Total'].sum().reset_index()
    fig = px.bar(customer_group, x='Customer ID', y='Total', title='Total Purchase by Customer')
    fig.write_html("charts/customer_bar.html")

    cross_tab = pd.crosstab(df['Customer ID'], df['StockCode'], values=df['Total'], aggfunc='sum', dropna=False)
    cross_tab.to_csv("reports/crosstab.csv")

    stats = df['Total'].describe()
    stats.to_csv("reports/statistics.csv")

    return customer_group, stats, cross_tab

# 4. Filter, sort, group
def filter_sort_group(df):
    df['Total'] = df['Quantity'] * df['Price']
    df_filtered = df[df['Total'] > 100]
    df_sorted = df_filtered.sort_values(by='Total', ascending=False)
    df_grouped = df_sorted.groupby('Customer ID').agg({'Total': ['sum', 'mean']}).reset_index()
    return df_grouped

# 5. Add summary line
def add_summary_line(df):
    df['Total'] = df['Quantity'] * df['Price']
    total = df['Total'].sum()
    return f"Total Sales Value: ₹{total:.2f}"

# 6. Export reports
def export_reports(df, summary):
    df.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in df.columns.values]

    df.to_excel("reports/report.xlsx", index=False)
    df.to_csv("reports/report.csv", index=False)
    df.to_xml("reports/report.xml", index=False)

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Sales Report", ln=True, align='C')
    pdf.ln(10)

    for _, row in df.iterrows():
        pdf.cell(200, 10, txt=str(row.to_dict()), ln=True)
        pdf.ln(5)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt=summary.encode('utf-8').decode('latin-1'), ln=True)
    pdf.output("reports/report.pdf")

# 7. Run everything
def run_report():
    df = pd.read_csv("data_analytics.csv", encoding='latin1')
    G, centrality = graph_analytics(df)
    customer_group, stats, crosstab = generate_charts(df)
    processed_data = filter_sort_group(df)
    summary = add_summary_line(df)
    export_reports(processed_data, summary)
    print("✅ Reports generated! Check the left file browser in Colab to download.")

# Run it
run_report()
