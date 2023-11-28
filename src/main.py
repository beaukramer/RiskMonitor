
import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
from data_fetchers import get_yahoo_data, get_shiller_data, get_fred_data
from constants import *
from absorptionratio import AbsorptionRatio
from turbulence import Turbulence
from kkt_attribution import KKT_Attribution
from visuals import PlotMaker
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio

# plotly app creation code
# Build the app components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

submit = [
    html.Button("Submit", id="submit-button"),
]

# Combine the form elements and placeholders for visualizations to form the app layout.
app.layout = dbc.Container(
    [
        html.H1("Risk Dashboard"),
        dbc.Row([dbc.Col(submit)]),
        # Charts
        dbc.Row(
            [
                dbc.Col([dcc.Graph(id='sector-absorption-ratio')], width=20),
                dbc.Col([dcc.Graph(id='asset-absorption-ratio')], width=20),
                dbc.Col([dcc.Graph(id="sector-turbulence")], width=20),
                dbc.Col([dcc.Graph(id="asset-turbulence")], width=20),
                # dbc.Col([dcc.Graph(id="robinson-ratios")], width=20),
                dbc.Col([dcc.Graph(id="kkt-recession-probability")], width=20),
                dbc.Col([dcc.Graph(id="kkt-variable-importance")], width=20),
                # dbc.Col([dcc.Graph(id="kkt-variable-importance-last10")], width=20),
            ]
        ),
    ]
)

# build the callback
@app.callback(
    [
        Output("sector-absorption-ratio", "figure"),
        Output("asset-absorption-ratio", "figure"),
        Output("sector-turbulence", "figure"),
        Output("asset-turbulence", "figure"),
        # Output("robinson-ratios", "figure"),
        Output("kkt-recession-probability", "figure"),
        Output("kkt-variable-importance","figure"),
        # Output("kkt-variable-importance-last10","figure"),
    ],
    [Input("submit-button", "n_clicks")],
    [],
)


def foo(n_clicks):

    # get sector data
    sector_prices = get_yahoo_data(SECTORS,start_date=START_DATE, end_date=END_DATE)
    sector_returns = sector_prices.pct_change().dropna()
    # get asset class data
    asset_prices = get_yahoo_data(ASSET_CLASSES,start_date=START_DATE, end_date=END_DATE)
    asset_returns = asset_prices.pct_change().dropna()
    # get robinson data
    # robinson_prices = get_yahoo_data(ROBINSON_RATIO, start_date=START_DATE, end_date=END_DATE)
    # get KKT data (fred and shiller then merge)
    kkt_data = get_fred_data(KKT_BUSINESS_CYCLE_INDICATOR_SERIES, start_date=None, end_date=END_DATE)
    kkt_data[[INDUSTRIAL_PRODCUTION, NONFARM_PAYROLLS]] = kkt_data[
        [INDUSTRIAL_PRODCUTION, NONFARM_PAYROLLS]].pct_change(12).dropna()
    kkt_data[TEN_YEAR_TREASURY_YIELD_MINUS_FF].ffill(inplace=True)
    kkt_data[TEN_YEAR_TREASURY_YIELD_MINUS_FF] = kkt_data[TEN_YEAR_TREASURY_YIELD_MINUS_FF].rolling(
        window=12).mean()
    kkt_data.dropna(subset=[INDUSTRIAL_PRODCUTION, NONFARM_PAYROLLS, TEN_YEAR_TREASURY_YIELD_MINUS_FF], how='any',
                    inplace=True)
    shiller_data = get_shiller_data(SHILLER_URL, SHILLER_SHEET_NAME, SHILLER_SKIP_ROW)
    sp500 = shiller_data['S&P'].copy()
    sp500 = sp500.pct_change(12).dropna()
    kkt_data = pd.merge(kkt_data, sp500, how='outer', left_index=True, right_index=True)
    kkt_data.ffill(inplace=True)
    kkt_data.dropna(how='any', inplace=True)

    # calculate absorption ratios
    sector_ar = AbsorptionRatio(sector_returns, window_size=WINDOW_SIZE)
    asset_ar = AbsorptionRatio(asset_returns, window_size=WINDOW_SIZE)
    # calculate turbulences
    sector_turbulence = Turbulence(sector_returns, window_size=WINDOW_SIZE, quantile=TURBULENCE_QUANTILE)
    asset_turbulence = Turbulence(asset_returns, window_size=WINDOW_SIZE ,quantile=TURBULENCE_QUANTILE)
    # calculate robinson ratios

    # calculate KKT indicator
    kkt_attribution = KKT_Attribution(kkt_data, econ_vars=ECON_VARIABLES, recession_var=NBER_RECESSION)


    # Absorption Ratio plots
    # plot of multi-asset absorption ratio
    asset_ar_plot = PlotMaker(asset_ar.absorption_ratio_standardized['Absorption_Ratio'], go_trace=go.Scatter, title='Asset Class Absorption Ratio',
              xaxis_title='Date', yaxis_title='Standardized Absorption Ratio', mode='lines+markers',
              color='blue').plot

    # plot of spyder equity sector absorption ratio
    sector_ar_plot = PlotMaker(sector_ar.absorption_ratio_standardized['Absorption_Ratio'], go_trace=go.Scatter, title='S&P Sector Absorption Ratio',
              xaxis_title='Date', yaxis_title='Standardized Absorption Ratio', mode='lines+markers',
              color='red').plot


    # Turbulence plots
    asset_turbulence_plot = PlotMaker(asset_turbulence.filtered_turbulence['Turbulence'], go_trace=go.Scatter, title='Asset Class Turbulence',
              xaxis_title='Date', yaxis_title='Turbulence', mode='lines',
              color='blue').plot
    sector_turbulence_plot = PlotMaker(sector_turbulence.filtered_turbulence['Turbulence'], go_trace=go.Scatter, title='S&P Sector Turbulence',
              xaxis_title='Date', yaxis_title='Turbulence', mode='lines',
              color='red').plot
    # Adam Robinson Plot (single plot)
    # copper/gold and #LQD/IEF and dicret/staples

    # KKT Plots
    # plot recession probability 'Probability1.0'
    kkt_recession_probability_plot = PlotMaker(kkt_attribution.df['Probability1.0'], go_trace=go.Scatter, title='KKT Recession Probability',
              xaxis_title='Date', yaxis_title='Probability', mode='lines',
              color='green').plot

    kkt_variable_importance_plot = go.Figure(layout=go.Layout(
        title='KKT Recession Probability Variable Importance',
        xaxis=dict(title='Date'),
        yaxis=dict(title='Variable Importance'), ), )

    kkt_variable_importance_plot.add_trace(go.Bar(y=kkt_attribution.variable_importance[INDUSTRIAL_PRODCUTION].iloc[-121:], x=kkt_attribution.variable_importance.index[-121:],
                                                  name=INDUSTRIAL_PRODCUTION, marker=dict(color='green')))
    kkt_variable_importance_plot.add_trace(go.Bar(y=kkt_attribution.variable_importance[NONFARM_PAYROLLS].iloc[-121:],
                                                  x=kkt_attribution.variable_importance.index[-121:],
                                                  name=NONFARM_PAYROLLS, marker=dict(color='blue')))
    kkt_variable_importance_plot.add_trace(go.Bar(y=kkt_attribution.variable_importance[TEN_YEAR_TREASURY_YIELD_MINUS_FF].iloc[-121:],
                                                  x=kkt_attribution.variable_importance.index[-121:],
                                                  name=TEN_YEAR_TREASURY_YIELD_MINUS_FF, marker=dict(color='yellow')))
    kkt_variable_importance_plot.add_trace(go.Bar(y=kkt_attribution.variable_importance["S&P"].iloc[-121:],
                                                  x=kkt_attribution.variable_importance.index[-121:],
                                                  name="S&P", marker=dict(color='red')))
    kkt_variable_importance_plot.update_layout(barmode='stack')

    return asset_ar_plot, sector_ar_plot, asset_turbulence_plot, sector_turbulence_plot, kkt_recession_probability_plot, kkt_variable_importance_plot

if __name__ == "__main__":
    app.run_server()

