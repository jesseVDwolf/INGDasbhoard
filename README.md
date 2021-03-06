# INGDashboard

A simple dashboard that helps you gain insight into your spendings
and gains over a period of time. The dashboard uses the export made
by ING. You can create this specific export at https://mijn.ing.nl/banking/overview

Make sure the export is of type "Comma seperated CSV" or "Puntkommagescheiden CSV"
in dutch and the language is "English". Open the dashboard using:
```bash
> streamlit run main.py
```

Upload your ING export and press submit to generate the dashboard.

![Dashboard](img/dashboard.png)

# Requirements

1. python 3.9.4 (tags/v3.9.4:1f2e308)
2. see requirements.txt

## Questions the dashboard should answer

1. How much did I spend / gain in total from period x to y
2. How much did I spend / gain per month from period x to y
3. How much did I spend / gain on item z from period x to y
4. How much did I spend / gain on item z per month from period x to y
5. How much did I spend / gain on items a - e from period x to y
6. How much did I spend / gain on items a - e per month from period x to y

## Filters

The filters in the dashboard filter out data from the original dataset which
in turn refreshes the graphs. The following filters are available:
1. A filter on year
2. A sub-filter on month
3. An optional switch to also filter on specific transactions
4. A filter on specific types of credit transactions
5. A filter on specific types of debit transactions

# Sources

1. https://plotly.com/python/builtin-colorscales/
2. https://plotly.com/python/bar-charts/
3. https://plotly.com/python/pie-charts/
4. https://blog.streamlit.io/introducing-new-layout-options-for-streamlit/
5. https://docs.streamlit.io/en/stable/api.html