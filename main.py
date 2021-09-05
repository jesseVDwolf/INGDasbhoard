import re
import plotly
import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import plotly.express as px
from datetime import datetime
from typing import Tuple, List
import plotly.graph_objects as go
from pandas.api.types import CategoricalDtype

# Work around to store states across reruns
import SessionState


def bytes_to_df(file: BytesIO) -> pd.DataFrame:
    """ create a cleaned up pandas dataframe from the streamlit BytesIO file """
    df_raw = pd.read_csv(filepath_or_buffer=file, sep=';', decimal=',', dtype={
        'Amount (EUR)': np.float64,
        'Resulting Balance': np.float64
    })
    columns = df_raw.columns
    columns = [re.sub(r"/|\(|\)", "", col) for col in columns]
    columns = [re.sub(r" +", " ", col) for col in columns]
    df_raw.columns = [col.replace(' ', '_').lower() for col in columns]
    df_raw['date'] = pd.to_datetime(df_raw['date'], format='%Y%m%d')
    df_raw['date_year'] = pd.DatetimeIndex(df_raw['date']).year
    df_raw['date_month'] = pd.DatetimeIndex(df_raw['date']).month
    df_raw['date_month_name'] = pd.DatetimeIndex(df_raw['date']).strftime("%B")

    # debitcredit column must be a categorical variable so that for each date
    # each category [Debit, Credit] is shown. This means that either one could
    # be zero if they didn't occur for a specific date, which is needed the chart
    df_raw['debitcredit'] = df_raw.debitcredit.astype(CategoricalDtype(categories=df_raw.debitcredit.unique()))
    return df_raw.copy(deep=True)


def get_filter_options_years(export_df: pd.DataFrame) -> List[int]:
    """ return filter options for years """
    years = list(set(pd.DatetimeIndex(export_df['date']).year))
    years.sort(reverse=True)
    return years


def get_filter_options_types(export_df: pd.DataFrame, state) -> Tuple[List[str], List[str]]:
    """ return filter options for credit_types and debit_types based on current filters set by user """
    df = export_df.copy(deep=True)

    # filter out transactions from years and months that are not selected by the user.
    _years = [int(year) for year in state.year_filter]
    df = df[df['date_year'].isin(_years) & df['date_month_name'].isin(state.month_filter)]

    df_credit = df[df['debitcredit'] == 'Credit']
    credit_types = list(set(df_credit.name_description.unique()))
    df_debit = df[df['debitcredit'] == 'Debit']
    debit_types = list(set(df_debit.name_description.unique()))
    return credit_types, debit_types


def get_filter_options_month(export_df: pd.DataFrame, years: List[int]) -> List[int]:
    """ get the month filter based on the current year filter """
    df = export_df.copy(deep=True)

    # filter out transactions from years that are not selected by the user.
    _years = [int(year) for year in years]
    df = df[df['date_year'].isin(_years)]

    # we want full month names ["Januari", "Februari" ...] but we also want it sorted correctly
    # 1. create a dataframe with only unique month name and corresponding month number
    # 2. convert to list of tuples
    # 3. let sorted() sort it ascending
    # 4. remove the month number
    df_month = df[['date_month', 'date_month_name']]
    df_month = df_month.drop_duplicates()
    months = list(df_month.itertuples(index=False, name=None))
    months = sorted(months, reverse=False)
    return [m for _, m in months]


def get_plotly_pie_fig(df: pd.DataFrame, title: str, inside_text: str) -> plotly.graph_objs.Figure:
    """ build a plotly pie figure with variable title and text in the hole """
    fig = px.pie(data_frame=df,
                 title=title + ': ' + str(round(df.amount_eur.sum(), 2)),
                 values='amount_eur',
                 names='name_description',
                 color_discrete_sequence=px.colors.sequential.Burg)
    fig.update_traces(
        textposition='inside',
        hoverinfo='label+percent+value',
        textinfo='percent',
        hole=0.2,
        marker=dict(
            colors=['honeydew'],
            line=dict(
                color='#000000',
                width=1.2
            )
        )
    )
    fig.update_layout(
        uniformtext_minsize=6,
        uniformtext_mode='hide',
        annotations=[
            dict(
                text=inside_text,
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(
                    color='black',
                    size=18
                )
            )
        ],
        legend=dict(
            font=dict(
                size=8,
                color='black'
            )
        )
    )
    return fig


def get_plotly_bar_fig(df: pd.DataFrame) -> plotly.graph_objs.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df.year_month.unique(),
        y=df[df['debitcredit'] == 'Debit'].amount_eur,
        name='Debit',
        textposition='auto',
        marker=dict(
            color='rgb(246, 51, 102)',
            line=dict(
                color='#000000',
                width=1.2
            )
        )
    ))
    fig.add_trace(go.Bar(
        x=df.year_month.unique(),
        y=df[df['debitcredit'] == 'Credit'].amount_eur,
        name='Credit',
        marker=dict(
            color='lightpink',
            line=dict(
                color='#000000',
                width=1.2
            )
        )
    ))
    fig.update_layout(
        title='Spendings vs Gains per month',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='Euros',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.25,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1  # gap between bars of the same location coordinate.
    )
    return fig


def main():
    # streamlit config
    st.set_page_config(layout='wide')

    st.markdown("<h1 style='text-align: center;'>ING export dashboard</h1>", unsafe_allow_html=True)
    filters = {'year_filter': None, 'month_filter': None, 'type_active_filter': False, 'credit_types_filter': None, 'debit_types_filter': None}
    state = SessionState.get(form_submitted=False, export_df=None, **filters)

    if not state.form_submitted:
        empty_one, form_col, empty_two = st.columns([2, 1, 2])
        with form_col:
            form = st.form(key='my_form')
            state.ing_export = form.file_uploader('Upload ING transaction export', type=['.csv'])
            submit = form.form_submit_button(label="Submit")
            if submit:
                state.form_submitted = True

    else:
        if not isinstance(state.export_df, pd.DataFrame):
            state.export_df = bytes_to_df(state.ing_export)

        try:
            file_name = state.ing_export.name[:state.ing_export.name.find('.csv')]
            _, _from, _to = file_name.split('_')
            export_date_from, export_date_to = (datetime.strptime(v, '%d-%m-%Y').strftime("%a %b %d, %Y") for v in [_from, _to])
        except ValueError:
            export_date_from = export_date_to = "[Unknown]"
        st.markdown(f"<div style='text-align: center;'>{export_date_from} <b> - </b> {export_date_to}</div>", unsafe_allow_html=True)

        filter_container = st.container()
        with filter_container:
            st.markdown("<h2 style='text-align: center;'>Date and type filters</h2>", unsafe_allow_html=True)
            expander = st.expander(label="", expanded=True)
            with expander:
                col_one_filter, col_two_filter = st.columns(2)
                years = get_filter_options_years(export_df=state.export_df)

                # year_filter must always contain a filter value otherwise month filter won't filter properly
                state.year_filter = col_one_filter.multiselect("Filter on years", years, default=years)

                months = get_filter_options_month(export_df=state.export_df, years=state.year_filter)
                state.month_filter = col_two_filter.multiselect("Filter on months", months, default=months)

            expander = st.expander(label="", expanded=True)
            with expander:
                col_one_filter, col_two_filter = st.columns(2)

                # type filters
                credit_types, debit_types = get_filter_options_types(export_df=state.export_df, state=state)
                state.type_active_filter = col_one_filter.checkbox('use type filter', value=False)
                state.credit_types_filter = col_two_filter.multiselect("Filter on credit types", credit_types)
                state.debit_types_filter = col_two_filter.multiselect("Filter on debit types", debit_types)

        summary_container = st.container()
        with summary_container:
            df_summary = state.export_df.copy(deep=True)
            df_summary = df_summary[df_summary['date_year'].isin([int(y) for y in state.year_filter])]
            df_summary = df_summary[df_summary['date_month_name'].isin(state.month_filter)]
            if state.type_active_filter:
                # filter out all types not specified by the type filters
                df_summary = df_summary[
                    (df_summary['name_description'].isin(state.credit_types_filter)) |
                    (df_summary['name_description'].isin(state.debit_types_filter))
                ]
            df_summary['year_month'] = df_summary['date_month_name'] + ' ' + df_summary['date_year'].astype(str)

            # create bar graph dataset and plotly bar object
            cols = ['year_month', 'debitcredit', 'amount_eur']
            df_summary_bar = df_summary.drop(columns=[col for col in df_summary.columns if col not in cols])
            df_summary_bar = df_summary_bar.groupby(by=['year_month', 'debitcredit']).sum().reset_index()
            st.plotly_chart(figure_or_data=get_plotly_bar_fig(
                df=df_summary_bar
            ), use_container_width=True)

            # create pie graph datasets
            cols = ['year_month', 'debitcredit', 'amount_eur', 'name_description']
            df_summary_pie = df_summary.drop(columns=[col for col in df_summary.columns if col not in cols])
            df_summary_pie = df_summary_pie.groupby(by=['year_month', 'debitcredit', 'name_description']).sum().reset_index()

            # plot the pie charts in the correct columns
            col_one, col_two = st.columns(2)
            col_one.plotly_chart(figure_or_data=get_plotly_pie_fig(
                df=df_summary_pie[df_summary_pie['debitcredit'] == 'Debit'],
                title='Total spend per type',
                inside_text='Debit'
            ), use_container_width=True)

            col_two.plotly_chart(figure_or_data=get_plotly_pie_fig(
                df=df_summary_pie[df_summary_pie['debitcredit'] == 'Credit'],
                title='Total gains per type',
                inside_text='Credit'
            ), use_container_width=True)


if __name__ == '__main__':
    main()
