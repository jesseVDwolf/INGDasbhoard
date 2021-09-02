import re
import plotly
import pandas as pd
import altair as alt
import streamlit as st
from typing import Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas.api.types import CategoricalDtype

# Work around to store states across reruns
import SessionState


def preprocess_df(original: pd.DataFrame) -> pd.DataFrame:
    """ modify the column names to something appropriate and make sure the right datatypes are used for the columns"""
    new_df = original.copy()

    chars = ['/', '(', ')']
    new_df.columns = [''.join([c for c in col if c not in chars]) for col in new_df.columns]
    new_df.columns = [re.sub(r' +', ' ', col).replace(' ', '_').lower() for col in new_df.columns]

    new_df['date'] = pd.to_datetime(new_df['date'], format='%Y%m%d')
    new_df['amount_eur'] = pd.to_numeric(new_df['amount_eur'].apply(lambda x: x.replace(',', '.')))
    new_df['resulting_balance'] = pd.to_numeric(new_df['resulting_balance'].apply(lambda x: x.replace(',', '.')))
    return new_df


def get_line_chart_df(original: pd.DataFrame) -> pd.DataFrame:
    """ convert the original dataframe to a dataframe usable for a line chart """
    new_df = original.copy()

    cols = ['date', 'debitcredit', 'amount_eur']
    new_df = new_df.drop(columns=[col for col in new_df.columns if col not in cols])

    # debitcredit column must be a categorical variable so that for each date
    # each category [Debit, Credit] is shown. This means that either one could
    # be zero if they didn't occur for a specific date, which is needed the chart
    new_df.debitcredit = new_df.debitcredit.astype(CategoricalDtype(categories=new_df.debitcredit.unique()))
    new_df = new_df.groupby(by=['date', 'debitcredit']).sum().reset_index()

    # make sure the dataframe is oriented properly
    # [date         creditdebit         amount_eur]
    # 2021-01-01    credit              5
    # 2021-01-01    debit               10
    #
    # to
    #
    # [date         debit               credit]
    # 2021-01-01    10                  5
    new_df = new_df.pivot(index='date', columns='debitcredit', values='amount_eur').reset_index()
    new_df = new_df.set_index('date')
    return new_df


def get_debitcredit_groupby(original: pd.DataFrame, aggr_col: str) -> pd.DataFrame:
    """ convert the original dataframe to a datafrmae useable for a pie chart """
    new_df = original.copy()

    cols = [aggr_col, 'debitcredit', 'amount_eur']
    new_df = new_df.drop(columns=[col for col in new_df.columns if col not in cols])

    # debitcredit column must be a categorical variable so that for each date
    # each category [Debit, Credit] is shown. This means that either one could
    # be zero if they didn't occur for a specific date, which is needed the chart
    new_df.debitcredit = new_df.debitcredit.astype(CategoricalDtype(categories=new_df.debitcredit.unique()))
    new_df = new_df.groupby(by=[aggr_col, 'debitcredit']).sum().reset_index()
    return new_df


def split_pie_chart_df(original: pd.DataFrame, aggr_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ split the original dataframe up into credit and debit dataframes """
    new_df = original.copy()

    new_df = get_debitcredit_groupby(new_df, aggr_col)
    mask = new_df['debitcredit'].isin(['Debit'])
    new_df_debit = new_df[mask]
    new_df_credit = new_df[~mask]
    return new_df_debit, new_df_credit


def get_pie_plotly_fig(credit: pd.DataFrame, debit: pd.DataFrame, aggr_col: str) -> plotly.graph_objs.Figure:
    """ create pie subplots using plotly """
    fig = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    fig.add_trace(go.Pie(labels=debit[aggr_col], values=debit.amount_eur, hole=0.3, name=f'Debit'), 1, 1)
    fig.add_trace(go.Pie(labels=credit[aggr_col], values=credit.amount_eur, hole=0.3, name=f'Credit'), 1, 2)
    fig.update_layout(
        # Add annotations in the center of the donut pies.
        annotations=[dict(text='Debit', x=0.17, y=0.5, font_size=15, showarrow=False),
                     dict(text='Credit', x=0.83, y=0.5, font_size=15, showarrow=False)])
    return fig


def aggregate_other(original: pd.DataFrame) -> pd.DataFrame:
    """ find the top 10 and aggregate everything else to {other} """
    trans_type = original.debitcredit.unique()[0]
    top_ten = original.sort_values(by='amount_eur', ascending=False).head(10)['name_description'].unique()

    df_other = original[~original['name_description'].isin(top_ten)]
    df_other.loc[:, 'name_description'] = 'other'
    df_other = df_other.groupby(['name_description', 'debitcredit']).sum().reset_index()
    df_other = df_other[df_other['name_description'].isin([trans_type])]
    return pd.concat([original[original['name_description'].isin(top_ten)], df_other])


def main():
    # stream configuration
    st.set_page_config(layout='wide')

    st.markdown("<h1 style='text-align: center;'>ING export dashboard</h1>", unsafe_allow_html=True)
    state = SessionState.get(form_submitted=False)

    if not state.form_submitted:
        empty_one, form_col, empty_two = st.beta_columns([2, 1, 2])
        with form_col:
            form = st.form(key='my_form')
            state.name = form.text_input(label="Name")
            state.ing_export = form.file_uploader('Upload ING transaction export', type=['.csv'])
            submit = form.form_submit_button(label="Submit")
            if submit:
                state.form_submitted = True

    else:
        style: str = f"<div style='text-align: center; font-family: \"Comic sans MS\"'>"
        st.markdown(f"{style}📁 {state.ing_export.name}</div>", unsafe_allow_html=True)
        st.markdown(f"{style}👤 {state.name}</div>", unsafe_allow_html=True)

        df: pd.DataFrame = pd.read_csv(state.ing_export, sep=';', quoting=True)
        df: pd.DataFrame = preprocess_df(df)

        # OVERAL SPENDINGS
        st.header("Overal spendings")
        st.subheader("Both incoming and outgoing payments over time")

        # line chart with date per day [2021-01-01 ... 2021-07-15]
        bar_chart_overal = alt.Chart(df).mark_bar().encode(x='date:T', y='amount_eur:Q', color='debitcredit')
        st.altair_chart(bar_chart_overal, use_container_width=True)

        # display pie charts in seperate columns
        pie_month_col, pie_type_col = st.beta_columns(2)
        pie_month_col.subheader("Transactions per month")
        pie_type_col.subheader("Transactions per type")

        # pie chart with amount per month [2021-01 ... 2021-07]
        df_chart_pie = df.copy()
        df_chart_pie['date'] = df_chart_pie['date'].apply(lambda x: x.strftime('%Y-%m'))
        df_chart_pie_debit, df_chart_pie_credit = split_pie_chart_df(df_chart_pie, 'date')
        fig = get_pie_plotly_fig(df_chart_pie_credit, df_chart_pie_debit, 'date')
        pie_month_col.plotly_chart(fig)

        # get monthly averages from pie chart dataframe
        avg_debit = df_chart_pie_debit.amount_eur.mean()
        avg_credit = df_chart_pie_credit.amount_eur.mean()

        # pie chart with amount per type [ALBERT HEIJN 1647 AMSTERDAM NLD ... TLS BV inz. OV-Chipkaart]
        df_chart_pie = df.copy()
        df_chart_pie_debit, df_chart_pie_credit = split_pie_chart_df(df_chart_pie, 'name_description')
        df_chart_pie_debit = aggregate_other(df_chart_pie_debit)
        df_chart_pie_credit = aggregate_other(df_chart_pie_credit)
        fig = get_pie_plotly_fig(df_chart_pie_credit, df_chart_pie_debit, 'name_description')
        pie_type_col.plotly_chart(fig)

        incoming_col, outgoing_col = st.beta_columns(2)
        columns = ['date', 'debitcredit', 'name_description', 'amount_eur', 'notifications']
        df_table = df.drop(columns=[col for col in df.columns if col not in columns])
        df_table['date'] = df_table['date'].apply(lambda x: x.strftime('%Y-%m-%d'))

        # INCOMING
        incoming_col.header(f"☝ Incoming -> On average per month: €{round(avg_credit, 2)}")

        # Incoming per month on average
        incoming_col.markdown("""
        #### **top 10 biggest transactions in the file**  
        ---
        """)
        incoming_col.write(df_table[df_table['debitcredit'].isin(['Credit'])].sort_values(
            by='amount_eur',
            ascending=False
        ).reset_index(drop=True).head(10))

        incoming_col.markdown("""
        #### **top 10 smallest transactions in the file**  
        ---
        """)
        incoming_col.write(df_table[df_table['debitcredit'].isin(['Credit'])].sort_values(
            by='amount_eur'
        ).reset_index(drop=True).head(10))

        # OUTGOING
        outgoing_col.header(f"👇 Outgoing -> On average per month: €{round(avg_debit, 2)}")

        outgoing_col.markdown("""
        #### **top 10 biggest transactions in the file**  
        ---
        """)
        outgoing_col.write(df_table[df_table['debitcredit'].isin(['Debit'])].sort_values(
            by='amount_eur',
            ascending=False
        ).reset_index(drop=True).head(10))

        # top 10 smallest transactions in the file
        outgoing_col.markdown("""
        #### **top 10 smallest transactions in the file**  
        ---
        """)
        outgoing_col.write(df_table[df_table['debitcredit'].isin(['Debit'])].sort_values(
            by='amount_eur'
        ).reset_index(drop=True).head(10))


if __name__ == '__main__':
    main()
