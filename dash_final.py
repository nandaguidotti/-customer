import streamlit as st
import pandas as pd
import plotly.express as px
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI
import os
from dotenv import load_dotenv

# Carregar as variáveis de ambiente do arquivo .env
load_dotenv()

# ============ 1. Set Page Config ============
st.set_page_config(layout="wide")
st.title("🔍 Comparative Model Analysis from January 2022 to December 2024.")

# Buscar a chave da API
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# if OPENAI_API_KEY is None:
#     st.error("API Key not set. Please set your OpenAI API key.")
# else:
#     st.success("API Key loaded successfully!")

# ============ 2. Load and adjust CSV ============
CSV_PATH = "summary_comparative_all_models_abs_en.csv"
df = pd.read_csv(CSV_PATH, sep=";")

# Harmonize datatypes
float_cols = [
    'forecast_deviation_abs%',
    'stock_deviation_abs%',
    'customer_demand',
    'forecast_ai',
    'difCD',
    'stock',
    'forecast_stock'
]

# Converte as colunas para tipo numérico
for col in float_cols:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')

# Criação de coluna de data
df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2))
df['yearmonth'] = df['date'].dt.to_period('M')

# # Verifique os nomes das colunas após a leitura
# st.write("Nomes das colunas após leitura:", df.columns.tolist())

# ============ 3. Helper: Get Best Model ============
def get_best_model(df_month):
    results = []
    for customer in df_month['customer'].unique():
        df_customer = df_month[df_month['customer'] == customer]
        if not df_customer.empty:
            best_row = df_customer.loc[df_customer['forecast_deviation_abs%'].abs().idxmin()]
            results.append({
                'Best Model': best_row['model'],
                'Customer': customer,
                'Absolute Deviation (%)': abs(best_row['forecast_deviation_abs%']),
                'Deviation (%)': best_row['forecast_deviation_abs%'],
                'Real': best_row['customer_demand'],
                'Forecast': best_row['forecast_ai'],
                'Input': best_row['input'],
                'Stock Real': best_row['stock'],
                'Stock Forecast': best_row['forecast_stock'],
                'Stock Deviation (%)': abs(best_row['stock_deviation_abs%'])
            })
    return pd.DataFrame(results)

# ============ 4. Dashboard Framework ============
tabs = st.tabs(["📈 Model Comparison", "📊 Performance Visualization", "🤖 AI Assistant",])

# ============ 1st Tab: Performance Visualization ============
with tabs[1]:
    st.markdown("### Results Visualization")
    st.markdown("""
    This dashboard presents model performance by customer for March and April 2025,
    automatically determining the best input type based on the lowest absolute percentage deviation.
    """)
    # Filter relevant months
    df_month_3 = df[df['yearmonth'] == '2025-03']
    df_month_4 = df[df['yearmonth'] == '2025-04']

    results_month_3 = get_best_model(df_month_3)
    results_month_4 = get_best_model(df_month_4)
    customers_ordered = sorted(results_month_3['Customer'].unique())

    st.markdown("### Results for March:")
    st.dataframe(
        results_month_3[['Customer', 'Best Model', 'Absolute Deviation (%)',
                         'Real', 'Forecast', 'Input', 'Stock Deviation (%)']]
        .sort_values('Absolute Deviation (%)')
    )

    st.markdown("### Results for April:")
    st.dataframe(
        results_month_4[['Customer', 'Best Model', 'Absolute Deviation (%)',
                         'Real', 'Forecast', 'Input', 'Stock Deviation (%)']]
        .sort_values('Absolute Deviation (%)')
    )

    results_month_3['Month/Year'] = 'March'
    results_month_4['Month/Year'] = 'April'
    df_results_bar = pd.concat([results_month_3, results_month_4])

    fig_bar = px.bar(
        df_results_bar,
        x='Customer',
        y='Absolute Deviation (%)',
        color='Best Model',
        barmode='group',
        facet_col='Month/Year',
        title='Absolute Deviation by Customer - March and April',
        category_orders={'Customer': customers_ordered}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    df_results = pd.concat([results_month_3, results_month_4])
    available_months = df_results['Month/Year'].unique()

    for month in available_months:
        df_month = df_results[df_results['Month/Year'] == month]
        fig = px.line(
            df_month,
            x='Customer',
            y='Absolute Deviation (%)',
            text='Best Model',
            title=f'Absolute Deviation by Customer - {month}',
            markers=True,
            category_orders={'Customer': customers_ordered}
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title='Customer', yaxis_title='Absolute Deviation (%)')
        st.plotly_chart(fig, use_container_width=True)


# ============ 2nd Tab: AI Assistant ============
with tabs[2]:
    st.header("💬 AI Assistant")

    user_query = st.text_input(
        "Ask a question (e.g., 'What is the best model for customer1 in March 2025?')"
    )

    if st.button("🔍 Ask AI"):
        if OPENAI_API_KEY is None:
            st.error("API Key is missing. Please set your OpenAI API key in .env file.")
        else:
            llm = OpenAI(api_token=OPENAI_API_KEY)
            sdf = SmartDataframe(df, config={"llm": llm})

        # Instruções adaptadas para sua estrutura
        llm_instructions = """CRITICAL DATA STRUCTURE:
        - Columns: customer; input; year; month; model; date; customer_demand; 
          forecast_ai; difCD; forecast_deviation_abs%; stock; forecast_stock; stock_deviation_abs%

        BUSINESS RULES:
        1. Best model = MIN(forecast_deviation_abs%)
        2. Date format: month-year (MM-YYYY)
        3. Values use comma as decimal separator

        RESPONSE TEMPLATE:
        "The best model for [customer] in [month]-[year] is [model] with [X]% deviation (actual: [customer_demand], forecast: [forecast_ai])"
        """

        with st.spinner("Processing..."):
            try:
                # Pré-processa a query para formato adequado
                processed_query = (
                    f"{llm_instructions}\n\n"
                    f"Using the EXACT column names above, answer:\n{user_query}"
                )

                response = sdf.chat(processed_query)
                st.success(response)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.write("Data sample for debugging:", df.sample(3))

# ============ 3rd Tab: Model Comparison Visualization ============
with tabs[0]:
    # st.header("Model Performance Visualization")

    # Sidebar for filters
    st.sidebar.header("🔧 Filters")
    year = st.sidebar.selectbox("Year", df['year'].unique(), key='year_filter')
    month = st.sidebar.selectbox("Month", df['month'].unique(), format_func=lambda x: f"{x:02d}", key='month_filter')
    df['month_name'] = df['month'].apply(lambda x: f"{int(x):02d}") + '-' + df['year'].astype(str)  # 'MM-YYYY'
    customer = st.sidebar.selectbox("Customer", sorted(df['customer'].unique()), key='customer_filter')

    available_models = df['model'].unique()
    selected_models = st.sidebar.multiselect("Models for analysis", options=available_models,
                                             default=available_models.tolist(), key='model_filter')

    # Filtered DataFrame
    filtered_df = df[(df['year'] == year) & (df['month'] == month) & (df['customer'] == customer) & (
        df['model'].isin(selected_models))]

    # Verifique se o DataFrame filtrado está vazio ou contém a coluna
    if filtered_df.empty:
        st.warning("No data found for the selected filters!")
    else:
        st.header(f"Comparison of Models - {customer} ({month:02d}/{year})")

        if 'forecast_deviation_abs%' in filtered_df.columns:
            # Ordena os dados conforme o desvio
            filtered_df = filtered_df.sort_values('forecast_deviation_abs%', ascending=False)

            # Cria um gráfico de barras simples para desvio percentual
            fig = px.bar(filtered_df,
                         x='model',
                         y='forecast_deviation_abs%',
                         color='forecast_deviation_abs%',
                         color_continuous_scale='RdYlGn_r',
                         labels={'forecast_deviation_abs%': 'Deviation (%)', 'model': 'Model'},
                         title=f"Percent Deviation of the Models - {customer} ({month:02d}/{year})")

            fig.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
            fig.update_layout(barmode='group', xaxis_title='Model', yaxis_title='Deviation (%)', title_x=0.5)

            st.plotly_chart(fig, use_container_width=True)
            # Exibir uma tabela com os dados filtrados
            st.subheader(f"Customer: {customer}, Month: {month:02d}")
            st.dataframe(filtered_df)  # Ou use st.table(filtered_df) para uma tabela mais simples

