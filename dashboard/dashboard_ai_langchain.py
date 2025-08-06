import difflib
import os
import pandas as pd
import streamlit as st
import plotly.express as px
import requests
from dotenv import load_dotenv
import plotly.graph_objects as go

from dashboard.load_result import load_all_results

# ========== CONFIGURATION ==========
st.set_page_config(layout="wide")
st.title("🔍 Comparative Model Analysis (multi-models, multi-customers)")

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output_forecasts'))

# with st.spinner("Loading data..."):
#     df_all_test, df_all_future = load_all_results(BASE_PATH)
def safe_load_all_results(base_path):
    try:
        return load_all_results(base_path)
    except Exception as e:
        st.warning(f"No results found yet. ({str(e)})")
        return pd.DataFrame(), pd.DataFrame()

with st.spinner("Loading data..."):
    df_all_test, df_all_future = safe_load_all_results(BASE_PATH)


if df_all_test.empty and df_all_future.empty:
    col1, col2 = st.columns(2)
    with col1:
        st.image("https://www.seg-automotive.com/image/sync/images/logo.svg", width=240)
    with col2:
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/c9/Webysther_20170627_-_Logo_ICMC-USP.svg", width=240)

    st.title("Welcome to the SEG Automotive Dashboard")
    st.markdown("""
    🚦 **No results found!**

    - Before you can view the dashboards, you need to run model training and prediction.
    - Use the **🛠️ Model Operations** tab to start training or prediction.
    - Results will appear here as soon as they are generated.

    ---
    """)
    st.info("Once results are available, reload this page to access all dashboards and analytics.")
    st.stop()


if not df_all_test.empty:
    if 'date' in df_all_test.columns:
        df_all_test['date'] = pd.to_datetime(df_all_test['date'], errors='coerce')
        df_all_test['year'] = df_all_test['date'].dt.year
        df_all_test['month'] = df_all_test['date'].dt.month
        df_all_test['month_year'] = df_all_test['date'].dt.to_period('M').astype(str)
else:
    st.warning("No data loaded.")

# ---- Garante coluna 'month_year' em ambos os DataFrames ----
if 'month_year' not in df_all_test.columns:
    if 'date' in df_all_test.columns:
        df_all_test['month_year'] = pd.to_datetime(df_all_test['date'], errors='coerce').dt.strftime('%Y-%m')
    else:
        df_all_test['month_year'] = None

if 'month_year' not in df_all_future.columns:
    if 'date' in df_all_future.columns:
        df_all_future['month_year'] = pd.to_datetime(df_all_future['date'], errors='coerce').dt.strftime('%Y-%m')
    else:
        df_all_future['month_year'] = None

# --------- REMOVE DUPLICATE 'data_type' COLUMNS -----------
for df in [df_all_test, df_all_future]:
    while 'data_type' in df.columns:
        df.drop(columns=['data_type'], inplace=True)

df_all_test['data_type'] = 'test'
df_all_future['data_type'] = 'future'

common_cols = [col for col in df_all_test.columns if col in df_all_future.columns and col != 'data_type']
df_combined = pd.concat([
    df_all_test[common_cols + ['data_type']],
    df_all_future[common_cols + ['data_type']]
], ignore_index=True)

# Extra: remove qualquer coluna duplicada
if not df_combined.columns.is_unique:
    st.error(f"Duplicate columns found after concat: {list(df_combined.columns)}")
    df_combined = df_combined.loc[:, ~df_combined.columns.duplicated()]

# ---------- API ENDPOINT ----------
API_URL = "http://127.0.0.1:5000"

# ---------- LOAD OPENAI KEY FOR AI ASSISTANT ----------
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

st.sidebar.image(
    "https://www.seg-automotive.com/image/sync/images/logo.svg",
    width=180,
)
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/c/c9/Webysther_20170627_-_Logo_ICMC-USP.svg",
    width=180,
)

st.sidebar.markdown("#### SEG Automotive Dashboard")

# ========== SIDEBAR MENU ==========
menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Dashboard",
        "🛠️ Model Operations",
        "📈 Model Comparison",
        "📊 Best Models",
        "🏆 Top 3 + Future",
        "🤖 AI Assistant"
    ]
)

# ========== DASHBOARD ==========
if menu == "🏠 Dashboard":
    st.header("Dashboard: Best Model Forecast vs Real Demand")

    # --- Carrega histórico fora do IF para não repetir ---
    df_historico = pd.read_csv("datasets/data_customer_selic_ipca.csv")
    if 'date' in df_historico.columns:
        df_historico['date'] = pd.to_datetime(df_historico['date'], format='%m/%Y', errors='coerce')
        df_historico['month_year'] = df_historico['date'].dt.strftime('%Y-%m')
    else:
        df_historico['month_year'] = df_historico['month_year'].astype(str)

    # --- Sidebar: filtro único de cliente para tudo ---
    clientes_all = sorted(set(df_all_test['customer']).union(df_historico['customer']))
    selected_customer = st.sidebar.selectbox("Customer", clientes_all, key="dashboard_customer")

    # --- Legenda dos Inputs/Modelos ---
    with st.expander("Legend – Inputs/Model Names"):
        st.markdown("""
        - **cd**: Customer Demand (demand history)
        - **po**: Production Orders (order history)
        - **s**: Stock (inventory level)
        - **ipca**: IPCA (Brazilian inflation rate)
        - **selic**: SELIC (Brazilian interest rate)
        - **model_input**: Combination of features used to train the model (e.g., `cd_po_ipca`)
        - **forecast_ai**: AI model's prediction for the month
        - **customer_demand**: Real units sold/delivered
        - **forecast_deviation_abs**: Absolute error between real demand and forecast
        """)

    st.markdown(
        "**Model for each point:** Hover to see the model name.<br>Future points use the current best model.",
        unsafe_allow_html=True
    )

    # --- Melhor modelo por mês no passado (test) ---
    df_best = df_all_test.loc[
        df_all_test.groupby(['customer', 'month_year'])['forecast_deviation_abs'].idxmin()
    ]
    df_best = df_best[df_best['customer'] == selected_customer].sort_values("month_year")
    df_best["month_year"] = df_best["month_year"].astype(str)

    # --- Últimos 12 meses para KPIs ---
    last_12 = df_best.tail(12) if len(df_best) > 12 else df_best.copy()

    # --- KPIs rápidos (linha superior) ---
    if not last_12.empty:
        total_demand = last_12['customer_demand'].sum()
        average_mae = last_12['forecast_deviation_abs'].mean()
        best_month = last_12.loc[last_12['customer_demand'].idxmax(), "month_year"]
        worst_month = last_12.loc[last_12['customer_demand'].idxmin(), "month_year"]
        most_accurate_model = last_12.groupby('model_input')['forecast_deviation_abs'].mean().idxmin()
        n_models = last_12['model_input'].nunique()

        kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
        kpi1.metric("Total Demand (12m)", f"{total_demand:,.0f} units")
        kpi2.metric("Avg MAE", f"{average_mae:.1f}")
        kpi3.metric("Best Month", best_month)
        kpi4.metric("Worst Month", worst_month)
        kpi5.metric("Best Model", most_accurate_model)

    # --- Melhor modelo do mês mais recente ---
    if len(df_best) > 0:
        last_month = df_best["month_year"].max()
        best_model_now = df_best[df_best["month_year"] == last_month]["model_input"].values[0]
    else:
        best_model_now = None

    # --- Previsão futura: só do best_model_now ---
    if best_model_now is not None:
        df_future = df_all_future[
            (df_all_future['customer'] == selected_customer) &
            (df_all_future['model_input'] == best_model_now)
        ].copy()
        df_future["month_year"] = pd.to_datetime(df_future["date"]).dt.to_period("M").astype(str)
        future_months = sorted(df_future["month_year"].unique())[:4]
        df_future = df_future[df_future["month_year"].isin(future_months)].sort_values("month_year")
    else:
        df_future = pd.DataFrame()
    # --- Junta para gráfico: passado e futuro ---
    df_plot = pd.DataFrame()
    if not df_best.empty:
        df_plot = df_best[["month_year", "forecast_ai", "customer_demand", "model_input"]].copy()
        df_plot["type"] = "Past"
    if not df_future.empty:
        future_plot = df_future[["month_year", "forecast_ai", "model_input"]].copy()
        future_plot["customer_demand"] = None
        future_plot["type"] = "Future"
        df_plot = pd.concat([df_plot, future_plot], ignore_index=True)

    # --- Gráfico Real vs Forecast ---
    df_real = df_plot[df_plot["type"] == "Past"][["month_year", "customer_demand"]].drop_duplicates()
    df_real = df_real.sort_values("month_year")
    df_forecast = df_plot[["month_year", "forecast_ai"]].drop_duplicates().sort_values("month_year")

    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_real["month_year"],
        y=df_real["customer_demand"],
        mode="lines+markers",
        name="Real Demand",
        line=dict(dash="solid", width=2),
        marker=dict(symbol="circle")
    ))
    fig.add_trace(go.Scatter(
        x=df_forecast["month_year"],
        y=df_forecast["forecast_ai"],
        mode="lines+markers",
        name="Best Model Forecast",
        line=dict(dash="dot", width=2),  # Linha pontilhada
        marker=dict(symbol="circle")
    ))
    fig.update_layout(
        title=f"Real Demand vs Best Model Forecast – {selected_customer}",
        xaxis_title="Month/Year",
        yaxis_title="Units",
        legend_title="Series",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Alertas Inteligentes ---
    erro_medio = last_12['forecast_deviation_abs'].mean() if not last_12.empty else 0
    meses_criticos = last_12[last_12['forecast_deviation_abs'] > 1.5 * erro_medio]
    if not meses_criticos.empty:
        st.warning(
            f"⚠️ {len(meses_criticos)} month(s) had forecast errors above 50% of the average! Check months: "
            + ", ".join(meses_criticos['month_year'])
        )
    else:
        st.success("All forecast errors are within the normal range.")

    # --- Gráfico Histórico do Customer Demand completo ---
    df_cliente_hist = df_historico[df_historico['customer'] == selected_customer].sort_values('date')
    df_cliente_hist = df_cliente_hist.reset_index(drop=True)
    df_cliente_hist["Periodo"] = df_cliente_hist.index + 1

    fig_hist = px.line(
        df_cliente_hist,
        x="Periodo",
        y="customer_demand",
        markers=True,
        title=f"Demand History for {selected_customer}",
        labels={"customer_demand": "Units", "Periodo": "Periods"}
    )
    fig_hist.update_layout(
        showlegend=False,
        xaxis_title="Periods",
        yaxis_title="Customer Demand (Units)",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

elif menu == "🛠️ Model Operations":
    st.header("Model Operations")
    st.markdown("Use the buttons below to trigger model training or prediction via the API.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("Train All Algorithms"):
            with st.spinner("Calling training API..."):
                try:
                    response = requests.post(f"{API_URL}/api/all/train_all_algorithms")
                    if response.status_code == 200:
                        st.success("Training started successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"API call failed: {str(e)}")
    with col2:
        if st.button("Predict All Algorithms"):
            with st.spinner("Calling prediction API..."):
                try:
                    response = requests.post(f"{API_URL}/api/all/predict_all_algorithms")
                    if response.status_code == 200:
                        st.success("Prediction started successfully!")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"API call failed: {str(e)}")
    with col3:
        if st.button("Stop Execution"):
            with st.spinner("Sending STOP flag..."):
                try:
                    response = requests.post(f"{API_URL}/api/all/stop")
                    if response.status_code == 200:
                        st.success("STOP flag sent!")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"API call failed: {str(e)}")
    with col4:
        if st.button("Clear Stop Flag"):
            with st.spinner("Clearing STOP flag..."):
                try:
                    response = requests.post(f"{API_URL}/api/all/clear_stop")
                    if response.status_code == 200:
                        st.success("STOP flag cleared!")
                        st.json(response.json())
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"API call failed: {str(e)}")

# ========== MODEL COMPARISON ==========
elif menu == "📈 Model Comparison":
    st.header("Model Comparison")
    st.sidebar.header("🔧 Filters")
    if not df_all_test.empty:
        months_years = sorted(df_all_test['month_year'].dropna().unique())
        customers = sorted(df_all_test['customer'].unique())
        models = sorted(df_all_test['model_input'].unique())

        month_year = st.sidebar.selectbox("Month/Year", months_years)
        customer = st.sidebar.selectbox("Customer", customers)
        selected_models = st.sidebar.multiselect("Models for analysis", models, default=models)

        filtered_df = df_all_test[
            (df_all_test['month_year'] == month_year) &
            (df_all_test['customer'] == customer) &
            (df_all_test['model_input'].isin(selected_models))
        ].sort_values("forecast_deviation_abs", ascending=True)

        if filtered_df.empty:
            st.warning("No data found for the selected filters!")
        else:
            # Arredondamento para visualização
            display_df = filtered_df.copy()
            if 'forecast_ai' in display_df.columns:
                display_df['forecast_ai'] = display_df['forecast_ai'].round(2)
            if 'forecast_deviation_abs' in display_df.columns:
                display_df['forecast_deviation_abs'] = display_df['forecast_deviation_abs'].round(2)

            # Só as colunas desejadas
            cols_to_show = [
                "customer", "month_year", "model_input", "customer_demand",
                "forecast_ai", "diff", "forecast_deviation_abs"
            ]
            st.subheader(f"Model Comparison - {customer} ({month_year})")
            st.dataframe(display_df[cols_to_show])

            fig = px.bar(
                display_df,
                x='model_input',
                y='forecast_deviation_abs',
                color='forecast_deviation_abs',
                color_continuous_scale='RdYlGn_r',
                labels={'forecast_deviation_abs': 'Deviation', 'model_input': 'Model'},
                title=f"Deviation of the Models - {customer} ({month_year})"
            )

            # Corrigir eixo Y para começar do zero e mostrar bem o range
            fig.update_layout(
                yaxis=dict(
                    range=[0, max(1.1 * display_df['forecast_deviation_abs'].max(), 1)]
                )
            )

            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Load the data to visualize.")

# ========== BEST MODELS ==========
elif menu == "📊 Best Models":
    st.header("Best Model per Customer and Month")
    if not df_all_test.empty:
        customers = sorted(df_all_test['customer'].unique())
        customer = st.sidebar.selectbox("Customer", customers)

        # Melhor modelo por cliente/mês
        df_best = df_all_test.loc[
            df_all_test.groupby(['customer', 'month_year'])['forecast_deviation_abs'].idxmin()
        ]

        # Só para o cliente selecionado
        filtered_best = df_best[df_best['customer'] == customer].sort_values("month_year")

        if filtered_best.empty:
            st.warning("No best model found for the selected customer!")
        else:
            display_df = filtered_best.copy()
            for col in ["forecast_ai", "forecast_deviation_abs", "diff", "customer_demand"]:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)

            cols_to_show = [
                "customer", "month_year", "model_input", "customer_demand",
                "forecast_ai", "diff", "forecast_deviation_abs"
            ]
            st.dataframe(display_df[cols_to_show])

            # Prepara o DataFrame para gráfico de comparação
            plot_df = display_df[["month_year", "customer_demand", "forecast_ai"]].copy()
            plot_df = plot_df.melt(id_vars="month_year", value_vars=["customer_demand", "forecast_ai"],
                                   var_name="Type", value_name="Value")

            fig = px.line(
                plot_df,
                x="month_year",
                y="Value",
                color="Type",
                line_dash="Type",
                markers=True,
                labels={'Value': 'Value', 'month_year': 'Month/Year', 'Type': 'Series'},
                title=f"Customer Demand vs Forecast (Best Model per Month) - {customer}"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Load the data to visualize.")

# ========== TOP 3 + FUTURE ==========
elif menu == "🏆 Top 3 + Future":
    st.header("Top 3 models per customer and forecast for the next 4 months")
    if not df_all_test.empty and not df_all_future.empty:
        selected_customer = st.sidebar.selectbox(
            "Select a customer",
            sorted(df_all_test['customer'].unique()),
            key="top3_customer"
        )
        df_cust = df_all_test[df_all_test['customer'] == selected_customer].copy()
        df_cust['month_year'] = df_cust['month_year'].astype(str).str[:7]
        last_2_months = sorted(df_cust['month_year'].unique())[-2:]
        top3_rows = (
            df_cust[df_cust['month_year'].isin(last_2_months)]
            .sort_values('forecast_deviation_abs')
            .groupby('month_year')
            .head(3)
        )

        st.subheader("Top 3 models per month - Test Results (last 2 months)")
        st.dataframe(top3_rows[["customer","month_year", "model_input", "customer_demand", "forecast_ai", "forecast_deviation_abs"]])

        unique_models = top3_rows['model_input'].unique()
        future_preds = []
        for model_input in unique_models:
            fut = df_all_future[
                (df_all_future['customer'] == selected_customer) &
                (df_all_future['model_input'] == model_input)
            ].sort_values("date").head(4)
            if not fut.empty:
                fut['model_input'] = model_input
                future_preds.append(fut)

        if future_preds:
            df_fut3 = pd.concat(future_preds)
            st.subheader("Future Forecast of Top 3 Models (from last 4 months)")
            st.dataframe(df_fut3[["customer","date", "model_input", "forecast_ai"]])
            last_4_months_real = sorted(df_cust['month_year'].unique())[-4:]
            plot_real = df_cust[df_cust['month_year'].isin(last_4_months_real)].copy()
            plot_real = plot_real[['month_year', 'customer_demand']]
            plot_real['type'] = 'Real Demand'
            plot_real['model_input'] = None
            plot_real.rename(columns={'customer_demand': 'value'}, inplace=True)
            df_fut3['month_year'] = pd.to_datetime(df_fut3['date'], errors='coerce').dt.strftime('%Y-%m')
            plot_forecast = df_fut3[['month_year', 'forecast_ai', 'model_input']].copy()
            plot_forecast['type'] = 'Forecast'
            plot_forecast.rename(columns={'forecast_ai': 'value'}, inplace=True)
            df_plot = pd.concat([plot_real, plot_forecast], ignore_index=True)
            df_plot = df_plot.sort_values(["type", "model_input", "month_year"])

            fig = px.line(
                df_plot,
                x="month_year",
                y="value",
                color="type",
                line_dash="model_input",
                markers=True,
                labels={"value": "Value", "month_year": "Month/Year", "type": "Series"},
                title="Last 4 Real Demands & Next 4-Month Forecasts (Top 3 Models)"
            )
            fig.for_each_trace(
                lambda t: t.update(
                    hovertemplate=(
                        "<b>Month</b>: %{x}<br>"
                        "<b>Value</b>: %{y}<br>"
                        "<b>Model</b>: %{customdata[0]}<extra></extra>"
                    )
                ) if t.name == "Forecast" else t
            )
            for trace in fig.data:
                if trace.name == "Forecast":
                    idx = (df_plot["type"] == "Forecast")
                    trace.customdata = df_plot[idx][["model_input"]].values

            fig.update_layout(
                xaxis_title="Month/Year",
                yaxis_title="Value",
                legend_title="Series",
                hovermode="x unified"
            )
            fig.update_xaxes(
                tickformat="%Y-%m",
                dtick="M1"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No future predictions available for the Top 3 models.")
    else:
        st.info("Load the data to visualize.")

# ========== AI ASSISTANT ==========
elif menu == "🤖 AI Assistant":
    from langchain_openai import ChatOpenAI
    from langchain_experimental.agents import create_pandas_dataframe_agent
    import re

    # --------------- SEG Context Guard Functions ---------------
    ALLOWED_KEYWORDS = [
        'seg', 'customer', 'forecast', 'model', 'demand', 'error', 'data', 'business', 'analysis',
        'revenue', 'sales', 'deviation', 'trend', 'mae', 'rmse', 'test', 'future', 'comparison'
    ]

    def is_seg_question(user_query):
        user_query_lower = user_query.lower()
        return any(kw in user_query_lower for kw in ALLOWED_KEYWORDS)

    GREETINGS_WHITELIST = [
        'hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how are you?', 'thanks', 'thank you', 'bye', 'goodbye',
        'oi', 'olá', 'tchau', 'obrigado', 'obrigada'
    ]

    def get_guarded_answer(user_query):
        query = user_query.lower().strip()
        def is_greeting(query):
            for greet in GREETINGS_WHITELIST:
                norm_query = query.replace('assistant', '').strip()
                if difflib.SequenceMatcher(None, norm_query, greet).ratio() > 0.8:
                    return True
                if greet in norm_query:
                    return True
            return False

        if is_greeting(query):
            return None

        if any(q in query for q in [
            'who is seg', 'what is seg', 'what does seg mean', 'what company is seg', 'about seg'
        ]):
            return (
                "SEG Automotive is a leading global supplier of automotive components and solutions, "
                "specializing in innovative technologies for efficient, reliable, and sustainable mobility. "
                "SEG Automotive develops and produces alternators, starters, and electrification components "
                "for major automotive manufacturers worldwide.\n\n"
                "Final Answer: SEG Automotive is a leading global supplier of automotive components."
            )
        if any(q in query for q in [
            'who are you', 'Who are you', 'what are you', 'what is you', 'who is this', 'who is the assistant',
            'who am i talking to', 'what can you do', 'who are u', 'what are u', 'what is u',
            'quem é você', 'o que é você', 'o que você faz', 'o que vc pode me ajudar'
        ]):
            return (
                "I am the official AI assistant of SEG Automotive, developed to support users with business intelligence, "
                "model comparison, and data analysis within the SEG Automotive context. My purpose is to help answer questions "
                "and provide insights based on business and analytical data from SEG Automotive.\n\n"
                "Final Answer: I am the official AI assistant of SEG Automotive."
            )
        if not is_seg_question(user_query):
            return (
                "I'm sorry, but I'm not authorized to answer questions outside the SEG Automotive business and analytical context. "
                "I can assist you with customer analysis, business metrics, forecasting, model comparisons, and data analytics for SEG Automotive.\n\n"
                "Final Answer: Please ask a question related to SEG Automotive business analytics."
            )
        return None

    st.header("💬 AI Assistant")
    st.markdown("""
    Ask anything about the models or results.  
    The assistant will answer based on your processed data (test and future CSV outputs).
    """)

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    user_query = st.text_input(
        "Ask a question about your data (e.g., 'What is the best forecast for customer1.0 in August 2025?')",
        key="ai_input"
    )

    guarded = get_guarded_answer(user_query)
    if guarded and user_query.strip():
        st.session_state['chat_history'].append({
            "role": "assistant",
            "content": guarded,
        })
        st.markdown(f"**Assistant:** {guarded}")
        st.stop()

    columns_for_ai = [
        'customer', 'model_input', 'date', 'month_year', 'customer_demand',
        'forecast_ai', 'forecast_deviation_abs', 'data_type'
    ]
    missing_cols = [col for col in columns_for_ai if col not in df_combined.columns]
    if missing_cols:
        st.error(f"Missing columns in your data: {missing_cols}")
    else:
        df_combined_ai = df_combined.loc[:, ~df_combined.columns.duplicated()][columns_for_ai].copy()
        for col in df_combined_ai.columns:
            if col not in ['forecast_ai', 'customer_demand', 'forecast_deviation_abs']:
                df_combined_ai[col] = df_combined_ai[col].astype(str)

        def extract_explanation_and_final(agent_response):
            """Extrai explicação antes do 'Final Answer:' e o resultado final separado."""
            if isinstance(agent_response, dict):
                text = agent_response.get("output", "")
            else:
                text = str(agent_response)
            split = re.split(r"Final Answer:\s*", text, maxsplit=1)
            if len(split) == 2:
                explanation = split[0].strip()
                final_answer = split[1].strip()
                return explanation, final_answer
            return text.strip(), None

        if st.button("🔍 Ask AI") and user_query.strip():
            trigger_keywords = [
                "forecast", "customer", "model", "demand", "deviation", "future", "test", "error"
            ]
            is_data_question = any(kw in user_query.lower() for kw in trigger_keywords)

            try:
                if is_data_question:
                    llm = ChatOpenAI(
                        api_key=OPENAI_API_KEY,
                        model="gpt-4o",
                        temperature=0.1,
                        streaming=False,
                    )
                    agent = create_pandas_dataframe_agent(
                        llm,
                        df_combined_ai,
                        verbose=True,
                        allow_dangerous_code=True,
                    )

                    custom_prompt = """
You are the official AI assistant for SEG Automotive, designed to support users in the domain of business intelligence and predictive analytics for the automotive sector.

Your purpose is to help users analyze and compare demand forecasts, models, and business metrics for different customers, models, and scenarios within the context of SEG Automotive.

Always respond in English, regardless of the user's question language.

---
"When filtering by month, always use the column date and the format 'YYYY-MM' (e.g., '2025-05'). Do not expect day values or use other columns for dates."

## Domain and Data Dictionary (SEG Automotive context):

- **customer**: The business client or unit being analyzed.
- **model_input**: The set of input variables (scenario) used to train the forecasting model.
    - `cd` = customer demand (historical demand)
    - `po` = production orders
    - `ipca` = Brazilian inflation rate (IPCA)
    - `selic` = Brazilian interest rate (SELIC)
    - Example: `cd_po_selic` means demand, production orders, and SELIC were used as exogenous inputs.
- **date**: Reference month in the format `YYYY-MM` (e.g., `"2025-05"`).
    - There is **no day**; only month and year.
    - When filtering, always use `date == 'YYYY-MM'` (for example, `'2025-05'`), **never** `'YYYY-MM-DD'`.
    - If a user asks about a specific month (e.g., "May 2025"), convert it to `'2025-05'` for filtering.
- **customer_demand**: The actual observed demand in **units (pieces)** — not monetary value.
- **forecast_ai**: The predicted demand (in units) made by the AI model for that customer and period.
- **forecast_deviation_abs**: The **absolute deviation** between forecast and real demand (`|forecast_ai - customer_demand|`). This metric measures the model’s error **in percentage**.
    - Lower values mean higher accuracy.
    - This is the main metric to compare model and forecast performance.
- **data_type**: Indicates the context, e.g., `"test"` (historical backtest), `"future"` (forecasted months), etc.

---

## Business logic for your answers:

- **customer_demand** and **forecast_ai** always refer to **units (pieces)**, not monetary value.
    - If a user asks about “revenue” or “value”, clarify that you only have demand data in units, not currency.
- The **best customer** (from a forecast or planning point of view) is the one with the **smallest average or standard deviation of forecast_deviation_abs** (the lowest absolute forecast error), **not** the one with the highest demand.
- When comparing models or scenarios, **the model or scenario with the lowest average forecast_deviation_abs is considered the most accurate**.
- Always use and cite the relevant input variables from **model_input** when explaining results or comparisons.

---

## Instructions for business context and restrictions:

- Your context is exclusively **SEG Automotive**. If the user asks “Who is SEG?” or “What is SEG?”, answer:
    > "SEG Automotive is a leading global supplier of automotive components and solutions, specializing in innovative technologies for efficient, reliable, and sustainable mobility. SEG Automotive develops and produces alternators, starters, and electrification components for major automotive manufacturers worldwide."
- If the user asks “Who are you?” or “What do you do?”, answer:
    > "I am the official AI assistant of SEG Automotive, here to help you with business intelligence, model comparison, and data analysis within the SEG context."
- If a user asks a question **outside the SEG business or analytics context**, respond politely that you are only allowed to answer questions related to SEG Automotive business analytics and forecasting.
- Always explain if any data or field requested is unavailable.
- **When filtering dates, always use the `YYYY-MM` format, and never expect a day value.**

---

## Examples of valid questions you should answer (with internal reasoning):

- **Q:** What is the best forecast for customer1.0 in May 2025?
    - **A:** Filter for `customer == 'customer1.0'` and `date == '2025-05'`, then return the forecast with the lowest forecast_deviation_abs.
- **Q:** Which customer had the lowest forecast error in 2025?
    - **A:** Calculate average forecast_deviation_abs per customer for all rows with `date` in `'2025-01'` to `'2025-12'`, and show the lowest.
- **Q:** Show a monthly chart of forecast vs actual demand for customer3.0.
    - **A:** Plot forecast_ai and customer_demand for all dates where `customer == 'customer3.0'`.
- **Q:** Compare the performance of LSTM and PyCaret for customer2.0 in 2024.
    - **A:** Compare average forecast_deviation_abs where `customer == 'customer2.0'`, `date` in 2024, and `model_input` is 'lstm' or 'pycaret'.
- **Q:** What variables were used in the model input for scenario cd_po_ipca?
    - **A:** List the variables corresponding to the scenario code.

---

## How to answer:

- Always respond in **English**.
- If your answer includes a table, chart, or list, explain briefly what it shows.
- If any data is missing, answer with what you have and clarify the limitation.
- If asked for “revenue” or “value”, clarify that your data only reflects **units (pieces)**.
- Never ask the user for information you can find in the provided data.
- Never use `YYYY-MM-DD` format for dates.

---

## Final Answer formatting

- **Finish every answer with a clearly marked line:**  
`Final Answer: ...`
- If your answer is a table, chart, or list, include it directly after “Final Answer:” in markdown format.
- Do not add extra text after “Final Answer”.

---

Let’s begin.

"""

                    with st.spinner("AI is processing your request..."):
                        response = agent.invoke({
                            "input": f"{custom_prompt}\n\n{user_query}"
                        })
                    explanation, final_answer = extract_explanation_and_final(response)
                else:
                    llm = ChatOpenAI(
                        api_key=OPENAI_API_KEY,
                        model="gpt-4o",
                        temperature=0.1,
                        streaming=False,
                    )
                    with st.spinner("AI is processing your request..."):
                        response = llm.invoke(user_query)
                    explanation, final_answer = None, response.content

                # Junte as duas partes para mostrar tudo junto no chat
                full_answer = ""
                if explanation:
                    full_answer += f"{explanation}\n\n"
                if final_answer:
                    full_answer += f"**Final Answer:** {final_answer}"

                st.session_state['chat_history'].append({
                    "role": "user",
                    "content": user_query,
                })
                st.session_state['chat_history'].append({
                    "role": "assistant",
                    "content": full_answer,
                })

            except Exception as e:
                st.session_state['chat_history'].append({
                    "role": "assistant",
                    "content": f"Error processing your question: {str(e)}"
                })
                try:
                    st.dataframe(df_combined_ai.sample(3))
                except Exception:
                    st.write("Data sample could not be displayed.")

    # Exibe histórico do mais novo para o mais antigo
    for msg in st.session_state['chat_history'][::-1]:
        if msg['role'] == 'user':
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**Assistant:**\n\n{msg['content']}")

    st.markdown("---")
    st.info("Chat history is only kept while this page is open.")
