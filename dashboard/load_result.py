import os
import pandas as pd

def load_all_results(base_path):
    test_list = []
    future_list = []
    test_filenames = ['test_results.csv', 'all_test_results.csv']
    future_filenames = ['future_forecast.csv', 'all_future_forecast.csv']

    for customer in os.listdir(base_path):
        customer_path = os.path.join(base_path, customer)
        if os.path.isdir(customer_path):
            for model_input in os.listdir(customer_path):
                model_path = os.path.join(customer_path, model_input)
                if os.path.isdir(model_path):
                    for dt in os.listdir(model_path):
                        dt_path = os.path.join(model_path, dt)
                        if os.path.isdir(dt_path):
                            # Lê todos os arquivos válidos de teste
                            for test_file in test_filenames:
                                full_path = os.path.join(dt_path, test_file)
                                if os.path.exists(full_path):
                                    df_test = pd.read_csv(full_path)
                                    df_test['customer'] = customer
                                    df_test['model_input'] = model_input
                                    test_list.append(df_test)
                            # Lê todos os arquivos válidos de futuro
                            for future_file in future_filenames:
                                full_path = os.path.join(dt_path, future_file)
                                if os.path.exists(full_path):
                                    df_future = pd.read_csv(full_path)
                                    df_future['customer'] = customer
                                    df_future['model_input'] = model_input
                                    future_list.append(df_future)

    df_all_test = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()
    df_all_future = pd.concat(future_list, ignore_index=True) if future_list else pd.DataFrame()
    df_all_test['date'] = pd.to_datetime(df_all_test['date'], errors='coerce')
    df_all_test = df_all_test[~df_all_test['date'].isnull()]
    df_all_test['month_year'] = df_all_test['date'].dt.to_period('M').astype(str)

    return df_all_test, df_all_future

