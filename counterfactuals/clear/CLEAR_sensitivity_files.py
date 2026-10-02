from __future__ import print_function

import csv
import os
import numpy as np
import pandas as pd
from . import CLEAR_settings


def Create_sensitivity(X_train, X_test_sample, model):
    print('\n Performing grid search - step 1 of CLEAR method \n')
    
    # Limpiar nombres
    X_train.columns = X_train.columns.str.strip()
    X_test_sample.columns = X_test_sample.columns.str.strip()
    
    feature_list = X_train.columns.tolist()
    
    # Identificar columnas numéricas reales (según configuración)
    numeric_cols = []
    for col in CLEAR_settings.numeric_features:
        col_clean = col.strip().lower()
        for real_col in X_train.columns:
            if real_col.strip().lower() == col_clean:
                numeric_cols.append(real_col)
                break
    
    # Si no hay columnas numéricas, usar todas las que sean numéricas
    if not numeric_cols:
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"  Columnas numéricas encontradas: {numeric_cols}")
    
    # Calcular min/max para numéricas
    if numeric_cols:
        feature_min = X_train[numeric_cols].quantile(.01)
        feature_max = X_train[numeric_cols].quantile(.99)
    else:
        feature_min = pd.Series()
        feature_max = pd.Series()
    
    # Identificar columnas categóricas según prefijos
    categorical_features = []
    if CLEAR_settings.category_prefix:
        categorical_features = [x for x in X_test_sample.columns if any(x.startswith(p) for p in CLEAR_settings.category_prefix)]
    
    print(f"  Columnas categóricas encontradas: {categorical_features}")
    
    # Eliminar archivos temporales
    try:
        os.remove(CLEAR_settings.CLEAR_path + 'numericTemp.csv')
        os.remove(CLEAR_settings.CLEAR_path + 'categoricalTemp.csv')
    except OSError:
        pass
    
    X_test_sample.reset_index(inplace=True, drop=True)
    
    # ========================================================================
    # SENSIBILIDAD NUMÉRICA
    # ========================================================================
    # Asegurar que todo sea numérico (ya debería serlo si pasamos datos codificados)
    full_df = X_test_sample.copy()
    for col in full_df.columns:
        full_df[col] = pd.to_numeric(full_df[col], errors='coerce')
    full_df = full_df.fillna(0)
    
    sensitivity_num = 250
    if numeric_cols:
        for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
            row_values = full_df.iloc[i, :].values.astype(float)
            for j in numeric_cols:
                te = np.tile(row_values, (sensitivity_num, 1))
                te_c = full_df.columns.get_loc(j)
                te[:, te_c] = np.linspace(feature_min.loc[j], feature_max.loc[j], sensitivity_num)
                with open(CLEAR_settings.CLEAR_path + 'numericTemp.csv', 'a') as f:
                    np.savetxt(f, te.astype(float), delimiter=',')
    else:
        print("  ⚠️ No hay columnas numéricas, se omite sensibilidad numérica.")
    
    # ========================================================================
    # SENSIBILIDAD CATEGÓRICA (si hay variables categóricas y prefijos)
    # ========================================================================
    if CLEAR_settings.category_prefix and categorical_features:
        print(f"  Generando sensibilidad categórica para: {categorical_features}")
        for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
            for j in CLEAR_settings.category_prefix:
                # Buscar columnas que comiencen con el prefijo
                cat_idx = [full_df.columns.get_loc(col) for col in full_df.columns if col.startswith(j)]
                if len(cat_idx) < 2:
                    continue
                cat_num = len(cat_idx)
                # Tomar la fila completa (todas las columnas, ya numéricas)
                te = np.tile(full_df.iloc[i, :].values, (cat_num, 1))
                k = 0
                for m in cat_idx:
                    te[k][cat_idx] = 0
                    te[k][m] = 1
                    k += 1
                with open(CLEAR_settings.CLEAR_path + 'categoricalTemp.csv', 'a') as f:
                    np.savetxt(f, te.astype(float), delimiter=',')
    else:
        print("  ℹ️ No hay columnas categóricas o no se definieron prefijos, se omite sensibilidad categórica.")
    
    # ========================================================================
    # LECTURA Y CREACIÓN DE ARCHIVOS DE SENSIBILIDAD
    # ========================================================================
    # Leer el archivo numérico (debe existir)
    num_file = CLEAR_settings.CLEAR_path + 'numericTemp.csv'
    if os.path.exists(num_file):
        sensit_df = pd.read_csv(num_file, header=None, names=full_df.columns)
        print(f"  Archivo numérico cargado: {num_file}")
    else:
        # Si no existe, crear uno vacío con todas las columnas
        print(f"  ⚠️ No se encontró {num_file}, se crea DataFrame vacío.")
        sensit_df = pd.DataFrame(columns=full_df.columns)
    
    # Multi-clase
    if len(CLEAR_settings.class_labels) > 2:
        if CLEAR_settings.multi_class_focus == 'All':
            num_class = len(CLEAR_settings.class_labels)
            multi_index = 0
        else:
            num_class = 1
            multi_index = [k for k, v in CLEAR_settings.class_labels.items() if v == CLEAR_settings.multi_class_focus][0]
        
        for c in range(num_class):
            predictions = model.predict_proba(sensit_df.values)
            temp_df = pd.DataFrame(columns=['observation', 'feature', 'newnn_class', 'probability', 'new_value'])
            cnt = 0
            for i in range(CLEAR_settings.first_obs, CLEAR_settings.last_obs + 1):
                for j in numeric_cols:
                    for k in range(sensitivity_num):
                        temp_df.loc[cnt, 'observation'] = i
                        temp_df.loc[cnt, 'feature'] = feature_list[(cnt // sensitivity_num) % len(numeric_cols)]
                        temp_df.loc[cnt, 'newnn_class'] = np.argmax(predictions[cnt])
                        temp_df.loc[cnt, 'probability'] = predictions[cnt, multi_index]
                        temp_df.loc[cnt, 'new_value'] = sensit_df.loc[cnt, j]
                        cnt += 1
            sensitivity_file = 'numSensitivity_m' + str(multi_index) + '.csv'
            filename1 = CLEAR_settings.CLEAR_path + sensitivity_file
            temp_df.to_csv(filename1, index=False)
            multi_index += 1
    else:
        # Clase binaria
        if not sensit_df.empty:
            values = sensit_df.values
            batch_size = 10000
            if len(values) <= batch_size:
                predictions = model.predict(values).flatten()
            else:
                chunks = []
                for start in range(0, len(values), batch_size):
                    batch = values[start:start + batch_size]
                    chunks.append(model.predict(batch).flatten())
                predictions = np.concatenate(chunks)
            sensitivity_file = 'numSensitivity.csv'
            init_cnt = sensitivity_num * CLEAR_settings.first_obs * len(numeric_cols)
            cnt = 0
            top_row = ['observation', 'feature', 'probability', 'new_value']
            temp = len(numeric_cols)
            with open(CLEAR_settings.CLEAR_path + sensitivity_file, 'w', newline='') as file1:
                writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
                writes.writerow(top_row)
                try:
                    while cnt < len(predictions):
                        feature = numeric_cols[((cnt // sensitivity_num)) % temp]
                        observation = (cnt + init_cnt) // (sensitivity_num * temp)
                        out_list = [observation, feature, predictions[cnt], sensit_df.loc[cnt, feature]]
                        cnt += 1
                        writes.writerow(out_list)
                    print(f"  ✅ {sensitivity_file} creado con {len(predictions)} filas.")
                except Exception as e:
                    print(f"  ⚠️ Error escribiendo {sensitivity_file}: {e}")
            file1.close()
        else:
            print("  ⚠️ No hay datos para sensibilidad numérica, se omite creación de numSensitivity.csv.")
        
        # Categóricas
        cat_file = CLEAR_settings.CLEAR_path + 'categoricalTemp.csv'
        if os.path.exists(cat_file) and categorical_features:
            catSensit_df = pd.read_csv(cat_file, header=None, names=full_df.columns)
            predictions = model.predict(catSensit_df.values).flatten()
            sensitivity_file = 'catSensitivity.csv'
            init_cnt = CLEAR_settings.first_obs * len(categorical_features)
            cnt = 0
            top_row = ['observation', 'feature', 'probability', 'new_value']
            temp = len(categorical_features)
            with open(CLEAR_settings.CLEAR_path + sensitivity_file, 'w', newline='') as file1:
                writes = csv.writer(file1, delimiter=',', skipinitialspace=True)
                writes.writerow(top_row)
                try:
                    while cnt < len(predictions):
                        feature = categorical_features[(cnt % temp)]
                        observation = (cnt + init_cnt) // temp
                        out_list = [observation, feature, predictions[cnt], catSensit_df.loc[cnt, feature]]
                        cnt += 1
                        writes.writerow(out_list)
                except Exception as e:
                    print(f"  ⚠️ Error escribiendo {sensitivity_file}: {e}")
            file1.close()
        else:
            print("  ℹ️ No hay archivo categórico o no hay variables categóricas, se omite catSensitivity.csv.")
    
    return