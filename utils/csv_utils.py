import pandas as pd
import os

def _build_metric_dataframe(keys, total_classes):
    df_all = pd.DataFrame(index=total_classes, columns=keys, dtype=float)
    return df_all


def _split_existing_dataframe(df):
    metric_rows = [idx for idx in df.index if idx != "mean" and not str(idx).startswith("param:")]
    param_rows = [idx for idx in df.index if str(idx).startswith("param:")]
    metric_df = df.loc[metric_rows].copy()
    param_df = df.loc[param_rows].copy() if param_rows else pd.DataFrame(columns=df.columns)
    return metric_df, param_df


def _stringify_param_value(value):
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value)
    return str(value)


def write_results(results: dict, cur_class, total_classes, csv_path, params: dict | None = None):
    keys = list(results.keys())

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
        metric_df, _ = _split_existing_dataframe(df)
        metric_df = metric_df.reindex(total_classes)
        for key in keys:
            if key not in metric_df.columns:
                metric_df[key] = pd.NA
        metric_df = metric_df[keys]
    else:
        metric_df = _build_metric_dataframe(keys, total_classes)

    for key in keys:
        metric_df.loc[cur_class, key] = results[key]

    metric_df = metric_df.apply(pd.to_numeric, errors="coerce")
    mean_row = metric_df.mean(axis=0, skipna=True).to_frame().T
    mean_row.index = ["mean"]

    combined_df = pd.concat([metric_df, mean_row], axis=0)
    combined_df["param_value"] = pd.NA

    if params:
        param_rows = []
        for key, value in params.items():
            row = {metric_key: pd.NA for metric_key in keys}
            row["param_value"] = _stringify_param_value(value)
            param_rows.append(pd.DataFrame(row, index=[f"param:{key}"]))
        if param_rows:
            param_df = pd.concat(param_rows, axis=0)
            combined_df = pd.concat([combined_df, param_df], axis=0)

    combined_df.to_csv(csv_path, header=True, float_format='%.2f')


def save_metric(metrics, total_classes, class_name, dataset, csv_path, params: dict | None = None):
    total_classes = list(total_classes)
    if dataset != 'mvtec':
        for indx in range(len(total_classes)):
            total_classes[indx] = f"{dataset}-{total_classes[indx]}"
        class_name = f"{dataset}-{class_name}"
    write_results(metrics, class_name, total_classes, csv_path, params=params)
