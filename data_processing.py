from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from labtools.signal_processing.resampling import resize_signal
from dotenv import load_dotenv

from common import get_data_file_path


def read_data(excel_path: Path, parameter: str = 'DFAa1', test: str = 'vt'):
    if parameter not in ['DFAa1', 'HR']:
        raise ValueError(f"Parameter {parameter} is not supported. Use 'DFAa1' or 'HR'.")
    if test not in ['ramp', 'vt']:
        raise ValueError(f"Test {test} is not supported. Use 'ramp' or 'vt'.")
    if test == 'ramp':
        test_abbr = 'R'
    else:
        test_abbr = 'W'

    sheet_suffix = f'{parameter} {test_abbr}'
    sheet_name_pre = f'LD2-{sheet_suffix}'
    sheet_name_post_24 = f'LD3-{sheet_suffix}'
    sheet_name_post_48 = f'LD4-{sheet_suffix}'

    xl = pd.ExcelFile(excel_path)
    df_pre = xl.parse(sheet_name=sheet_name_pre)
    df_post_24 = xl.parse(sheet_name=sheet_name_post_24)
    df_post_48 = xl.parse(sheet_name=sheet_name_post_48)

    # Reset index to Time column
    df_pre.set_index('Time', inplace=True)
    df_post_24.set_index('Time', inplace=True)
    df_post_48.set_index('Time', inplace=True)

    return df_pre, df_post_24, df_post_48


def filter_dataframes(df_pre: pd.DataFrame, df_post_24: pd.DataFrame, df_post_48: pd.DataFrame, min_length: int = 120):
    # Ensure all dataframes have the same length
    df_pre = df_pre.iloc[:min_length, :]
    df_post_24 = df_post_24.iloc[:min_length, :]
    df_post_48 = df_post_48.iloc[:min_length, :]
    # Filter all-NaN columns (those which do not contain FULL data until min_length)
    df_pre = df_pre.dropna(axis=1, how='any')
    df_post_24 = df_post_24.dropna(axis=1, how='any')
    df_post_48 = df_post_48.dropna(axis=1, how='any')
    # Get unique common participants
    participants = list(set(df_pre.columns) & set(df_post_24.columns) & set(df_post_48.columns))
    print(f'Participants left after filtering: {len(participants)}')
    # Drop participants not in both dataframes
    df_pre = df_pre[participants]
    df_post_24 = df_post_24[participants]
    df_post_48 = df_post_48[participants]

    return df_pre, df_post_24, df_post_48


def trim_signal(s, remove=60):
    arr = s.dropna().to_numpy()
    if remove > 0:
        return arr[remove:-remove] if len(arr) > 2 * remove else np.array([])
    return arr


def time_normalize_dataframes(df1, df2, df3, remove=60, new_length=101):
    # Create a mask for non-NaN values in both dataframes
    df_mask = df1.notna() & df2.notna() & df3.notna()
    # Bring both dataframes to the same length for each participant
    df1 = df1[df_mask]
    df2 = df2[df_mask]
    df3 = df3[df_mask]
    # Trim the signals to remove the first and last 60 samples
    trimmed1 = {col: trim_signal(df1[col], remove) for col in df1.columns}
    trimmed2 = {col: trim_signal(df2[col], remove) for col in df2.columns}
    trimmed3 = {col: trim_signal(df3[col], remove) for col in df3.columns}

    lengths1 = [len(arr) for arr in trimmed1.values()]
    max_length = np.max(lengths1)
    min_len = 60  # minimum length for the signals

    df1_norm = pd.DataFrame(columns=df1.columns)
    df2_norm = pd.DataFrame(columns=df2.columns)
    df3_norm = pd.DataFrame(columns=df3.columns)
    for (col1, arr1), (col2, arr2), (col3, arr3) in zip(trimmed1.items(), trimmed2.items(), trimmed3.items()):
        if col1 != col2 or col1 != col3 or col2 != col3:
            raise ValueError(f"Column names do not match: {col1=}, {col2=}, {col3=}")
        if len(arr1) < min_len:
            print(f"Signal {col1} is too short after trimming: {len(arr1)} samples. Skipping this signal.")
            continue
        if new_length > max_length:
            ValueError(f"The requested length ({new_length})is longer than the maximum possible length ({max_length}).")
        arr1_norm = resize_signal(arr1, new_length=new_length)
        arr2_norm = resize_signal(arr2, new_length=new_length)
        arr3_norm = resize_signal(arr3, new_length=new_length)
        df1_norm[col1] = pd.Series(arr1_norm)
        df2_norm[col2] = pd.Series(arr2_norm)
        df3_norm[col3] = pd.Series(arr3_norm)
    # remove the columns with all NaN values
    df1_norm.dropna(axis=1, how='all', inplace=True)
    df2_norm.dropna(axis=1, how='all', inplace=True)
    df3_norm.dropna(axis=1, how='all', inplace=True)
    return df1_norm, df2_norm, df3_norm


def plot_raw_signals(df_pre, df_post, path_plot):
    # Plot the signals
    for col in df_pre.columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.lineplot(data=df_pre[col], label=f'{col} pre', ax=ax, color='blue', linestyle='-')
        sns.lineplot(data=df_post[col], label=f'{col} post', ax=ax, color='red', linestyle='--')
        ax.legend()
        ax.set_xlabel('Samples')
        ax.set_ylabel('DFA alpha 1')
        plt.title('col')
        path_plot_file = path_plot.joinpath(f'{col}.png')
        plt.savefig(path_plot_file)
        plt.close()


def plot_normalized_signals(df1, df2, df3, path_plot, plot_individual_signals=False):
    # Plot the normalized signals
    if plot_individual_signals:
        for col in df1.columns:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            sns.lineplot(data=df1[col], label=f'{col} pre', ax=ax, color='blue', linestyle='-')
            sns.lineplot(data=df2[col], label=f'{col} post 24h', ax=ax, color='red', linestyle='--')
            sns.lineplot(data=df3[col], label=f'{col} post 48h', ax=ax, color='green', linestyle='-.')
            ax.legend()
            ax.set_xlabel('Samples')
            ax.set_ylabel('DFA alpha 1')
            plt.title(f'Normalized {col}')
            path_plot_file = path_plot.joinpath(f'{col}_normalized.png')
            plt.savefig(path_plot_file)
            plt.close()
    # plot the mean of all signals with sd shaded
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sns.lineplot(data=df1.mean(axis=1), label='pre', ax=ax, color='blue', linestyle='-')
    sns.lineplot(data=df2.mean(axis=1), label='post 24h', ax=ax, color='red', linestyle='--')
    sns.lineplot(data=df3.mean(axis=1), label='post 48h', ax=ax, color='green', linestyle='-.')
    ax.fill_between(df1.index, df1.mean(axis=1) - df1.std(axis=1), df1.mean(axis=1) + df1.std(axis=1), color='blue',
                    alpha=0.2)
    ax.fill_between(df2.index, df2.mean(axis=1) - df2.std(axis=1), df2.mean(axis=1) + df2.std(axis=1), color='red',
                    alpha=0.2)
    ax.fill_between(df3.index, df3.mean(axis=1) - df3.std(axis=1), df3.mean(axis=1) + df3.std(axis=1), color='green',
                    alpha=0.2)
    # add horizontal lines at 0.5 and 0.75
    ax.axhline(y=0.5, color='gray', linestyle='--')
    ax.axhline(y=0.75, color='gray', linestyle='--')
    ax.legend()
    ax.set_xlabel('Samples')
    ax.set_ylabel('DFA alpha 1')
    plt.title('Normalized mean of all signals')
    path_plot_file = path_plot.joinpath('0_mean_all_signals.png')
    plt.savefig(path_plot_file)


def plot_participant_params(df_pre: pd.DataFrame,
                            df_post_24: pd.DataFrame,
                            df_post_48: pd.DataFrame,
                            path: Path,
                            param_name: str):
    if path is None:
        raise ValueError("Path must be provided to save the plots.")
    # Plot the parameters for each participant
    lw = 2  # line width
    path_out = path.joinpath(param_name)
    path_out.mkdir(parents=True, exist_ok=True)
    columns = set(df_pre.columns) | set(df_post_24.columns) | set(df_post_48.columns)
    max_length = max(len(df_pre), len(df_post_24), len(df_post_48))
    for col in columns:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        sns.lineplot(data=df_pre[col], label='pre', ax=ax, linewidth=lw)
        sns.lineplot(data=df_post_24[col], label='post 24h', ax=ax, linestyle='--', linewidth=lw)
        sns.lineplot(data=df_post_48[col], label='post 48h', ax=ax, linestyle='-.', linewidth=lw)
        ax.legend()
        ax.set_xlabel('Samples')
        ax.set_ylabel(param_name)
        ax.set_xlim(0, max_length - 1)
        plt.title(col)
        path_plot_file = path_out.joinpath(f'{col}_{param_name}.png')
        plt.savefig(path_plot_file)
        plt.close()


def get_dataframes(parameter: str = 'DFAa1', test: str = 'vt'):
    path = get_data_file_path()
    path_plot = path.parent.joinpath('plots')
    # GET THE PRE AND POST DATAFRAMES

    make_plots = False

    df_pre, df_post_24, df_post_48 = read_data(path, parameter=parameter, test=test)
    # Check if missing data exist:
    for name, df in zip(["pre", "post_24", "post_48"], [df_pre, df_post_24, df_post_48]):
        valid = df.notna()
        gaps = valid.cumprod().sum() != valid.sum()
        if gaps.any():
            raise ValueError(f"Missing data in {name}: columns {gaps[gaps].index.tolist()}")
    #
    # SOME PLOTTING
    #
    # plot_participant_params(df_pre, df_post_24, df_post_48, param_name=param_name, path=path_plot)
    # plot_raw_signals(df_pre, df_post, path_plot)

    #
    #
    # time normalize or filter the dataframes
    #
    if test == 'vt':
        # FILTER THE DATAFRAMES
        df_pre, df_post_24, df_post_48 = filter_dataframes(df_pre, df_post_24, df_post_48, min_length=440)
        # save the filtered dataframes to excel
        path_out = path.parent.joinpath('filtered_data_VT.xlsx')
        with pd.ExcelWriter(path_out) as writer:
            df_pre.to_excel(writer, sheet_name='Pre')
            df_post_24.to_excel(writer, sheet_name='Post 24h')
            df_post_48.to_excel(writer, sheet_name='Post 48h')
        if make_plots:
            plot_participant_params(df_pre, df_post_24, df_post_48, path_plot, param_name=parameter)
            plot_raw_signals(df_pre, df_post_24, path_plot)

        # EXPERIMENTAL:
        df_pre, df_post_24, df_post_48 = time_normalize_dataframes(df_pre, df_post_24, df_post_48, remove=0)
    else:
        # TIME NORMALIZE THE DATAFRAMES
        df_pre_norm, df_post_24_norm, df_post_48_norm = time_normalize_dataframes(df_pre, df_post_24, df_post_48,
                                                                                  remove=0)
        if make_plots:
            plot_normalized_signals(df_pre_norm, df_post_24_norm, df_post_48_norm, path_plot,
                                    plot_individual_signals=True)
        # save the normalized dataframes to excel
        path_out = path.parent.joinpath('normalized_data_RAMP.xlsx')
        # Save the normalized dataframes to Excel
        with pd.ExcelWriter(path_out) as writer:
            df_pre_norm.to_excel(writer, sheet_name='Pre')
            df_post_24_norm.to_excel(writer, sheet_name='Post 24h')
            df_post_48_norm.to_excel(writer, sheet_name='Post 48h')
        df_pre, df_post_24, df_post_48 = df_pre_norm, df_post_24_norm, df_post_48_norm
    return df_pre, df_post_24, df_post_48


if __name__ == '__main__':
    df_pre, df_post_24, df_post_48 = get_dataframes(test='ramp', parameter='RAMP')
