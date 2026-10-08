from pathlib import Path
from sys import platform

import numpy as np
import pandas as pd
import seaborn as sns
import spm1d
from matplotlib import pyplot as plt
from labtools.signal_processing.resampling import resize_signal
from dotenv import load_dotenv

load_dotenv()


def get_data_file_path():
    import os
    one_drive_root = Path(os.getenv('ONE_DRIVE_ROOT'))
    path = one_drive_root.joinpath('Dokumente')
    path = path.joinpath('projects')
    path = path.joinpath('PetersDaten')
    path = path.joinpath('continuousstatistics')
    path = path.joinpath('DatenPeter.xlsx')
    assert path.exists()
    return path


def get_dataframes(excel_path: Path, parameter: str = 'DFAa1', test: str = 'vt'):
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


def time_normalize_dataframes(df1, df2, df3, remove=60):
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
        new_length = 101
        # new_length = max_length
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


def perform_spm1D(df1: pd.DataFrame,
                  df2: pd.DataFrame,
                  df3: pd.DataFrame,
                  x_label: str,
                  y_label: str,
                  alpha: float = 0.05,
                  path_plot=None):
    # Ensure that the dataframes have the same column order.
    df2 = df2[df1.columns]
    df3 = df3[df1.columns]

    # Convert dataframes to numpy arrays
    yA = df1.to_numpy().T
    yB = df2.to_numpy().T
    yC = df3.to_numpy().T

    # (2) Plot:
    plt.close('all')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot means and standard deviations:
    color_A = 'k'  # gray
    color_B = 'r'  # red
    color_C = 'g'  # green
    spm1d.plot.plot_mean_sd(yA, ax=ax1, linecolor=color_A, facecolor=color_A, edgecolor=color_A, label='PRE')
    spm1d.plot.plot_mean_sd(yB, ax=ax1, linecolor=color_B, facecolor=color_B, edgecolor=color_B, label='POST24h')
    spm1d.plot.plot_mean_sd(yC, ax=ax1, linecolor=color_C, facecolor=color_C, edgecolor=color_C, label='POST48h')
    # ax1.set_title('Means and SDs')
    ax1.set_ylabel(y_label, fontsize=16)
    ax1.legend(fontsize=14)

    # Perform one-way repeated measures ANOVA using the explicit method
    n_subjects = yA.shape[0]
    n_conditions = 3

    # Stack the data:
    signals = np.vstack([yA, yB, yC])

    # Define the repeated measures factor (A) and the subject ID factor (SUBJ)
    rm_factor = np.array([0] * n_subjects + [1] * n_subjects + [2] * n_subjects)
    subj = np.array(list(range(n_subjects)) * n_conditions)

    # The anova1rm function requires data in (observations x variables) format.
    # Here, observations are (subjects x conditions), and variables are the time points.
    F = spm1d.stats.anova1rm(signals, rm_factor, subj, equal_var=True)
    Fi = F.inference(alpha=alpha)

    # ax2.set_title('One-way repeated measures ANOVA')
    Fi.plot(ax=ax2)
    Fi.plot_threshold_label(fontsize=14)
    ax2.set_xlabel(x_label, fontsize=16)

    # set x and y tick labels font size to 16
    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)

    # Perform post-hoc paired t-tests if the ANOVA is significant
    # if Fi.h0reject:
    if True:
        print("ANOVA was significant, performing post-hoc tests...")

        # Bonferroni correction for multiple comparisons
        adj_p = alpha / 3
        print(f"Applying Bonferroni correction. New alpha: {adj_p:.4f}")

        # Create a new figure for post-hoc tests
        # fig_posthoc, axes_posthoc = plt.subplots(3, 1, figsize=(8, 15))
        fig_posthoc, axes_posthoc = plt.subplots(1, 3, figsize=(15, 8))
        fig_posthoc.suptitle(f'Post-Hoc Paired t-tests (Bonferroni adjusted α={adj_p:.3f})', fontsize=16)

        # Post-hoc test: pre vs. post 24h
        spmi_AB = spm1d.stats.ttest_paired(yA, yB).inference(alpha=adj_p, two_tailed=True)
        ax_AB = axes_posthoc[0]
        spmi_AB.plot(ax=ax_AB)
        spmi_AB.plot_threshold_label(ax=ax_AB, fontsize=14)
        ax_AB.set_title('PRE vs. POST24h', fontsize=16)

        # Post-hoc test: pre vs. post 48h
        spmi_AC = spm1d.stats.ttest_paired(yA, yC).inference(alpha=adj_p, two_tailed=True)
        ax_AC = axes_posthoc[1]
        spmi_AC.plot(ax=ax_AC)
        spmi_AC.plot_threshold_label(ax=ax_AC, fontsize=14)
        ax_AC.set_title('PRE vs. POST48h', fontsize=16)

        # Post-hoc test: post 24h vs. post 48h
        spmi_BC = spm1d.stats.ttest_paired(yB, yC).inference(alpha=adj_p, two_tailed=True)
        ax_BC = axes_posthoc[2]
        spmi_BC.plot(ax=ax_BC)
        spmi_BC.plot_threshold_label(ax=ax_BC, fontsize=14)
        ax_BC.set_title('POST24h vs. POST48h', fontsize=16)

        # increase axis labels and tick labels font size for x and y axes
        for ax in axes_posthoc:
            ax.tick_params(axis='x', labelsize=16)
            ax.tick_params(axis='y', labelsize=16)
            ax.set_xlabel(x_label, fontsize=16)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if path_plot:
            plt.savefig(path_plot.joinpath('spm_posthoc_tests.png'))
            print(f"Saved post-hoc tests plot to {path_plot.joinpath('spm_posthoc_tests.png')}.")
            plt.close()
    else:
        print("ANOVA was not significant, no post-hoc tests needed.")

    if path_plot:
        fig.tight_layout()
        fig.savefig(path_plot.joinpath('spm_anova1rm.png'))
        print(f"Saved ANOVA plot to {path_plot.joinpath('spm_anova1rm.png')}.")
    plt.close()
    # plt.show()


def perform_spm1D_OLD(df1, df2, df3, alpha: float = 0.05):
    #
    # perform statistical parametric mapping between df_pre_norm and df_post_norm
    #

    # make sure that the dataframes have the same column order. Assumes that df1, df2, df3 have the same columns
    df2 = df2[df1.columns]
    df3 = df3[df1.columns]

    # convert dataframes to numpy arrays
    yA = df1.to_numpy().T
    yB = df2.to_numpy().T
    yC = df3.to_numpy().T

    # (2) Plot:
    plt.close('all')
    plt.figure(figsize=(10, 6))

    ### plot means and standard deviations:
    ax = plt.subplot(211)
    color_A = 'k'  # gray
    color_B = 'r'  # red
    color_C = 'g'  # green
    spm1d.plot.plot_mean_sd(yA, linecolor=color_A, facecolor=color_A, edgecolor=color_A)
    spm1d.plot.plot_mean_sd(yB, linecolor=color_B, facecolor=color_B, edgecolor=color_B)
    spm1d.plot.plot_mean_sd(yC, linecolor=color_C, facecolor=color_C, edgecolor=color_C)
    spm1d.plot.legend_manual(ax,
                             labels=['Pre', 'Post 24h', 'Post 48h'],
                             colors=[color_A, color_B, color_C],
                             linestyles=['-'] * 3,
                             linewidths=[10, 10, 10],
                             loc='upper right',
                             fontsize=12)
    ax.set_title('Means and SDs', fontsize=18)

    ### plot hypothesis test results:
    # create signal matrix
    signals = np.vstack([yA, yB, yC])
    # create repeated measures factor
    rm_factor = np.array([1] * len(yA) + [2] * len(yB) + [3] * len(yC))
    # create numeric subject IDs
    n_subj = df1.shape[1]
    subj = np.tile(np.arange(n_subj), 3)

    ax = plt.subplot(212)
    F = spm1d.stats.anova1rm(signals, rm_factor, subj, equal_var=True)
    Fi = F.inference(alpha=0.05)
    Fi.plot(ax=ax)
    Fi.plot_threshold_label(fontsize=12)

    # spmi = spm1d.stats.ttest_paired(yA, yB).inference(alpha, two_tailed=True)
    # spmi.plot(ax=ax)
    # spmi.plot_threshold_label()
    ax.set_title('Hypothesis test', fontsize=18)
    # ax.text(0.6, 0.2, 'Datum: zero\nCriterion:  $t ^*$', transform=ax.transAxes)

    plt.savefig(path_plot.joinpath('spm_anova1rm.png'))


if __name__ == '__main__':
    path = get_data_file_path()
    path_plot = path.parent.joinpath('plots')
    # GET THE PRE AND POST DATAFRAMES
    param_name = 'DFAa1'  # 'DFAa1' or 'HR'
    test_type = 'ramp'  # 'ramp' or 'vt'
    make_plots = False

    df_pre, df_post_24, df_post_48 = get_dataframes(path, parameter=param_name, test=test_type)
    #
    # SOME PLOTTING
    #
    # plot_participant_params(df_pre, df_post_24, df_post_48, param_name=param_name, path=path_plot)
    # plot_raw_signals(df_pre, df_post, path_plot)

    #
    #
    # time normalize or filter the dataframes
    #
    if test_type == 'vt':
        # FILTER THE DATAFRAMES
        df_pre, df_post_24, df_post_48 = filter_dataframes(df_pre, df_post_24, df_post_48, min_length=440)
        # save the filtered dataframes to excel
        path_out = path.parent.joinpath('filtered_data_VT.xlsx')
        with pd.ExcelWriter(path_out) as writer:
            df_pre.to_excel(writer, sheet_name='Pre')
            df_post_24.to_excel(writer, sheet_name='Post 24h')
            df_post_48.to_excel(writer, sheet_name='Post 48h')
        if make_plots:
            plot_participant_params(df_pre, df_post_24, df_post_48, path_plot, param_name=param_name)
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

    print(len(df_pre))
    print(len(df_post_24))
    print(len(df_post_48))
    perform_spm1D(df_pre,
                  df_post_24,
                  df_post_48,
                  x_label="Normalized Test Duration [%]",
                  y_label='DFAa1' if param_name == 'DFAa1' else 'Heart Rate [bpm]',
                  alpha=0.05,
                  path_plot=path_plot)