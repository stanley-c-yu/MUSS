import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.gridspec as gridspec
import seaborn as sns


def table_3(summary_file):
    output_filepath = summary_file.replace(
        'prediction_sets', 'publication_figures_and_tables').replace('summary.csv', 'table3.csv')
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    summary = pd.read_csv(summary_file)
    filter_condition = (summary['FEATURE_TYPE'] == 'MO') & (
        summary['NUM_OF_SENSORS'] == 2)
    table3_data = summary.loc[filter_condition, [
        'SENSOR_PLACEMENT', 'POSTURE_AVERAGE', 'Lying', 'Sitting', 'Upright']]
    table3_data.columns = ['Sensor placements',
                           'Average', 'Lying', 'Sitting', 'Upright']
    table3_data = table3_data.sort_values(by=['Average'], ascending=False)
    table3_data.loc[:, 'Sensor placements'] = table3_data['Sensor placements'].transform(
        lambda s: s.replace('_', ' and '))
    table3_data.to_csv(output_filepath, float_format='%.2f', index=False)


def table_4(summary_file):
    output_filepath = summary_file.replace(
        'prediction_sets', 'publication_figures_and_tables').replace('summary.csv', 'table4.csv')
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    summary = pd.read_csv(summary_file)
    filter_condition = (summary['FEATURE_TYPE'] == 'MO') & (
        summary['NUM_OF_SENSORS'] == 2)
    table4_data = summary.loc[filter_condition, [
        'SENSOR_PLACEMENT', 'ACTIVITY_AVERAGE',  'Between activity groups', 'Within activity groups']]
    table4_data.columns = ['Sensor placements',
                           'Average', 'Between activity groups', 'Within activity groups']
    table4_data = table4_data.sort_values(by=['Average'], ascending=False)
    table4_data.loc[:, 'Sensor placements'] = table4_data['Sensor placements'].transform(
        lambda s: s.replace('_', ' and '))
    table4_data.to_csv(output_filepath, float_format='%.2f', index=False)


def figure_1(summary_file):
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    rcParams['font.serif'] = ['Times New Roman']
    # setup configurations
    figure_file_extensions = ['.png', '.svg', '.pdf', '.eps']
    output_filepaths = [summary_file.replace(
        'prediction_sets', 'publication_figures_and_tables').replace('summary.csv', 'figure1' + extension) for extension in figure_file_extensions]
    os.makedirs(os.path.dirname(output_filepaths[0]), exist_ok=True)

    # read data
    summary = pd.read_csv(summary_file)

    #  prepare line plot data
    line_plot_data = summary[[
        'NUM_OF_SENSORS', 'FEATURE_TYPE', 'POSTURE_AVERAGE', 'ACTIVITY_AVERAGE']]
    line_plot_data.columns = [
        'Number of sensors', 'Feature set', 'Posture', 'PA']
    line_plot_data.loc[:, 'Feature set'] = line_plot_data['Feature set'].map({
        'M': 'Motion features only',
        'O': 'Orientation related features only',
        'MO': 'Motion + orientation related features'
    })

    # prepare point plot data
    point_plot_data = summary.loc[summary['FEATURE_TYPE'] == 'MO', [
        'SENSOR_PLACEMENT', 'NUM_OF_SENSORS', 'POSTURE_AVERAGE', 'ACTIVITY_AVERAGE']]
    point_plot_data.columns = ['Sensor placements',
                               'Number of sensors', 'Posture', 'PA']

    point_plot_data = pd.melt(point_plot_data, id_vars=['Sensor placements', 'Number of sensors'], value_vars=[
                              'Posture', 'PA'], var_name='Classification task', value_name='F1-score')
    point_plot_data['Include dominant wrist'] = point_plot_data['Sensor placements'].transform(
        lambda s: 'DW' in s.split('_'))
    point_plot_data['Include nondominant wrist'] = point_plot_data['Sensor placements'].transform(
        lambda s: 'NDW' in s.split('_'))

    point_plot_data = point_plot_data.sort_values(
        by=['Number of sensors', 'Classification task', 'F1-score'], ascending=True)

    point_plot_data = point_plot_data.groupby(['Number of sensors', 'Classification task']).apply(
        lambda rows: pd.concat((rows.head(5), rows.tail(5))))
    point_plot_data = point_plot_data.reset_index(drop=True).drop_duplicates()

    # draw plots
    g, axes = plt.subplots(2, 2, figsize=(12, 8))
    for task, index in zip(['Posture', 'PA'], [0, 1]):
        # draw swarm and line for MO feature set
        axes[index][0].set_ylim(0, 1.2)
        axes[index][0].yaxis.grid(linestyle='--')
        swarm_data = point_plot_data.loc[point_plot_data['Classification task'] == task, :]
        if task == 'Posture':
            swarm_data_wrist = swarm_data.loc[(swarm_data['Include dominant wrist'] == True) | (
                swarm_data['Include nondominant wrist'] == True), :]
            swarm_data_nonwrist = swarm_data.loc[(swarm_data['Include dominant wrist'] == False) & (
                swarm_data['Include nondominant wrist'] == False), :]
        else:
            swarm_data_wrist = swarm_data.loc[swarm_data['Include dominant wrist'] == True, :]
            swarm_data_nonwrist = swarm_data.loc[swarm_data['Include dominant wrist'] == False, :]
        line_data_mo = line_plot_data.loc[line_plot_data['Feature set'] == 'Motion + orientation related features', [
            'Number of sensors', 'Feature set', task]].rename(columns={task: 'F1-score'})
        sns.stripplot(x='Number of sensors', y='F1-score', data=swarm_data_wrist,
                      ax=axes[index][0], marker='o', jitter=True, color='dimgrey')
        sns.stripplot(x='Number of sensors', y='F1-score', data=swarm_data_nonwrist,
                      ax=axes[index][0], marker='o', jitter=True, color='white', linewidth=1)
        sns.pointplot(x='Number of sensors', y='F1-score', data=line_data_mo,
                      dodge=True, ax=axes[index][0], color='black', markers='x')
        
        if task == 'Posture':
            axes[index][0].annotate('Motion + orientation related features', xy=(4.5, 0.95), xycoords='data', xytext=[2.5, 0.7],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='black'), horizontalalignment='left')
            axes[index][0].annotate('"o": Non-wrist sensors', xy=(0.05, 0.93), xycoords='data', xytext=[0.15, 1.1],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='dimgrey'), horizontalalignment='left', color='dimgrey')
            axes[index][0].annotate('"●": Including wrist sensors', xy=(0.05, 0.58), xycoords='data', xytext=[0.2, 0.4],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='dimgrey'), horizontalalignment='left', color='dimgrey')
        else:
            axes[index][0].annotate('Motion + orientation related features', xy=[4.5, 0.7], xycoords='data', xytext=(
                2.5, 0.95), textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='black'), horizontalalignment='left')
            axes[index][0].annotate('"●": Including dominant wrist sensor', xy=(1, 0.7), xycoords='data', xytext=[1.6, 0.8],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='dimgrey'), horizontalalignment='center', color='dimgrey')
            axes[index][0].annotate('"o": Not including dominant wrist sensor', xy=(1.2, 0.4), xycoords='data', xytext=[1.8, 0.25],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='dimgrey'), horizontalalignment='center', color='dimgrey')

        # draw line for other feature set
        
        line_data_others = line_plot_data.loc[line_plot_data['Feature set'] != 'Motion + orientation related features', [
            'Number of sensors', 'Feature set', task]].rename(columns={task: 'F1-score'})
        sns.pointplot(x='Number of sensors', y='F1-score', data=line_data_others,
                      dodge=True, ax=axes[index][1], hue='Feature set', color='black', linestyles=['-', '-.'], markers='x')
        
        if task == 'Posture':
            axes[index][1].annotate('Motion features only', xy=(4.3, 0.75), xycoords='data', xytext=[4.5, 0.6],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='black'), horizontalalignment='center')
            axes[index][1].annotate('Orientation related features only', xy=(4.5, 1), xycoords='data', xytext=[4.5, 1.1],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='black'), horizontalalignment='center', color='black')
        else:
            axes[index][1].annotate('Motion features only', xy=(4.3, 0.6), xycoords='data', xytext=[4.5, 0.75],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='black'), horizontalalignment='center')
            axes[index][1].annotate('Orientation related features only', xy=(4.5, 0.45), xycoords='data', xytext=[4.5, 0.25],
                                    textcoords='data', arrowprops=dict(arrowstyle='->', connectionstyle="arc3", facecolor='black'), horizontalalignment='center', color='black')
        axes[index][1].legend().remove()
        axes[index][1].set_ylim(0, 1.2)
        axes[index][1].set_yticklabels([])
        axes[index][1].yaxis.set_major_formatter(plt.NullFormatter())
        axes[index][1].set_ylabel('')
        axes[index][1].yaxis.grid(linestyle='--')
        axes[index][1].spines['left'].set_color('grey')

    g.subplots_adjust(wspace=0, hspace=0.35)
    plt.figtext(0.5, 0.5, '(a) Posture recognition performances', ha='center', va='top')
    plt.figtext(0.5, 0.06, '(b) PA recognition performances', ha='center', va='top')
    # save figure in different formats
    for output_filepath in output_filepaths:
        plt.savefig(output_filepath, dpi=300, orientation='landscape')


if __name__ == '__main__':
    dataset_folder = 'D:/data/spades_lab/'
    summary_file = os.path.join(
        dataset_folder, 'DerivedCrossParticipants', 'location_matters', 'prediction_sets', 'summary.csv')
    # table_3(summary_file)
    # table_4(summary_file)
    figure_1(summary_file)
