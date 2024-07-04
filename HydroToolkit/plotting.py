import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import theilslopes, pearsonr
from scipy.optimize import curve_fit
import functools
from . import utils
from . import core

cm = 1/2.54 # inch/cm conversion factor
page = (21.0 - 1.65 - 1.25)*cm # A4 page
colors = ['#0099ff', '#89b576', '#f27977', '#ff6699',
          '#99e6ff', '#006600', '#fae6ff']

def save_plot(func):
    @functools.wraps(func)
    def wrapper(*args, save:bool = False, path:str = None, **kwargs):
        fig, ax = func(*args, **kwargs)
        if save:
            fig.savefig(path, dpi=300)
        plt.show()
        return fig, ax
    return wrapper

class Plotter:
    def __init__(self, df: pd.DataFrame, df_resampled: pd.DataFrame, 
                 metadata: dict, translate_unit: dict):
        """
        Initialize the Plotter with dataframe and metadata.

        Args:
            df (pd.DataFrame): Dataframe with data.
            metadata (dict): Metadata dictionary.
            translate_unit (dict): Dictionary for unit translations.
        """
        self.df = df
        self.df_resampled = df_resampled
        self.metadata = metadata
        self.translate_unit = translate_unit

    @save_plot
    def plot_timeseries(self, variables: list, filter: bool = False,
                        trend: bool = False, daily: bool = True):
        """
        Plot time series for the specified variables. You may select variables to plot,
        apply a filter to the data, and plot a seasonal Mann-Kendall trend line.
        The trend is only plotted if p-value < 0.05.

        Args:
            variables (list): List of variables to plot.
            filter (bool): Whether to plot filtered data. Defaults to False.
            trend (bool): Whether to plot seasonal Mann-Kendall trend line. Defaults to False.
            daily (bool): Whether to use daily data. If set False, the native 
            resolution of the data will be used. Defaults to True.

        Returns:
            tuple: (fig, axes) Matplotlib figure and axes.
        """

        if daily:
            data = self.df_resampled
        else:
            data = self.df

        fig, axes = plt.subplots(len(variables), 1, figsize=(page, page*0.5), sharex=True)

        for i, variable in enumerate(variables):
            axes[i].plot(data[variable], color=colors[i], label=variable)
            axes[i].set_ylabel(self.translate_unit.get(variable, variable))
            
            if filter and f'{variable}_filtered' in data.columns:
                axes[i].plot(data[f'{variable}_filtered'], label=f'Filtered {variable}')
            
            if trend:
                y_data = data[variable].resample('D').mean()
                x_data = y_data.reset_index().index/365
                trend = utils.mann_kendall_test(y_data, period=365)
                y_pred = x_data * trend['slope'] + trend['intercept']
                print(f'Mann-Kendall: {variable} = {trend["slope"]:.2f} * year + {trend["intercept"]:.2f}, p-value: {trend["p-value"]:.2f}')
                
                if trend['p-value'] < 0.05:
                    axes[i].plot(y_data.index, y_pred, c='black', ls='--',
                                 label='Mann-Kendall slope')
        plt.tight_layout()
        return fig, axes

    @save_plot
    def plot_histogram(self, variables: list, daily: bool = True):
        """
        Plot histograms for the specified variables.

        Args:
            variables (list): List of variables to plot.
            save (bool): Whether to save the plot. Defaults to False.
            path (str): Path to save the plot. Defaults to None.

        Returns:
            tuple: (fig, axes) Matplotlib figure and axes.
        """
        if daily:
            data = self.df_resampled
        else:
            data = self.df

        fig, axes = plt.subplots(1, len(variables), 
                                 figsize=(page*0.3*len(variables), 5*cm))

        for i, variable in enumerate(variables):
            data[variable].hist(ax=axes[i], color=colors[i], edgecolor='k', 
                                   density=True)
            x, y = utils.create_kde(data[variable])
            axes[i].plot(x, y, c='k', ls='--')
            axes[i].set_title('')
            axes[i].set_xlabel(self.translate_unit.get(variable, variable))

        axes[0].set_ylabel('Density')
        plt.tight_layout()
        return fig, axes
    
    @save_plot
    def plot_ts_hist(self, variables: list, trend: bool = False, daily: bool = True):
        """plot combined timeseries and histogram for the specified variables
        in a panel plot.

        Args:
            variables (list): List of variables to plot
            trend (bool, optional): If ture, the Theil-Sen Slope of the 
            timeseries is also plotted. Defaults to False.

        Returns:
            tuple: (fig, axes) Matplotlib figure and axes.
        """
        if daily:
            df = self.df_resampled
        else:
            df = self.df

        fig, axes = plt.subplots(len(variables), 2,
                                 figsize=(page, page * 0.2 * len(variables)),
                                 gridspec_kw={'width_ratios': [4, 1]})

        # Determine the overall min and max time span with some padding
        all_dates = pd.Series(df.index.unique())
        min_date, max_date = all_dates.min(), all_dates.max()
        date_range = max_date - min_date
        padding = date_range * 0.05
        padded_min_date = min_date - padding
        padded_max_date = max_date + padding

        for i, variable in enumerate(variables):
            # Time series plot
            ax_ts = axes[i, 0]
            ax_ts.plot(df[variable], color=colors[i])
            if trend:
                y_data = df[variable].resample('D').mean()
                x_data = y_data.reset_index().index/365
                trend = utils.mann_kendall_test(y_data, period=365)
                y_pred = x_data * trend['slope'] + trend['intercept']
                print(f'Mann-Kendall: {variable} = {trend["slope"]:.2f} * year + {trend["intercept"]:.2f}, p-value: {trend["p-value"]:.2}')
                
                if trend['p-value'] < 0.05:
                    ax_ts.plot(y_data.index, y_pred, c='black', ls='--', 
                               label='Mann-Kendall slope')

                # data = df[variable].bfill().dropna()
                # slope, intercept, _, _ = theilslopes(data, data.index.to_julian_date())
                # y = intercept + slope * data.index.to_julian_date()
                # ax_ts.plot(data.index, y, c='black', ls='--', label='Theil-Sen slope')
                # print(f'Theil: {variable}(year) = {slope * 365:.2f} * year + {intercept:.2f}')

            ax_ts.set_ylabel(self.translate_unit.get(variable, variable))
            ax_ts.set_xlim(padded_min_date, padded_max_date)

            # Histogram with KDE plot
            ax_hist = axes[i, 1]
            sns.histplot(df[variable], kde=True, ax=ax_hist, color=colors[i], bins=15)
            ax_hist.yaxis.tick_right()
            ax_hist.yaxis.set_label_position("right")
            ax_hist.set(xlabel=self.translate_unit.get(variable, variable), ylabel='Anzahl')
            

        plt.tight_layout()
        return fig, axes
    
        # if daily:
        #     df = self.df_resampled
        # else:
        #     df = self.df

        # fig, axes = plt.subplots(len(variables), 2,
        #                          figsize=(page, page * 0.2 * len(variables)),
        #                          gridspec_kw={'width_ratios': [4, 1]})

        # # Determine the overall min and max time span with some padding
        # all_dates = pd.Series(df.index.unique())
        # min_date, max_date = all_dates.min(), all_dates.max()
        # date_range = max_date - min_date
        # padding = date_range * 0.05
        # padded_min_date = min_date - padding
        # padded_max_date = max_date + padding

        # for i, variable in enumerate(variables):
        #     # Time series plot
        #     ax_ts = axes[i, 0]
        #     ax_ts.plot(df[variable], color=colors[i])
        #     if trend:
        #         data = df[variable].bfill().dropna()
        #         slope, intercept, _, _ = theilslopes(data, data.index.to_julian_date())
        #         y = intercept + slope * data.index.to_julian_date()
        #         ax_ts.plot(data.index, y, c='black', ls='--', label='Theil-Sen slope')
        #         print(f'Theil: {variable}(year) = {slope * 365:.2f} * year + {intercept:.2f}')
        #     ax_ts.set_ylabel(self.translate_unit.get(variable, variable))
        #     ax_ts.set_xlim(padded_min_date, padded_max_date)

        #     # Histogram with KDE plot
        #     ax_hist = axes[i, 1]
        #     sns.histplot(data[variable], kde=True, ax=ax_hist, color=colors[i], 
        #                  bins=15)
        #     ax_hist.yaxis.tick_right()
        #     ax_hist.yaxis.set_label_position("right")
        #     ax_hist.set(xlabel=self.translate_unit.get(variable, variable), 
        #                 ylabel='Anzahl')

        # plt.tight_layout()
        # return fig, axes
    
    @save_plot
    def plot_confidence_intervals(self, variable: str, stats: pd.DataFrame):
        """Plot the annual distribution of a variable as a function of day of 
        the year.

        Args:
            variable (str): identifier of variable to plot. this is only used 
            for labelling.
            stats (pd.DataFrame): dataframe containing the distribution metric
            for each day of the year. use 'create_stats()' method to generate

        Returns:
            tuple: (fig, axes) Matplotlib figure and axes.
        """

        fig, ax = plt.subplots(figsize=(page, page*0.4))
        ax.fill_between(stats.index, stats['min'], stats['max'], 
                        color='#0099ff', alpha=0.5, label='Min/Max')
        ax.plot(stats.index, stats['mean'], color='black', label='Mean')
        ax.plot(stats.index, stats['median'], color='black', linestyle='--', label='Median')

        months = ['FEB', 'APR', 'JUN', 'AUG', 'OCT', 'DEC']
        month_days = [32, 91, 152, 213, 274,335]
        ax.set_xticks(ticks=month_days, labels=months)
        ax.minorticks_off()
        ax.set_ylabel(self.translate_unit.get(variable, variable))
        ax.legend()

        plt.tight_layout()
        return fig, ax
    
    @save_plot
    def plot_ci_panel(self, stats_list: list, variables: list):
        """
        Plot confidence intervals for multiple variables in separate subplots.

        Args:
            variables (list): List of variables to plot.

        Returns:
            tuple: (fig, axes) Matplotlib figure and axes.
        """

        if len(variables) < 2:
            raise ValueError('list of variables has length < 2. Parse a longer list or use plot_confidence_interval() instead.')
        
        fig, axes = plt.subplots(len(stats_list), 1, sharex=True, 
                                 figsize=(page, page*0.2*len(stats_list)))
                                
        for i, stats in enumerate(stats_list):

            ax = axes[i]
            
            # Plot min/max envelope
            ax.fill_between(stats.index, stats['min'], stats['max'],
                            color=colors[i], alpha=0.5, label='Min/Max')
            
            # Plot mean
            ax.plot(stats.index, stats['mean'], color=colors[i], label='Mean')
            
            # Plot median
            ax.plot(stats.index, stats['median'], color=colors[i], linestyle='--',
                    label='Median')
            
            ax.set_ylabel(self.translate_unit.get(variables[i], variables[i]))
            ax.legend()

        axes[-1].set_xlabel('Time')
        months = ['FEB', 'APR', 'JUN', 'AUG', 'OCT', 'DEC']
        month_days = [32, 91, 152, 213, 274,335]
        axes[-1].set_xticks(ticks=month_days, labels=months)
        axes[-1].minorticks_off()
        plt.tight_layout()
        return fig, axes
    
    @save_plot
    def plot_autocorrelations(self, autocorrelations: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(page*0.4, 5*cm))
        for column in autocorrelations.columns:
            ax.plot(autocorrelations.index, autocorrelations[column],
                    label=self.translate_unit.get(column, column))
        
        ax.hlines([0], 0, 365, colors=['black'], ls='--')
        ax.set(xlabel='Versatz [Tage]', ylabel='Autokorrelation',
               ylim=(-1,1))
        ax.legend()
        plt.tight_layout()
        return fig, ax
    
    @save_plot
    def plot_cumulative_distribution(self, variables: list):
        """
        Plot cumulative distribution functions for the specified variables.

        Args:
            variables (list): List of variables to plot.

        Returns:
            tuple: (fig, axes) Matplotlib figure and axes.
        """
        data = self.df_resampled[variables]
        data['year'] = data.index.year

        fig, axes = plt.subplots(1, len(variables), 
                                 figsize=(len(variables)*page*0.3, 5*cm))

        for i, variable in enumerate(variables):
            sns.ecdfplot(data=data, x=variable, hue='year', palette='viridis', 
                         ax=axes[i], legend=False)
            axes[i].set_xlabel(self.translate_unit.get(variable, variable))

        axes[0].set_ylabel('Cumulative Frequency')

        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=data['year'].min(),
                                                      vmax=data['year'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Jahr')

        plt.tight_layout()
        return fig, axes
    
    @save_plot
    def plot_parde_coefficients(self, variable: str='Q'):
        """
        Plot Pardé coefficients for the specified variable.

        Args:
            variable (str): The variable to plot.

        Returns:
            tuple: (fig, ax) Matplotlib figure and axis.
        """
        fig, ax = plt.subplots()
        ax.set_title(self.metadata['Messstelle'] + ' ' + self.translate_unit.get(variable, variable))

        # Annual coefficients
        data = self.df_resampled
        data['year'] = data.index.year
        data['month'] = data.index.month
        years = data['year'].unique()

        annual = pd.Series(dtype='float')
        for year in years:
            coeffs = utils.calculate_parde_coefficients(data[variable].resample('M').mean().loc[str(year)])
            annual = pd.concat([annual, coeffs])

        # annual = []
        # for year in years:
        #     coeffs = utils.calculate_parde_coefficients(data[variable].resample('M').mean().loc[str(year)])
        #     annual.append(coeffs)

        # annual = pd.concat(annual)
        # annual.index = pd.to_datetime(annual.index, format='%Y-%m')

        df = pd.DataFrame(annual, columns=['parde'])
        df['year'] = df.index.year
        df['month'] = df.index.month
        sns.lineplot(data=df, x='month', y='parde', hue='year', legend=False, 
                     palette='viridis', lw=0.5, ax=ax)
        
        # All-time mean
        mean = utils.calculate_parde_coefficients(data.groupby('month')[variable].mean())
        ax.plot(mean, lw=2, color='k')

        # Style plot
        ax.axhline(1, ls='--', color='k')
        ax.set(xlabel='', ylabel='Pardé Koeff.',
               xticks=np.linspace(1, 11, 6),
               xticklabels=['Jan', 'Mär', 'Mai', 'Jul', 'Sep', 'Nov'])

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=df['year'].min(), vmax=df['year'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Year')
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

        return fig, ax
    
    @save_plot
    def plot_heatmap(self, variable: str='TEMP'):
        """
        Plot heatmap for the specified variable.

        Args:
            variable (str): The variable to plot.

        Returns:
            tuple: (fig, ax) Matplotlib figure and axis.
        """

        data = self.df_resampled
        data['year'] = data.index.year
        data['day_of_year'] = data.index.day_of_year

        # create square matrix for plotting
        heatmap_data = data.pivot_table(index='year', columns='day_of_year',
                                        values=variable)
        
        # create heatmap plot
        fig, ax = plt.subplots(figsize=(page, 8*cm))
        sns.heatmap(heatmap_data, cmap='viridis', cbar=False, ax=ax)

        # add colorbar
        sm = plt.cm.ScalarMappable(cmap='viridis', 
                                   norm=plt.Normalize(vmin=heatmap_data.min().min(),
                                                      vmax=heatmap_data.max().max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label(self.translate_unit.get(variable, variable))

        # format ticks and ticklabels
        months = ['FEB', 'APR', 'JUN', 'AUG', 'OCT', 'DEC']
        month_days = [32, 91, 152, 213, 274, 335]
        ax.set_xticks(ticks=month_days, labels=months)
        ax.invert_yaxis()
        ax.set(xlabel='', ylabel='Jahr')
        ax.minorticks_off()
        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(True)

        plt.tight_layout()
        return fig, ax
    
    @save_plot
    def plot_autocorr_timeseries(self, variable: str='Q'):
        """
        Plots the timeseries of autocorrelation stats for a given variable.

        Parameters:
        - variable (str): The name of the variable to plot the autocorrelation timeseries for. Default is 'Q'.

        Returns:
        - fig (matplotlib.figure.Figure): The generated figure object.
        - ax (matplotlib.axes.Axes): The generated axes object.
        """
        series = self.df_resampled[variable]
        lags = range(1, 366)
        years = series.index.year.unique()
        
        # container variable
        dic = {}

        # get yearly stats in loop
        for year in years:
            values = series.loc[str(year)]
            
            # check if there's data in this year
            if np.nansum(values) == 0:
                pass
            else:
                autocorrelations = np.array([values.autocorr(lag=lag) for lag in lags])

                # check if autocorrelation function drops below thresholds
                if autocorrelations[~np.isnan(autocorrelations)].min() > 0.05:
                    pass

                # if yes, get threshold crossings and write data to DF
                else:
                    kd = np.where(autocorrelations < 0.05)[0][0] # Korrelationsdauer KD (first index i.e. [0] where condition is true)
                    rt = np.where(autocorrelations < 0.2)[0][0] # Reaktionszeit RT
                    iak = np.nansum(np.absolute(autocorrelations))/365 # integrierte Autokorrelation
                    # print(str(year) + f': KD = {kd}, RT = {rt}, IAK = {iak}')
                    
                    # store data
                    row = [kd, rt, iak]
                    dic[year] = row
        
        # Dataframe for better handling
        df = pd.DataFrame(dic).T
        df.columns = ['kd', 'rt', 'iak']
        df.index = pd.to_datetime(df.index, format='%Y')

        fig, ax = plt.subplots(figsize=(7*cm,5*cm))
        ax.plot(df.kd, label='Korrelationsdauer')
        ax.plot(df.rt, label='Reaktionszeit')
        ax.set(ylabel='Tage')
        plt.xticks(rotation=45)
        ax.legend()

        plt.tight_layout()
        return fig, ax
    
    @save_plot
    def plot_cross_correlation(self, variables: list):
        """
        Plot cross correlation for specified variables.

        Args:
            variables (list): List of variables to plot cross correlation (should be pairs).

        Returns:
            tuple: (fig, ax) Matplotlib figure and axis.
        """

        fig, ax = plt.subplots(figsize=(page*0.4, 5*cm))

        for var1, var2 in variables:
            lags, correlation = utils.compute_cross_correlation(self.df_resampled[var1], self.df_resampled[var2])
            ax.fill_between(lags, correlation, np.zeros(correlation.shape), 
                            alpha=0.5, label=f'{self.translate_unit.get(var1, var1)} vs. {self.translate_unit.get(var2, var2)}')

        # Indicator lines
        ax.axhline(0, color='black', linestyle='--')
        for lag in [-365, -365/2, -30, -7, 0, 7, 30, 365/2, 365]:
            ax.axvline(lag, color='0.5', linestyle='--', linewidth=0.5)


        # indicator lines
        ax.axhline(0, color='black', linestyle='--')
        for lag in [-365, -365/2, -30, -7,0,7, 30, 365/2, 365]:
            ax.axvline(lag, color='0.5', linestyle='--', linewidth=0.5)

        # Style plot
        ax.legend()
        ax.set(xlabel='Versatz [Tage]', ylabel='Korrelation', xlim=(-400, 400), ylim=(-1, 1))
        plt.grid(False)

        plt.tight_layout()
        return fig, ax

    @save_plot
    def plot_scatter(self, variable_pairs: list, regression: bool = True,
                     print_stats: bool = True):
        """
        Plot scatter plots for specified variable pairs.

        Args:
            variable_pairs (list): List of variable pairs to plot scatter (e.g., [('Q', 'LF'), ('Q', 'TEMP')]).
            regression (bool): Whether to include regression lines. Defaults to True.

        Returns:
            tuple: (fig, axes) Matplotlib figure and axes.
        """
        df = self.df_resampled.dropna(subset=[var for pair in variable_pairs for var in pair])

        fig, axes = plt.subplots(1, len(variable_pairs), 
                                 figsize=(page*0.4*len(variable_pairs), 6*cm))

        for i, (x_var, y_var) in enumerate(variable_pairs):
            ax = axes[i]
            ax.plot(df[x_var], df[y_var], '. ', alpha=0.5, mew=0)
            if regression:
                sns.regplot(data=df, x=x_var, y=y_var, ax=ax, scatter=False, 
                            line_kws={'color': 'k', 'ls': '--'})
                r, p = pearsonr(df[x_var], df[y_var])
                if print_stats:
                    ax.text(.5, .8, f'''R²={r**2:.2f}\np={p:.2g}''', 
                            transform=ax.transAxes)
            ax.set(xlabel=self.translate_unit.get(x_var, x_var), 
                   ylabel=self.translate_unit.get(y_var, y_var))

        plt.tight_layout()
        return fig, axes