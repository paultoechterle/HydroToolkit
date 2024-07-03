import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import theilslopes, pearsonr
from scipy.optimize import curve_fit
from . import utils
from .plotting import Plotter, colors, page, cm

class DataLoader:
    """
    A class for loading data from files or a DataFrame.

    Attributes:
        paths (dict): Dictionary containing variable names as keys and file paths as values.
        data_frame (pd.DataFrame): DataFrame containing the data directly.
    """

    def __init__(self, paths: dict = None, data_frame: pd.DataFrame = None):
        """
        Initialize the DataLoader with paths to data files or a DataFrame.

        Args:
            paths (dict): Dictionary containing variable names as keys and file paths as values.
            data_frame (pd.DataFrame): DataFrame containing the data directly.
        """
        self.paths = paths
        self.data_frame = data_frame

    def load_metadata(self, path: str):
        """
        Load metadata from a given path.

        Args:
            path (str): Path to the metadata file.
        """
        return utils.read_metadata(path)

    def load_data(self):
        """
        Load data from the paths specified during initialization or from a DataFrame.

        Returns:
            dict: Dictionary with variable names as keys and dataframes as values.
        """
        if self.data_frame is not None:
            # Ensure 'time' column is present and set it as the index
            if 'time' in self.data_frame.columns:
                self.data_frame.set_index('time', inplace=True)
            # Create a dictionary of DataFrames for each column
            data = {col: self.data_frame[[col]] for col in self.data_frame.columns}
            return data
        elif self.paths is not None:
            data = {}
            for variable, path in self.paths.items():
                df = utils.read_data(path, variable=variable)
                df.set_index('time', inplace=True)
                data[variable] = df
            return data
        else:
            raise ValueError("Either 'paths' or 'data_frame' must be provided")

class Analysis:
    def __init__(self, data: dict, resolution: str = 'D'):
        """
        Initialize the Analysis with data and desired resolution.

        Args:
            data (dict): Dictionary with variable names as keys and dataframes as values.
            resolution (str): Desired temporal resolution (e.g., 'D' for daily, 'H' for hourly).
        """
        self.data = data
        self.resolution = resolution
        self.df = self.combine_data()
        self.df_resampled = self.resample_data()

    def combine_data(self):
        """
        Combine data from different variables into a single dataframe.

        Returns:
            pd.DataFrame: Combined dataframe.
        """
        combined_df = pd.concat(self.data.values(), axis=1)
        return combined_df

    def resample_data(self):
        """
        Resample data to the desired temporal resolution.

        Returns:
            pd.DataFrame: Resampled dataframe.
        """
        return self.df.resample(self.resolution).mean()

    def apply_filter(self, column):
        """
        Apply a low-pass filter to a specific column.

        Args:
            column (str): The column to filter.
        """
        self.df_resampled[f'{column}_filtered'] = utils.apply_filter(self.df_resampled[column])

    def create_stats(self, variable: str, freq: str = 'D'):
        """
        Create statistics for a given variable at the specified frequency.
        The statistics include:
        - minimum value
        - maximum value
        - mean value
        - median value
        - standard deviation
        - lower bound of 1-sigma range (mean - std)
        - upper bound of 1-sigma range (mean + std)

        Args:
            variable (str): The variable for which to create statistics.
            freq (str): Frequency for the statistics ('D' for daily, 'M' for monthly).

        Returns:
            pd.DataFrame: Dataframe containing the statistics.
        """
        if freq == 'D':
            grouped = self.df_resampled.groupby(self.df_resampled.index.day_of_year)[variable]
        elif freq == 'M':
            grouped = self.df_resampled.groupby(self.df_resampled.index.month)[variable]
        else:
            raise ValueError("Invalid frequency. Use 'D' for daily or 'M' for monthly.")

        stats = grouped.agg(['min', 'max', 'mean', 'median', 'std'])
        stats['1-sigma_min'] = stats['mean'] - stats['std']
        stats['1-sigma_max'] = stats['mean'] + stats['std']

        return stats
    
    def calculate_autocorrelations(self):
            """
            Calculates the autocorrelation function for the resampled data.

            Returns:
                The autocorrelation function of the resampled data.
            """
            return utils.calculate_autocorrelation_function(self.df_resampled)

    def calculate_mann_kendall(self, variable):
        """
        Calculate the Mann-Kendall test for trend detection.

        Args:
            variable (str): The variable for which to calculate the test.

        Returns:
            float: The Mann-Kendall test statistic.
            float: The p-value associated with the test statistic.
        """
        return utils.mann_kendall_test(self.df_resampled[variable], period=365)
    
class Station:
    def __init__(self, data_paths: dict = None, metadata_path: str = None, 
                 data_frame: pd.DataFrame = None, metadata: dict = None,
                 resolution: str = 'D'):
        """
        Initialize the Station class.

        Args:
            data_paths (dict): Paths to the data files.
            metadata_path (str): Path to the metadata file.
            data_frame (pd.DataFrame): DataFrame containing the data directly.
            metadata (dict): Dictionary containing metadata information.
            resolution (str): Resolution for data resampling. Defaults to 'D' (daily).
        """
        if data_frame is not None:
            self.loader = DataLoader(data_frame=data_frame)
            self.metadata = metadata
        else:
            self.loader = DataLoader(paths=data_paths)
            self.metadata = self.loader.load_metadata(metadata_path)

        self.data = self.loader.load_data()
        self.variables = list(self.data.keys())

        self.processor = Analysis(self.data, resolution)
        self.plotter = Plotter(self.processor.df, self.processor.df_resampled,
                               self.metadata, self.get_translate_unit())
        
        self.df = self.processor.df
        self.df_resampled = self.processor.df_resampled

        self.stats = self.df.describe()
        self.stats.loc['max_min_ratio'] = self.stats.loc['max'] / self.stats.loc['min']
        self.stats = self.stats.round(1)

        self.autocorrelation_stats = utils.calculate_autocorrelation_stats(utils.calculate_autocorrelation_function(self.df))

        keys = ['Messstelle', 'HZB-Nummer', 'coordinates']
        self.stammdaten = {key: self.metadata[key] for key in keys}

    def get_translate_unit(self):
        """
        Return a dictionary with units for the variables.
        This should be implemented in subclasses to provide specific units.
        """
        raise NotImplementedError("Subclasses should implement this method to provide specific units.")

    def apply_filter(self, variable: str) -> None:
        """
        Applies a lowpass filter to the given variable.

        Args:
            variable (str): The variable to apply the filter to.

        Returns:
            None
        """
        self.processor.apply_filter(variable)

    def create_stats(self, variable: str, freq: str) -> object:
        """
        Create timeseries statistics dataframe for a given variable.

        Args:
            variable (str): The name of the variable for which statistics are to be created.
            freq (str): The frequency at which the statistics should be calculated.

        Returns:
            object: The statistics object containing the calculated statistics.

        """
        return self.processor.create_stats(variable, freq)
    
    def mann_kendall_test(self, variable: str = 'Q') -> dict:
        """
        Perform the Mann-Kendall test for trend detection on a given variable.

        Args:
            variable (str): Name of the variable to perform the test on. Defaults to 'Q'.

        Returns:
            dict: Dictionary containing the test statistic, p-value, and trend 
            (positive, negative, or no trend).
        """
        return self.processor.calculate_mann_kendall(variable)
    
    def pull_spartacus(self, t0: pd.Timestamp, tn: pd.Timestamp, parameter: str = 'RR'):
        """
        Fetch data from the Spartacus database. 
        (https://data.hub.geosphere.at/dataset/spartacus-v2-1d-1km)

        Args:
            t0 (pd.Timestamp): Start timestamp.
            tn (pd.Timestamp): End timestamp.
            parameter (str): Parameter to fetch (e.g., 'RR', 'TN', 'TX', 'SA').

        Returns:
            pd.DataFrame: Data fetched from Spartacus.
        """
        assert t0 < tn, "t0 must be before tn"
        valid_parameters = ['RR', 'TN', 'TX', 'SA']
        assert parameter in valid_parameters, f"parameter must be one of {valid_parameters}"

        lat_lon = f"{self.metadata['coordinates'][0]},{self.metadata['coordinates'][1]}"
        start = t0.strftime('%Y-%m-%dT%H:%M:%SZ')
        end = tn.strftime('%Y-%m-%dT%H:%M:%SZ')

        params = {
            "parameters": parameter,
            "start": start,
            "end": end,
            "lat_lon": lat_lon,
            "format": "json"
        }

        data = utils.read_spartacus(params)
        return data

    def calc_catchment_area(self):
        """calculate the minimum catchment area from mean dicharge and 
        precipitation estimates. Mean precipitation values are calulated from
        spartacus dataset and averaged over the observation period.
        Evapotranspiration is neglected

        Returns:
            float: catchment area in km²
        """
        t0 = self.df.index.min()
        tn = self.df.index.max()
        mean_discharge = self.df.Q.mean()
        mean_precip = self.pull_spartacus(t0, tn)['RR'].resample('Y').sum().mean()
        seconds_in_year = 60 * 60 * 24 * 365
        mean_discharge_lpy = mean_discharge * seconds_in_year
        catchment_area = mean_discharge_lpy / mean_precip  # in m²
        catchment_area = catchment_area / 1000 ** 2  # in km²

        print(f'Catchment area @ {mean_discharge:.0f} l/s and {mean_precip:.0f} mm/m² = {catchment_area:.2f} km² ')
        return catchment_area
    
    def recession_curve(self, t0: pd.Timestamp, tn: pd.Timestamp, 
                         model: str='Maillet', plot: bool = False):
        """
        Fit a recession model to the discharge data and optionally plot the fit.

        Args:
            t0 (pd.Timestamp): Start timestamp.
            tn (pd.Timestamp): End timestamp.
            model (str): Recession Model used. 'Maillet' or 'Boussinesq'. Default is 'Maillet
            plot (bool): Whether to plot the results. Defaults to False.

        Returns:
            tuple: If plot is True, returns (fig, ax, t, fit, alpha). Otherwise, returns (t, fit, slice, alpha).
        """
        if model not in ['Maillet', 'Boussinesq']:
            raise ValueError('Invalid model input. Use Maillet or Boussinesq')
        
        if model == 'Maillet':
            mod = utils.maillet_model
            initial_guess = [self.df.loc[t0:tn, 'Q'].max(), 0.1]
        elif model == 'Boussinesq':
            mod = utils.boussinesq_model
            initial_guess = [self.df.loc[t0:tn, 'Q'].max(), 0.1]

        y_data = self.df.Q[t0:tn].interpolate().dropna()#.resample('D').mean()
        time_data = (y_data.index - y_data.index[0]).total_seconds() / (24 * 3600)  # Convert to days

        params, covariance = curve_fit(mod, time_data, y_data, p0=initial_guess)
        Q0, alpha = params
        y_pred = mod(time_data, Q0, alpha)
        # goodness of fit
        residuals = y_data-y_pred
        # Sum of squares
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y_data - y_data.mean())**2)
        # R and R^2
        r = np.sqrt(1 - (ss_res / ss_tot))
        r_squared = 1 - (ss_res / ss_tot)
        print(f'{model} Model: Q0 = {Q0:.0f} l/s, alpha = {alpha:.2}, R² = {r_squared:.4f}')

        if plot:
            fig, ax = plt.subplots()
            ax.plot(time_data, y_data, '. ', label='data')
            ax.plot(time_data, y_pred, ls='--', lw=2,
                    label=fr'{model} Modell: $Q_0$={Q0:.0f} l/s, $\alpha$={alpha:.2}, R²={r_squared:.4f}')
            ax.legend()
            ax.set(ylabel='Schüttung [l/s]', xlabel='Trockentage')
            return fig, ax
        else:
            return time_data, y_pred, y_data, alpha

        # slice = self.df_resampled.Q[t0:tn].dropna()
        # dt = (slice.index - t0).days
        # t = np.array(dt)

        # params, covariance = curve_fit(model, t, slice)
        # Q0, alpha = params

        # fit = model(t, Q0, alpha)

        # if plot:
        #     fig, ax = plt.subplots()
        #     ax.plot(t, slice, label='Schüttungsverlauf')
        #     ax.plot(t, fit, ls='--', c='k', label=fr'Modell: $Q_0$={Q0:.0f} l/s, $\alpha$={alpha:.2}')
        #     ax.set(ylabel='Schüttung [l/s]', xlabel='Trockentage', title=self.metadata['Messstelle'])
        #     ax.legend()
        #     fig.autofmt_xdate()
        #     return fig, ax, t, fit, alpha
        # else:
        #     return t, fit, slice, alpha

    def plot_recession_curves(self, timeslices: list, model:str='Maillet', 
                               save: bool = False, path: str = 'TWFL.png'):
        """
        Plot multiple periods of discharge recession. see Station.recession_curve()

        Args:
            timeslices (list): List of tuples with start and end timestamps.
            model (str): Discharge Recession Model used for the fiting. 
            'maillet' or 'boussinesq' are valid input. for details refer to 
            utils.maillet_model or utils.boussinesq_model
            save (bool): Whether to save the plot. Defaults to False.
            path (str): Path to save the plot. Defaults to None.

        Returns:
            tuple: (fig, ax) Matplotlib figure and axis.
        """
        fig, axes = plt.subplots(2, 1, figsize=(page, 10*cm), 
                                 gridspec_kw={'height_ratios': [1, 3]})

        ax = axes[0]
        ax.plot(self.df_resampled['Q'])
        ax.set(ylabel='Q [l/s]')

        for counter, (t0, tn) in enumerate(timeslices):
            ax.axvspan(t0, tn, color=colors[counter+1 % len(colors)], alpha=0.7)

        ax = axes[1]
        alphas = []
        for counter, (t0, tn) in enumerate(timeslices):
            time_data, y_pred, y_data, alpha=self.recession_curve(t0, tn, model=model)
            ax.plot(time_data, y_data, c=colors[counter+1 % len(colors)], lw=2, label='Schüttungsverlauf')
            ax.plot(time_data, y_pred, c='k', lw=1, ls='--', label='Model')
            alphas.append(alpha)

        ax.set(ylabel='Schüttung [l/s]', xlabel='Trockentage')
        mean_alpha = np.mean(alphas)
        ax.text(0.75, 0.8, fr'$\overline{{\alpha}} = {mean_alpha:.2}$ /Tag', transform=ax.transAxes, fontsize=7)

        plt.tight_layout()
        if save:
            savepath = path if path else f'plots/{self.metadata["Messstelle"]}_TWFLplot.png'
            plt.savefig(savepath, dpi=300)

        return fig, ax

    def plot_timeseries(self, variables: list=None, filter:bool=False, daily:bool=True,
                        save:bool=False, path:str='ts.png'):
        if variables is None:
            variables = self.variables
        fig, axes = self.plotter.plot_timeseries(variables, filter=filter, 
                                                  save=save, path=path, daily=daily)
        return fig, axes

    def plot_histogram(self, variables: list=None, daily: bool=True, save:bool=False, path:str='hist.png'):
        if variables is None:
            variables = self.variables
        fig, axes = self.plotter.plot_histogram(variables, daily=daily, 
                                                save=save, path=path)
        return fig, axes

    def plot_ts_hist(self, variables: list=None, trend:bool=False, daily: bool=True, 
                     save:bool=False, path:str='ts_hist.png'):
        if variables is None:
            variables = self.variables
        fig, axes = self.plotter.plot_ts_hist(variables, trend=trend, save=save, 
                                              daily=daily, path=path)
        return fig, axes

    def plot_confidence_intervals(self, variable: str='Q', save:bool=False, 
                                  path:str='ci.png'):
        stats = self.create_stats(variable, freq='D')
        fig, axes = self.plotter.plot_confidence_intervals(variable, stats=stats,
                                                           save=save, path=path)
        return fig, axes
    
    def plot_ci_panel(self, variables: list=None, save:bool=False, path:str='ci_panel.png'):
        if variables is None:
            variables = self.variables
        stats_list = [self.create_stats(variable, freq='D') for variable in variables]
        fig, axes = self.plotter.plot_ci_panel(stats_list, variables, save=save, path=path)
        return fig, axes
    
    def plot_autocorrelation(self, save:bool=False, path:str='acf.png'):
        autocorrelations = self.processor.calculate_autocorrelations()
        fig, ax = self.plotter.plot_autocorrelations(autocorrelations[self.variables],
                                                     save=save, path=path)
        return fig, ax
    
    def plot_cumulative_distribution(self, variables: list=None, save:bool=False, path:str='cumdist.png'):
        if variables is None:
            variables = self.variables
        fig, axes = self.plotter.plot_cumulative_distribution(variables,
                                                              save=save,
                                                              path=path)
        return fig, axes
    
    def plot_parde_coefficients(self, variable='Q', save:bool=False, path:str='parde.png'):
        fig, ax = self.plotter.plot_parde_coefficients(variable, save=save,
                                                       path=path)
        return fig, ax
    
    def plot_heatmap(self, variable='TEMP', save:bool=False, path:str='heatmap.png'):
        fig, ax = self.plotter.plot_heatmap(variable, save=save, path=path)
        return fig, ax

    def plot_autocorr_timeseries(self, variable='Q', save:bool=False, path:str='acf_ts.png'):
        fig, ax = self.plotter.plot_autocorr_timeseries(variable, save=save, path=path)
        return fig, ax

    def plot_cross_correlation(self, variable_pairs=[('Q', 'LF'), ('Q', 'TEMP')], 
                               save:bool=False, path:str='crosscorr.png'):
        fig, ax = self.plotter.plot_cross_correlation(variable_pairs, save=save, path=path)
        return fig, ax
    
    def plot_scatter(self, variable_pairs=[('Q', 'LF'), ('Q', 'TEMP')], 
                     regression:bool=True, save:bool=False, path:str='scatter.png'):
        fig, axes = self.plotter.plot_scatter(variable_pairs, regression=regression, 
                                              save=save, path=path)
        return fig, axes

class Spring(Station):
    def __init__(self, data_paths: dict = None, metadata_path: str = None, 
                 data_frame: pd.DataFrame = None, metadata: dict = None, resolution: str = 'D'):
        super().__init__(data_paths, metadata_path, data_frame, metadata, resolution)
        
        try:
            self.isotopes = utils.iso_df_mean.loc[int(self.stammdaten['HZB-Nummer'])]
        except ValueError:
            print('No isotope data found for this station')
            self.isotopes = None
        except KeyError:
            print('No isotope data found for this station')
            self.isotopes = None

    def mean_catchment_elevation(self, spring_elevation:float, plot:bool=False):
        """
        Calculate the mean catchment elevation based on precipitation isotope data.

        Parameters:
        - spring_elevation (float): The elevation of the spring in meters.
        - plot (bool, optional): Whether to plot the regression line and data points. Default is False.

        Returns:
        - catchment_elevation (float): The calculated mean catchment elevation in meters.
        - fig, ax (matplotlib.figure.Figure, matplotlib.axes.Axes): The figure and axes objects if plot=True.

        Note:
        - This method requires the 'isotopes' attribute to be set with sping isotope data.

        """
        if self.isotopes is None:
            print('No isotope data found for this station. Cannot calculate catchment elevation.')
            return None
        
        # get the lapse rate from precipitation stations:
        # load precipitation isotope data
        pfad = r"M:\WASSERRESSOURCEN - GQH Stufe 1 2024 - 2400613\C GRUNDLAGEN\01-Daten\06-Isotopendaten\Qualitaetsdatenabfrage_20240703_1337.csv"
        cols = {
            'GZÜV-ID': 'gzuv_id',
            'Name': 'name',
            'Gemeindename': 'gemeindename',
            'Höhe Messpunkt': 'hohe',
            'Monat': 'monat',
            'Lfd. Nummer': 'lfd_nummer',
            'Probenahme-Beginn TT-MM-JJJJ': 'probenahme_beginn',
            'Probenahme-(Ende) TT-MM-JJJJ': 'probenahme_ende',
            'Flaschengewicht g': 'flaschengewicht',
            'Niederschlag - Monatssumme (mm)': 'niederschlag_monatssumme',
            'Laborcode - Niederschlag I104': 'laborcode_niederschlag',
            'Niederschlag - Art': 'niederschlag_art',
            'Lufttemperatur - Monatsmittel (°C)': 'lufttemperatur_monatsmittel',
            'Laborcode - Lufttemperatur I107': 'laborcode_lufttemperatur',
            'Potentielle Verdunstung - Monatssumme (mm)': 'potentielle_verdunstung_monatssumme',
            'Laborcode - Pot. Verdunstung I110': 'laborcode_verdunstung',
            'Pegelstand cm': 'pegelstand_cm',
            'Durchfluss m³/s': 'durchfluss_m3_s',
            'Delta 18O (? VSMOW)': 'd18o',
            'Messgenauigkeit Delta 18O (?)': 'messgenauigkeit_delta_18o',
            'Laborcode - 18O I114': 'laborcode_18o',
            'Delta  2H (? VSMOW)': 'd2h',
            'Messgenauigkeit Delta 2H (?)': 'messgenauigkeit_delta_2h',
            'Laborcode - Deuterium I117': 'laborcode_deuterium',
            'Tritium - 3H (TE)': 'tritium',
            'Messgenauigkeit Tritium 3H': 'messgenauigkeit_tritium',
            'Laborcode - Tritium I120': 'laborcode_tritium',
            'freies Luftvolumen (ml) bis 2003': 'freies_luftvolumen',
            'Unnamed: 28': 'unnamed_28'
        }
        df = pd.read_csv(pfad, sep=';', encoding='latin1', skiprows=36)
        df = df.rename(columns=cols)
        
        # create timestamps
        df['datetime'] = pd.to_datetime(df['probenahme_ende'], dayfirst=True, errors='coerce')
        df['month'] = df['datetime'].dt.month

        # create monthly averages for each station
        dfs = []
        for station in df.gzuv_id.unique():
            station_df = df[df.gzuv_id == station]
            station_means = station_df.groupby('month').d18o.agg(['mean', 'std'])
            # add elevation and station name
            station_means['elevation'] = np.full_like(station_means['mean'], station_df.hohe.unique()[0])
            station_means['name'] = [station_df.name.unique()[0] for i in station_means.index]
            dfs.append(station_means)
        # concatenate all stations
        df_monthly = pd.concat(dfs)
        df_monthly.reset_index(inplace=True)

        # create annual averages for each station
        df_annual = df_monthly.groupby('name')['mean'].agg(['mean', 'std'])
        df_annual['elevation'] = df_monthly.groupby('name')['elevation'].min()

        # calculate lapse rate
        # use linear regression on annual d18o values vs elevation
        from sklearn.linear_model import LinearRegression
        x = df_annual['elevation'].values.reshape(-1, 1)
        y = df_annual['mean'].values.reshape(-1, 1)
        reg = LinearRegression().fit(x, y)
        slope = reg.coef_[0][0]
        intercept = reg.intercept_[0]
        r_squared = reg.score(x, y)
        print(f'slope per 100m: {slope*100:.2f}, intercept: {intercept:.2f}, R²: {r_squared:.2f}')

        # project measured spring d18O onto precipitation lapse rate
        catchment_elevation = (self.isotopes.d18o['mean'] - intercept) / slope
        print(f'Mean Catchment elevation: {catchment_elevation:.0f} m')

        if plot:
            fig, ax = plt.subplots()
            ax.plot(df_annual['elevation'], df_annual['mean'], 'o', label='Niederschlagsstationen')
            ax.plot(x, reg.predict(x), 'k-', label='Regressionslinie')
            ax.errorbar(x=spring_elevation, y=self.isotopes.d18o['mean'], 
                        yerr=self.isotopes.d18o['std'], marker='o', label='Quelle')
            ax.set(xlabel='Elevation [m]', ylabel=r'$\delta^{18}O$ [‰]')
            return fig, ax
        
        return catchment_elevation

    def get_translate_unit(self):
        return {
            'Q': 'Schüttung [l/s]',
            'LF': 'el. Leitf. [µS/cm]',
            'TEMP': 'Temperatur [°C]'
        }
    
    def calc_storage_volume(self, alpha:float, Q0: float=None):
        if Q0 == None:
            Q_hat = self.df.Q.mean()
        else:
            Q_hat = Q0
        return utils.calculate_storage_volume(Q_hat, alpha)
    
    def calc_halflife(self, alpha:float):
        return utils.calculate_halflife(alpha)
    
    def calc_tau(self, alpha:float, Q_annual: float=None, Q0: float=None):
        if Q_annual == None:
            Q_annual = self.df_resampled.Q.resample('Y').sum().median() * 24*62**2
        if Q0 == None:
            Q0 = self.df.Q.mean()

        V0 = self.calc_storage_volume(alpha, Q0)
        tau = utils.calculate_tau(V0, Q_annual)
        return tau
    
    def print_aquifer_metrics(self, alpha:float, Q_annual: float=None, 
                              Q0: float=None):
        """Calculate the hydrogeological halflife (t05), mean storage volume (V0)
        and the mean residence time (tau) based on the discharge parameters
        and the spring specific coefficient alpha.

        Args:
            alpha (float): spring specific coefficient - obtain through dry 
            weather fall lines

        Returns:
            tuple: (t05, V0, tau)
        """
        if Q_annual == None:
            Q_annual = self.df_resampled.Q.resample('Y').sum().median() * 24*62**2
        if Q0 == None:
            Q0 = self.df.Q.mean()

        t05 = self.calc_halflife(alpha)
        V0 = self.calc_storage_volume(alpha, Q0)
        tau = self.calc_tau(alpha, Q_annual, Q0)

        print(f'''Mean discharge = {Q0:.0f} l/s \nMean annual Discharge = {Q_annual/1000:.2} m³ \nMean storage Volume = {V0:.4} m³ \nt(1/2) = {t05:.0f} days \nMean residence time = {tau*365.25:.0f} days''')
        return t05, V0, tau
    
    def plot_isotope_crossplot(self, HZBnr: int=None, save: bool=False, path: str='ISOCP.png') -> tuple:
        """
        Plots the isotope crossplot for a given HZB number.

        Parameters:
        - HZBnr (int): The HZB number of the station. If None, the HZB number from the 'stammdaten' attribute will be used.

        Returns:
        - tuple: A tuple containing the figure and axis objects of the plot.

        Note:
        - The function requires the 'stammdaten' attribute to be set before calling this function.
        - The function reads data from the file '24-06-11 Isotopendaten Quellen Tirol_DatenBML.xlsx' located at 'M:\WASSERRESSOURCEN - GQH Stufe 1 2024 - 2400613\C GRUNDLAGEN\01-Daten\'.
        - The function plots the isotope data for all stations, the Global Meteoric Water Line (GMWL), and the specific station based on the HZB number.
        """
        
        if HZBnr is None:
            HZBnr = int(self.stammdaten['HZB-Nummer'])

        file = r"M:\WASSERRESSOURCEN - GQH Stufe 1 2024 - 2400613\C GRUNDLAGEN\01-Daten\24-06-11 Isotopendaten Quellen Tirol_DatenBML.xlsx"
        df = pd.read_excel(file)
        station_df = df[df['HZB-Nr.'] == HZBnr].dropna(subset=['Sauerstoff-18 [‰ V-SMOW]', 'Deuterium [‰ V-SMOW]'])
        gmwl = utils.GMWL((df['Sauerstoff-18 [‰ V-SMOW]'].min(),
                        df['Sauerstoff-18 [‰ V-SMOW]'].max()))
        
        if station_df.empty:
            print('No data for station found')
        else:
            fig, ax = plt.subplots(figsize=(page*0.5, 0.4*page))
            ax.plot(df['Sauerstoff-18 [‰ V-SMOW]'], df['Deuterium [‰ V-SMOW]'], 
                    '.', label='Alle Stationen', color='gray', alpha=0.5)
            ax.plot(gmwl['d18o'], gmwl['d2h'], 'k--', label='GMWL')
            ax.plot(station_df['Sauerstoff-18 [‰ V-SMOW]'], 
                    station_df['Deuterium [‰ V-SMOW]'], 'o', 
                    label=self.stammdaten['Messstelle'])
            ax.set_xlabel(r'$\delta^{18}O$ [‰]')
            ax.set_ylabel(r'$\delta^{2}H$ [‰]')
            
            ax.legend()
            plt.tight_layout()
            if save:
                savepath = path if path else f'plots/{self.stammdaten["Messstelle"]}_ISOCP.png'
                plt.savefig(savepath, dpi=300)
            return fig, ax
        return None
    
class River(Station):
    def __init__(self, data_paths: dict = None, metadata_path: str = None, 
                 data_frame: pd.DataFrame = None, metadata: dict = None, resolution: str = 'D'):
        super().__init__(data_paths, metadata_path, data_frame, metadata, resolution)

    def get_translate_unit(self):
        return {
            'Q': 'Abfluss [m³/s]',
            'TEMP': 'Temperatur [°C]'
        }

if __name__=='__main__':

    path = r'M:\WASSERRESSOURCEN - GQH Stufe 1 2024 - 2400613\C GRUNDLAGEN\01-Daten\05-HD Tirol - 15min\Anfrage Juni 2024'

    searchstring = 'Schrei'

    path_lf, path_q, path_t = utils.filter_by_string(utils.list_files(path), searchstring)
    X = Spring({'Q':path_q, 'LF': path_lf, 'TEMP':path_t}, path_q)
    X.plot_timeseries()