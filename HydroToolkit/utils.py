import os
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from typing import Union
import requests
import matplotlib.pyplot as plt
import pymannkendall as mk

cm = 1/2.54 # inch/cm conversion factor
page = (21.0 - 1.65 - 1.25)*cm # A4 page

def read_data(path: str, variable: str = 'x') -> pd.DataFrame:
    """
    Takes the path to a csv file that is provided via eHYD (https://ehyd.gv.at/) and returns
    a pandas DataFrame with the columns 'time' and variable (default 'x').

    Args:
        path (str): Path to csv file. Attention: must be formatted exactly like the default from eHYD.
        variable (str, optional): Name of the variable (used as column name). Defaults to 'x'.

    Returns:
        pd.DataFrame: DataFrame with columns 'time' and variable.
    """
    with open(path, 'r', encoding='Windows 1252') as file:
        lines = file.readlines()

        # Find the line with "Werte:"
        for i, line in enumerate(lines):
            if "Werte:" in line:
                start_line = i + 1
                break
        else:
            raise ValueError("Fehler beim Einlesen der Daten. Header nicht korrekt identifiziert.")

    df = pd.read_csv(path, skiprows=start_line, sep=';', names=['time', variable, 'blank'],
                     encoding='Windows 1252', skipinitialspace=True,
                     converters={variable: lambda x: x.strip().replace(',', '.')})

    df.replace('Lücke', np.nan, inplace=True)  # correct NaN values
    df[variable] = df[variable].astype('float')  # reformat str to float
    df['time'] = pd.to_datetime(df.time, dayfirst=True)  # reformat datetime

    return df[df.columns[:-1]]

def read_metadata(path: str) -> dict:
    """
    Extracts the station metadata header from a csv file that is provided via eHYD (https://ehyd.gv.at/)
    and returns a dictionary with the respective station metadata.

    Args:
        path (str): Path to csv file. Attention: must be formatted exactly like the default from eHYD.

    Returns:
        dict: Metadata dictionary.
    """
    with open(path, 'r', encoding='Windows 1252') as file:
        lines = file.readlines()

        # Find the line with "Werte:"
        for i, line in enumerate(lines):
            if "Werte:" in line:
                start_line = i + 1
                break
        else:
            raise ValueError("Fehler beim Einlesen der Daten. Header nicht korrekt identifiziert.")

    metadata = {}
    with open(path, 'r', encoding='Windows 1252') as file:
        for line in (next(file) for _ in range(start_line)):
            if ':' in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().lstrip(';')

    metadata['coordinates'] = extract_coordinates(path)

    return metadata

def read_TIWAG(paths):
    dfs = []
    for path in paths:
        df = pd.read_csv(path, encoding='ANSI', delimiter=';', decimal=',',
                        dayfirst=True, skiprows=11, na_values='---')
        df['date'] = df['Datum'].astype(str)
        df['time'] = df['Uhrzeit'].astype(str)
        df['date_time'] = pd.to_datetime(df['date'] + ' ' + df['time'], dayfirst=True)
        df.set_index('date_time', inplace=True)
        dfs.append(df.iloc[:,[2]])

    df = pd.concat(dfs)
    df.columns = ['Q', 'LF', 'TEMP']
    return df

def list_files(directory: str, filetype: str = '.csv') -> list:
    """
    Searches a directory (including subfolders) for *.csv files 
    and outputs a list of the relative paths.

    Args:
        directory (str): Directory to look for files.
        filetype (str, optional): File type to look for. Defaults to '.csv'.

    Returns:
        list: List of relative paths.
    """
    csv_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(filetype):
                csv_files.append(os.path.join(root, file))
    return csv_files

def filter_by_string(input_list:list, substring:str) -> list:
    """Select items from a list of files that contain a substring (e.g. HZB Nr).

    Args:
        input_list : list of file paths
        substring : string that must be contained in the path to be selected

    Returns:
        list: list of paths that contain the substring
    """

    file_list = [item for item in input_list if substring in os.path.basename(item)]

    if len(file_list) == 0:
        print(f'No files containing \"{substring}\" in filename found!')
    # else:
    #     print(f'{len(file_list)} file(s) found')
    return file_list

def create_kde(data: Union[pd.DataFrame, pd.Series], res: int = 100) -> tuple:
    """
    Creates kernel density function (KDE) for a Series or DataFrame and returns
    the function domain and corresponding function values (x, y).

    Args:
        data (Union[pd.DataFrame, pd.Series]): Input (timeseries) data.
        res (int, optional): Domain resolution at which the function is evaluated. Defaults to 100.

    Returns:
        tuple: Domain values, KDE function values.
    """
    data = data.dropna().squeeze()
    x = np.linspace(data.min(), data.max(), res)
    kde = gaussian_kde(data, bw_method=0.5)
    return x, kde(x)

def calculate_autocorrelation_function(df:pd.DataFrame) -> pd.DataFrame:
    """calculates the autocorrelation values for lag times between 1 and 366 
    steps for each column of a pandas dataframe.

    Args:
        df (pd.DataFrame): Dataframe with timeseries data (typically at daily 
        resolution). Attention:;needs to be equally spaced to give meaningful
        results.

    Returns:
        pd.DataFrame: Autocorrelations as Dataframe
    """

    # select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    # resample daily averages
    numeric_df = numeric_df.resample('D').mean()

    # calulate autocorrelation function for all variables
    lags = range(1, 366) # function time domain (1 Year)
    dic = {variable: np.array([numeric_df[variable].autocorr(lag=lag) for lag in lags])
           for variable in numeric_df}
    
    # Autocorrelation function to DataFrame
    autocorrelations = pd.DataFrame(dic)
    return autocorrelations

def calculate_autocorrelation_stats(df:pd.DataFrame) -> pd.DataFrame:
    """Extension to "calculate_autocorrelation_function. Takes autocorrelation
    DataFrame as an input and returns another DataFrame that contains the 
    following statistical measures of the autocorrleation function:

    KD: Korrelationsdauer (timesteps until function value drops below 0.05)
    RT: ReactionTime (timesteps until function drops below 0.2 - specific for
    karstic springs)
    IAK: Integrated Autocorrelation (absolute area below the autocorrelation 
    function)

    Args:
        df (pd.DataFrame): Pandas DataFrame containting autocorrelation functions

    Returns:
        pd.DataFrame: Dataframe containing kd, rt and iak values for each column
    """
    dic = {}
    for column in df.columns:
        series = df[column]
        kd = series.index[series < 0.05].min() # Korrelationsdauer KD
        rt = series.index[series < 0.2].min() # Reaktionszeit RT
        iak = series.abs().sum()/365 # integrierte Autokorrelation
        dic[column] = [iak, kd, rt]
        # print(column + f': KD = {kd}, RT = {rt}')

    autocorrelation_stats = pd.DataFrame(dic, index=['iak','kd','rt'])
    return autocorrelation_stats

def calculate_storage_volume(Q0:float, alpha:float) -> float:
    """calculate the dischargable water volume in storage for an aquifer based
    on a given spring discharge and specific storage coefficient (alpha).

    Args:
        Q0 (float): Discharge in [l/s]
        alpha (float): Spring specific storage coefficient

    Returns:
        float: dischargable water volume
    """
    seconds_per_day = 60*60*24
    liter_to_m3 = 1/1000

    V0 = Q0 / alpha * seconds_per_day * liter_to_m3
    return V0

def calculate_tau(V:float, Q:float) -> float:
    """estimate the mean residence time according to the water budget method:
    residence time = amount of water in reservoir / discharge at steady state.

    Args:
        V (float): Water Volume in reservoir (m³)
        Q (float): Steady state discharge (liters per year)

    Returns:
        float: residence time in years
    """    
    tau = V/(Q/1000)
    return tau

def calculate_halflife(alpha:float):
    """Calculate the hydrogeological half-life from the spring specific
    coefficient. This measure quantifies the time (in days) it takes for spring 
    discharge at a given time Q(0) to drop to 50 % of that level Q(1/2).
    t(1/2) = ln(2)/alpha = 0.693/alpha

    Args:
        alpha (float): Spring specific koefficient

    Returns:
        float: hydrogeological half-life
    """    
    return np.log(2)/alpha

def compute_cross_correlation(ts1:pd.Series, ts2:pd.Series, 
                              plot:bool=False) -> Union[tuple, None]:
    """
    Calculate the cross-correlation between two timeseries (pandas objects).
    For details see numpy.correlate()

    Args:
        ts1 (pd.Series): First timeseries.
        ts2 (pd.Series): Second timeseries.
        plot (bool, optional): If True, the output is a matplotlib figure and 
        axes tuple (fig, ax) with the cross correlation plot. Defaults to False.

    Returns:
        tuple: Arrays for lags and corresponding correlation values, or (fig, ax) if plot is True.
    """

    # Ensure both time series are clean and aligned before computation
    ts1_cleaned = ts1.resample('D').mean().interpolate(method='linear').dropna()
    ts2_cleaned = ts2.resample('D').mean().interpolate(method='linear').dropna()

    # Ensure both series have the same date range
    common_index = ts1_cleaned.index.intersection(ts2_cleaned.index)
    ts1_cleaned = ts1_cleaned.reindex(common_index)
    ts2_cleaned = ts2_cleaned.reindex(common_index)

    # Center the data
    ts1_centered = ts1_cleaned - ts1_cleaned.mean()
    ts2_centered = ts2_cleaned - ts2_cleaned.mean()

    # Compute cross-correlation
    correlation = np.correlate(ts1_centered, ts2_centered, mode='full')

    # Normalize the correlation
    correlation /= (np.std(ts1_cleaned) * np.std(ts2_cleaned) * len(ts1_cleaned))

    # Create an array of lag values
    lags = np.arange(-len(ts1_cleaned) + 1, len(ts1_cleaned))

    # Generate plot
    if plot == True:
        
        fig, ax = plt.subplots(figsize=(page*0.4, 5*cm))
        ax.fill_between(lags, correlation, np.zeros(correlation.shape), 
                        alpha=0.5)
        
        # indicator lines
        ax.axhline(0, color='black', linestyle='--')
        for lag in [-365, -365/2, -30, -7,0,7, 30, 365/2, 365]:
            ax.axvline(lag, color='0.5', linestyle='--', linewidth=0.5)

        # style plot
        ax.set(xlabel='Lags [Tage]', ylabel='Korrelation', xlim=(-400,400), 
               ylim=(-1,1))
        plt.grid(False)
        plt.tight_layout()

        return fig, ax
    return lags, correlation

def apply_filter(series: pd.Series, N: int = 3, Wn: float = 0.01) -> pd.Series:
    """
    Apply a Butterworth lowpass filter to a timeseries.

    Args:
        series (pd.Series): Input series to be filtered.
        N (int, optional): The order of the filter. Defaults to 3.
        Wn (float, optional): The critical frequency. Defaults to 0.01.

    Returns:
        pd.Series: Filtered series.
    """
    from scipy.signal import butter, filtfilt

    b, a = butter(N, Wn, 'lowpass')
    filtered_series = filtfilt(b, a, series.interpolate())
    return filtered_series

def calculate_parde_coefficients(data: Union[pd.DataFrame, pd.Series]) -> Union[pd.DataFrame, pd.Series]:
    """
    Calculate Parde coefficients for a pandas DataFrame or Series.

    Args:
        data (Union[pd.DataFrame, pd.Series]): Input data for which to calculate Parde coefficients.

    Returns:
        Union[pd.DataFrame, pd.Series]: Parde coefficients.
    """
    Qa = data.mean()    # annual mean
    Qi = data           # momentary value (monthly in most cases)
    parde_coeff = Qi / Qa
    return parde_coeff

def maillet_model(t: float, Q0: float, alpha: float) -> float:
    """
    Maillet (1905) model function for exponential decay of spring discharge.
    Details: doi:10.1016/S0022-1694(02)00418-3

    Args:
        t (float): Time variable.
        Q0 (float): Initial quantity.
        alpha (float): Decay constant.

    Returns:
        float: Result of the Maillet equation.
    """
    return Q0 * np.exp(-alpha * t)

def boussinesq_model(t: float, Q0: float, alpha: float) -> float:
    """
    Boussinesq (1903) model function for exponential decay of spring discharge.
    Details: doi:10.1016/S0022-1694(02)00418-3
    
    Args:
        t (float): Time variable.
        Q0 (float): Initial quantity.
        alpha (float): Decay constant.

    Returns:
        float: Result of the Maillet equation.
    """
    return Q0 / (1+alpha*t)**2

def read_spartacus(params:dict) -> pd.DataFrame:
    """request point timeseries data from spartacus server

    Args:
        params (dict):     
        Define the parameters (example parameters, adjust as needed)
    params = {
        "parameters": "RR",                             # RR (precip), SA (sunshine duration), TN (minimum temp), TX (maximum temp)
        "start": "2023-01-01T00:00:00Z",                # Start date
        "end": "2023-01-31T23:59:59Z",                  # End date
        "lat_lon": "47.1215,11.3985",
        "format": "json"                                # Ensure the format is set to json to get JSON data
    }

    Returns:
        pd.DataFrame
    """

    # Define the endpoint URL
    url = "https://dataset.api.hub.geosphere.at/v1/timeseries/historical/spartacus-v2-1d-1km"

    # Make the request to download the JSON data
    response = requests.get(url, params=params)

    if response.status_code == 200:
            data = response.json()
    else:
        print("Error:", response.status_code, response.text)

    dic = {}
    dic[params["parameters"]] = data['features'][0]['properties']['parameters'][params["parameters"]]['data']
    df = pd.DataFrame(dic, pd.to_datetime(data['timestamps']))
    df.index = df.index.tz_localize(None)
    return df

def dms_to_dd(degrees: float, minutes: float, seconds: float, 
              direction: str) -> float:
    """
    Convert degrees, minutes, seconds to decimal degrees.

    Args:
        degrees (float): Degrees.
        minutes (float): Minutes.
        seconds (float): Seconds.
        direction (str): Direction ('N', 'S', 'E', 'W').

    Returns:
        float: Decimal degrees.
    """
    dd = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        dd *= -1
    return dd

def extract_coordinates(file_path: str) -> list:
    """
    Extract coordinates from a file.

    Args:
        file_path (str): Path to the file containing coordinates.

    Returns:
        list: List of coordinates [latitude, longitude].
    """
    with open(file_path, 'r', encoding='Windows 1252') as file:
        content = file.readlines()

    for index, line in enumerate(content):
        if "Geographische Koordinaten" in line:
            coord_line = content[index + 2].strip()
            lon_str, lat_str = coord_line.split(';')[1:3]
            lon_deg, lon_min, lon_sec = map(float, lon_str.split())
            lat_deg, lat_min, lat_sec = map(float, lat_str.split())
            longitude_dd = dms_to_dd(lon_deg, lon_min, lon_sec, 'E')
            latitude_dd = dms_to_dd(lat_deg, lat_min, lat_sec, 'N')
            return [latitude_dd, longitude_dd]
    return None

def mann_kendall_test(data: pd.Series, period: int) -> dict:
    """
    Perform seasonal Mann-Kendall test (Hirsch, R. M., Slack, J. R. 1984)
    on a time series. See "pymannkendall" documentation for details.

    Args:
        data (pd.Series): Input time series.
        period (int): Seasonal period.

    Returns:
        dict: Dictionary containing the trend, p-value, slope, and intercept.
    """
    result = mk.seasonal_test(data, period=period)
    trend = result.trend
    p_value = result.p
    slope = result.slope
    intercept = result.intercept
    return {'trend': trend, 'p-value': p_value, 'slope': slope, 'intercept': intercept}

def get_unique_colors(n: int) -> list[str]:
    """
    Generate a list of unique colors.

    Args:
        n (int): The number of unique colors to generate.

    Returns:
        List[str]: A list of unique colors.

    Raises:
        ValueError: If the requested number of colors exceeds the available CSS4 colors.
    """
    import random
    from matplotlib import colors

    xkcd_colors = list(colors.XKCD_COLORS.values())
    if n > len(xkcd_colors):
        raise ValueError("Requested number of colors exceeds the available CSS4 colors.")
    return random.sample(xkcd_colors, n)

def GMWL(limit = (-25,5)) -> dict:
    """
    Generate the Global Meteoric Water Line (GMWL) based on the given limit.

    Parameters:
    limit (tuple): A tuple representing the range of d18o values. Default is (-25, 5).

    Returns:
    dict: A dictionary containing the d18o and d2h values of the GMWL.

    """
    d18o = np.linspace(limit[0], limit[1], 5)
    d2h = 8 * d18o + 10
    gmwl = {'d18o': d18o, 'd2h': d2h}
    return gmwl

# GZÜV Readout from excel sheets and data Handling
class GZUV:
    def __init__(self, path:str, top:int=38, bottom:int=25):
        """
        Handles GZÜV data from csv files provided by "Qualitätsdatenabfrage".

        Parameters:
        - path (str): The path to the data file.
        - top (int): The number of rows to skip from the top of the file (default: 38).
        - bottom (int): The number of rows to read from the file (default: 25).
        """
        self.path = path
        column_mapping = {'GZÜV-ID': 'gzuev_id',
                                'Gemeindename': 'gemeindename',
                                'Quartal': 'quartal',
                                'ENTNAHME-DATUM': 'entnahme_datum',
                                'LUFTTEMPERATUR IN °C': 'lufttemperatur',
                                'GERUCH': 'geruch',
                                'FAERBUNG': 'faerbung',
                                'TRUEBUNG': 'truebung',
                                'WASSERTEMPERATUR °C': 'wassertemperatur',
                                'PH-WERT': 'pH',
                                'SAUERSTOFFGEHALT mg/l': 'O2_gehalt',
                                'REDOXPOTENTIAL mV': 'redoxpotential',
                                'GESAMTHAERTE °dH': 'gesamt_haerte',
                                'KARBONATHAERTE °dH': 'karbonat_haerte',
                                'FREIE KOHLENSAEURE mg/l': 'CO3',
                                'CALCIUM mg/l': 'Ca',
                                'MAGNESIUM mg/l': 'Mg',
                                'NATRIUM mg/l': 'Na',
                                'KALIUM mg/l': 'K',
                                'EISEN mg/l': 'Fe',
                                'MANGAN mg/l': 'Mn',
                                'BOR mg/l': 'B',
                                'AMMONIUM mg/l': 'NH4',
                                'NITRIT mg/l': 'NO2',
                                'NITRAT mg/l': 'NO3',
                                'CHLORID mg/l': 'Cl',
                                'SULFAT mg/l': 'SO4',
                                'HYDROGENK. mg/l': 'HCO3',
                                'ORTHOPHOSPHAT mg/l': 'PO4',
                                'KIESELSAEURE mg/l': 'SiO2',
                                'FLUORID mg/l': 'F',
                                'DOC mg/l': 'doc',
                                'TOC mg/l': 'toc',
                                'QUELLSCHÜTTUNG l/s': 'quellschuettung',
                                'TRITIUM TE': 'H3',
                                'DEUTERIUM d 0/00': 'D2',
                                'SAUERSTOFF 18 d 0/00': 'O18',
                                'RADON-222 Bq/l': 'Rn222',
                                'ELEKTR. LEITF. (bei 20°C) µS/cm': 'elektr_leitf',
                                'CADMIUM µg/l': 'Cd',
                                'QUECKSILBER µg/l': 'Hg',
                                'ZINK µg/l': 'Zn',
                                'KUPFER µg/l': 'Cu',
                                'ALUMINIUM µg/l': 'Al',
                                'BLEI µg/l': 'Pb',
                                'CHROM-GESAMT (filtriert) µg/l': 'Cr_gesamt',
                                'NICKEL µg/l': 'Ni',
                                'ARSEN µg/l': 'As',
                                'URAN µg/l': 'U'}

        # read data sheet
        self.df_raw = pd.read_csv(path, sep=';', encoding='latin1', skiprows=top)[:-bottom]
        self.df_raw.rename(columns=column_mapping, inplace=True)
        self.df_raw['date'] = pd.to_datetime(self.df_raw['entnahme_datum'], dayfirst=True)
        
        # Clean the data
        self.cleaned_df = self._clean_data(self.df_raw)

        # join with station ids
        self.stations = pd.read_excel(r"M:\WASSERRESSOURCEN - GQH Stufe 1 2024 - 2400613\C GRUNDLAGEN\01-Daten\Stationen_ids.xlsx")
        self.df = pd.merge(self.stations, self.cleaned_df,
                           left_on='gzuev_id', right_on='gzuev_id')

    def _clean_data(self, df: pd.DataFrame):
        import re
        # Columns to exclude from cleaning
        exclude_columns = ['gzuev_id','quartal','gemeindename','entnahme_datum',
                        'date','geruch','faerbung','truebung',]
        
        # Replace no data values, below detection limit and [] in relevant columns only
        for col in df.columns:
            if col not in exclude_columns:
                # Fix double decimal issues (e.g., '0.0.2' -> '0.2')
                df[col] = df[col].apply(lambda x: re.sub(r'(\d+)\.(\d+)\.(\d+)', r'\1.\3', x) if isinstance(x, str) and re.match(r'\d+\.\d+\.\d+', x) else x)
                
                # Replace values below detection limit with half the numeric value
                df[col] = df[col].apply(lambda x: re.sub(r'<(\d+)', lambda m: str(float(m.group(1)) * 0.5), x) if isinstance(x, str) and re.match(r'<\d+', x) else x)
                
                # Replace bracketed numeric values with NaN
                df[col].replace(to_replace=r'\[\d+,\d+\]', value=np.nan, regex=True, inplace=True)
                
                # Remove other brackets
                df[col].replace(to_replace=r'\[|\]', value='', regex=True, inplace=True)
                
                # Replace 'n.a.' with NaN
                df[col].replace(to_replace='n.a.', value=np.nan, regex=False, inplace=True)
                
                # Convert commas to dots for numeric values
                df[col] = df[col].apply(lambda x: str(x).replace(',', '.') if isinstance(x, str) else x)
                
                # Convert columns to numeric, coercing errors
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
   
def GZUV_format_for_plot(df:pd.DataFrame, label:str='label', color:str='k', marker:str='o', size:int=30, alpha:float=1) -> pd.DataFrame:
    """Take a cleaned GZÜV dataframe and add necessary columns 
    so it can be used as input for wqchartpy diagrams.

    Args:
        df (pd.DataFrame): dataframe with GZÜV data
        label (str, optional): _description_. Defaults to 'label'.
        color (str, optional): _description_. Defaults to 'k'.
        marker (str, optional): _description_. Defaults to 'o'.
        size (int, optional): _description_. Defaults to 30.
        alpha (float, optional): _description_. Defaults to 1.

    Returns:
        pd.DataFrame: Dataframe with correct format for wqchartpy.
    """
    cols = ['Ca', 'Mg', 'Na', 'K', 'HCO3', 'SO4', 'Cl', 'CO3', 'pH', 'date', 'Name']
    station_df = df[cols].copy()
    station_df['TDS'] = station_df[cols[:8]].sum(axis=1)
    station_df['Sample'] = station_df['Name']
    station_df['Label'] = label
    station_df['Color'] = color
    station_df['Marker'] = marker
    station_df['Size'] = size
    station_df['Alpha'] = alpha
    station_df.replace(np.nan, 0, inplace=True)
    return station_df

# CTUA Readout from excel sheets and data Handling
def process_CTUA_data(file_path:str)->pd.DataFrame:
    """Read data from CTUA excel Lab Reports

    Args:
        file_path (str): path to CTUA lab report

    Returns:
        pd.DataFrame: cleaned dataframe
    """
    # Load the Excel file
    df = pd.read_excel(file_path)

    # Determine the number of header rows
    header_row_count = df[df.iloc[:,0] == 'Datum Probenahme'].index[0]

    # Separate the dataframe into left and right parts
    left_names = df.iloc[header_row_count, :4].values
    left = df.iloc[header_row_count+1:, :4]
    left.columns = left_names
    left['Datum Probenahme'] = pd.to_datetime(left['Datum Probenahme'])
    right = df.iloc[header_row_count+1:, 4:]

    # Combine left and right parts
    df_join = pd.concat([left, right], axis=1)

    # Process the unnamed columns to handle detection limit markers
    unnamed_columns = [col for col in right.columns if "Unnamed" in str(col)]

    for col in unnamed_columns:
        marker_col = df_join[col]
        value_col_index = df_join.columns.get_loc(col) + 1
        if value_col_index < len(df_join.columns):
            value_col_name = df_join.columns[value_col_index]
            
            # # Set the corresponding values to 0 where markers are found
            # df_join.loc[marker_col.isin(['<', '<<']), value_col_name] = 0
            df_join.loc[marker_col.isin(['<<']), value_col_name] = np.nan
            df_join.loc[marker_col.isin(['<']), value_col_name] /= 2

    # Convert measurement columns to float
    measurement_columns = [col for col in df_join.columns if col not in left_names and "Unnamed" not in str(col)]
    df_join[measurement_columns] = df_join[measurement_columns].astype(float)

    # Drop all unnamed columns
    df_cleaned = df_join.loc[:, ~df_join.columns.str.contains('^Unnamed')]
    return df_cleaned

def combine_CTUA_data(path:str) -> pd.DataFrame:
    """scrapes a directory for files with "CTUA" in the name and combines 
    the data of all files into a single DataFrame. See also utils.process_CTUA_data()

    Args:
        path (str): search directory

    Returns:
        pd.DataFrame: combined dataframe
    """
    files = filter_by_string(list_files(path, '.xlsx'), 'CTUA')

    combined_df = pd.DataFrame()

    for i, file in enumerate(files):
        processing =file.split('\\')[-1]
        print(f'Processing file {i}: {processing}')
        cleaned_df = process_CTUA_data(file)
        if cleaned_df.columns.duplicated().any():
            print(f"Duplicate columns found in {file}: {cleaned_df.columns[cleaned_df.columns.duplicated()].tolist()}")
        combined_df = pd.concat([combined_df, cleaned_df], ignore_index=True)

    return combined_df

def CTUA_format_for_plot(df:pd.DataFrame, label:str='label', color:str='k', 
                         marker:str='o', size:int=30, alpha:float=1) -> pd.DataFrame:
    """
    Takes a cleaned (combined) dataframe from CTUA data and adds some columns so it
    can be used as input for wqchartpy diagrams, which is a package for
    plotting of hydrochemistry data.

    Args:
        df (pd.DataFrame): Input dataframe with hydrochemistry data. Derived 
            from combine_CTUA_data() function.
        label (str, optional): Label for the data points. Defaults to 'label'.
        color (str, optional): Color of the data points. Defaults to 'k' (black).
        marker (str, optional): Marker style for the data points. Defaults to 'o' (circle).
        size (int, optional): Size of the data points. Defaults to 30.
        alpha (float, optional): Transparency of the data points. Defaults to 0.6.

    Returns:
        pd.DataFrame: Dataframe with correct format for wqchartpy.

    Example Use:
        station_id = 'KK71130012'
        file_path = r"path/to/Qualitätsdatenabfrage.xlsx"
        df = GZUV(path=file_path).get_station_data(station_id=station_id)
        df = format_for_plot(df, station_id)
    """

    chemical_symbols_mapping = {'pH (Labor)': 'pH','LF (Labor) (25°C)': 'LF',
                                'Calcium': 'Ca','Magnesium': 'Mg','Natrium': 'Na',
                                'Kalium': 'K','Hydrogenkarbonat': 'HCO3','Chlorid': 'Cl',
                                'Sulfat': 'SO4','Nitrat': 'NO2','Nitrit': 'NO3',
                                'Ammonium': 'NH3','Orthophosphat': 'PO4',
                                'DOC (gel. org. C)': 'DOC','Lithium': 'Li',
                                'Uran': 'U','Zink': 'Zn','Quecksilber': 'Hg',
                                'Nickel': 'Ni','Kupfer': 'Cu','Chrom': 'Cr',
                                'Cadmium': 'Cd','Blei': 'Pb','Arsen': 'As',
                                'Aluminium': 'Al','Mangan': 'Mn','Eisen': 'Fe',
                                'Bor': 'B',
                                # Non-element columns retain their original names
                                'Datum Probenahme': 'Datum Probenahme',
                                'Probennr.': 'Probennr.','Probestelle': 'Probestelle',
                                'Bezeichnung': 'Sample','Gesamthärte': 'Gesamthärte',
                                'Karbonathärte': 'Karbonathärte',
                                'Kultivierbare MO, 22': 'Kultivierbare MO, 22',
                                'Kultivierbare MO, 37': 'Kultivierbare MO, 37',
                                'Escherichia coli /100ml': 'Escherichia coli /100ml',
                                'Coliforme /100ml': 'Coliforme /100ml',
                                'Intest. Enterok. /100ml': 'Intest. Enterok. /100ml'}

    df.rename(columns=chemical_symbols_mapping, inplace=True)
    
    df['Label'] = [label for i in range(df.index.size)]
    df['Color'] = [color for i in range(df.index.size)]
    df['Marker'] = [marker for i in range(df.index.size)]
    df['Size'] = [size for i in range(df.index.size)]
    df['Alpha'] = [alpha for i in range(df.index.size)]
    df['CO3'] = [0 for i in range(df.index.size)]
    df['TDS'] = df.Ca + df.Mg + df.Na + df.K + df.Cl + df.SO4 + df.HCO3

    return df

if __name__ == "__main__":

    path_ctua = r"M:\WASSERRESSOURCEN - GQH Stufe 1 2024 - 2400613\C GRUNDLAGEN\01-Daten\01-Laborprüfberichte"
    df = combine_CTUA_data(path_ctua)
    df.to_excel(path_ctua + '/CTUA_combined.xlsx', index=False)