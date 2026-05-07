#!/usr/bin/env python3
"""
Climate Simulation Results Analysis
Analyzes EnergyPlus simulation results from HTML tables to extract key metrics
and generate trend visualizations comparing different climate scenarios.

Key Metrics:
- Energy consumption per area
- Heating/cooling load unmet hours
- Peak demand

Comparison:
- sTMY vs Annex80 TMY data
- Climate scenarios: 2001, 2041, 2081
- Current TMY as baseline
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyPlusResultsAnalyzer:
    """Analyze EnergyPlus simulation results from HTML tables"""

    def __init__(self, output_base_dir: str = None):
        """Initialize the analyzer"""
        if output_base_dir is None:
            self.output_base_dir = Path(__file__).parent / "output"
        else:
            self.output_base_dir = Path(output_base_dir)

        # Cities to analyze
        self.cities = [
            "Los_Angeles",
            "Miami",
            "Montreal",
            "Toronto"
        ]

        # Climate scenarios
        self.scenarios = ["current", "stmy_2001", "stmy_2041", "stmy_2081"]
        self.annex80_scenarios = ["annex80_2001", "annex80_2041", "annex80_2081"]

        # Metrics to extract
        self.metrics = {
            "energy_per_area": "Total Site Energy per Total Floor Area",
            "heating_unmet_hours": "Time Setpoint Not Met During Occupied Heating",
            "cooling_unmet_hours": "Time Setpoint Not Met During Occupied Cooling",
        }

        self.results_data = {}

        print(f"Initialized EnergyPlus Results Analyzer")
        print(f"Output base directory: {self.output_base_dir}")

    def find_latest_results(self) -> Optional[Path]:
        """Find the latest simulation results directory (excluding smym_ prefixed ones)"""
        if not self.output_base_dir.exists():
            logger.error(f"Output directory not found: {self.output_base_dir}")
            return None

        # Find all timestamp directories WITHOUT smym_ prefix
        timestamp_dirs = []
        for d in self.output_base_dir.iterdir():
            if d.is_dir():
                # Only match YYYYMMDD_HHMMSS pattern (exclude smym_ prefixed ones)
                if re.match(r'\d{8}_\d{6}$', d.name) and not d.name.startswith('smym_'):
                    timestamp_dirs.append(d)

        if not timestamp_dirs:
            logger.error("No simulation results directories found (excluding smym_ prefixed)")
            return None

        # Return the latest one
        latest_dir = sorted(timestamp_dirs)[-1]
        logger.info(f"Using latest results from (non-smym): {latest_dir}")
        return latest_dir

    def extract_eplus_tables(self, html_file: Path) -> Dict[str, pd.DataFrame]:
        """
        Extract specific tables from eplustbl.htm file
        Focus on: Site and Source Energy, and Unmet Degree-Hours tables
        """
        tables = {}

        print(f"\n=== Processing {html_file.name} ===")
        print(f"File path: {html_file}")
        print(f"File exists: {html_file.exists()}")

        if not html_file.exists():
            logger.warning(f"HTML file not found: {html_file}")
            return tables

        try:
            print("Reading file content...")
            with open(html_file, 'r', encoding='utf-8') as f:
                content = f.read()

            print(f"File size: {len(content)} characters")

            print("Parsing HTML with BeautifulSoup...")
            soup = BeautifulSoup(content, 'html.parser')

            # Find all tables with their preceding text
            all_tables = soup.find_all('table')
            print(f"Found {len(all_tables)} tables in HTML")

            for table in all_tables:
                # Look for the comment before the table that contains the table name
                comment = table.find_previous(string=re.compile(r'FullName:'))
                if comment:
                    table_name = comment.strip().replace('FullName:', '').strip()

                    # Extract specific tables we care about
                    if any(keyword in table_name.lower() for keyword in
                          ['site and source energy', 'unmet degree-hours']):

                        print(f"Found target table: {table_name}")

                        # Convert HTML table to pandas DataFrame
                        try:
                            df = pd.read_html(str(table))[0]
                            tables[table_name] = df
                            print(f"  Successfully parsed table with shape: {df.shape}")

                            # Save table as CSV for debugging
                            if hasattr(self, '_csv_output_dir') and self._csv_output_dir:
                                # Get city name from html file path
                                city_name = self._extract_city_from_path(html_file)
                                scenario = self._extract_scenario_from_path(html_file)

                                safe_name = re.sub(r'[^\w\s-]', '_', table_name)
                                csv_filename = f"{safe_name}_{city_name}_{scenario}.csv"
                                csv_file = self._csv_output_dir / csv_filename
                                df.to_csv(csv_file, index=False)
                                print(f"  Saved CSV: {csv_file}")

                        except Exception as e:
                            print(f"  Error parsing table {table_name}: {e}")

            print(f"Extracted {len(tables)} target tables:")
            for table_name in tables.keys():
                print(f"  - {table_name}")

        except Exception as e:
            logger.error(f"Error extracting tables from {html_file}: {str(e)}")
            import traceback
            traceback.print_exc()

        return tables

    def _extract_city_from_path(self, html_file: Path) -> str:
        """Extract city name from file path"""
        path_parts = html_file.parts
        for part in path_parts:
            for city in self.cities:
                if city in part:
                    return city
        return "unknown"

    def _extract_scenario_from_path(self, html_file: Path) -> str:
        """Extract scenario from file path"""
        path_parts = html_file.parts
        for part in path_parts:
            if "_current" in part:
                return "current"
            elif "_stmy_" in part:
                if "2001" in part:
                    return "stmy_2001"
                elif "2041" in part:
                    return "stmy_2041"
                elif "2081" in part:
                    return "stmy_2081"
            elif "_annex80_" in part:
                if "2001" in part:
                    return "annex80_2001"
                elif "2041" in part:
                    return "annex80_2041"
                elif "2081" in part:
                    return "annex80_2081"
        return "unknown"

    def extract_metrics_from_tables(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Extract specific metrics from the parsed tables"""
        metrics_data = {}

        try:
            # Extract from Site and Source Energy table
            site_energy_table = None
            for table_name, df in tables.items():
                if 'site and source energy' in table_name.lower():
                    site_energy_table = df
                    break

            if site_energy_table is not None:
                print(f"Processing Site and Source Energy table:")
                print(site_energy_table.head())
                print(f"Table shape: {site_energy_table.shape}")
                print(f"Column names: {list(site_energy_table.columns)}")

                # Look for "Energy Per Conditioned Building Area [MJ/m2]" column
                energy_per_area_col_idx = None
                for col_idx in range(1, site_energy_table.shape[1]):
                    if pd.notna(site_energy_table.iloc[0, col_idx]):
                        header_text = str(site_energy_table.iloc[0, col_idx])
                        if 'Energy Per Conditioned Building Area [MJ/m2]' in header_text:
                            energy_per_area_col_idx = col_idx
                            print(f"  Found Energy Per Conditioned Building Area column at index {col_idx}")
                            break

                if energy_per_area_col_idx is not None:
                    # Look for "Total Source Energy" row
                    for row_idx in range(1, site_energy_table.shape[0]):
                        if pd.notna(site_energy_table.iloc[row_idx, 0]):
                            row_label = str(site_energy_table.iloc[row_idx, 0])
                            if 'Total Source Energy' == row_label:
                                value_cell = site_energy_table.iloc[row_idx, energy_per_area_col_idx]
                                if pd.notna(value_cell):
                                    value = self._extract_numeric_value(str(value_cell))
                                    if value is not None:
                                        metrics_data['energy_per_area'] = value
                                        print(f"  Found energy_per_area: {value} from Total Source Energy")
                                        break

            # Extract from Unmet Degree-Hours table
            unmet_table = None
            for table_name, df in tables.items():
                if 'unmet degree-hours' in table_name.lower():
                    unmet_table = df
                    break

            if unmet_table is not None:
                print(f"Processing Unmet Degree-Hours table:")
                print(unmet_table.head())
                print(f"Table shape: {unmet_table.shape}")
                print(f"Column names: {list(unmet_table.columns)}")

                # Find specific columns
                heating_col_idx = None
                cooling_col_idx = None

                for col_idx in range(1, unmet_table.shape[1]):
                    if pd.notna(unmet_table.iloc[0, col_idx]):
                        header_text = str(unmet_table.iloc[0, col_idx])
                        if 'Heating Setpoint Unmet Degree-Hours [°C·hr]' in header_text:
                            heating_col_idx = col_idx
                            print(f"  Found Heating Setpoint Unmet Degree-Hours column at index {col_idx}")
                        elif 'Cooling Setpoint Unmet Degree-Hours [°C·hr]' in header_text:
                            cooling_col_idx = col_idx
                            print(f"  Found Cooling Setpoint Unmet Degree-Hours column at index {col_idx}")

                # Look for "Sum" row to get total values
                for row_idx in range(unmet_table.shape[0]):
                    if pd.notna(unmet_table.iloc[row_idx, 0]):
                        row_label = str(unmet_table.iloc[row_idx, 0])
                        if 'Sum' == row_label:
                            # Extract heating unmet hours
                            if heating_col_idx is not None:
                                heating_cell = unmet_table.iloc[row_idx, heating_col_idx]
                                if pd.notna(heating_cell):
                                    heating_value = self._extract_numeric_value(str(heating_cell))
                                    if heating_value is not None:
                                        metrics_data['heating_unmet_hours'] = heating_value
                                        print(f"  Found heating_unmet_hours: {heating_value} from Sum row")

                            # Extract cooling unmet hours
                            if cooling_col_idx is not None:
                                cooling_cell = unmet_table.iloc[row_idx, cooling_col_idx]
                                if pd.notna(cooling_cell):
                                    cooling_value = self._extract_numeric_value(str(cooling_cell))
                                    if cooling_value is not None:
                                        metrics_data['cooling_unmet_hours'] = cooling_value
                                        print(f"  Found cooling_unmet_hours: {cooling_value} from Sum row")
                            break

        except Exception as e:
            logger.error(f"Error extracting metrics from tables: {str(e)}")
            import traceback
            traceback.print_exc()

        return metrics_data

    def extract_peak_demand_from_csv(self, csv_file: Path) -> Optional[Dict[int, float]]:
        """
        Extract monthly peak demand from eplusmtr.csv file
        Process: hourly facility electricity -> convert J to kW -> daily max -> monthly average
        """
        if not csv_file.exists():
            logger.warning(f"CSV file not found: {csv_file}")
            return None

        try:
            print(f"Processing peak demand from: {csv_file}")

            # Read CSV file
            df = pd.read_csv(csv_file)

            # Find facility electricity column (should contain "Facility" and "Electricity")
            facility_col = None
            for col in df.columns:
                if 'Electricity:Facility' in col and '[J]' in col:
                    facility_col = col
                    break

            if facility_col is None:
                logger.warning(f"No facility electricity column found in {csv_file}")
                print(f"Available columns: {list(df.columns)}")
                return None

            print(f"Found facility electricity column: {facility_col}")

            # Convert from J (Joules) to kW (assuming hourly data)
            # 1 J/hr = 1/3600000 kW (1 Wh = 3600 J, 1 kW = 1000 W)
            df['electricity_kw'] = df[facility_col] / 3600000

            # Handle datetime column - EnergyPlus format is typically "MM/DD  HH:MM:SS"
            if 'Date/Time' in df.columns:
                # EnergyPlus date format: "01/01  01:00:00" (MM/DD  HH:MM:SS)
                # Need to add year and parse carefully
                try:
                    # Add a year (2023 is arbitrary) and clean up the format
                    df['date_time_str'] = df['Date/Time'].astype(str)
                    # Replace multiple spaces with single space and add year
                    df['date_time_clean'] = df['date_time_str'].str.replace(r'\s+', ' ', regex=True)
                    df['date_time_clean'] = '2023/' + df['date_time_clean']

                    # Parse with explicit format
                    df['datetime'] = pd.to_datetime(df['date_time_clean'], format='%Y/%m/%d %H:%M:%S')

                except Exception as e:
                    print(f"Error parsing Date/Time with format method: {e}")
                    # Fallback: try a more flexible approach
                    try:
                        # Extract month and day, add year, then parse
                        df['date_parts'] = df['Date/Time'].str.strip().str.split(r'\s+', expand=True)
                        df['date_part'] = df['date_parts'][0]  # MM/DD
                        df['time_part'] = df['date_parts'][1]  # HH:MM:SS
                        df['full_date_str'] = '2023/' + df['date_part'] + ' ' + df['time_part']
                        df['datetime'] = pd.to_datetime(df['full_date_str'], format='%Y/%m/%d %H:%M:%S')

                    except Exception as e2:
                        print(f"Error with fallback parsing: {e2}")
                        # Create datetime from row index as last resort
                        hours_in_year = len(df)
                        start_date = pd.Timestamp('2023-01-01 01:00:00')
                        df['datetime'] = pd.date_range(start=start_date, periods=hours_in_year, freq='H')

            else:
                # If no Date/Time column, create from index assuming hourly data starting Jan 1
                hours_in_year = len(df)
                start_date = pd.Timestamp('2023-01-01 01:00:00')
                df['datetime'] = pd.date_range(start=start_date, periods=hours_in_year, freq='H')

            # Extract month from datetime
            df['month'] = df['datetime'].dt.month
            df['date'] = df['datetime'].dt.date

            # Calculate daily maximum for each day
            daily_max = df.groupby('date')['electricity_kw'].max().reset_index()
            daily_max['month'] = pd.to_datetime(daily_max['date']).dt.month

            # Calculate monthly average of daily peaks
            monthly_peak_demand = daily_max.groupby('month')['electricity_kw'].mean().to_dict()

            print(f"Extracted monthly peak demands: {monthly_peak_demand}")
            return monthly_peak_demand

        except Exception as e:
            logger.error(f"Error processing peak demand from {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def analyze_simulation_results(self, results_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Analyze all simulation results in the directory"""
        results = {}

        for city in self.cities:
            results[city] = {}

            # Analyze current TMY (baseline)
            current_dir = results_dir / f"{city}_current"
            html_file = current_dir / "eplustbl.htm"
            if html_file.exists():
                tables = self.extract_eplus_tables(html_file)
                metrics = self.extract_metrics_from_tables(tables)
                results[city]["current"] = metrics

            # Analyze sTMY scenarios
            for scenario in ["stmy_2001", "stmy_2041", "stmy_2081"]:
                scenario_dir = results_dir / f"{city}_{scenario}"
                html_file = scenario_dir / "eplustbl.htm"
                if html_file.exists():
                    tables = self.extract_eplus_tables(html_file)
                    metrics = self.extract_metrics_from_tables(tables)
                    results[city][scenario] = metrics

            # Analyze Annex80 scenarios (if available)
            for scenario in ["annex80_2001", "annex80_2041", "annex80_2081"]:
                scenario_dir = results_dir / f"{city}_{scenario}"
                html_file = scenario_dir / "eplustbl.htm"
                if html_file.exists():
                    tables = self.extract_eplus_tables(html_file)
                    metrics = self.extract_metrics_from_tables(tables)
                    results[city][scenario] = metrics

        self.results_data = results
        return results

    def create_trend_plots(self, output_dir: Path = None):
        """Create trend plots for each metric and city"""
        if output_dir is None:
            output_dir = self.output_base_dir / "analysis_plots"
        output_dir.mkdir(exist_ok=True)

        metrics = ["energy_per_area", "heating_unmet_hours", "cooling_unmet_hours"]
        metric_labels = {
            "energy_per_area": "Energy Consumption per Area (MJ/m²)",
            "heating_unmet_hours": "Heating Unmet Degree-Hours (°C·hr)",
            "cooling_unmet_hours": "Cooling Unmet Degree-Hours (°C·hr)"
        }

        years = [2001, 2041, 2081]

        for city in self.cities:
            if city not in self.results_data:
                continue

            city_data = self.results_data[city]

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Climate Impact Trends - {city.replace("_", " ")}', fontsize=16, fontweight='bold')

            for idx, metric in enumerate(metrics):
                ax = axes[idx]

                # Extract data for sTMY scenarios
                stmy_values = []
                annex80_values = []

                for year in years:
                    # sTMY data
                    stmy_key = f"stmy_{year}"
                    if stmy_key in city_data and metric in city_data[stmy_key]:
                        stmy_values.append(city_data[stmy_key][metric])
                    else:
                        stmy_values.append(np.nan)

                    # Annex80 data
                    annex80_key = f"annex80_{year}"
                    if annex80_key in city_data and metric in city_data[annex80_key]:
                        annex80_values.append(city_data[annex80_key][metric])
                    else:
                        annex80_values.append(np.nan)

                # Plot trend lines
                if not all(np.isnan(stmy_values)):
                    ax.plot(years, stmy_values, 'o-', label='sTMY', linewidth=2, markersize=8)

                if not all(np.isnan(annex80_values)):
                    ax.plot(years, annex80_values, 's-', label='Annex80 TMY', linewidth=2, markersize=8)

                # Add current TMY baseline
                if "current" in city_data and metric in city_data["current"]:
                    current_value = city_data["current"][metric]
                    ax.axhline(y=current_value, color='red', linestyle='--',
                              label='Current TMY (Baseline)', linewidth=2)

                ax.set_xlabel('Year')
                ax.set_ylabel(metric_labels[metric])
                ax.set_title(f'{metric_labels[metric]}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Set x-axis ticks
                ax.set_xticks(years)
                ax.set_xticklabels([str(y) for y in years])

            plt.tight_layout()

            # Save plot
            plot_file = output_dir / f"{city}_climate_trends.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved trend plot: {plot_file}")

    def create_summary_table(self, output_dir: Path = None) -> pd.DataFrame:
        """Create summary table of all results"""
        if output_dir is None:
            output_dir = self.output_base_dir / "analysis_results"
        output_dir.mkdir(exist_ok=True)

        # Prepare data for DataFrame
        summary_data = []

        for city in self.cities:
            if city not in self.results_data:
                continue

            city_data = self.results_data[city]

            for scenario, metrics in city_data.items():
                row = {
                    'City': city.replace('_', ' '),
                    'Scenario': scenario,
                    'Data_Source': self._get_data_source(scenario),
                    'Year': self._get_year_from_scenario(scenario)
                }

                # Add metric values
                for metric in ["energy_per_area", "heating_unmet_hours", "cooling_unmet_hours"]:
                    row[metric] = metrics.get(metric, np.nan)

                summary_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(summary_data)

        # Save to CSV
        csv_file = output_dir / "climate_simulation_results.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved summary table: {csv_file}")

        return df

    def _extract_numeric_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text, handling various formats"""
        try:
            # Remove common non-numeric characters and whitespace
            cleaned_text = re.sub(r'[^\d.\-+eE]', '', str(text).strip())

            if not cleaned_text:
                return None

            # Try to convert to float
            value = float(cleaned_text)
            return value

        except (ValueError, TypeError):
            return None

    def _get_data_source(self, scenario: str) -> str:
        """Get data source from scenario name"""
        if scenario == "current":
            return "Current TMY"
        elif scenario.startswith("stmy_"):
            return "sTMY"
        elif scenario.startswith("annex80_"):
            return "Annex80 TMY"
        else:
            return "Unknown"

    def _get_year_from_scenario(self, scenario: str) -> Optional[int]:
        """Extract year from scenario name"""
        if scenario == "current":
            return None

        year_match = re.search(r'(\d{4})', scenario)
        if year_match:
            return int(year_match.group(1))
        return None

    def create_comparison_plots(self, output_dir: Path = None):
        """Create comparison plots between sTMY and Annex80"""
        if output_dir is None:
            output_dir = self.output_base_dir / "analysis_plots"
        output_dir.mkdir(exist_ok=True)

        # Keep only 3 metrics, remove peak_demand
        metrics = ["energy_per_area", "heating_unmet_hours", "cooling_unmet_hours"]
        metric_labels = {
            "energy_per_area": "Energy Consumption per Area (MJ/m²)",
            "heating_unmet_hours": "Heating Unmet Degree-Hours (°C·hr)",
            "cooling_unmet_hours": "Cooling Unmet Degree-Hours (°C·hr)"
        }

        years = [2001, 2041, 2081]

        # Define colors for each city
        city_colors = {
            "Los_Angeles": "#1f77b4",  # Blue
            "Miami": "#ff7f0e",        # Orange
            "Montreal": "#2ca02c",     # Green
            "Toronto": "#d62728"       # Red
        }

        # Create 1x3 subplot layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Climate Data Source Comparison Across All Cities', fontsize=16, fontweight='bold')

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for city in self.cities:
                if city not in self.results_data:
                    continue

                city_data = self.results_data[city]
                city_color = city_colors.get(city, "#000000")  # Default to black if city not found
                city_display_name = city.replace("_", " ")

                # Extract sTMY and Annex80 data
                stmy_values = []
                annex80_values = []

                for year in years:
                    stmy_key = f"stmy_{year}"
                    annex80_key = f"annex80_{year}"

                    if stmy_key in city_data and metric in city_data[stmy_key]:
                        stmy_values.append(city_data[stmy_key][metric])
                    else:
                        stmy_values.append(np.nan)

                    if annex80_key in city_data and metric in city_data[annex80_key]:
                        annex80_values.append(city_data[annex80_key][metric])
                    else:
                        annex80_values.append(np.nan)

                # Plot lines for this city
                if not all(np.isnan(stmy_values)):
                    label = f'{city_display_name} sTMY'
                    ax.plot(years, stmy_values, 'o-', color=city_color, alpha=0.8,
                           label=label, linewidth=2, markersize=6)

                if not all(np.isnan(annex80_values)):
                    label = f'{city_display_name} Annex80'
                    ax.plot(years, annex80_values, 's--', color=city_color, alpha=0.8,
                           label=label, linewidth=2, markersize=6)

            ax.set_xlabel('Year')
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(metric_labels[metric])
            ax.grid(True, alpha=0.3)
            ax.set_xticks(years)

            # Add legend to each subplot
            ax.legend(loc='best', fontsize='small')

        plt.tight_layout()

        # Save comparison plot
        comparison_file = output_dir / "data_source_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved comparison plot: {comparison_file}")

    def collect_peak_demand_data(self, results_dir: Path) -> Dict[str, Dict[int, float]]:
        """
        Collect monthly peak demand data from all simulation results
        Returns: {scenario_label: {month: peak_demand_kw}}
        """
        peak_demand_data = {}

        for city in self.cities:
            # Process all scenarios for this city
            all_scenarios = ["current", "stmy_2001", "stmy_2041", "stmy_2081",
                           "annex80_2001", "annex80_2041", "annex80_2081"]

            for scenario in all_scenarios:
                scenario_dir = results_dir / f"{city}_{scenario}"
                csv_file = scenario_dir / "eplusmtr.csv"

                if csv_file.exists():
                    monthly_peaks = self.extract_peak_demand_from_csv(csv_file)
                    if monthly_peaks:
                        # Create scenario label for heatmap: City_DataSource_Year
                        if scenario == "current":
                            scenario_label = f"{city}_Current"
                        else:
                            data_source = "sTMY" if scenario.startswith("stmy_") else "Annex80"
                            year = self._get_year_from_scenario(scenario)
                            scenario_label = f"{city}_{data_source}_{year}"

                        peak_demand_data[scenario_label] = monthly_peaks

        return peak_demand_data

    def create_peak_demand_heatmap(self, output_dir: Path = None):
        """
        Create monthly peak demand heatmaps in 2x2 subplot layout
        Each city gets its own subplot with different color schemes:
        - Los Angeles & Miami: Red gradients
        - Montreal & Toronto: Blue gradients
        """
        if output_dir is None:
            output_dir = self.output_base_dir / "analysis_plots"
        output_dir.mkdir(exist_ok=True)

        # Find latest results directory
        results_dir = self.find_latest_results()
        if results_dir is None:
            logger.error("No results directory found for peak demand analysis.")
            return

        # Collect peak demand data from all simulations
        logger.info("Collecting peak demand data from all simulations...")
        peak_data = self.collect_peak_demand_data(results_dir)

        if not peak_data:
            logger.warning("No peak demand data found.")
            return

        # Setup months
        months = list(range(1, 13))  # 1-12
        month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

        # Define color schemes for different cities
        city_colors = {
            'Los_Angeles': 'Reds',
            'Miami': 'Reds',
            'Montreal': 'Blues',
            'Toronto': 'Blues'
        }

        # City display names
        city_display_names = {
            'Los_Angeles': 'Los Angeles',
            'Miami': 'Miami',
            'Montreal': 'Montreal',
            'Toronto': 'Toronto'
        }

        # Create 2x2 subplot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()  # Make it easier to iterate

        # Process each city
        for idx, city in enumerate(self.cities):
            ax = axes[idx]

            # Filter data for this city
            city_peak_data = {}
            for scenario_label, monthly_data in peak_data.items():
                if scenario_label.startswith(f"{city}_"):
                    city_peak_data[scenario_label] = monthly_data

            if not city_peak_data:
                ax.text(0.5, 0.5, f'No data available for\n{city_display_names[city]}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(city_display_names[city], fontsize=14, fontweight='bold')
                continue

            # Create matrix for this city's heatmap
            heatmap_data = []
            scenario_labels = []

            # Sort scenarios by type and year for better display
            sorted_scenarios = sorted(city_peak_data.keys(), key=lambda x: (
                'Current' in x,           # Current first
                'sTMY' in x,             # Then sTMY
                'Annex80' in x,          # Then Annex80
                x.split('_')[-1] if x.split('_')[-1].isdigit() else '0'  # Then by year
            ))

            for scenario_label in sorted_scenarios:
                monthly_data = city_peak_data[scenario_label]
                # Ensure all 12 months are present, fill missing with NaN
                row_data = [monthly_data.get(month, np.nan) for month in months]

                # Only include if we have some valid data
                if not all(np.isnan(row_data)):
                    heatmap_data.append(row_data)
                    # Clean up scenario label for display
                    display_label = scenario_label.replace(f"{city}_", "").replace("_", " ")
                    scenario_labels.append(display_label)

            if not heatmap_data:
                ax.text(0.5, 0.5, f'No valid data for\n{city_display_names[city]}',
                       ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(city_display_names[city], fontsize=14, fontweight='bold')
                continue

            # Create DataFrame for this city's heatmap
            city_heatmap_df = pd.DataFrame(heatmap_data,
                                         index=scenario_labels,
                                         columns=month_labels)

            # Create heatmap for this city
            cmap = city_colors.get(city, 'YlOrRd')

            # Use independent color scale for each city
            all_city_values = []
            for monthly_data in city_peak_data.values():
                all_city_values.extend([v for v in monthly_data.values() if not np.isnan(v)])

            if all_city_values:
                vmin, vmax = min(all_city_values), max(all_city_values)
            else:
                vmin, vmax = 0, 1

            sns.heatmap(city_heatmap_df,
                       annot=True,
                       fmt='.1f',
                       cmap=cmap,
                       cbar_kws={'label': 'Peak Demand (kW)'},
                       linewidths=0.5,
                       vmin=vmin,
                       vmax=vmax,
                       ax=ax)

            ax.set_title(city_display_names[city], fontsize=14, fontweight='bold')
            ax.set_xlabel('Month', fontsize=10)
            ax.set_ylabel('Scenario', fontsize=10)

            # Improve label readability
            ax.tick_params(axis='x', rotation=0, labelsize=9)
            ax.tick_params(axis='y', rotation=0, labelsize=8)

        # Overall title
        fig.suptitle('Monthly Peak Demand by Climate Scenario and City\n(Daily Peak Averaged by Month)',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # Make room for suptitle

        # Save heatmap
        heatmap_file = output_dir / "monthly_peak_demand_heatmap_by_city.png"
        plt.savefig(heatmap_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved city-based peak demand heatmap: {heatmap_file}")

        # Also save individual city data as CSV files
        for city in self.cities:
            city_peak_data = {}
            for scenario_label, monthly_data in peak_data.items():
                if scenario_label.startswith(f"{city}_"):
                    clean_label = scenario_label.replace(f"{city}_", "").replace("_", " ")
                    city_peak_data[clean_label] = monthly_data

            if city_peak_data:
                # Create DataFrame for this city
                city_data_rows = []
                for scenario, monthly_data in city_peak_data.items():
                    row_data = [monthly_data.get(month, np.nan) for month in months]
                    city_data_rows.append(row_data)

                city_df = pd.DataFrame(city_data_rows,
                                     index=list(city_peak_data.keys()),
                                     columns=month_labels)

                csv_file = output_dir / f"monthly_peak_demand_{city}.csv"
                city_df.to_csv(csv_file)
                logger.info(f"Saved {city} peak demand data: {csv_file}")

        return peak_data

    def run_complete_analysis(self):
        """Run the complete analysis workflow"""
        logger.info("Starting complete analysis workflow...")

        # Find latest results
        results_dir = self.find_latest_results()
        if results_dir is None:
            logger.error("No results directory found. Cannot proceed with analysis.")
            return

        # Create analysis output directory in the timestamp folder
        analysis_dir = results_dir / "_analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Create CSV output directory for extracted tables within _analysis
        csv_dir = analysis_dir / "tables"
        csv_dir.mkdir(exist_ok=True)
        self._csv_output_dir = csv_dir

        # Set up debug output directory for saving extracted tables (keeping both for compatibility)
        self._debug_output_dir = analysis_dir

        # Analyze simulation results
        logger.info("Analyzing simulation results...")
        results = self.analyze_simulation_results(results_dir)

        if not results:
            logger.error("No results extracted. Cannot proceed with analysis.")
            return

        # Generate summary table
        logger.info("Creating summary table...")
        df = self.create_summary_table(analysis_dir)
        print("\nSummary of extracted results:")
        print(df.head(10))

        # # Create trend plots
        # logger.info("Creating trend plots...")
        # self.create_trend_plots(analysis_dir)
        #
        # # Create comparison plots
        # logger.info("Creating comparison plots...")
        # self.create_comparison_plots(analysis_dir)

        # Create peak demand heatmap
        logger.info("Creating peak demand heatmap...")
        self.create_peak_demand_heatmap(analysis_dir)

        logger.info(f"Analysis complete! Results saved in: {analysis_dir}")
        logger.info(f"CSV tables saved in: {csv_dir}")

        return results

    def analyze_smym_results(self, smym_folder_name: str) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Analyze sMYM simulation results from a specific smym_ prefixed folder
        Args:
            smym_folder_name: Name of the smym folder (e.g., 'smym_20251102_162142')
        """
        smym_dir = self.output_base_dir / smym_folder_name

        if not smym_dir.exists():
            logger.error(f"sMYM directory not found: {smym_dir}")
            return {}

        logger.info(f"Analyzing sMYM results from: {smym_dir}")

        results = {}

        # Look for city subdirectories in the smym folder
        for city in self.cities:
            results[city] = {}

            # Find all year directories for this city
            city_pattern = f"{city}_*"
            city_dirs = list(smym_dir.glob(city_pattern))

            for city_dir in city_dirs:
                if city_dir.is_dir():
                    # Extract year from directory name (assuming format like City_YYYY)
                    dir_name = city_dir.name
                    year_match = re.search(r'(\d{4})', dir_name)
                    if year_match:
                        year = year_match.group(1)

                        html_file = city_dir / "eplustbl.htm"
                        if html_file.exists():
                            tables = self.extract_eplus_tables(html_file)
                            metrics = self.extract_metrics_from_tables(tables)
                            results[city][year] = metrics
                            logger.info(f"Extracted metrics for {city} year {year}")
                        else:
                            logger.warning(f"HTML file not found: {html_file}")

        return results


    def create_smym_trend_plots(self, smym_results: Dict[str, Dict[str, Dict[str, float]]],
                               smym_folder_name: str, output_dir: Path = None):
        """
        Create trend plots for sMYM results showing year-over-year changes for each city
        Args:
            smym_results: Results from analyze_smym_results
            smym_folder_name: Name of the smym folder for plot titles
            output_dir: Output directory for plots
        """
        if output_dir is None:
            output_dir = self.output_base_dir / smym_folder_name / "_analysis"
        output_dir.mkdir(exist_ok=True)

        metrics = ["energy_per_area", "heating_unmet_hours", "cooling_unmet_hours"]
        metric_labels = {
            "energy_per_area": "Energy Consumption per Area (MJ/m²)",
            "heating_unmet_hours": "Heating Unmet Degree-Hours (°C·hr)",
            "cooling_unmet_hours": "Cooling Unmet Degree-Hours (°C·hr)"
        }

        # Create individual plots for each city
        for city in self.cities:
            if city not in smym_results or not smym_results[city]:
                logger.warning(f"No sMYM data found for {city}")
                continue

            city_data = smym_results[city]

            # Extract years and sort them
            years = sorted([int(year) for year in city_data.keys() if year.isdigit()])
            if not years:
                logger.warning(f"No valid years found for {city}")
                continue

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'sMYM Climate Trends - {city.replace("_", " ")}\n({smym_folder_name})',
                        fontsize=16, fontweight='bold')

            for idx, metric in enumerate(metrics):
                ax = axes[idx]

                # Extract metric values for available years
                metric_values = []
                valid_years = []

                for year in years:
                    year_str = str(year)
                    if year_str in city_data and metric in city_data[year_str]:
                        metric_values.append(city_data[year_str][metric])
                        valid_years.append(year)
                    else:
                        # Skip missing data points
                        continue

                if len(valid_years) > 0:
                    ax.plot(valid_years, metric_values, '-', linewidth=2,
                           color='#2E86AB', label='sMYM Data')

                    # Add trend line if we have enough points
                    if len(valid_years) > 2:
                        z = np.polyfit(valid_years, metric_values, 1)
                        p = np.poly1d(z)
                        ax.plot(valid_years, p(valid_years), "--", alpha=0.8, color='red',
                               label=f'Trend (slope: {z[0]:.2e})')

                ax.set_xlabel('Year')
                ax.set_ylabel(metric_labels[metric])
                ax.set_title(f'{metric_labels[metric]}')
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Set x-axis ticks to show every 10 years to avoid overlapping
                if valid_years:
                    # Find the range of years and create ticks every 10 years
                    min_year = min(valid_years)
                    max_year = max(valid_years)

                    # Round to nearest decade
                    start_decade = (min_year // 10) * 10
                    end_decade = ((max_year // 10) + 1) * 10

                    # Create tick positions every 10 years
                    decade_ticks = list(range(start_decade, end_decade + 1, 10))

                    # Filter to only include decades that have data nearby
                    filtered_ticks = [tick for tick in decade_ticks
                                    if any(abs(year - tick) <= 5 for year in valid_years)]

                    ax.set_xticks(filtered_ticks)
                    ax.set_xticklabels([str(y) for y in filtered_ticks])

            plt.tight_layout()

            # Save plot
            plot_file = output_dir / f"{city}_smym_trends.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved sMYM trend plot: {plot_file}")

        # Create combined comparison plot for all cities
        self.create_smym_comparison_plot(smym_results, smym_folder_name, output_dir)

    def create_smym_comparison_plot(self, smym_results: Dict[str, Dict[str, Dict[str, float]]],
                                   smym_folder_name: str, output_dir: Path):
        """
        Create comparison plot showing all cities on the same plots
        """
        metrics = ["energy_per_area", "heating_unmet_hours", "cooling_unmet_hours"]
        metric_labels = {
            "energy_per_area": "Energy Consumption per Area (MJ/m²)",
            "heating_unmet_hours": "Heating Unmet Degree-Hours (°C·hr)",
            "cooling_unmet_hours": "Cooling Unmet Degree-Hours (°C·hr)"
        }

        # Define colors for each city
        city_colors = {
            "Los_Angeles": "#1f77b4",  # Blue
            "Miami": "#ff7f0e",        # Orange
            "Montreal": "#2ca02c",     # Green
            "Toronto": "#d62728"       # Red
        }

        # Create 1x3 subplot layout
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'sMYM Climate Trends Comparison - All Cities\n({smym_folder_name})',
                    fontsize=16, fontweight='bold')

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for city in self.cities:
                if city not in smym_results or not smym_results[city]:
                    continue

                city_data = smym_results[city]
                city_color = city_colors.get(city, "#000000")
                city_display_name = city.replace("_", " ")

                # Extract years and sort them
                years = sorted([int(year) for year in city_data.keys() if year.isdigit()])

                # Extract metric values for available years
                metric_values = []
                valid_years = []

                for year in years:
                    year_str = str(year)
                    if year_str in city_data and metric in city_data[year_str]:
                        metric_values.append(city_data[year_str][metric])
                        valid_years.append(year)

                if len(valid_years) > 0:
                    ax.plot(valid_years, metric_values, '-', color=city_color,
                           linewidth=2, label=city_display_name, alpha=0.8)

            ax.set_xlabel('Year')
            ax.set_ylabel(metric_labels[metric])
            ax.set_title(metric_labels[metric])
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save comparison plot
        comparison_file = output_dir / f"smym_trends_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved sMYM comparison plot: {comparison_file}")

    def create_smym_summary_table(self, smym_results: Dict[str, Dict[str, Dict[str, float]]],
                                 smym_folder_name: str, output_dir: Path = None) -> pd.DataFrame:
        """Create summary table for sMYM results"""
        if output_dir is None:
            output_dir = self.output_base_dir / smym_folder_name / "_analysis"
        output_dir.mkdir(exist_ok=True)

        # Prepare data for DataFrame
        summary_data = []

        for city in self.cities:
            if city not in smym_results:
                continue

            city_data = smym_results[city]

            for year_str, metrics in city_data.items():
                if year_str.isdigit():
                    row = {
                        'City': city.replace('_', ' '),
                        'Year': int(year_str),
                        'Data_Source': 'sMYM'
                    }

                    # Add metric values
                    for metric in ["energy_per_area", "heating_unmet_hours", "cooling_unmet_hours"]:
                        row[metric] = metrics.get(metric, np.nan)

                    summary_data.append(row)

        # Create DataFrame
        df = pd.DataFrame(summary_data)

        if not df.empty:
            # Sort by city and year
            df = df.sort_values(['City', 'Year'])

            # Save to CSV
            csv_file = output_dir / f"smym_results_summary.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f"Saved sMYM summary table: {csv_file}")

        return df

    def run_smym_analysis(self, smym_folder_name: str):
        """
        Run complete analysis for sMYM results
        Args:
            smym_folder_name: Name of the smym folder (e.g., 'smym_20251102_162142')
        """
        logger.info(f"Starting sMYM analysis for folder: {smym_folder_name}")

        # Analyze sMYM simulation results
        smym_results = self.analyze_smym_results(smym_folder_name)

        if not smym_results:
            logger.error("No sMYM results extracted. Cannot proceed with analysis.")
            return

        # Create analysis output directory
        analysis_dir = self.output_base_dir / smym_folder_name / "_analysis"
        analysis_dir.mkdir(exist_ok=True)

        # Create CSV output directory for extracted tables
        csv_dir = analysis_dir / "csv"
        csv_dir.mkdir(exist_ok=True)
        self._csv_output_dir = csv_dir

        # Generate summary table
        logger.info("Creating sMYM summary table...")
        df = self.create_smym_summary_table(smym_results, smym_folder_name, analysis_dir)
        if not df.empty:
            print(f"\nsMYM Summary for {smym_folder_name}:")
            print(df.head(10))

        # Create trend plots
        logger.info("Creating sMYM trend plots...")
        self.create_smym_trend_plots(smym_results, smym_folder_name, analysis_dir)

        logger.info(f"sMYM Analysis complete! Results saved in: {analysis_dir}")

        return smym_results


def main():
    """Main function to run the analysis"""
    analyzer = EnergyPlusResultsAnalyzer()

    print("\n=== EnergyPlus Results Analyzer ===")
    print("Please choose analysis type:")
    print("1. Regular analysis (latest non-smym results)")
    print("2. sMYM analysis (specific smym folder)")

    while True:
        choice = input("\nEnter your choice (1 or 2): ").strip()

        if choice == "1":
            # Regular analysis
            results = analyzer.run_complete_analysis()

            if results:
                print("\nRegular analysis completed successfully!")
                print(f"Number of cities analyzed: {len([c for c in results if results[c]])}")
                total_scenarios = sum(len(results[city]) for city in results if results[city])
                print(f"Total scenarios analyzed: {total_scenarios}")
            else:
                print("Regular analysis failed or no results found.")
            break

        elif choice == "2":
            # sMYM analysis
            print("\nAvailable sMYM folders:")
            output_dir = analyzer.output_base_dir
            if output_dir.exists():
                smym_folders = [d.name for d in output_dir.iterdir()
                              if d.is_dir() and d.name.startswith('smym_')]

                if smym_folders:
                    smym_folders.sort(reverse=True)  # Show newest first
                    for i, folder in enumerate(smym_folders, 1):
                        print(f"  {i}. {folder}")

                    print(f"\nOr enter folder name manually (e.g., smym_20251102_162142)")

                    folder_choice = input("Enter folder number or folder name: ").strip()

                    # Check if it's a number (selecting from list)
                    if folder_choice.isdigit():
                        folder_num = int(folder_choice)
                        if 1 <= folder_num <= len(smym_folders):
                            smym_folder_name = smym_folders[folder_num - 1]
                        else:
                            print(f"Invalid choice. Please enter 1-{len(smym_folders)}")
                            continue
                    else:
                        # Manual input
                        smym_folder_name = folder_choice
                        if not smym_folder_name.startswith('smym_'):
                            print("Folder name should start with 'smym_'")
                            continue

                    # Run sMYM analysis
                    smym_results = analyzer.run_smym_analysis(smym_folder_name)

                    if smym_results:
                        print(f"\nsMYM analysis completed successfully for {smym_folder_name}!")
                        cities_with_data = [city for city in smym_results if smym_results[city]]
                        print(f"Number of cities with data: {len(cities_with_data)}")

                        total_years = sum(len([year for year in smym_results[city].keys() if year.isdigit()])
                                        for city in cities_with_data)
                        print(f"Total city-year combinations analyzed: {total_years}")
                    else:
                        print(f"sMYM analysis failed or no results found for {smym_folder_name}.")
                else:
                    print("No sMYM folders found in output directory.")
                    print(f"Looking in: {output_dir}")
            else:
                print(f"Output directory not found: {output_dir}")
            break

        else:
            print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()
