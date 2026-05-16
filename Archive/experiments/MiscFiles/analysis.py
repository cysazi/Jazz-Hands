import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import find_peaks

def analyze_accel_data(filepath='accel_data.csv'):
    """
    Reads accelerometer data from a CSV file, performs FFT analysis, and visualizes the data
    using interactive Plotly figures.

    Args:
        filepath (str): The path to the CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(filepath)
        # Convert time from microseconds to seconds
        time = df['time'].values / 1_000_000.0
        accel_x = df['accel_x'].values
        accel_y = df['accel_y'].values
        accel_z = df['accel_z'].values
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")

    # --- Analysis ---
    # Calculate total acceleration magnitude for each datapoint
    accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

    # Calculate average sampling interval
    avg_sampling_interval = np.mean(np.diff(time))
    fs = 1 / avg_sampling_interval
    print(f"Average Sampling Interval: {avg_sampling_interval:.6f} s ({fs:.2f} Hz)")

    # --- FFT Analysis & Plotting ---
    signals_to_analyze = [
        (accel_x, 'Accel X'), (accel_y, 'Accel Y'),
        (accel_z, 'Accel Z'), (accel_mag, 'Total Accel')
    ]

    # Create a 2x2 subplot figure for the power spectrums
    fig_fft = make_subplots(
        rows=2, cols=2,
        subplot_titles=[name for _, name in signals_to_analyze]
    )

    for i, (signal, name) in enumerate(signals_to_analyze):
        N = len(signal)
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(N, 1 / fs)

        # Power Spectrum
        power_spectrum = np.abs(yf)**2 / N

        # Find peaks in the power spectrum
        peaks, _ = find_peaks(power_spectrum[:N//2], height=np.mean(power_spectrum[:N//2]))

        print(f"\n--- {name} Analysis ---")
        print(f"Average Magnitude: {np.mean(np.abs(signal)):.4f}")

        # Suggest a cutoff frequency
        if len(peaks) > 0:
            main_freq_component = xf[peaks[-1]]
            cutoff_freq = main_freq_component * 1.2
            print(f"Significant frequency components found at: {xf[peaks]} Hz")
            print(f"Suggested Cutoff Frequency: {cutoff_freq:.2f} Hz")
        else:
            print("No significant frequency peaks found.")
            cutoff_freq = fs / 4
            print(f"Suggested Cutoff Frequency (default): {cutoff_freq:.2f} Hz")

        # Add the power spectrum trace to the subplot
        row, col = (i // 2) + 1, (i % 2) + 1
        fig_fft.add_trace(
            go.Scatter(x=xf[:N//2], y=power_spectrum[:N//2], mode='lines', name=name),
            row=row, col=col
        )

    # Update layout for the FFT figure
    fig_fft.update_layout(
        title_text='Power Spectrums',
        showlegend=False,
        height=800
    )

    # --- Time Series Plotting ---
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(x=time, y=accel_x, mode='lines', name='Accel X'))
    fig_time.add_trace(go.Scatter(x=time, y=accel_y, mode='lines', name='Accel Y'))
    fig_time.add_trace(go.Scatter(x=time, y=accel_z, mode='lines', name='Accel Z'))
    fig_time.add_trace(go.Scatter(
        x=time, y=accel_mag, mode='lines', name='Total Accel Magnitude',
        line=dict(dash='dash', color='black')
    ))

    # Update layout for the time series figure
    fig_time.update_layout(
        title_text='Accelerometer Signals vs. Time',
        xaxis_title='Time (s)',
        yaxis_title='Acceleration',
        legend_title='Signal'
    )

    # Show both interactive figures
    fig_fft.show()
    fig_time.show()


if __name__ == '__main__':
    analyze_accel_data()
