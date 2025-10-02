
import pyvista as pv
pv.set_jupyter_backend('static')

import os
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
import json
from datetime import datetime

import matplotlib.pyplot as p
import numpy as np
import xarray as xr
from IPython.display import Image, display
from pint.errors import UnitStrippedWarning

import cedalion
import cedalion.dataclasses as cdc
import cedalion.datasets
import cedalion.geometry.registration
import cedalion.geometry.segmentation
import cedalion.imagereco.forward_model as fw
import cedalion.imagereco.tissue_properties
import cedalion.io
import cedalion.plots
import cedalion.sigproc.quality as quality
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.vis.plot_sensitivity_matrix
from cedalion import units
from cedalion.imagereco.solver import pseudo_inverse_stacked
from cedalion.io.forward_model import FluenceFile, load_Adot

import cedalion.xrutils as xrutils
xrutils.unit_stripping_is_error()

# xr.set_options(display_expand_data=False);


def load_dataset_config(dataset_name, config_path="datasets/datasets_info.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Datasets configuration file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_datasets = json.load(f)
    
    if dataset_name not in all_datasets['datasets']:
        available = list(all_datasets['datasets'].keys())
        raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")
    
    return all_datasets['datasets'][dataset_name]

def list_available_datasets(config_path="datasets/datasets_info.json"):
    """List all available dataset configurations."""

    if not os.path.exists(config_path):
        print("No datasets.json file found.")
        return []
    
    with open(config_path, 'r', encoding='utf-8') as f:
        all_datasets = json.load(f)
    
    return list(all_datasets['datasets'].keys())

def calculate_total_hours(rec):
    try:
        time_data = rec['amp'].time.values
        total_seconds = float(time_data[-1] - time_data[0])
        total_hours = total_seconds / 3600
        return round(total_hours, 3)
    except Exception as e:
        print(f"Warning: Could not calculate total hours: {e}")
        return 0.0

def get_n_channels(rec):
    try:
        return int(rec['amp'].channel.size)
    except Exception as e:
        print(f"Warning: Could not extract number of channels: {e}")
        return 0

def get_landmark_names(geo3d):
    try:
        # valid landmarks: Nz, Iz, LPA, RPA which are obtained from head.landmarks.label
        # Get unique landmark labels
        landmarks = list(geo3d.label.values)
        # Filter to get only unique landmarks and sort them
        unique_landmarks = sorted(list(set(landmarks)))
        return ', '.join(unique_landmarks)
    except Exception as e:
        print(f"Warning: Could not extract landmark names: {e}")
        return 'Not available'

def get_wavelength_info(rec):
    try:
        wavelengths = rec['amp'].wavelength.values
        unique_wavelengths = sorted(list(set(wavelengths)))
        wavelength_names = ', '.join([f"{w}nm" for w in unique_wavelengths])
        return wavelength_names
    except Exception as e:
        print(f"Warning: Could not extract wavelength info: {e}")
        return 'Not available'

def get_trial_types_info(rec):
    try:
        event_stimulus = rec.stim.groupby("trial_type")[["onset"]].count()
        
        trial_info = []
        for trial_type, count in event_stimulus['onset'].items():
            trial_info.append(f"{trial_type}({count})")
        
        trial_types_str = ', '.join(trial_info)
        
        return trial_types_str
    except Exception as e:
        print(f"Warning: Could not extract trial types info: {e}")
        return 'Not available'

def get_n_trials(rec):
    try:
        total_trials = len(rec.stim)
        return total_trials
    except Exception as e:
        print(f"Warning: Could not extract number of trials: {e}")
        return 0

def log_dataset_info(dataset_info, rec, geo3d, log_file_path="datasets/dataset_analysis_log.csv"):
    """Log dataset information to CSV file.
        If data with the same ID exists, it will be overridden.
    
    Parameters:
    -----------
    dataset_info : dict
        Manual dataset information
    rec : dict
        Recording data from SNIRF file
    geo3d : xarray.DataArray
        Geometry data with landmark information
    log_file_path : str
        Path to the log file
    """
    
    # Calculate extracted information and override defaults if data is available
    if rec is not None and geo3d is not None:
        n_channels = get_n_channels(rec)
        total_hrs = calculate_total_hours(rec)
        landmarks = get_landmark_names(geo3d)
        wavelength_names = get_wavelength_info(rec)
        trial_types = get_trial_types_info(rec)
        n_trials = get_n_trials(rec)
    else:
        # Use default values from dataset_info when data is not available
        n_channels = dataset_info['n_channels']
        total_hrs = dataset_info['total_hrs']
        landmarks = dataset_info['landmarks']
        wavelength_names = dataset_info['wavelength_names']
        trial_types = dataset_info['trial_types']
        n_trials = dataset_info.get('n_trials', 'Unknown')
    
    # Prepare the log entry
    log_entry = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ID': dataset_info['ID'],
        'dataset_title': dataset_info['dataset_title'],
        'paradigm_purpose': dataset_info['paradigm_purpose'],
        'modalities': dataset_info['modalities'],
        'n_subjects': dataset_info['n_subjects'],
        'n_channels': n_channels,
        'total_hrs': total_hrs,
        'head_coverage': dataset_info['head_coverage'],
        'organization': dataset_info['organization'],
        'authors': dataset_info['authors'],
        'upload_date': dataset_info['upload_date'],
        'data_link': dataset_info['data_link'],
        'publication': dataset_info['publication'],
        'aquisition_system': dataset_info['aquisition_system'],
        'format': dataset_info['format'],
        'additional_details': dataset_info['additional_details'],
        'landmarks': landmarks,
        'wavelength_names': wavelength_names,
        'trial_types': trial_types,
        'n_trials': n_trials
    }
    
    # Check if file exists and read existing data
    file_exists = os.path.exists(log_file_path)
    existing_data = []
    dataset_updated = False
    
    if file_exists:
        # Read existing data and filter out entries with the same ID
        with open(log_file_path, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['ID'] != dataset_info['ID']:
                    existing_data.append(row)
                else:
                    dataset_updated = True
                    print(f"Overriding existing entry for dataset: {dataset_info['ID']}")
    
    # Write all data (existing + new/updated entry)
    with open(log_file_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(log_entry.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write existing data (excluding the overridden entry)
        for row in existing_data:
            writer.writerow(row)
        
        # Write the new/updated entry
        writer.writerow(log_entry)
        
        if not file_exists:
            print(f"Created new log file: {log_file_path}")
        elif dataset_updated:
            print(f"Dataset entry updated in: {log_file_path}")
        else:
            print(f"New dataset entry added to: {log_file_path}")
    
    # Print summary
    print("\n=== Dataset Information Summary ===")
    for key, value in log_entry.items():
        if key != 'timestamp':
            print(f"{key}: {value}")
    print("===================================\n")


HEAD_MODEL = "colin27" # "icbm152"
FORWARD_MODEL = "NIRFASTER" # "MCX"
VISUALIZE = False  # Set to False to skip all visualizations
SAVE_FIG = False
IMAGE_RECON = False

JSON_PATH = "datasets/datasets_info.json"
LOG_FILE_PATH = "datasets/dataset_analysis_log.csv"

# TO be configured for each dataset
dataset_name = "openneuro_spatial_attention_decoding" # could be different from snirf file name
snirf_path = os.path.join("datasets/openneuro_spatial_attention_decoding/sub-08_task-03_nirs.snirf")


try:
    DATASET_INFO = load_dataset_config(dataset_name, JSON_PATH)
    snirf_path = snirf_path
    print(f"Loaded configuration for dataset: {DATASET_INFO['ID']}")
except (FileNotFoundError, KeyError) as e:
    print(f"Error: {e}")
    print("Available datasets:")
    for dataset in list_available_datasets("datasets/datasets_info.json"):
        print(f"  - {dataset}")
    exit(1)


# Load the data
try:
    recordings = cedalion.io.read_snirf(snirf_path)
    rec = recordings[0]
    geo3d = rec.geo3d
    log_dataset_info(DATASET_INFO, rec, geo3d)
        
except Exception as e:
    print(f"Error loading data: {e}")
    print("Logging dataset info with default values...")
    # Log with default values when data loading fails
    log_dataset_info(DATASET_INFO, None, None)
    print("Exiting due to data loading failure.")
    exit(1)



# check the wavelength
print(rec['amp'].wavelength)
rec['amp'] = rec['amp'].assign_coords(wavelength=[760, 850])

meas_list = rec._measurement_lists['amp'].copy()
# wavelength_mapping = {785: 760, 830: 850}
wavelength_mapping = {757: 760, 844: 850}
meas_list['wavelength'] = meas_list['wavelength'].map(wavelength_mapping)
rec._measurement_lists['amp'] = meas_list

print(rec._measurement_lists['amp'].wavelength.unique())


# check the landmarks
geo3d = rec.geo3d

# all(label in geo3d.label.data for label in ["Nz", "Iz", "LPA", "RPA", "Cz"])
for label in ["Nz", "Iz", "LPA", "RPA", "Cz", "NASION"]:
    print(f"- {label}: {label in geo3d.label.data}")

#  match label names
subject_nasion_mask = geo3d['label'].data == 'NASION'
geo3d[subject_nasion_mask]

new_labels = geo3d['label'].data.copy()
new_labels[subject_nasion_mask] = 'Nz'

# Create new geo3d with updated labels
geo3d = geo3d.assign_coords(label=new_labels)
print(geo3d['label'].data)

# Vizualize the montage
fig = cedalion.plots.plot_montage3D(rec["amp"], geo3d)
    
print(rec['amp'].source.size, rec['amp'].detector.size, rec['amp'].channel.size )
print(rec['amp'].channel)

for i, (type, x) in enumerate(geo3d.groupby("type")):
    print(type)
    print(x)


# check event stimulus
event_stimulus = rec.stim.groupby("trial_type")[["onset"]].count()
print(event_stimulus)

trial_types = rec.stim.trial_type.unique().tolist()
print(trial_types)
    
# Preprocessing
# 1. Convert to optical density
# convert to optical density
rec["od"] = cedalion.nirs.int2od(rec["amp"])

# 2. Motion correction
rec["od_tddr"] = motion_correct.tddr(rec["od"])

# 3. Wavelet-based motion correction
rec["od_wavelet"] = motion_correct.wavelet(rec["od_tddr"])

# 4. Bandpass filter
rec["od_freqfiltered"] = rec["od_wavelet"].cd.freq_filter(
    fmin=0.01, fmax=0.5, butter_order=4
)


# Block averaging (in optical density)
epochs = rec["od_freqfiltered"].cd.to_epochs(
    rec.stim,  # stimulus dataframe
    trial_types = trial_types,  #["Left", "Right"], select fingertapping events, discard others
    before=5 * units.s,  # seconds before stimulus
    after=30 * units.s,  # seconds after stimulus
)

# Baseline correction
print(epochs.reltime)

# calculate baseline
baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")

# subtract baseline
epochs_blcorrected = epochs - baseline

# group trials by trial_type. For each group individually average the epoch dimension
blockaverage = epochs_blcorrected.groupby("trial_type").mean("epoch")

# vizualise
if VISUALIZE:
    noPlts2 = int(np.ceil(np.sqrt(len(blockaverage.channel))))
    f,ax = p.subplots(noPlts2,noPlts2, figsize=(12,10))
    ax = ax.flatten()
    for i_ch, ch in enumerate(blockaverage.channel):
        for ls, trial_type in zip(["-", "--"], blockaverage.trial_type):
            ax[i_ch].plot(blockaverage.reltime, blockaverage.sel(wavelength=760, trial_type=trial_type, channel=ch), "r", lw=2, ls=ls)
            ax[i_ch].plot(blockaverage.reltime, blockaverage.sel(wavelength=850, trial_type=trial_type, channel=ch), "b", lw=2, ls=ls)

        ax[i_ch].grid(1)
        ax[i_ch].set_title(ch.values)
        ax[i_ch].set_ylim(-.02, .02)
        ax[i_ch].set_axis_off()
        ax[i_ch].axhline(0, c="k")
        ax[i_ch].axvline(0, c="k")

    p.suptitle("760nm: r | 850nm: b | left: - | right: --")
    p.tight_layout()
    p.show()


if IMAGE_RECON:
    
    # Head modeling and co-registration
    # 1. Load segmentation, masks and landmarks
    if HEAD_MODEL == "colin27":
        SEG_DATADIR, mask_files, landmarks_file = cedalion.datasets.get_colin27_segmentation()
        PARCEL_DIR = cedalion.datasets.get_colin27_parcel_file()
    elif HEAD_MODEL == "icbm152":
        SEG_DATADIR, mask_files, landmarks_file = cedalion.datasets.get_icbm152_segmentation()
        PARCEL_DIR = cedalion.datasets.get_icbm152_parcel_file()
    else:
        raise ValueError(f"Unknown HEAD_MODEL: {HEAD_MODEL}")

    # 2. Compute head model
    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
        scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
        landmarks_ras_file=landmarks_file,
        parcel_file=PARCEL_DIR,
        brain_face_count=None,
        scalp_face_count=None
    )

    print(head.landmarks.label)

    # 3. Optode geometry registration
    # Snap to scalp (coregister)
    geo3d_snapped_ijk = head.align_and_snap_to_scalp(geo3d) # trans_rot_isoscale, or general

    # plot the registered montage
    if VISUALIZE:
        plt = pv.Plotter()
        cedalion.plots.plot_surface(plt, head.brain, color="w")
        cedalion.plots.plot_surface(plt, head.scalp, opacity=.1)
        cedalion.plots.plot_labeled_points(plt, geo3d_snapped_ijk)
        plt.show()


    
    # Forward model
    # 1. Create forward model (simulate light propagation in tissue)
    fwm = cedalion.imagereco.forward_model.ForwardModel(head, geo3d_snapped_ijk, meas_list)

    temporary_directory = TemporaryDirectory()
    tmp_dir_path = Path(temporary_directory.name)

    fluence_fname = tmp_dir_path / "fluence.h5"

    if FORWARD_MODEL == "NIRFASTER":
        fwm.compute_fluence_nirfaster(fluence_fname)
    elif FORWARD_MODEL == "MCX":
        fwm.compute_fluence_mcx(fluence_fname)
    else:
        raise ValueError(f"Unknown FORWARD_MODEL: {FORWARD_MODEL}")

    
    ## Plot fluence
    # # for plotting use a geo3d without the landmarks
    # geo3d_plot = geo3d_snapped_ijk[geo3d_snapped_ijk.type != cdc.PointType.LANDMARK]
    geo3d_plot = geo3d_snapped_ijk

    if VISUALIZE:
        plt = pv.Plotter()

        src, det, wl = "S1", "D1", 760

        # fluence_file.get_fluence returns a 3D numpy array with the fluence
        # for a specified source and wavelength.
        with FluenceFile(fluence_fname) as fluence_file:
            f = fluence_file.get_fluence(src, wl) * fluence_file.get_fluence(det, wl)

        f[f <= 0] = f[f > 0].min()
        f = np.log10(f)
        vf = pv.wrap(f)

        plt.add_volume(
            vf,
            log_scale=False,
            cmap="plasma_r",
            clim=(-10, 0),
        )
        cedalion.plots.plot_surface(plt, head.brain, color="w")
        cedalion.plots.plot_labeled_points(plt, geo3d_plot, show_labels=False)

        cog = head.brain.vertices.mean("label")
        cog = cog.pint.dequantify().values
        plt.camera.position = cog + [-300, 30, 100]
        plt.camera.focal_point = cog
        plt.camera.up = [0, 0, 1]

        plt.show()

    
    # Calcualte sensitivity matrices
    sensitivity_fname = tmp_dir_path / "sensitivity.h5"
    fwm.compute_sensitivity(fluence_fname, sensitivity_fname)

    # load and display sensitivity matrix
    Adot = load_Adot(sensitivity_fname)

    # Plot sensitivity matrix
    if VISUALIZE:
        plotter = cedalion.vis.plot_sensitivity_matrix.Main(
            sensitivity=Adot,
            brain_surface=head.brain,
            head_surface=head.scalp,
            labeled_points=geo3d_plot,
        )
        plotter.plot(high_th=0, low_th=-3)
        plotter.plt.show()


    
    # Inverse sensitivity matrix

    # Compute stacked sensitivity matrix
    Adot_stacked = fwm.compute_stacked_sensitivity(Adot)

    # Compute inverse matrix (with Tikhonov regularization?)
    B = pseudo_inverse_stacked(Adot_stacked, alpha = 0.01, alpha_spatial = 0.001)

    # Apply inverse matrix to data
    dC_brain, dC_scalp = fw.apply_inv_sensitivity(blockaverage, B)

    
    # helper function to display gifs in rendered notbooks
    def display_image(fname : str):
        display(Image(data=open(fname,'rb').read(), format='png'))

    
    # Plot results on brain and scalp
    # Channel space
    if VISUALIZE:
        from cedalion.plots import scalp_plot_gif

        # configure the plot
        data_ts = blockaverage.sel(wavelength=850, trial_type=trial_types[0])
        data_ts = data_ts.rename({"reltime": "time"})

        filename_scalp = "scalp_plot_ts"

        # call plot function
        scalp_plot_gif(
            data_ts,
            geo3d,
            filename=filename_scalp,
            time_range=(-5, 30, 0.5) * units.s,
            scl=(-0.01, 0.01),
            fps=6,
            optode_size=6,
            optode_labels=True,
            str_title="OD 850 nm",
        )
        display_image(f"{filename_scalp}.gif")

     
    # Image space plot
    # Single-View Animations of Activitations on the Brain
    if VISUALIZE:
        from cedalion.plots import image_recon_view

        filename_view = 'image_recon_view'

        X_ts = xr.concat([dC_brain.sel(trial_type=trial_types[0]), dC_scalp.sel(trial_type=trial_types[0])], dim="vertex")
        X_ts = X_ts.rename({"reltime": "time"})
        X_ts = X_ts.transpose("vertex", "chromo", "time")
        X_ts = X_ts.assign_coords(is_brain=('vertex', Adot.is_brain.values))

        scl = np.percentile(np.abs(X_ts.sel(chromo='HbO').values.reshape(-1)),99)
        clim = (-scl,scl)

        image_recon_view(
            X_ts,  # time series data; can be 2D (static) or 3D (dynamic)
            head,
            cmap='seismic',
            clim=clim,
            view_type='hbo_brain',
            view_position='left',
            title_str='HbO / uM',
            filename=filename_view,
            save=SAVE_FIG,
            time_range=(-5,30,0.5)*units.s,
            fps=6,
            geo3d_plot = geo3d_plot,
            wdw_size = (1024, 768)
        )

    
    #  Select a single time point and plot activity as a still image at that time
    if VISUALIZE:
        X_ts_plot = X_ts.sel(time=4, method="nearest") # note: sel does not accept quantified units

        filename_view = 'image_recon_view_still'

        image_recon_view(
            X_ts_plot,  # time series data; can be 2D (static) or 3D (dynamic)
            head,
            cmap='seismic',
            clim=clim,
            view_type='hbo_brain',
            view_position='left',
            title_str='HbO / uM',
            filename=filename_view,
            save=SAVE_FIG,
            time_range=(-5,30,0.5)*units.s,
            fps=6,
            geo3d_plot = geo3d_plot,
            wdw_size = (1024, 768)
        )

        display_image(f"{filename_view}.png")

    
    # Multi-View Animations of Activitations on the Brain
    if VISUALIZE:
        from cedalion.plots import image_recon_multi_view

        filename_multiview = 'image_recon_multiview'

        # prepare data
        X_ts = xr.concat([dC_brain.sel(trial_type=trial_types[0]), dC_scalp.sel(trial_type=trial_types[0])], dim="vertex")
        X_ts = X_ts.rename({"reltime": "time"})
        X_ts = X_ts.transpose("vertex", "chromo", "time")
        X_ts = X_ts.assign_coords(is_brain=('vertex', Adot.is_brain.values))

        scl = np.percentile(np.abs(X_ts.sel(chromo='HbO').values.reshape(-1)),99)
        clim = (-scl,scl)


        image_recon_multi_view(
            X_ts,  # time series data; can be 2D (static) or 3D (dynamic)
            head,
            cmap='seismic',
            clim=clim,
            view_type='hbo_brain',
            title_str='HbO / uM',
            filename=filename_multiview,
            save=SAVE_FIG,
            time_range=(-5,30,0.5)*units.s,
            fps=6,
            geo3d_plot = None, #  geo3d_plot
            wdw_size = (1024, 768)
        )
        display_image(f"{filename_multiview}.gif")

    
    # Parcel Space Plot
    # subtract baseline
    baseline =  dC_brain.isel(reltime=0) # first sample along reltime dimension
    dC_brain_blsubtracted = dC_brain - baseline

    # average over parcels
    avg_HbO = dC_brain_blsubtracted.sel(chromo="HbO").groupby('parcel').mean()

    avg_HbR = dC_brain_blsubtracted.sel(chromo="HbR").groupby('parcel').mean()

    display(dC_brain_blsubtracted.rename("dC_brain_blsubtracted"))
    display(avg_HbO.rename("avg_HbO"))

    selected_parcels = [
        "SomMotA_1_LH", "SomMotA_3_LH", "SomMotA_4_LH", "SomMotA_5_LH", "SomMotA_9_LH", "SomMotA_10_LH",
        "SomMotA_1_RH", "SomMotA_2_RH", "SomMotA_3_RH", "SomMotA_4_RH", "SomMotA_6_RH", "SomMotA_7_RH"
    ]

    # map parcel labels to colors
    parcel_colors = {
        parcel: p.cm.jet(i / (len(selected_parcels) - 1))
        for i, parcel in enumerate(selected_parcels)
    }

    if VISUALIZE:
        # assign colors to vertices
        b = cdc.VTKSurface.from_trimeshsurface(head.brain)
        b = pv.wrap(b.mesh)
        b["parcels"] = np.asarray([
            parcel_colors.get(parcel, (0.8, 0.8, 0.8, .3))
            for parcel in head.brain.vertices.parcel.values
        ])

        plt = pv.Plotter()

        plt.add_mesh(
            b,
            scalars="parcels",
            rgb=True,
            smooth_shading=False
        )

        cedalion.plots.plot_labeled_points(plt, geo3d_plot)

        legends = [a for a in parcel_colors.items()]
        plt.add_legend(labels= legends, face='o', size=(0.3,0.3))

        cog = head.brain.vertices.mean("label")
        cog = cog.pint.dequantify().values

        plt.camera.position = cog + [0,0,400]
        plt.camera.focal_point = cog
        plt.camera.up = [0,1,0]
        plt.reset_camera()

        plt.show()


    
    # Plot averaged time traces in each parcel for the event conditions
    if VISUALIZE:
        f,ax = p.subplots(2,6, figsize=(20,5))
        ax = ax.flatten()
        for i_par, par in enumerate(selected_parcels):
            ax[i_par].plot(avg_HbO.sel(parcel = par, trial_type = trial_types[0]).reltime, avg_HbO.sel(parcel = par, trial_type = trial_types[0]).values, "r", lw=2, ls='-')
            ax[i_par].plot(avg_HbR.sel(parcel = par, trial_type = trial_types[0]).reltime, avg_HbR.sel(parcel = par, trial_type = trial_types[0]).values, "b", lw=2, ls='-')

            ax[i_par].grid(1)
            ax[i_par].set_title(par, color=parcel_colors[par])
            ax[i_par].set_ylim(-.05, .2)

        p.suptitle(f"Parcellations: HbO: r | HbR: b | {trial_types[0]}", y=1)
        p.tight_layout()
        p.show()

        f,ax = p.subplots(2,6, figsize=(20,5))
        ax = ax.flatten()
        for i_par, par in enumerate(selected_parcels):
            ax[i_par].plot(avg_HbO.sel(parcel = par, trial_type = trial_types[1]).reltime, avg_HbO.sel(parcel = par, trial_type = trial_types[1]).values, "r", lw=2, ls='-')
            ax[i_par].plot(avg_HbR.sel(parcel = par, trial_type = trial_types[1]).reltime, avg_HbR.sel(parcel = par, trial_type = trial_types[1]).values, "b", lw=2, ls='-')

            ax[i_par].grid(1)
            ax[i_par].set_title(par, color=parcel_colors[par])
            ax[i_par].set_ylim(-.05, .2)

        p.suptitle(f"Parcellations: HbO: r | HbR: b | {trial_types[1]}", y=1)
        p.tight_layout()
        p.show()
        
    temporary_directory.cleanup()
    
        

print("---DONE---")