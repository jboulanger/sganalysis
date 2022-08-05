# Stress Granule analysis

Python scripts to process stress granule screen.

Image are acquired on a Nikon wide-field system.

Processing is done on a HPC cluster.

## Installation
To run the code on a cluster, you might need to enable your slurm account and storage account.

Then you'll need to enable a password less access to the cluster typing  in a terminal ```ssh-keygen``` on your system (press enter to all questions) followed by ```ssh-copy-id -i ~/.ssh/id_rsa.pub username@host```

You need to install a python environement on the cluster. Connect in ssh to the host and download miniconda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
conda init tcsh
```
Then create an environement and install the packages
```
conda create -n sganalysis
conda activate sganalysis
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
conda install tifffile scikit-image pandas matplotlib seaborn
python -m pip install edt cellpose nd2reader
```

Download the ImageJ macro from this repository:
[Launch_SGA_jobs.ijm](https://raw.githubusercontent.com/jboulanger/sganalysis/master/Launch_SGA_jobs.ijm)

You need to connect your local system to the share drive accessible from the cluster using for example "connect to server" on a mac or "map network drive" in windows.

## Usage
To process the data, you need to follow several steps which are all done from the ImageJ macro by choosing the "Action" parameter in the macro. The other parameters will not change:
- Username : your username on the cluster (eg bob)
- Host: the access node that you used previoulsy
- Folder : the path to the nd2 files on your local system
- Local share: the local mounting point of the network drive (eg X: or /Volumes/bob)
- Remote share: the path to the drive on the cluster (eg /cephfs/bob)

Actions:

1. Install: downloads and copies the pythonscript from github into a "job" folder. This can also be used to update the script.
2. Scan: scans nd2 files located in the data folder and create a spreadsheet filelist.csv listing the files and field of views. You can open this file to remove the field of views that you want to discard. By default, the condition column is the well number but you can change it to the corresponding condition at this point or later before creating figures.
3. Config: configure the channel order
4. Process: process all the field of views listed in filelist.csv and export the results in a "result" folder as csv and vignette files. Each field of view is processed in separate job on the cluster.
5. Figure: collate all the csv files in the "result" folder into a cells.csv file and generate a boxplot for each measurements by condition as defined in the filelist.csv.
6. List Jobs: list the running jobs in a table

## Images and features

Images have 4 channels
1. nuclei (DAPI)
2. membrane (WGA)
3. other
4. granules

Segmentations:
- Cells are segmented using cellpose with membrane and nulcei markers
- Nuclei are segmented using cellpose with nuclei marker
- Stress granules are segmented using the granule channel
- Other label is segmented with the other channel

We define 3 masks:
- Cells: cells without nuclei
- Particles: stress granules in cell and not in nuclei

Measurement per cells:
1. Cell ID
2. area of the whole cell
3. area of the cell without nuclei
4. area of the particles in cell not in nuclei
5. area of the cytosol (in cell not particle not nuclei)
6. area ratio particles:cell
7. area ratio cytosol:cell
8. number of particle per cells
9. Mean intensity in cell of granule channel
10. Mean intensity in cell of other channel
11. Total intensity in cell of granule channel
12. Total intensity in cell of other channel
13. Mean intensity in cytosol of granule channel
14. Mean intensity in cytosol of other channel
15. Total intensity in cytosol of granule channel
16. Total intensity in cytosol of other channel
17. Mean intensity in particle of granule channel
18. Mean intensity in particle of other channel
19. Total intensity in particle of granule channel
20. Total intensity in particle of other channel
21. Mean intensity ratio of other particle:cytosol
22. Mean intensity ratio of granule particle:cytosol
23. Spearman granule:other
24. Pearson granule:other
25. Number of particle
26. Spread of particle (trace of the 2nd moment matrix)
27. Mean particle area
28. Mean particle perimeter
29. Mean particle distance to nuclei
30. Mean particle distance to edge
31. Mean particle circularity
32. Mean particle aspect ratio
33. Mean particle solidity
34. Mean particle roundness


Measurement per particle:
1. Particle ID
2. Cell ID
3. Mean intensity of channel granules
4. Total intensity of channel granules
5. Mean Intensity of granule ratio of particle:cytosol
6. Mean intensity of channel other
7. Total intensity of channel other
8. Mean Intensity of other ratio of particle:cytosol
9. Area
10. Perimeter
11. Distance to nuclei
12. Distance to edge of the cell
13. Circularity (4 PI area / perimeter)
14. Aspect ratio (minor axis/ major axis)
15. Solidity
16. Roundness (4 Area / (PI major_axis^2))

Note:
- There is not difference between particle and SG  the only difference is how we report the results : per cell or per granule
- Particle Perimeter/Particle Area = 1/circularity
