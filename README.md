

Images have 4 channels
1. DAPI
2. WGA
3. Other
4. Granules

Segmentations:
- Cells are segmented using channels 1 DAPI and 2 WGA
- Nuclei are segmented using channel 1 DAPI
- Stress granules are segmented using channel 4
- Other label

We define 3 masks:
- Cells: cells without nuclei
- Particles: stress granules in cell and not in nuclei
- Cytosol: cells without nuclei and not particle

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
