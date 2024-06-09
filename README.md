# af_tools

A suite of tools designed for the analysis of single/multiple AlphaFold 3
and ColabFold predictions:

- Plotting pLDDT (per residue and atom) for individual predictions
- Plotting PAE (Predicted Aligned Error) for individual predictions
- Generating mean pLDDT per model histogram for multiple predictions
- Calculating RMSDs relative to a reference of multiple predictions using multiprocessing
- Plotting RMSD versus pLDDT and performing clustering with HDBSCAN
- Performing random subsampling of ```.a3m``` multiple sequence alignments for ColabFold
