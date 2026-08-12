This simple test is to ensure the turbulence simulator gives the correct WFE. The following process was taken:
	1. Set up the data logger
	2. Run the turbulence simulator 
	3. Log the DM
	4. Run script to find np.std(DM)
	5. Calculate strehl (use T and V in palila)
    6. Calculate WFE
    7. Repeat for a new turbulence amplitude 