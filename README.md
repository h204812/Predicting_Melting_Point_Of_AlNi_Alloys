steps 

1) Download NiAl_Mishin.eam.alloy
Link : https://www.ctcms.nist.gov/potentials/

2) Run the file : in.alni_melt.lmp
    script : lmp -in in.alni_melt.lmp

3) Create a csv file from dump.alni
     script : python analyze thermo.py

4) Download ovitio ( Need some strucutral information)
    Link : https://www.ovito.org/#download
4.1) 
   select these in Add mofications dropdown menu
   1) custom neighbour analysis
   2) Atomic Strain

4.2)
   1) Go to file option  and click
   2) click export file and save as Table of Global Attributes
   3) select the following before you save
       Timestep
      CommonNeighborAnalysis.counts.BCC
      CommonNeighborAnalysis.counts.FCC
      CommonNeighborAnalysis.counts.HCP
      CommonNeighborAnalysis.counts.OTHER

5) run the file combined_data.py
   script : python combined_data.py

Data is ready as final_ml_dataset.py
   
