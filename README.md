## Welcome to Angela's Software Portfolio!

### About me
My name is Angela, and I graduated with a B.S. in Chemical Engineering (Energy and Environment concentration) from UC Berkeley. I also (almost) hold a Master's degree in Molecular Science and Software Engineering. With these two degrees, I hope to be a liaison between both classical science and data science, and apply physical significance to results obtained via software solutions. Currently, my career field of choice is in the clean energy space!

In my spare time, I love dancing, doing arts and crafts, and gaming! I also thoroughly enjoy trying out new recipes and playing the daily Wordle.

### Software packages

#### [Reproduction of Smith et al 2017](https://github.com/angelahou/angelahou.github.io/tree/main/ANI_1)
As part of a group project in our Machine Learning class, we attempted to reproduce the results of [Smith et al 2017](https://www.nature.com/articles/sdata2017193), which developed a neural network potential for organic molecules called ANI_1. We were able to recreate the atomization energy vectors (AEV's) and begin training on the dataset. However, this implementation is limited by its speed -- the project scope did not allow for parallelization, so even though we had access to Cori at NERSC, we were not able to use it to its full potential and train on all the data.

#### [Analyzing global video game sales](https://github.com/angelahou/angelahou.github.io/tree/main/vg_sales_analysis)
[Camille Huang](https://github.com/cmshuang) and I analyzed a dataset containing video game sales, release platform, and genre to discover global sales trends, and then predict a video game's genre based on its supporting data. For game developers, knowing which genres are popular in which regions could be helpful with marketing; for consumers, knowing what genres are enjoyable can help with the search for more interesting or fun content. The results of this analysis reflect the limitations of current video game genre classification.

#### [Quantum computing with CNDO/2](https://github.com/angelahou/angelahou.github.io/tree/main/CNDO2)
In our Numerical Algorithms appled to Computational Quantum Chemistry class, we learned about and implemented the Complete Neglect of Differential Overlap (CNDO/2) method, which aims to predict a molecule's energy given its Cartesian coordinates. CNDO/2 only includes valence electrons in its analysis, and implements the zero-differential overlap assumption. This project focused on evaluating hydrocarbons but can potentially be extended to molecules containing other atoms. My [implementation](https://docs.google.com/presentation/d/1Rk6E4Uifkgr0W22fsw7yY3rAkx2it7cbWLR5DomoyQY/edit?usp=sharing) focused on serial performance optimization, i.e. cutting down the time to generate the Fock and density matrices, as well as speed up matrix multiplication.

### Drop me a line!

Feel free to email me at angela.hou@berkeley.edu or drop me a message on [LinkedIn](https://www.linkedin.com/in/angela-hou-3b444817b/)!
