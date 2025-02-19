## Purpose
This repo contains a demonstration of using different machine learning models to build a system that could support the 
diagnosis of Otitis Media.

The repo serves as a demonstration for the following work:

[A multimodal machine learning algorithm improved diagnostic accuracy for otitis media in a school aged aboriginal population](https://doi.org/10.1016/j.jbi.2025.104801)

----

## Structure

1. `%d_otitis_media_%s.ipynb`: Running the models with different numbers of output classes and settings
2. `image-utilities\`: Extracting images from videos, pre-processing the images for model training
3. `exp\`: Experiment scripts
4. `fragments\`: Code fragments that were used in the Thesis [Developing a deep learning algorithm to improve diagnosis of otitis media](https://theses.flinders.edu.au/view/be922010-567d-4d38-8aa9-d0dc26d4eacf/1)

----

## How to run
Scripts are built in python notebook format (`.ipynb`), using Python 3 language.
These files can be imported and run within any Jupyter environments.
- Run locally with the jupyter environment
   https://jupyter.org/install
- Run online with a hosted service like https://colab.google/

----

## License
Copyright 2021 Phu Nguyen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.