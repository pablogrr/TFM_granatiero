<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="logo_velada.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">A restaurant recommender system for a new-born app-based gastronomic guide</h3>

  <p align="center">
    FUNDAMENTAL PRINCIPLES OF DATA SCIENCE MASTER’S THESIS - Pablo Granatiero 
    
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Repository structure</a></li>
      </ul>
    </li>
        <li><a href="#the-data">The data</a></li>
    <li>
      <a href="#notebooks">The notebooks</a>
      <ul>
        <li><a href="#Initializing-data">Initializing data</a></li>
        <li><a href="#Analysis">Analysis</a></li>
        <li><a href="#Collaborative">Collaborative</a></li>
        <li><a href="#Content-Based">Content Based</a></li>
        <li><a href="#Models-comparison">Models comparison</a></li>
        <li><a href="#Time-Evolution">Time evolution</a></li>          
      </ul>
    </li>
    <li><a href="#The-codes">The Codes</a></li>
    <li><a href="#The-report">The report</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this repository you will find all the materials of the experimental final project done in collaboration with the Product Officer of the gastronomic app Velada. We will describe in detail how we built a recommender for the app, trying to give attention not just on the restaurant suggested, but also on the moment of the suggestion. We will analyze the data, in particular all the aspects related to the needs of a recommender system. We will show all the models we tested, mainly using a Content Based model and a Collaborative Filtering. We will describe a very first approach to the the temporal behavior and the recurrent activities of users. You will find the following folders and files:


### Repository structure

* [The data](TFM_Granatiero_Data)
* [The notebooks](TFM_Granatiero_Notebooks)
* [The programs](TFM_Granatiero_Notebooks/TFM_Granatiero_Utils)
* [The report](TFM_REPORT_GRANATIERO.pdf)

<!-- The Data -->
## The data

The first thing you can notice is that this folder is empty. Indeed the data of this project are propriety of the app Velada and can't be public. The data are indispensable to run the models contained in the notebook. The user interested can contact me at pablogranatiero@gmail.com

<!-- notebooks -->
## Notebooks
In the five notebooks inside the repository you will find the data analysis, the preprocess, all the experiments performed with the recommenders, the analysis and a very first approach trying to manage recurrent restaurants.

### Initializing data 

In the <a href="TFM_Granatiero_Notebooks/TFM_Initializing_data.ipynb"> TFM_Initializing_data notebook </a> there are all the code to preprocess the raw data coming from the two main sources: Google Analytics and Velada app itself.

### Analysis
In this <a href="TFM_Granatiero_Notebooks/TFM_Analysis.ipynb"> commented notebook </a> are shown all the steps we did to retrieve all the information and plots useful in order to build the recommenders and to take important desicions.

### Collaborative
<a href="TFM_Granatiero_Notebooks/TFM_Collaborative.ipynb"> Here </a> you will find described all the experiments we performed with the Collaborative Filtering model. In particular we tested the model:

1. Rising the minimum number of likes

2. Introducing a neighbor parameter

3. Trying to add features

4. Rising the volume of data 

### Content Based
In <a href="TFM_Granatiero_Notebooks/TFM_Content_Based.ipynb"> TFM_Content_Based </a> you will find described all the experiments we performed with the Content Based model. In particular we tested the model:

1. In the binary case

2. Inspecting feature's importance

3. Adding restaurants' rates thanks to a qualitative assignation rule

### Models comparison
<a href="TFM_Granatiero_Notebooks/TFM_all_models_comparison.ipynb"> Here </a> is described how we compare together the best setup for the Collaborative and for the Content Based, comparing them with two base models: a totally random one and another simply based on ranking (we called it Most Popular model).  

### Time Evolution
In this <a href="TFM_Granatiero_Notebooks/TFM_Time_Evolution.ipynb"> notebook </a> there is a very first approach to the recurrency of restaurants and the introduction of time parameter in the recommender. 

<!-- CODES -->
## The Codes 

In this <a href="TFM_Granatiero_Notebooks/TFM_Granatiero_Utils/"> folder </a> you will find all the codes needed to make the models properly work. In particular:

1. The codes to make the preprocess

3. The program to test the base models

3. The codes of the Collabaritve Filtering in all its versions

4. The codes of the Content Based Model in all its versions
 
<!-- Report -->
## The report

Finally this is the <a href="TFM_REPORT_GRANATIERO.pdf/">  report </a>. A detailed description of the whole project in six chapters:

1. Intro
2. The App
3. The Data
4. The Recommenders
5. Future implementations
6. Conclusion

