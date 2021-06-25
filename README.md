<!-- PROJECT LOGO -->
<br />
<p align="center">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">Recommender for restaurants</h3>

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
        <li><a href="#Initializing_data">Initializing data</a></li>
        <li><a href="#Analysis">Analysis</a></li>
        <li><a href="#Collaborative">Collaborative</a></li>
        <li><a href="#Content-Based">Content Based</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

In this repository you will find all the materials of the experimental final project done in collaboration wit the owner of the restaurant app Velada. We will describe in detail how we built a recommender for the app, trying to give attention not just on the restaurant suggested, but also on the moment of the suggestion. We will analyze the data, giving attention to all the aspect related to the needs of a recommender system. We will show all the models we tested, mainly using a Content Based model and a Collaborative Filter. We will describe a very first approach to the the temporal behavior and the recurrent activities of users. You will find the following folders and files:


### Repository structure

* [The data](TFM_Granatiero_Data)
* [The notebooks](TFM_Granatiero_Notebooks)
* [The programs](TFM_Granatiero_Notebooks/TFM_Granatiero_Utils)
* [The report](TFM_DS_GRANATIERO.pdf)

<!-- The Data -->
## The data

The first thing you can notice is that this folder is empty. Indeed the data of this project are propriety of the app Velada and can't be public. The data are indispensable to run the models contained in the notebook. The user interested can contact me at pablogranatiero@gmail.com

<!-- notebooks -->
## Notebooks
In the five notebooks inside the repository you will find the data analysis, the preprocess, all the experiments performed with the recommenders, the analysis and the codes of the time recommender.

### Initializing_data 

In the TFM_Analysis notebook there are all the code to preprocess the raw data coming from the two main sources: Google Analytics and Velada app itself.

### Analysis
In this commented notebook are shown all the steps we did to retrieve all the information and plots useful in order to build the recommenders and to take important desicions.

### Collaborative
Here you will find described all the experiments we performed with the Collaborative Filter model. In particular we tested the model:

1. Rising the minimum number of likes

2. Introducing a neighbor parameter

3. Trying to add features

4. Rising the volume of data 

### Content Based
In TFM_Content_Based you will find described all the experiments we performed with the Content Based model. In particular we tested the model:

1. Rising the minimum number of likes

2. Introducing a neighbor parameter

3. Trying to add features

4. Rising the volume of data 
<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/github_username/repo_name/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email

Project Link: [https://github.com/github_username/repo_name](https://github.com/github_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* []()
* []()
* []()





<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/github_username/repo.svg?style=for-the-badge
[contributors-url]: https://github.com/github_username/repo/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/github_username/repo.svg?style=for-the-badge
[forks-url]: https://github.com/github_username/repo/network/members
[stars-shield]: https://img.shields.io/github/stars/github_username/repo.svg?style=for-the-badge
[stars-url]: https://github.com/github_username/repo/stargazers
[issues-shield]: https://img.shields.io/github/issues/github_username/repo.svg?style=for-the-badge
[issues-url]: https://github.com/github_username/repo/issues
[license-shield]: https://img.shields.io/github/license/github_username/repo.svg?style=for-the-badge
[license-url]: https://github.com/github_username/repo/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/github_username
