<a name="readme-top"></a>



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
-->
<!--[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->

<div align="center">
  <img src="images/rustyOnix.png" alt="Logo" width="120" height="120">

  # RustONNX: Gotta Infer 'em All!

  <p align="center">
  Rust-based ONNX inference engine with a .onnx and .pb parser under the hood. Delivered with a set of validated operators, neural networks and inputs. Easy to add new operators to extend model compatibility. Dataset (images & labels) serialization can be used to extend the set of available inputs. Features Rayon-powered image-based and intra-network parallelization, Python bindings.
    <br/>
    TODO- Screenshot here
    <!--screenshot here-->
</p>
</div>



<!-- TABLE OF CONTENTS, TO DO AT THE END -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Why-this-logo-and-this-name-for-the-project">Why this logo and this name for the project</a></li>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <!-- <li><a href="#contributing">Contributing</a></li> -->
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

### Features

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Why this logo and this name for the project
We like playing with words.  
Onix is a Pokémon whose name is fairly similar to ONNX, so we chose it for the logo. 
On top of this, we made it rusty because our project is Rust based.  
The name of the project, "RustONNX: Gotta Infer 'em All!", includes another reference 
to the Pokémon series.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Supported models

| Model name           | Description                                                                        | Version | Opset version |
| -------------------- | ---------------------------------------------------------------------------------- | ------- | ------------- |
| **mnist-12**         | Handwritten digits recognition model specifically designed for the MNIST dataset.  | N/A     | 12            |
| **resnet18-v1-7**    | Image classification model with 18 layers for ImageNet-like datasets.              | 1       | 7             |
| **resnet34-v2-7**    | Image classification model with 34 layers for ImageNet-like datasets.              | 2       | 7             |
| **mobilenetv2-7**    | Efficient and lightweight image classification model for ImageNet-like datasets.   | 2       | 7             |

#### :warning: ONNX Model Zoo is undergoing some changes
During the development of this project, the ONNX Model Zoo was being expanded by incorporating additional models. Since new models had not been validated yet, the models supported by our _RustONNX: Gotta Infer 'em All!_ were taken from the set of _validated_ networks.

Please, be aware that new models may be more performant and that old models, included the ones supported by our _RustONNX: Gotta Infer 'em All!_, may be deleted in future updates of the ONNX Model Zoo.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Supported operators

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!--### Built With

This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section.

* [![Rust][Rust]][Rust-url]
* [![CLion][CLion]][CLion-url]

TO-COMPLETE

<p align="right">(<a href="#readme-top">back to top</a>)</p>-->

## How to install

TODO 

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
  ```sh
  npm install npm@latest -g
  ```

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
2. Install NPM packages
   ```sh
   npm install


### :warning: Attention to spaces in project path
Depending on the Rust toolchain you decide to use, please be aware that you may experience build-time errors caused by spaces in the project path.

To avoid any build issues, it's recommended to clone or place the project in a directory without spaces in its path.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES TODO -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Project structure

TODO

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make will benefit everybody else and are **greatly appreciated**.

If you're interested in opening an issue, please be sure to make it:

* _Scoped to a single topic_. One topic per issue.
* _Specific_. Include as many details as possible. 
* _Unique_. Do not duplicate existing opened issues: **search for the topic you're interested in before creating any issues**.
* _Reproducible_. If you're reporting a problem, include the necessary steps to reproduce it.

If you have a suggestion that would make this better, please fork the repo and create a pull request.

Don't forget to give the project a star! Thanks again!

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Authors

Claudio Tancredi (s292523):
- [LinkedIn profile](https://www.linkedin.com/in/claudio-tancredi/) 
- Email: s292523@studenti.polito.it

Francesca Russo (s287935):
- [LinkedIn profile](https://www.linkedin.com/in/francesca-russo-1a4a2b228/)
- Email: s287935@studenti.polito.it

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS TODO -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
<!--[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/ProgrammazioneDiSistema2023-IA-ZZ/Group01/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew -->
[product-screenshot]: images/screenshot.png
[Rust]: https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white
[Rust-url]: https://www.rust-lang.org/
[CLion]: https://img.shields.io/badge/CLion-%231572B6?style=for-the-badge&logo=clion&logoColor=white
[CLion-url]: https://www.jetbrains.com/clion/