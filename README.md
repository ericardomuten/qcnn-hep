<div align="center">
  
  # Quantum Convolutional Neural Networks<br>for High Energy Physics Analysis at the LHC
  
  <a href="https://summerofcode.withgoogle.com/projects/#5612096894533632" target="_blank"><img src="https://img.shields.io/badge/GSoC-2021-fbbc05?style=flat&logo=data%3Aimage%2Fpng%3Bbase64%2CiVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAMAAABHPGVmAAAALVBMVEVHcEz7vQD7vQD8vQD7vQD8vQD7vQD8vQD8vQD7vQD7vQD8vQD7vQD7vQD7vQAgxtLpAAAADnRSTlMAZvVQ6QrVPhl6oSmHvzL6LQUAAASGSURBVHjatdnZdusgDAVQELMY%2Fv9zb2%2Bwc%2BIKDzQLvTXB3gYBFqmaDVeKU4sCBlFyy43WqLjlBpR1BpR1BpR1xjoFxmIFBpSVBpSVBpSVBpSVBpQ1xvdK1oPgblhfOWltjNaJq7ddYT2IfImYJqMDrENUChGDZn%2FWQ%2FMHxBcD4BMyBc5XCHkNQTq60vfIgXAx5xByju6T8V8itsT3%2FUPi6r39Ce8rp%2FCWYrHfIDXs95FZJs%2FvTob6Z4T2buQE4eikvHeG%2FoZY7TpRfDsNWzrjtP0L4s12NYhh%2BO1ZjJ9HfOjdYGo3QZx7YvwEAgOPdx3eQJlArMFA3wXSZ%2BwMQvplJGoPY6sqNU0gxcGYUVx5jtSIx3oS6HysTxEbMMDPAmkM9iFSXnPXt8nwuQ%2FYI8TH%2F425TQe7%2FnBPEH2bECI6T4t%2Bgvh4N1istR50FJdeIX1Ek%2FqJdGGQOWmAa4u7rn18vuuIzUq52gbxvpiSuzIau%2BuO9FUUfTvvCjcoQ4MMltRnEOqF0pdD%2FwiBZWxoqGCn8r2VGKIUCHOoTyHK2g7y1bsJRRqNe3%2FlXv5GbNhWEWXxbsf1UITRF4kYcM4KiI%2FbeFIevNNq7P2EIg0bVL%2BfqCcyYV2rbDdExWSPjUPPGBRh9JTowTscW0Dqf%2BwLXGmPthgKKMJo1f1OSQ29hf1Mbdlmg5NFV1H7KoICA3mruIQ4vl4TTFhvuAlxxrdb1J55KMJoBatEPCv6mr3sJzK%2F9RQKDAx49Ji5ctSLwsxAxgyuiduOAeVtIG14zppPKtAka9lcMZz71IHyNoAcCpvIx6UfxGLleCim3ggUpe0dQhe7I86mWvQERZmCIocryAqPsdYOSQlVIjCgyMRbLSaXxi3GD4LEw4AipzCyyvS5a5ThMpJTGAYUuQljhiWL53R11FN5BxhQsK0UWbE747E7evGV2FaEAUWmDave0H4LQxg6nErl1IEBBRdmOzjkBPpdqFB%2BpUtUGb0tDKloZP44hQLthQoDwXYiXlowpMJIymExdARL8SViYzymhGEMFR%2FR3cOyNoRCpQcZFu1s6AsNhlQuSiJP%2B1Kk90dNRHW9BYyhwlszhNgdb05CjmGcKDb3DotAoYIYV9wWxjDSZcHNmN%2Fj0KpPm3R7dMjq7HlrSokvjIqjww3SEhb4XJDpg3CLvM9%2BPG%2FMHOcaOwzYRFScNe8QHJb9nOEDhvkGwV48eZC3BgfzWwSHZaXthKEVMvkMaQnKhKESzSCkJ37uQqlJ7RmCIcbr%2By5qUEjiIwQK3q4yZKHqYDxEUIo4U6%2BNahxKr0kEZwv8HC%2BDqo69UaI2ieBAujN2RNhOoPybQjBr9oNSKNXSoQ%2B2luCUQuk1iSCIg9oiZl24Vv8TtXLROaotAtO3%2F9ooWSFcjDnH6BQio2SZQSRz%2FpsPfsifQ2RY1tmNBM3oxQRCbRjkOZn%2FEACT2J%2B1vkZiGESyG1SZS%2FqJ1wTogE1hEFHNh9yNCbvvREwqCwwoawwoKw0oKw0oKw0oKw0oKw0oKw0oMFYqMFYqMFYqMBYq88Y%2FxB7wiOJRvWkAAAAASUVORK5CYII%3D" /></a>
  
  <a href="https://ml4sci.org/" target="_blank"><img alt="gsoc@ml4sci" height="200px" src="https://raw.githubusercontent.com/eraraya-ricardo/GSoC-QCNN/main/assets/gsoc%40ml4sci.jpeg" /></a>
    
</div>

## Organization

[Machine Learning for Science (ML4Sci)](https://ml4sci.org/)

## Mentors

[Prof. Sergei V. Gleyzer](http://sergeigleyzer.com/), [Dr. Emanuele Usai](https://orcid.org/0000-0001-9323-2107), and [Raphael Koh](https://www.raphaelkoh.me/)

## Abstract

One of the challenges in High-Energy Physics (HEP) is events classification, which is to predict whether an image of particle jets belongs to events being sought after or just background signals. Classical Convolutional Neural Network (CNN) has been proven a powerful algorithm in image classification, including jets image. As quantum computers promise many advantages over classical computing, comes a question on whether quantum machine learning (QML) can give any improvement in solving the problem.

This project aims to demonstrate quantum machine learning's potential, specifically Quantum Convolutional Neural Network (QCNN), in HEP events classification from image data. Although many previous works have tried to classify images with QCNN, none of them is fully quantum. They were still incorporating classical fully-connected layers after variational circuits. This project will be one of the first to try classifying images with a fully quantum implementation of QCNN and probably the first one to do so with particle jets images.
