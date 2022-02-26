# Non-traditional approaches to input corruption in Collaborative Denoising AutoEncoders

Thesis MSc. in Computer Science & Engineering @ Politecnico di Milano, November 2020. Defended on December 15, 2020.

Mirrors:
- https://www.politesi.polimi.it/handle/10589/170595

## Abstract 
Recommender systems became part of everyday life. They are present when choosing what movie to watch on Netflix, what book to buy on Amazon, and even to find friends on Facebook. Progressively, people are turning to these systems to help them interact more effectively with overwhelming amounts of content, and find the information that is most valuable to them.

Deep learning-based recommendation systems has increased in the past recent years to overcome the limitations of traditional models, especially when dealing with the enormous volume, complexity and dynamics of data. One successful application is to consider the collaborative filtering from an autoencoder perspective in order to achieve better results.

The collaborative denoising autoencoder (CDAE) is a flexible top-n recommender that uses a corrupted version of the user-item interactions to make recommendations. The type of process used to perform the corruption is an important aspect that substantially affects the final representation learned from the input, and therefore has an impact in the performance.

There are two core purposes for this work. To begin with, it is aimed to replicate the results of the experiments for the collaborative denoising autoencoder (CDAE) model proposed by Wu Y., DuBois C., Zheng A. X. and Ester M. [13], so as to have a clear starting point for further analysis and be able to understand the components of this recommender. Moreover, it proposes novel ways to introduce noise to the input data supplied to the CDAE aiming to overcome the issues inherent in state-of-the-art choices for corruption and improve performance. Usually, in such traditional approaches the noise is stochastically applied to the input, either by some additive mechanism or randomly masking some of the input values, and so, it is hard to have control over it.

**Further details**
 - *Collaborative Denoising Autoencoder (CDAE) (Wu, Y., DuBois, C.,
   Zheng, A. X., & Ester, M. (2016, February)*. Collaborative denoising auto-encoders for top-n recommender systems. In Proceedings of the Ninth ACM International Conference on Web Search and Data Mining (pp. 153-162). ACM.)
   

## Setup details

### Datasets

Inside `src/apps/data/bin` the following binary encoded datasets are available:

 * `movielens_100k.bin`
 * `movielens_10m.bin`

Besides, also a default 80-20% split is present,

 * `movielens_100k.train.bin` 
 * `movielens_100k.test.bin`
 * `movielens_10m.train.bin` 
 * `movielens_10m.test.bin`

To add the non-encoded datasets, 

1. Create a folder in `src/apps/data` for each movielens dataset to be considered, as follows:

 * `movielens_100k_dataset/`
 * `movielens_10m_dataset/`
 * `movielens_1m_dataset/`

 2. Add the corresponding rating matrices at [https://www.google.com](https://www.google.com)

 * `movielens_100k_dataset/` -> `u.data`
 * `movielens_10m_dataset/` -> `ratings.dat`
 * `movielens_1m_dataset/` -> `ratings.dat`


### Testing

Run test file

    cd test 
      
Run Makefile, which generates the executable test main, with the command:

    make test (or VERBOSE=1 make test)
    
### Setup

Run main file

    cd src/apps 
      
Run Makefile, which generates the executable main, with the command:

    make test (or VERBOSE=1 make test)
   

