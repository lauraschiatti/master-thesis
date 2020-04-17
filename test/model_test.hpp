#include <iostream>
#include <numeric>
#include <algorithm>

#include "gtest/gtest.h"

#include <base/data.hpp>
// #include <model/linear_model.hpp>
// #include <model/factor_model.hpp>
// #include <model/evaluation.hpp>
// #include <model/recsys/popularity.hpp>
// #include <model/recsys/itemcf.hpp>
// #include <model/recsys/usercf.hpp>
// #include <model/recsys/bpr.hpp>
#include <model/recsys/cdae.hpp>
#include <solver/solver.hpp>
// #include <solver/sgd.hpp>

TEST(model, sample_movielens_data) {
  using namespace libcf;

  /*
  * load data
  */
  std::string movie_lens = "./test_data/sample_movielens_data.txt";
  
  int line_size = 4; // Dataset format: UserID::MovieID::Rating::Timestamp
  
  std::string sample_data(movie_lens);
  
  auto line_parser = [&](const std::string& line) {
    auto rets = split_line(line, ": ");
    CHECK_EQ(rets.size(), line_size);
    return std::vector<std::string>(std::make_move_iterator(rets.begin()),
                                    std::make_move_iterator(rets.begin() + 3));
  };
  
  Data data;
  data.load(sample_data, RECSYS, line_parser);
  
  Data train;
  Data test;
  data.random_split(train, test);
  LOG(INFO) << train;
  LOG(INFO) << test;

  /*
  * test models
  */

  // {
  //   LinearModelConfig lm_config;
  //   LinearModel lm(lm_config);
  //   SGDConfig sgd_config;
  //   sgd_config.learn_rate = 0.01;
  //   SGD<LinearModel> sgd(lm, sgd_config);
  //   sgd.train(train, test, {RMSE, MAE});
  // }

  // {
  //   FactorModelConfig fm_config;
  //   FactorModel fm(fm_config);
  //   SGDConfig sgd_config;
  //   sgd_config.learn_rate = 0.01;
  //   SGD<FactorModel> sgd(fm, sgd_config);
  //   sgd.train(train, test, {RMSE, MAE});
  // }

  // {
    // Popularity pop_model;
    // Solver<Popularity> solver(pop_model);
    // solver.train(train, test, {TOPN});
  // }

//   {
//     ItemCF itemcf_model(Jaccard, 50);
//     Solver<ItemCF> solver(itemcf_model);
//     solver.train(train, test, {TOPN});
//   }

//   {
//     UserCF itemcf_model(Jaccard, 50);
//     Solver<UserCF> solver(itemcf_model);
//     solver.train(train, test, {TOPN});
//   }

//   {
//     BPR bpr_model;
//     SGDConfig sgd_config;
//     sgd_config.learn_rate = 0.1;
//     SGD<BPR> sgd(bpr_model, sgd_config);
//     sgd.train(train, test, {TOPN});
//   }

}