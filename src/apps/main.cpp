#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

#include <glog/logging.h>
#include <gflags/gflags.h>

#include <base/data.hpp>
#include <base/io.hpp>
#include <base/io/file.hpp>
#include <base/timer.hpp>
#include <base/random.hpp>

#include <solver/solver.hpp>
#include <model/recsys/cdae.hpp>


/**
 * Data files
*/ 
std::string dataset_dir = "data/";
std::string dataset_bin_dir = "data/bin/";

std::string dataset = "movielens_100k";
const char * log_file = "log/movielens_100k_implicit.log";
std::string dataset_filepath = dataset_dir + dataset + "_dataset/u.data";

// std::string dataset = "movielens_10m"; // movielens_1m
// const char * log_file = "log/movielens_10m_implicit.log";
// std::string dataset_filepath = dataset_dir + dataset + "_dataset/ratings.dat";

// std::string dataset = "politic_old"; // politic_old
// const char * log_file = "log/politic_old_implicit.log";
// std::string dataset_filepath = dataset_dir + dataset + "_dataset/politic_old.txt";

// use sample dataset
// std::string dataset_filepath = dataset_dir + dataset + "_dataset/" + "s_" + dataset + "_data.txt";
// std::string dataset_filepath = dataset_dir + dataset + "_dataset/ratings.csv";
// use whole dataset
// std::string dataset_filepath = dataset_dir + dataset + "_dataset/" + dataset + "_data.txt";

DEFINE_string(input_file, dataset_filepath, "input data"); 
 
// dataset binaries 
// DEFINE_string(cache_file, dataset_bin_dir + dataset + ".bin", "cache file"); 
// DEFINE_string(train_cache_file, dataset_bin_dir + dataset + ".train.bin", "cached train file"); 
// DEFINE_string(test_cache_file, dataset_bin_dir + dataset + ".test.bin", "cached test file"); 

DEFINE_string(cache_file, dataset_bin_dir + dataset + ".txt", "cache file"); 
DEFINE_string(train_cache_file, dataset_bin_dir + dataset + ".train.txt", "cached train file"); 
DEFINE_string(test_cache_file, dataset_bin_dir + dataset + ".test.txt", "cached test file"); 

// ============================================================================================= //
// ============================================================================================= //

DEFINE_string(task, "train", "Task type");  
DEFINE_int32(seed, 20141119, "Random Seed");  // default 
DEFINE_string(method, "CDAE", "Which Method to use"); // "NONE"
DEFINE_string(model_variant, "M1", "Which Model to train"); // "M1", "M2", "M3", "M4"

/**
 * Model parameters
*/
DEFINE_int32(num_dim, 50, "Num of latent dimensions"); // K : num of latent dimensions (hidden neurons)
DEFINE_int32(num_neg, 5, "Num of negative samples");  // NS

// corruption level
DEFINE_int32(cnum, 1, "Num of Corruptions"); // default
DEFINE_double(cratio, 0.0, "Corruption Ratio");

// scaled input
DEFINE_bool(scaled, false, "scaled input"); // default

// training using SGD (and AdaGrad)
DEFINE_int32(max_iteration, 50, "Max num of iterations"); // default
DEFINE_double(learn_rate, 0.1, "Learning Rate"); //  η
DEFINE_bool(adagrad, true, "Use AdaGrad"); 
DEFINE_double(beta, 1., "Beta for adagrad"); // β

// holdout data
DEFINE_double(holdout_perc, 0.2, "Holdout percentage"); 

// user factor: include user input node (CDAE) or not (DAE)
DEFINE_bool(user_factor, true, "using user factor"); // false=DAE, true=CDAE

// asymmetric DAE: tied weights (TW) or non-tied weights (NTW)
DEFINE_bool(asym, true, "Asymmetric DAE"); // false=TW, true=NTW 


// ============================================================================================= //
// ============================================================================================= //

int main(int argc, char* argv[]) {
  using namespace libcf;
  
  /**
   * Set google's logging library.
  */
  FLAGS_log_dir = "./log"; // set directory to save log files
  // FLAGS_logtostderr = 1; // log messages to the console instead of logfiles.
  google::SetLogDestination(google::GLOG_INFO, log_file);
  google::InitGoogleLogging(argv[0]); // Initialize Google's logging library.

  // gflags::SetUsageMessage("movielens");
  // gflags::ParseCommandLineFlags(&argc, &argv, true);

// ============================================================================================= //
// ============================================================================================= //

  /**
   * Data preparation
  */

  int line_size;
  std::string split;
  bool skip_header;

  if (dataset == "movielens_10m" || dataset == "movielens_1m") {
    // data format: UserID::MovieID::Rating::Timestamp
    line_size = 4; 
    split = ": ";
    skip_header = false;

  } else if (dataset == "movielens_100k") {
    // data format: UserID\tMovieID\tRating\tTimestamp
    line_size = 4; 
    split = "\t ";
    skip_header = false;
  
  } else if (dataset == "politic_old" || dataset == "politic_new"){   
    // data format: legislatorID\tbillID\tcount
    line_size = 3; 
    split = "\t ";
    skip_header = false; 

  } else if(dataset == "yelp"){
    // data format: user_id,business_id,stars,date
    line_size = 4; 
    split = ", ";
    skip_header = true;
  
  } else if(dataset == "netflix_prize"){
    // data format: userId,movieId,rating,timestamp
    line_size = 4; 
    split = ", ";
    skip_header = true;
  }


  // create data binary
  std::ifstream data_file;
  data_file.open(FLAGS_cache_file);
  
  if (!data_file){ // data binary file does not exist 
    std::cout<<"TASK: prepare \n";

    auto line_parser = [&](const std::string& line) {
      auto rets = split_line(line, split); // rets[2KNPtV5E44vAiEr5BvMkUA,eJpr6Ks8pr4bmvDVPTN-Xg,2,2012-06-11]
      // LOG(ERROR) << "rets" << rets;
      // LOG(ERROR) << "rets" << rets.size();
      // LOG(ERROR) << "line_size" << line_size;
      CHECK_EQ(rets.size(), line_size); 
      return std::vector<std::string>{rets[0], rets[1], "1"};
    };

    Data data;
    // give format to "Data set summary" (on logs file)
    data.load(FLAGS_input_file, RECSYS, line_parser, skip_header); 

    // generate dataset(.bin) file
    save(data, FLAGS_cache_file, false); // binary_format = false
  }

  // split data
  std::ifstream train_file;
  train_file.open(FLAGS_train_cache_file);
   
  if (!train_file){ // train binary file does not exist
    std::cout<<"TASK: split \n";

    Random::seed(FLAGS_seed); // use the same seed to split the data 

    Data data;
    load(FLAGS_cache_file, data, false);
    LOG(INFO) << data; 
    
    Data train, test;
    data.random_split_by_feature_group(train, test, 0, FLAGS_holdout_perc);
    LOG(INFO) << train;
    LOG(INFO) << test;

    save(train, FLAGS_train_cache_file);
    save(test, FLAGS_test_cache_file);
  }  

  /**
   * Loading data  
  */

  std::cout << "TASK: loading data \n";
  Data train, test;

  if (FLAGS_task == "train") {
    std::cout << "TASK: train \n";

    Random::seed(FLAGS_seed); // use the same seed to split the data 
    
    Data data;
    load(FLAGS_cache_file, data);
    LOG(INFO) << data; 
    data.random_split_by_feature_group(train, test, 0, 0.2);
    LOG(INFO) << train;
    LOG(INFO) << test;

  } if (FLAGS_task == "test") {
    std::cout << "TASK: test \n";
    load(FLAGS_train_cache_file, train);
    load(FLAGS_test_cache_file, test);
  
  }

  // ============================================================================================= //
  // ============================================================================================= //

  /**
  * Model  
  */

  // experiments models
  struct model_setup { 
    // h(.): activation function on the hidden layer
    // identity, linear_function, tanh/sigmoid 
    bool linear; // true=identity, false=check tanh/sigmoid
    bool tanh; // true=tanh, false=sigmoid
    bool linear_function; // true=linear_mapping

    // f(.) activation function on the output layer
    // identity, sigmoid 
    bool sigmoid_output; // true=sigmoid , false=identity
    string loss_type; // loss function type l(.)
  } ; 

  model_setup model;

  if (FLAGS_model_variant == "M1"){
    std::cout << "MODEL: M1 \n";
    // Model M1: h(.) = identity, f(.) = identity, l(.) = SQUARE
    model = {.linear = true, .tanh = false, .linear_function = false, .sigmoid_output = false, .loss_type = "SQUARE"};

  } else if(FLAGS_model_variant == "M2"){
    std::cout << "MODEL: M2 \n";
    // Model M2: h(.) = identity, f(.) = sigmoid, l(.) = LOGISTIC
    model = {.linear = true, .tanh = false, .linear_function = false, .sigmoid_output = true, .loss_type = "LOGISTIC"};
  
  } else if(FLAGS_model_variant == "M3"){
    std::cout << "MODEL: M3";
    // Model M3: h(.) = sigmoid, f(.) = identity, l(.) = SQUARE
    model = {.linear = false, .tanh = false, .linear_function = false, .sigmoid_output = false, .loss_type = "SQUARE"};
  
  } else if(FLAGS_model_variant == "M4"){
    std::cout << "MODEL: M4";
    // Model M4: h(.) = sigmoid, f(.) = sigmoid, l(.) = LOGISTIC
    model = {.linear = false, .tanh = false, .linear_function = false, .sigmoid_output = true, .loss_type = "LOGISTIC"};
  }

  Random::timed_seed();

  if (FLAGS_method == "CDAE") {
    std::cout << "METHOD: CDAE\n";
    
    CDAEConfig config;
    config.learn_rate = FLAGS_learn_rate;
    config.num_dim = FLAGS_num_dim;
    config.using_adagrad = FLAGS_adagrad;
    config.asymmetric = FLAGS_asym;
    config.num_corruptions = FLAGS_cnum;
    config.corruption_ratio = FLAGS_cratio;
    config.linear = model.linear;
    config.scaled = FLAGS_scaled;
    config.num_neg = FLAGS_num_neg;
    config.user_factor = FLAGS_user_factor;
    config.beta = FLAGS_beta; 
    config.linear_function = model.linear_function;
    config.tanh = model.tanh;
    config.sigmoid_output = model.sigmoid_output;
    
    if (model.loss_type == "SQUARE") {
      config.lt = SQUARE;
    } else if (model.loss_type == "LOG") {
      config.lt = LOG;
    } else if (model.loss_type == "HINGE") {
      config.lt = HINGE;
    } else if (model.loss_type == "LOGISTIC") {
      config.lt = LOGISTIC;
    } else if (model.loss_type == "CE") {
      config.lt = CROSS_ENTROPY;
    } else {
      LOG(FATAL) << "UNKNOWN LOSS";
    }
    
    CDAE model(config);
    Solver<CDAE> solver(model, FLAGS_max_iteration);
    solver.train(train, test, {TOPN}); // train, validation
    // solver.test(test, {TOPN});
  }

  return 0;
}