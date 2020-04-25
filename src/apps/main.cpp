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
std::string dataset = "movielens";

DEFINE_string(input_file, dataset_dir + dataset +"_data.txt", "input data"); 

// dateaset binaties 
DEFINE_string(cache_file, dataset_bin_dir + dataset + ".bin", "cache file"); 
DEFINE_string(train_cache_file, dataset_bin_dir + dataset + ".train.bin", "cached train file"); 
DEFINE_string(test_cache_file, dataset_bin_dir + dataset + ".test.bin", "cached test file"); 

/**
 * Task type: train, test
*/
DEFINE_string(task, "train", "Task type"); 
DEFINE_double(holdout_perc, 0.2, "Holdout percentage");  

/**
 * Recsys models: NONE, CDAE
*/
DEFINE_int32(seed, 20141119, "Random Seed");
DEFINE_string(method, "CDAE", "Which Method to use"); // "NONE"

/**
 * Model parameters
*/
DEFINE_int32(num_dim, 50, "Num of latent dimensions"); // K hidden neurons
DEFINE_int32(num_neg, 5, "Num of negative samples");  // NS

// corruption level
DEFINE_int32(cnum, 1, "Num of Corruptions"); // default
DEFINE_double(cratio, 0, "Corruption Ratio");

// scaled input
DEFINE_bool(scaled, false, "scaled input"); // default

// user factor (CDAE vs DAE)
DEFINE_bool(user_factor, true, "using user factor"); // default

// training using SGD (and AdaGrad)
DEFINE_int32(max_iteration, 50, "Max num of iterations"); // default
DEFINE_double(learn_rate, 0.1, "Learning Rate"); //  η
DEFINE_bool(adagrad, true, "Use AdaGrad"); 
DEFINE_double(beta, 1., "Beta for adagrad"); // β

// activation function h(.)
DEFINE_bool(linear_function, true, "Using Linear Mapping Function");
DEFINE_bool(tanh, false, "Using tanh NonLinear Function"); // sigmoid/tanh

// loss and regularization
DEFINE_string(loss_type, "SQUARE", "Loss function type"); // l(.)
// by default: lambda = 0.01;  // λ   
// and PenaltyType pt = L2;  // L2-norm regularization
  
// DAE
DEFINE_bool(asym, false, "Asymmetric DAE"); // default
DEFINE_bool(linear, false, "Linear DAE");  // with linear activation function


int main(int argc, char* argv[]) {
  using namespace libcf;
  
  /**
   * Set google's logging library.
  */
  FLAGS_log_dir = "./log"; // set directory to save log files
  // FLAGS_logtostderr = 1; // log messages to the console instead of logfiles.
  google::SetLogDestination(google::GLOG_INFO, "log/movielens_implicit.log");
  google::InitGoogleLogging(argv[0]); // Initialize Google's logging library.

  // gflags::SetUsageMessage("movielens");
  // gflags::ParseCommandLineFlags(&argc, &argv, true);

  //////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * Data preparation
  */

  int line_size;
  std::string split;

  if (dataset == "movielens") {
    // data format: UserID::MovieID::Rating::Timestamp
    line_size = 4; 
    split = ": ";
  }

  // get data
  std::ifstream data_file;
  data_file.open(FLAGS_cache_file);
  
  if (!data_file){ // dataset file does not exist 
    std::cout<<"TASK: prepare\n";

    auto line_parser = [&](const std::string& line) {
      // auto rets = split_line(line, " "); // yelp
      auto rets = split_line(line, split);
      CHECK_EQ(rets.size(), line_size); 
      return std::vector<std::string>{rets[0], rets[1], "1"};
    };

    Data data;
    data.load(FLAGS_input_file, RECSYS, line_parser, false); // skip_header=true
    save(data, FLAGS_cache_file);  
  }

  // split data
  std::ifstream train_file;
  train_file.open(FLAGS_train_cache_file);
   
  if (!train_file){ // train dataset file does not exist
    std::cout<<"TASK: split\n";

    Random::seed(FLAGS_seed); // use the same seed to split the data 

    Data data;
    load(FLAGS_cache_file, data);
    LOG(INFO) << data; 
    
    Data train, test;
    data.random_split_by_feature_group(train, test, 0, FLAGS_holdout_perc);
    LOG(INFO) << train;
    LOG(INFO) << test;
    
    save(train, FLAGS_train_cache_file);
    save(test, FLAGS_test_cache_file);
  }  


  //////////////////////////////////////////////////////////////////////////////////////////////////

  /**
   * Models 
  */

  Data train, test;

  if (FLAGS_task == "train") {
    std::cout << "TASK: train\n";

    Random::seed(FLAGS_seed); // use the same seed to split the data 
    
    Data data;
    load(FLAGS_cache_file, data);
    LOG(INFO) << data; 
    data.random_split_by_feature_group(train, test, 0, 0.2);
    LOG(INFO) << train;
    LOG(INFO) << test;

  } if (FLAGS_task == "test") {
    std::cout << "TASK: test\n";
    load(FLAGS_train_cache_file, train);
    load(FLAGS_test_cache_file, test);
  
  }

  // std::cout << "TASK: loading data ... \n";

  // Data train, test;
  // load(FLAGS_train_cache_file, train);
  // load(FLAGS_test_cache_file, test);

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
    config.linear = FLAGS_linear;
    config.scaled = FLAGS_scaled;
    config.num_neg = FLAGS_num_neg;
    config.user_factor = FLAGS_user_factor;
    config.beta = FLAGS_beta; 
    config.linear_function = FLAGS_linear_function;
    config.tanh = FLAGS_tanh;
    
    if (FLAGS_loss_type == "SQUARE") {
      config.lt = SQUARE;
    } else if (FLAGS_loss_type == "LOG") {
      config.lt = LOG;
    } else if (FLAGS_loss_type == "HINGE") {
      config.lt = HINGE;
    } else if (FLAGS_loss_type == "LOGISTIC") {
      config.lt = LOGISTIC;
    } else if (FLAGS_loss_type == "CE") {
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