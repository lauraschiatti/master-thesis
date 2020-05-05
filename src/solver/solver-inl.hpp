#include <solver/solver.hpp>

using namespace std;

namespace libcf {
  
/*
* Train model
*/ 
template<class Model>
void Solver<Model>::train(const Data& train_data, 
                              const Data& validation_data,
                              const vector<EvalType>& eval_types) {


  double train_loss = 0;

  // Create evaluation metrics array
  vector< shared_ptr<Evaluation<Model>>> evaluations(eval_types.size());
  for (size_t idx = 0; idx < eval_types.size(); ++idx) {
    evaluations[idx] = Evaluation<Model>::create(eval_types[idx]);
  }

  // Initialize parameters with random values
  model_->reset(train_data);

  // iter ← 0
  size_t iteration = 0;

  // pre_train(train_data, validation_data); // not implemented!

  Timer t;
  
  // Log: iterations, time and train loss
  LOG(INFO) <<  string(110, '-') <<  endl;
  {
    stringstream ss;
    ss <<  setfill(' ') <<  setw(5) << "Iters" << "|"
        <<  setw(8) << "Time"  << "|" 
        <<  setw(10) << "Train Loss" << "|";
    
    if(validation_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluation_type() << "|";
    } 

    LOG(INFO) << ss.str();
  }

  if (iteration % eval_iterations == 0)
  {
    stringstream ss;
    ss <<  setw(5) << iteration << "|"
        <<  setw(8) <<  setprecision(3) << t.elapsed() << "|"
        <<  setw(10) <<  setprecision(5) << train_loss << "|";
    
    if (validation_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluate(*model_, validation_data, train_data) << "|";
    }
    
    LOG(INFO) << ss.str();
  }

  /*
  * Training: learning algorithm
  */

  bool stop = false;
  while(!stop) {

    // one iteration
    train_one_iteration(train_data);

    // iter ← iter + 1
    iteration ++;

    // log: iteration, time and train loss
    train_loss = model_->current_loss(train_data);
    
    if (iteration % eval_iterations == 0)
    {
      stringstream ss;
      ss <<  setw(5) << iteration << "|"
          <<  setw(8) <<  setprecision(3) << t.elapsed() << "|"
          <<  setw(10) <<  setprecision(5) << train_loss << "|";
      
      if (validation_data.size() > 0) {
        for (size_t idx = 0; idx < eval_types.size(); ++idx) 
          ss << evaluations[idx]->evaluate(*model_, validation_data, train_data) << "|";
      }
      
      LOG(INFO) << ss.str();
    }

    // while iter < maxIter or error on validation set decreases (early stopping not implemented!)
    // check conditions
    if (iteration >= max_iteration_) {
      stop = true;
    }

    // other conditions
  }

  LOG(INFO) <<  string(110, '-') <<  endl;
}


/**
 * Test model
*/

template<class Model>
void Solver<Model>::test(const Data& test_data,
                         const vector<EvalType>& eval_types) {

  Timer t;
  vector< shared_ptr<Evaluation<Model>>> evaluations(eval_types.size());
  for (size_t idx = 0; idx < eval_types.size(); ++idx) {
    evaluations[idx] = Evaluation<Model>::create(eval_types[idx]);
  }

  LOG(INFO) <<  string(100, '-') <<  endl;
  {
    stringstream ss;
    ss <<  setfill(' ') 
        <<  setw(8) << "Time"  << "|";
    if(test_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluation_type() << "|";
    }
    LOG(INFO) << ss.str();
  }

  {
    stringstream ss;
    ss <<  setw(8) <<  setprecision(3) << t.elapsed() << "|";
    if (test_data.size() > 0) {
      for (size_t idx = 0; idx < eval_types.size(); ++idx) 
        ss << evaluations[idx]->evaluate(*model_, test_data) << "|";
      cout << "ok";
    }
    LOG(INFO) << ss.str();
  }
}

} // namespace