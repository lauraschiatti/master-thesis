#ifndef _LIBCF_CDAE_HPP_
#define _LIBCF_CDAE_HPP_

#include <base/random.hpp>
#include <base/mat.hpp>
#include <base/instance.hpp>
#include <base/data.hpp>
#include <base/parallel.hpp>
#include <model/recsys/recsys_model_base.hpp>

namespace libcf {

struct CDAEConfig {
  CDAEConfig() = default;
  double lambda = 0.01;    
  double learn_rate = 0.1;   
  LossType lt = LOGISTIC; 
  PenaltyType pt = L2;  
  size_t num_dim = 10; 
  bool using_adagrad = true;
  std::string corruption_type = "mask_out";
  double corruption_ratio = 0.5; 
  size_t num_removed_interactions = 1;
  bool remove_same_interaction = true;
  size_t num_corrupted_versions = 10;
  size_t num_corruptions = 1;
  bool asymmetric = false; 
  bool user_factor = true;
  bool linear = false;
  size_t num_neg = 5;
  bool scaled = true;
  double beta = 0.;
  bool linear_function = false;
  bool tanh = false;  
  bool sigmoid_output = false; 
};

/* 
 * Denoising Auto-Encoder
 */
class CDAE : public RecsysModelBase {

 public:
  CDAE(const CDAEConfig& mcfg) {  
    lambda_ = mcfg.lambda;
    learn_rate_ = mcfg.learn_rate;
    num_dim_ = mcfg.num_dim; 
    loss_ = Loss::create(mcfg.lt);
    penalty_ = Penalty::create(mcfg.pt);
    using_adagrad_ = mcfg.using_adagrad;
    corruption_type_ = mcfg.corruption_type;
    corruption_ratio_ = mcfg.corruption_ratio;
    num_removed_interactions_ = mcfg.num_removed_interactions;
    remove_same_interaction_ = mcfg.remove_same_interaction;
    num_corrupted_versions_ = mcfg.num_corrupted_versions;
    num_corruptions_ = mcfg.num_corruptions;
    asymmetric_ = mcfg.asymmetric;
    user_factor_ = mcfg.user_factor;
    linear_ = mcfg.linear;
    num_neg_ = mcfg.num_neg;
    scaled_ = mcfg.scaled;
    beta_ = mcfg.beta;
    linear_function_ = mcfg.linear_function; 
    tanh_ = mcfg.tanh; 
    sigmoid_output_ = mcfg.sigmoid_output, 

    LOG(INFO) << "CDAE Configure: \n" 
        << "\t{lambda: " << lambda_ << "}, "
        << "{Loss: " << loss_->loss_type() << "}, "
        << "{Penalty: " << penalty_->penalty_type() << "}\n"
        << "\t{Dim: " << num_dim_ << "}, "
        << "{LearnRate: " << learn_rate_ << "}, "
        << "{Using AdaGrad: " << using_adagrad_ << "}\n"
        << "\t{Corruption Type: " << corruption_type_ << "}, "
        << "{Corruption Ratio: " << corruption_ratio_ << "}, \n"
        << "\t{Num of Removed Interactions : " << num_removed_interactions_ << "}, "
        << "{Remove Same Interaction: " << remove_same_interaction_ << "}, "
        << "{Num of Corrupted Versions: " << num_corrupted_versions_ << "}, \n"
        << "{Num Corruptions: " << num_corruptions_ << "}, \n"
        << "\t{Asymmetric: " << asymmetric_ << "} "
        << "{UserFactor: " << user_factor_ << "}, "
        << "{Linear: " << linear_ << "}, " 
        << "{Num Negative: " << num_neg_ << "}, "
        << "{Scaled: " << scaled_ << "}\n"
        << "\t{Beta: " << beta_ << "}, "
        << "{LinearFunction: " << linear_function_ << "}, "
        << "{tanh: " << tanh_ << "} "
        << "{Sigmoid in output: " << sigmoid_output_ << "}"; 
  }

  CDAE() : CDAE(CDAEConfig()) {}
  
  /**
   * Prediction error on training data
  */
  double data_loss(const Data& data_set, size_t sample_size=0) const {
     atomic<double> rets(0.);
    
    if(corruption_type_ == "mask_out" || corruption_type_ == "without_replacement"){
      
      parallel_for (0, num_users_, [&](size_t uid) {
        
        // get rated items for user u
        auto fit = user_rated_items_.find(uid);
        CHECK(fit != user_rated_items_.end());
        auto& item_set = fit->second;
        double user_rets = 0;

        for (size_t jid = 0; jid < num_corruptions_; ++jid) {
          
          double scale = 1;
          // std::cout << "data_loss get_corrupted_input_without_replacement2: " << corrpted_item_set.size() << "\n";
        
          if (scaled_) {
            scale /=  (1. - corruption_ratio_) ;
          }    

          // CDAE input: sample corrupted rating vector 
          unordered_map<size_t, double> corrupted_item_set;
          corrupted_item_set.reserve(static_cast<size_t>(item_set.size())); 
          
          if(corruption_type_ == "mask_out"){
            corrupted_item_set = get_corrupted_input(uid, item_set, corruption_ratio_);
          
          } else if(corruption_type_ == "without_replacement") {
            corrupted_item_set = get_corrupted_input_without_replacement(uid, item_set);

            // remove n-1 interactions more
            for (size_t idx = 1; idx < num_removed_interactions_; ++idx) {
              corrupted_item_set = get_corrupted_input_without_replacement(uid, corrupted_item_set);
            }
          }
          
          auto z = get_hidden_values(uid, corrupted_item_set, scale);
          
          for (auto& p : item_set) {
            size_t iid = p.first;
            user_rets += loss_->evaluate(get_output_values(z, iid), 1.);
          }
        }
        rets = rets + user_rets / num_corruptions_;
      
      });
    
    } else if(corruption_type_ == "with_replacement"){
    
      parallel_for (0, num_users_, [&](size_t uid) {

        // get rated items for user u
        auto fit = user_rated_items_.find(uid);
        CHECK(fit != user_rated_items_.end());
        auto& item_set = fit->second;
        double user_rets = 0;

        for (size_t jid = 0; jid < num_corruptions_; ++jid) {
          
          double scale = 1;
        
          if (scaled_) {
            scale /=  (1. - corruption_ratio_) ;
          }    

          // CDAE input: corrupted rating vector 
          unordered_map<size_t, double> corrupted_item_set;
          corrupted_item_set.reserve(static_cast<size_t>(item_set.size())); 

          // create several corrupted versions for user uid
          for (size_t idx = 1; idx < num_corrupted_versions_; ++idx) {

            corrupted_item_set = get_corrupted_input_without_replacement(uid, item_set);

            // remove n-1 interactions more
            for (size_t idx = 1; idx < num_removed_interactions_; ++idx) {
              corrupted_item_set = get_corrupted_input_without_replacement(uid, corrupted_item_set);
            }

            auto z = get_hidden_values(uid, corrupted_item_set, scale);
          
            for (auto& p : item_set) {
              size_t iid = p.first;
              user_rets += loss_->evaluate(get_output_values(z, iid), 1.);
            }
          }
          
          
        }
        rets = rets + user_rets / num_corruptions_;
      
      });
      
    }

    return rets;
  }
   
  /**
   * Regularization Loss
  */ 
  double penalty_loss() const {
    return 0.5 * lambda_ * (penalty_->evaluate(W) + penalty_->evaluate(V)
                            + penalty_->evaluate(Wu) + penalty_->evaluate(b) 
                            + penalty_->evaluate(b_prime)); 
  }

  /**
   * Reset the model parameters 
  */
  void reset(const Data& data_set) {
    RecsysModelBase::reset(data_set);

    double init_scale = 4. *  sqrt(6. / static_cast<double>(num_items_ + num_dim_));
    W = DMatrix::Random(num_items_, num_dim_) * init_scale;
    W_ag = DMatrix::Constant(num_items_, num_dim_, 0.0001);
    
    // TW or NTW
    if (asymmetric_) {
      V = DMatrix::Random(num_items_, num_dim_) * init_scale;
      V_ag = DMatrix::Constant(num_items_, num_dim_, 0.0001);
    } 

    // DAE or CDAE
    if (user_factor_) {
      Wu = DMatrix::Random(num_users_, num_dim_) * init_scale;
      Wu_ag = DMatrix::Constant(num_users_, num_dim_, 0.0001);
    }

    b = DVector::Zero(num_dim_);
    b_ag = DVector::Ones(num_dim_) * 0.0001;
    b_prime = DVector::Zero(num_items_);
    b_prime_ag = DVector::Ones(num_items_) * 0.0001;
    // bu = DVector::Zero(num_users_);
    // bu_ag = DVector::Ones(num_users_) * 0.0001;

    // user-specific matrix Uu
    if (linear_function_) { 
      Uu = DMatrix::Constant(num_users_, num_dim_, 1.);
      Uu_ag = DMatrix::Constant(num_users_, num_dim_, 0.0001);
    }

    LOG(INFO) << "CDAE parameters initialization: \n" 
        << "\t{init_scale: " << init_scale << "}, "
        << "{W size: " << W.size() << "}, "
        << "{W_ag size: " << W_ag.size() << "}\n"
        << "\t{V size: " << V.size() << "}, "
        << "{V_ag size: " << V_ag.size() << "}, "
        << "{Wu size: " << Wu.size() << "}\n"
        << "\t{Wu_ag size: " << Wu_ag.size() << "}, "
        << "{b size: " << b.size() << "}, "
        << "{b_ag size: " << b_ag.size() << "}\n"
        << "\t{b_prime size: " << b_prime.size() << "}, "
        << "{b_prime_ag size: " << b_prime_ag.size() << "}, "; 

  } 

  /**
   * One CDEA learning algorithm iteration 
  */
  void train_one_iteration(const Data& train_data) {

    if(corruption_type_ == "mask_out" || corruption_type_ == "without_replacement"){

      // overall % of corruption of the original item_set 
      double overall_item_set_corruption = 0.0;

      // for each of the users
      for (size_t uid = 0; uid < num_users_; ++uid) { 

        double user_item_set_corruption = 0.0;
          
        // get rated items for user u
        auto it = user_rated_items_.find(uid);  // iterator (key/value, first/second)
        CHECK(it != user_rated_items_.end()); // iterator one past the end 
        auto& item_set = it->second; // value in the map

        // for each corruption
        for (size_t idx = 0; idx < num_corruptions_; ++idx) {
          
          // CDAE input: corrupted rating vector 
          unordered_map<size_t, double> corrupted_item_set;
          corrupted_item_set.reserve(static_cast<size_t>(item_set.size())); 

          if(corruption_type_ == "mask_out"){
            corrupted_item_set = get_corrupted_input(uid, item_set, corruption_ratio_);
        
          } else if(corruption_type_ == "without_replacement"){
            corrupted_item_set = get_corrupted_input_without_replacement(uid, item_set);

            // remove n-1 interactions more
            for (size_t idx = 1; idx < num_removed_interactions_; ++idx) {
              corrupted_item_set = get_corrupted_input_without_replacement(uid, corrupted_item_set);
            }

          }

          // current user % of corruption of the original item_set 
          user_item_set_corruption = (double)corrupted_item_set.size() / (double)item_set.size();

          // train CDAE on user's corrupted input
          train_one_user_corruption(uid, corrupted_item_set, item_set);
          
        }

        if(uid == 0){
          std::cout << "user_item_set_corruption: " << user_item_set_corruption << std::endl;
        }

        overall_item_set_corruption =  overall_item_set_corruption + user_item_set_corruption;
      }

      std::cout << std::endl; 
      double corruption_ratio = 1 - overall_item_set_corruption/num_users_; 
      std::cout << " ---> overall_item_set_corruption: " << overall_item_set_corruption/num_users_ << std::endl;
      std::cout << " ---> corruption_ratio = 1 - overall_item_set_corruption: " << corruption_ratio << std::endl;
      std::cout << std::endl; 
    

    /**
     * WITH REPLACEMENT CORRUPTION: 
     * for each user repeat the without_corruption several times,
     * each of them producing a different corrupted replica of the original user.
    */ 
    } else if(corruption_type_ == "with_replacement"){
      
      // for each of the users
      for (size_t uid = 0; uid < num_users_; ++uid) { 

        // get rated items for user u
        auto it = user_rated_items_.find(uid);  // iterator (key/value, first/second)
        CHECK(it != user_rated_items_.end()); // iterator one past the end 
        auto& item_set = it->second; // value in the map

        // for each corruption
        for (size_t idx = 0; idx < num_corruptions_; ++idx) {
          
          // CDAE input: corrupted rating vector 
          unordered_map<size_t, double> corrupted_item_set;
          corrupted_item_set.reserve(static_cast<size_t>(item_set.size())); 

          // create several corrupted versions for user uid
          for (size_t idx = 1; idx < num_corrupted_versions_; ++idx) {
            corrupted_item_set = get_corrupted_input_without_replacement(uid, item_set);

            // remove n-1 interactions more
            for (size_t idx = 1; idx < num_removed_interactions_; ++idx) {
              corrupted_item_set = get_corrupted_input_without_replacement(uid, corrupted_item_set);
            }

            // train CDAE on the current corrupted input
            train_one_user_corruption(uid, corrupted_item_set, item_set);
          }

          // current user % of corruption of the original item_set 
          // user_item_set_corruption = (double)corrupted_item_set.size() / (double)item_set.size();
        }

      }
    }
  }
  
  /**
   * DEFAULT CORRUPTION: multiplicative mask-out/drop-out
   * original corruption mechanism of randomly 
   * masking entries of the input by making them zero. 

   * keeps 1-corruption_level entries of the inputs the same 
   * and zero-out randomly selected subset of size corruption_level
  */ 

  unordered_map<size_t, double> get_corrupted_input(int uid, const  unordered_map<size_t, double>& input_set, 
                                          double corruption_ratio) const {

    if (uid == 0){                                        
      std::cout << "The original input_set. Size: " << input_set.size() << "\n";
      for (auto it = input_set.begin(); it != input_set.end(); ++it ){
        std::cout << " " << it->first;
      }
      std::cout << std::endl;
    } 

    // corrupted_item_set                                            
    unordered_map<size_t, double> rets;
    rets.reserve(static_cast<size_t>(input_set.size() * (1. - corruption_ratio))); 

    // this will produce an array of 0s and 1s where 1 has a
    // probability of 1 - ``corruption_level`` and 0 with
    // ``corruption_level``

    for (auto& p : input_set) {
      // generate a random number in a range [min,max)
      // by default: min = 0., max = 1.
      auto random = Random::uniform(); 
      if (random > corruption_ratio) {
        rets.insert(p);
      }
    }

    if (uid == 0){
      std::cout << "Now the new corrupted_item_set. Size: " << rets.size() << "\n";
      for (auto it = rets.begin(); it != rets.end(); ++it ){
        std::cout << " " << it->first; 
      }
      std::cout << std::endl;
    }
    
    return rets;
  }

  /**
   * WITHOUT REPLACEMENT CORRUPTION: (VERSION 1)  << used for experiments >>
   * for each user with at least 2 interactions,
   * select a random interaction and remove it from its original item_set
  */ 
  unordered_map<size_t, double> get_corrupted_input_without_replacement(int uid,
                                          const unordered_map<size_t, double>& item_set) const {

    // 1) create copy of the original item_set as the corrupted_item_set as a 

    if (uid == 0){
      std::cout << "uid 0\n";
      std::cout << "The original item_set. Size: " << item_set.size() << "\n";
      for (auto it = item_set.begin(); it != item_set.end(); ++it ){
        std::cout << " " << it->first;
      }
      std::cout << std::endl;
    }

    // different ways to copy a vector ==> use vector::insert 
    // Because vectors use an array as their underlying storage, 
    // inserting elements in positions other than the vector end causes the container 
    // to relocate all the elements that were after position to their new positions.                              
    unordered_map<size_t, double> corrupted_item_set;
    corrupted_item_set.reserve(static_cast<size_t>(item_set.size())); 
    for (auto& p : item_set) {
      corrupted_item_set.insert(p);
    }

    if (uid == 0){
      std::cout << "The copy of the original item_set (corrupted_item_set). Size:" << corrupted_item_set.size() << "\n";
      for (auto it = corrupted_item_set.begin(); it != corrupted_item_set.end(); ++it ){
        std::cout << " " << it->first;
      }
      std::cout << std::endl;
    }

    // ==========================================================================
    
    // 2) sample a random interaction from the item_set of user uid

    int interaction_idx;

    if(remove_same_interaction_){  // variation b) get always the same interaction for each user
      // NOTE: remove last interaction 
      // same obtained results when removed 1st and last interaction
      interaction_idx = item_set.size() - 1; 
    
    } else {  // variation a) generate random number between 0 to input_set size
      interaction_idx = rand() % item_set.size() - 1;  
    }

    if (uid == 0){
      std::cout << "\nElement at position " << interaction_idx << " will be removed\n";
    }
    
    // get an iterator to the position in item_set corresponding to the element to be removed
    // vector using std::next
    int index = interaction_idx;
    auto it2 =  std::next(corrupted_item_set.begin(), index);
    if (uid == 0){ 
      cout << "Element at index " << index << " is: " << *it2 << '\n';
    }

    // ==========================================================================

    // 3) remove the random interaction from the corrupted_item_set
    
    int count=0;
    
    for (auto it = corrupted_item_set.begin(); it != corrupted_item_set.end(); it++) {
      if (it == it2){
        //check that the user has at least 2 interactions
        if(item_set.size() >= 2) { 
          corrupted_item_set.erase(it);
          count++;
        } else{
          std::cout << "uid" << uid << " has few interactions\n" ;
        }
        break; // exit the loop
      }
    }

    if (uid == 0){
      if(count == 0){
        std::cout << "Element not found..!!\n";
      } else{
        std::cout << "Element deleted successfully..!!\n";
        
        std::cout << "Now the new corrupted_item_set. Size: " << corrupted_item_set.size() << "\n";
        for (auto it = corrupted_item_set.begin(); it != corrupted_item_set.end(); ++it ){
          std::cout << " " << it->first; 
        }
        std::cout << std::endl;
      }
    }
    
    return corrupted_item_set;
  }

  
  /**
   * WITHOUT REPLACEMENT CORRUPTION: (VERSION 2) << lower results w.r.t. version 1>>
   * for each user with at least 2 interactions,
   * select a random interaction and copy into the originally empty corrupted_item_set
   * all interactions but the one to be removed.
  */ 
  unordered_map<size_t, double> get_corrupted_input_without_replacement2(int uid,
                                          const unordered_map<size_t, double>& item_set) const {

    if (uid == 0){
      std::cout << "uid 0\n";
      std::cout << "The original item_set. Size: " << item_set.size() << "\n";
      for (auto it = item_set.begin(); it != item_set.end(); ++it ){
        std::cout << " " << it->first;
      }
      std::cout << std::endl;
    }

    // 1) sample a random interaction from the item_set of user uid

    int interaction_idx;

    if(remove_same_interaction_){  // variation b) get always the same interaction for each user
      // NOTE: remove last interaction 
      // same obtained results when removed 1st and last interaction
      interaction_idx = item_set.size() - 1; 
    
    } else {  // variation a) generate random number between 0 to input_set size
      interaction_idx = rand() % item_set.size() - 1;  
    }

    if (uid == 0){
      std::cout << "\nElement at position " << interaction_idx << " will be removed\n";
    }
    
    // get an iterator to the position in item_set corresponding to the element to be removed
    // vector using std::next
    int index = interaction_idx;
    auto it2 =  std::next(item_set.begin(), index);
    if (uid == 0){ 
      cout << "Element at index " << index << " is: " << *it2 << '\n';
    }

    // ==========================================================================

    // 2) append all the items to the the corrupted_item_set but the interaction to be removed 

    unordered_map<size_t, double> corrupted_item_set;
    corrupted_item_set.reserve(static_cast<size_t>(item_set.size())); 

    // int count = 0;

    if(item_set.size() >= 2) { 
      for (auto it = item_set.begin(); it != item_set.end(); ++it ){
        if(it != it2){
          corrupted_item_set.insert(*it);
          // count++;
        }
      }

    } else {
      std::cout << "uid" << uid << " has few interactions\n" ;
      for (auto it = item_set.begin(); it != item_set.end(); ++it ){
        corrupted_item_set.insert(*it);
      }
    }


    if (uid == 0){
      // if(count == 0){
      //   std::cout << "Element not found..!!\n";
      // } else{
      //   std::cout << "Element deleted successfully..!!\n";
        std::cout << "uid 0\n";
        std::cout << "Now the new corrupted_item_set. Size: " << corrupted_item_set.size() << "\n";
        for (auto it = corrupted_item_set.begin(); it != corrupted_item_set.end(); ++it ){
          std::cout << " " << *it; 
        }
        std::cout << std::endl;
      // }
    }

    return corrupted_item_set;
  }


  /**
   * Train CDAE on corrupted input of a given user
  */
  void train_one_user_corruption(size_t uid, 
                                const  unordered_map<size_t, double>& input_set,  // corrupted_item_set
                                const  unordered_map<size_t, double>& output_set)  // input item_set 
                                {
    
    // make corruption unbiased: 
    // set uncorrupted item_set to 1/(1-q) its original value
    double scale = 1.;
    if (scaled_) {
      scale /= (1. - corruption_ratio_);
    }

    // std::cout << "train_one_user_corruption input_set size: " <<   input_set.size() << "\n";

    // map corrupted input into a hidden representation Zu 
    // apply h(.) mapping function
    DVector z = get_hidden_values(uid, input_set, scale);

    // std::cout << "get_hidden_values \n";
    // for(int i=0; i < z.size(); i++)
    // std::cout << z[i] << ' ';
    
    // ????
    DVector z_1_z =  DVector::Ones(num_dim_);
    if (! linear_) { 
      if (! tanh_) {
        // sigmoid activation
        z_1_z = z - z.cwiseProduct(z);
      } else {
        // tanh activation
        z_1_z = DVector::Ones(num_dim_) - z.cwiseProduct(z); 
      }
    }
    
    // Sample a subset of negative items Su 
    vector<size_t> negative_samples(output_set.size() * num_neg_);
    
    for (size_t idx = 0; idx < negative_samples.size(); ++idx) {
      negative_samples[idx] = sample_negative_item(output_set);
    }
    

    /**
     * Learn the parameters of CDAE: update W,W,V,b,b′
     * NOTE: no need to compute the gradients on all the outputs, 
     *       compute the gradients on the items in Ou ∪ Su  
    */

    unordered_map<size_t, DVector> input_gradient;
    DVector hidden_gradient = DVector::Zero(num_dim_);

    // Update Wi' and bi' using items in the Ou(user training items) 
    // and Su (subsample of negative items
    // items that user didn't interact with

    for (auto& p : output_set)  // items in Ou set (item_id, rating)
    {
      size_t iid = p.first; // item_id

      // Compute output values yui^ = f(.)
      double y = get_output_values(z, iid); // yui^ 

      // std::cout << "output_set get_output_values" << y <<"\n";
      // for(int i=0; i < z.size(); i++)

      // Get loss gradient
      double target = 1.;   // implicit rating where all the yui are 1
      double gradient = loss_->gradient(y, target);  // prediction, target value (yui^, yui)

      // Update bi'
      // ==========
      {
        // compute ∂l/∂bi' gradient
        double grad = gradient + lambda_ * b_prime(iid); //  gradient + λ bi'

        // update bi' using AdaGrad to adapt step size (Goodfellow AdaGrad form)
        if (using_adagrad_) { 
          b_prime_ag(iid) += grad * grad; // accumulate squared gradients: sum(grad^2)
          grad /= (beta_ +  sqrt(b_prime_ag(iid))); // bi' = grad/(β + sqrt(sum(grad^2)))
        }

        // update bi' without using AdaGrad
        b_prime(iid) -= learn_rate_ * grad; // bi' = bi' - η.grad
      }

      // Update Wi'
      // ==========
      if (asymmetric_) { // ????
        hidden_gradient += gradient * V.row(iid);
        DVector grad = gradient * z + lambda_ * V.row(iid).transpose();
        if (using_adagrad_) {
          V_ag.row(iid) += grad.cwiseProduct(grad);
          grad = grad.cwiseQuotient((V_ag.row(iid).transpose().cwiseSqrt().array() + beta_).matrix());
        }
        V.row(iid) -= learn_rate_ * grad;
      } else {
        
        hidden_gradient += gradient * W.row(iid); 
        
        // compute ∂l/∂Wi' gradient
        if (input_set.count(iid)) {
          input_gradient[iid] = gradient * z;

        } else {
          DVector grad = gradient * z + lambda_ * W.row(iid).transpose();

          // update Wi' using adagrad to adapt step size
          if (using_adagrad_) {
            W_ag.row(iid) += grad.cwiseProduct(grad); // accumulate squared gradients: sum(grad^2)
            // Wi' = grad/(sqrt(sum(Wi')) + β)
            grad = grad.cwiseQuotient((W_ag.row(iid).transpose().cwiseSqrt().array()+ beta_).matrix());
          }

          // update Wi' without using adagrad
          W.row(iid) -= learn_rate_ * grad;  // Wi' = Wi' - η.grad
        }
      }
    }

    for (auto& iid : negative_samples) // items in Su set
    {
      // Compute output values yu^ = f(.)
      double y = get_output_values(z, iid); // yu^

      // Get loss gradient
      double target = 0.;   // items the user didn't interact with
      double gradient = loss_->gradient(y, target);  // prediction, target value (yui^, yui)

      // Update bi'
      // ==========
      {
        // compute ∂l/∂bi' gradient
        double grad = gradient + lambda_ * b_prime(iid); //  gradient + λ bi'

        // update bi' using AdaGrad to adapt step size (Goodfellow AdaGrad form)
        if (using_adagrad_) { 
          b_prime_ag(iid) += grad * grad;  // accumulate squared gradients: sum(grad^2)
          grad /= (beta_ +  sqrt(b_prime_ag(iid))); // bi' = grad/(β + sqrt(sum(grad^2)))
        }

        // update bi' without using AdaGrad
        b_prime(iid) -= learn_rate_ * grad; // bi' = bi' - η.grad
      }

      // Update Wi'
      // ==========
      if (asymmetric_) { // ????
        hidden_gradient += gradient * V.row(iid);
        DVector grad = gradient * z + lambda_ * V.row(iid).transpose();
        if (using_adagrad_) {
          V_ag.row(iid) += grad.cwiseProduct(grad);
          grad = grad.cwiseQuotient((V_ag.row(iid).transpose().cwiseSqrt().array() + beta_).matrix());
        }
        V.row(iid) -= learn_rate_ * grad;

      } else {
        
        hidden_gradient += gradient * W.row(iid);

        // compute ∂l/∂Wi' gradient
        DVector grad = gradient * z + lambda_ * W.row(iid).transpose();

        // update Wi' using adagrad to adapt step size
        if (using_adagrad_) {
          W_ag.row(iid) += grad.cwiseProduct(grad); // accumulate squared gradients: sum(grad^2)
          // Wi' = grad/(sqrt(sum(Wi')) + β)
          grad = grad.cwiseQuotient((W_ag.row(iid).transpose().cwiseSqrt().array() + beta_).matrix());
        }

        // update Wi' without using adagrad
        W.row(iid) -= learn_rate_ * grad; // Wi' = Wi' - η.grad
      }
    }

    // user-specific matrix Uu
    DVector Uu_grad;
    if (linear_function_) {
      Uu_grad = DVector::Zero(num_dim_);
      Uu_grad += Uu.row(uid).transpose() * lambda_; 
    }

    // b ???
    {
      DVector grad = DVector::Zero(num_dim_);
      //if (!linear_function_) {
        grad = hidden_gradient.cwiseProduct(z_1_z) + lambda_ * b;
      //} else {
        //grad = Uu[uid].tranpose() * hidden_gradient.cwiseProduct(z_1_z) + lambda_ * b;
        //Uu_grad += hidden_gradient.cwiseProduct(z_1_z) * b.transpose();
      //}
      if (using_adagrad_) {
        b_ag += grad.cwiseProduct(grad);
        grad = grad.cwiseQuotient((b_ag.cwiseSqrt().array() + beta_).matrix());
      }
      b -= learn_rate_ * grad;
    }
   
    // Update Vu
    if (user_factor_)
    {   
      DVector grad = DVector::Zero(num_dim_);
      //if (!linear_function_) {
        grad = hidden_gradient.cwiseProduct(z_1_z) + lambda_ * Wu.row(uid).transpose();
      //} else {
      //  grad = Uu[uid].transpose() * hidden_gradient.cwiseProduct(z_1_z) + lambda_ * Wu.row(uid).transpose();
      //  Uu_grad += hidden_gradient.cwiseProduct(z_1_z) * Wu.row(uid);
      //}
      if (using_adagrad_) {
        Wu_ag.row(uid) += grad.cwiseProduct(grad);
        grad = grad.cwiseQuotient((Wu_ag.row(uid).transpose().cwiseSqrt().array() + beta_).matrix());
      }
      Wu.row(uid) -= learn_rate_ * grad;
    }

    // Update Wj
    // std::cout << "user input_set";
    for (auto& p : input_set) {
      // std::cout << p  << "\n";
      
      size_t jid = p.first;
      
      DVector grad = DVector::Zero(num_dim_);
      
      if (!linear_function_) {
        grad = hidden_gradient.cwiseProduct(z_1_z) * scale + lambda_ * W.row(jid).transpose();
      } else {
        grad = Uu.row(uid).transpose().cwiseProduct(hidden_gradient.cwiseProduct(z_1_z)) * scale + lambda_ * W.row(jid).transpose();
        Uu_grad += hidden_gradient.cwiseProduct(z_1_z).cwiseProduct(W.row(jid).transpose());
      }

      if (input_gradient.count(jid))
        grad += input_gradient[jid];
      
      if (using_adagrad_) {
        W_ag.row(jid) += grad.cwiseProduct(grad);
        grad = grad.cwiseQuotient((W_ag.row(jid).transpose().cwiseSqrt().array() + beta_).matrix());
      }
      W.row(jid) -= learn_rate_ * grad;
    }
    
    if (linear_function_) {
      if (using_adagrad_) {
        Uu_ag.row(uid) += Uu_grad.cwiseProduct(Uu_grad).transpose(); 
        Uu_grad = Uu_grad.cwiseQuotient((Uu_ag.row(uid).transpose().cwiseSqrt().array() + beta_).matrix());
      }
      Uu.row(uid) -= learn_rate_ * Uu_grad;
    }
  }

  /**
   * Compute latent representation Zu = h(W^T.Yu~ + Vu + b)
   * using the sum of weighted vectors
  */
  DVector get_hidden_values(size_t uid, const  unordered_map<size_t, 
                            double>& item_set, // corrupted input set
                            double scale = 1.0) const {

    // std::cout << "get_hidden_values corrupted_item_set: " << item_set.size() << "\n";
    
    // initialize h vector to zero
    DVector h1 = DVector::Zero(num_dim_); 

    // std::cout << "get_hidden_values corrupted input set: " << item_set.size() << "\n";
    
    // += W^T.yu~
    for (auto& p : item_set) {
      size_t iid = p.first; 
      h1 += W.row(iid) * scale; // DVectorSlice object representing the iid row
    }
    
    if (linear_function_) { 
      h1 = Uu.row(uid).transpose().cwiseProduct(h1);
    }

    // += Vu
    if (user_factor_) {
      h1 += Wu.row(uid);
    }

    // +=b
    h1 += b; 

    if (! linear_) {
      if (! tanh_) {
        // sigmoid activation
        h1 = h1.unaryExpr([](double x) { // apply a unary operator coefficient-wise
                        if (x > 18.) {
                        return 1.;
                        } 
                        if (x < -18.) {
                        return 0.;
                        }
                        return 1. / (1. +  exp(-x));  // 1 / (1 + exp(-x))
                        });
      } 

      else {
        // tanh activation
        h1 = h1.unaryExpr([](double x) {
                          if (x > 9.) {
                            return 1.;
                          }
                          if (x < -9.) {
                            return -1.;
                          }  
                          double r =  exp(-2. * x);
                          return (1. - r) / (1. + r);  // (1 - exp(-2x)) / (1 + exp(-2x))
                        });
      }
    }

    return h1;
  }

  /**
   * Latent representation is mapped back to the orignal input space
   * Compute output yu^ = f(Wi'^T.Zu + bi')
   * f(.) is identity function or sigmoid function
  */ 
  double get_output_values(const DVector& z, size_t idx) const {
    double h2 = 0; 

    if (asymmetric_) {
      h2 += V.row(idx).dot(z) + b_prime(idx); // Vi'^T.Zu + bi'
    } else {
      h2 += W.row(idx).dot(z) + b_prime(idx); // Wi^T.Zu + bi'
    }

    if(sigmoid_output_){
      double x = h2;
      if (x > 18.) {
        h2 = 1.;
      } 
      if (x < -18.) {
        h2 = 0.;
      }
      h2 =  1. / (1. +  exp(-x));  // 1 / (1 + exp(-x))
    }

    return h2;
  }


  DMatrix get_user_representations() {
    
    DMatrix user_vec(num_users_, num_dim_);

    for (size_t uid = 0; uid < num_users_; ++uid) {
      auto fit = user_rated_items_.find(uid);
      CHECK(fit != user_rated_items_.end());
      user_vec.row(uid) = get_hidden_values(uid, fit->second);
    }
  
    return  move(user_vec);
  }

  /**
   * Recommendation
  */
  // required by evaluation measure TOPN
   vector<size_t> recommend(size_t uid, size_t topk,
                                const  unordered_map<size_t, double>& rated_item_set) const {
    size_t item_id = 0;
    size_t item_id_end = item_id + data_->feature_group_total_dimension(1);
     
    DVector z = DVector::Zero(num_dim_);
    if (corruption_ratio_ != 1.) { 
      z = get_hidden_values(uid, rated_item_set);
    } else { 
      z = get_hidden_values(uid,  unordered_map<size_t, double>{});
    }

    Heap< pair<size_t, double>> topk_heap(sort_by_second_desc<size_t, double>, topk);
    double pred;
    for (; item_id != item_id_end; ++item_id) {
      if (rated_item_set.count(item_id)) {
        continue;
      }
      pred = get_output_values(z, item_id);
      if (topk_heap.size() < topk) {
        topk_heap.push({item_id, pred});
      } else {
        topk_heap.push_and_pop({item_id, pred});
      }
    }
    CHECK_EQ(topk_heap.size(), topk);
    auto topk_heap_vec = topk_heap.get_sorted_data();
     vector<size_t> ret(topk);
     transform(topk_heap_vec.begin(), topk_heap_vec.end(),
                   ret.begin(),
                   [](const  pair<size_t, double>& p) {
                   return p.first;
                   });
    return  move(ret);
  }


 private:

  // Model params: W, W', V, b, b'
  DMatrix W; // W: weights vector between the item input nodes and the nodes in the hidden layer
  DMatrix V; // W': weights between nodes in the hidden layer and the output layer
  
  // --  when asymmetric true => W != W'. Otherwise W = W' (tied weights) --

  DMatrix Wu; // Vu: weight vector for the user input node 
  DVector b; // b: weight vector for the bias node in the hidden layer
  DVector b_prime; // b': offset vector for the output layer
  DVector bu; // bu: not used!
  DMatrix Uu; // user-specific matrix Uu (K×K transform matrix) on the hidden layer

  // Model params updated using Adagrad: W, W', V, b, b'
  DMatrix W_ag, V_ag, Wu_ag, Uu_ag;
  DVector b_ag, b_prime_ag, bu_ag;
 
  size_t num_dim_ = 0.;  
  double learn_rate_ = 0.;
  double lambda_ = 0.;  
  std::string corruption_type_ = "mask_out";
  double corruption_ratio_ = 0.5;
  size_t num_removed_interactions_ = 1;
  bool remove_same_interaction_ = true;
  size_t num_corrupted_versions_ = 10;
  size_t num_corruptions_ = 10;
  size_t num_neg_ = 5;
  bool using_adagrad_ = true;
  bool asymmetric_ = false;  
  bool user_factor_ = true;
  bool linear_ = false;
  bool scaled_ = true;
  double beta_ = 0.; 
  bool linear_function_ = false;
  bool tanh_ = false; 
  bool sigmoid_output_ = false;
};

} // namespace

#endif // _LIBCF_CDAE_HPP_