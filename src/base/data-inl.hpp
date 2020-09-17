#include <base/data.hpp>

#include <random>
#include <algorithm>

#include <base/io.hpp>
#include <base/timer.hpp>
#include <base/utils.hpp>
#include <base/random.hpp>

#include <boost/archive/text_oarchive.hpp>

namespace libcf {

void Data::load(const std::string& filename, 
                const DataFormat& df, 
                const LineParser& parser,
                bool skip_header) {
  
  if (data_info_ == nullptr) {
    data_info_ = std::make_shared<DataInfo>(new DataInfo());
  }

  // give format to Data set summary (on logs file)
  switch (df) {
    case VECTOR : {
      FileLineReader f(filename);
      add_feature_group(DENSE);
      set_label_type(EMPTY);
      f.set_line_callback(
         [&](const std::string& line, size_t line_num) {
          if (skip_header && line_num == 0) return;
          auto rets = parser(line);
          if (rets.size() == 0) return;
          Instance ins;
          std::vector<double> vec(rets.size());
          std::transform(rets.begin(), rets.end(), vec.begin(),
            [&](const std::string& str) { return std::stod(str); });
          ins.add_feat_group(data_info_->feature_group_infos_[0], vec);
          instances_.push_back(ins);
        });
      f.load();
      break; 
    }
    case LIBSVM : {
      //TODO
      break;
    }
    case RECSYS : {
      FileLineReader f(filename);
      
      // instantiate 3 FeatureGroupInfo: users, items and ratings
      add_feature_group(SPARSE_BINARY);  // data_info_->feature_group_infos_.push_back(FeatureGroupInfo(ft));  
                                         // users: {type : SPARSE_BINARY}, {size: 0}
      add_feature_group(SPARSE_BINARY);  // items: {type : SPARSE_BINARY}, {size: 0}
      set_label_type(CONTINUOUS); // ratings: {type: DENSE}, {size: 0}
      
      // process each line in the file
      f.set_line_callback( 
          // [&] to write an inline function 
          [&](const std::string& line, size_t line_num) {
          
            if (skip_header && line_num == 0) return;
  
            // std::cout << "line " << line;
            auto rets = parser(line);
            // std::cout << " rets " << rets; 
            /*
            line 69    12    5   882145567      rets [69,12,1]1
            line 237  494    4   879376553      rets [237,494,1]1
            line 85   133    4   879453876      rets [85,133,1]1
            line 276  85     3   874791871      rets [276,85,1]1
            */

            if (rets.size() == 0) return;

            // Each line is an instance, default label is 0.
            Instance ins; 
            ins.add_feat_group(data_info_->feature_group_infos_[0], rets[0]); // users: {type : SPARSE_BINARY}, {size: x}
            // std::cout << ins << std::endl;
            /* ins {Label: 0}, {Feature Groups: [{0: [(8:1)]}]] 
              ins {Label: 0}, {Feature Groups: [{0: [(162:1)]}] */

            // preprocess item ids => transform (feature 1)
            ins.add_feat_group(data_info_->feature_group_infos_[1], rets[1]); // items: {type : SPARSE_BINARY}, {size: x}
            // std::cout << ins << std::endl;
            /* {Label: 0}, {Feature Groups: [{0: [(8:1)]}, {1: [(359:1)]}]1
               {Label: 0}, {Feature Groups: [{0: [(162:1)]}, {1: [(309:1)]}]1 */

            // assign converted string to ins.label
            ins.set_label(std::stod(rets[2])); // std::stod => converts (rating) string to double; 
            // std::cout << ins << std::endl;
            /* {Label: 1}, {Feature Groups: [{0: [(351:1)]}, {1: [(321:1)]}]
               {Label: 1}, {Feature Groups: [{0: [(363:1)]}, {1: [(492:1)]}] */

            // append new instance
            instances_.push_back(ins);
          }
        );
      
       // count lines loaded (line_num)  and lines skipped (line_skipped) from data file
      f.load(); 
      break;
    }
    default : {
      break;
    }
  }

  data_info_->total_dimensions_ = 0;

  // assigns new contents to feature_group_global_idx_ vector, replacing its current content
  data_info_->feature_group_global_idx_.assign(num_feature_groups(), 0);  //num_feature_groups() => data_info_->feature_group_infos_.size();

  size_t idx = 0;
  for (auto& fg_info : data_info_->feature_group_infos_) {
    data_info_->feature_group_global_idx_[idx++] = data_info_->total_dimensions_;
    data_info_->total_dimensions_ += fg_info.size();
  }

  LOG(INFO) << "Data loaded successfully.\n";
  LOG(INFO) << *this; // operator<< overload
}

// overload operator<< to display an object
std::ostream& operator<< (std::ostream& stream, const Data& data) {

  stream << "\nData set summary : \n";
  stream << "\tNum of Instance: " << data.instances_.size() << std::endl;
  stream << "\tNum of feature groups: " << data.data_info_->feature_group_infos_.size() << std::endl;
  stream << "\tTotal feature dimensions: " << data.data_info_->total_dimensions_ << std::endl;
  stream << "\tFeature group idx scope: [";
  for (size_t idx = 0; idx < data.data_info_->feature_group_global_idx_.size(); ++idx) {
    if (idx > 0) stream << " ";
    stream << data.data_info_->feature_group_global_idx_[idx];   
  }
  stream << "]\n";
  size_t idx = 0;
  for (auto& fg_info : data.data_info_->feature_group_infos_) {
    stream << "\tFeature group " << idx++ << " -> " << fg_info << std::endl;
  }

  stream << "Head of the data set:\n"; 
  size_t num_lines = std::min(size_t{2}, data.instances_.size());
  for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
    auto& ins = data.instances_[line_idx];
    stream << "  " << ins << std::endl;
  }
  return stream;
}

class Data::instance_iterator {
 public:
  instance_iterator(const Data& data, const Instance& ins) :
      data_cref_(&data), instance_cref_(&ins),
      fg_idx_(0), feat_idx_(0) 
  {}

  instance_iterator(const Data& data, const Instance& ins,
                    size_t fg_idx, size_t feat_idx) : 
      data_cref_(&data), instance_cref_(&ins),
      fg_idx_(fg_idx), feat_idx_(feat_idx) 
  {}

  instance_iterator(const instance_iterator&) = default;
  instance_iterator(instance_iterator&&) = default;

  size_t feature_group_idx() const { return fg_idx_; }

  size_t index() const { 
    CHECK_LT(fg_idx_, instance_cref_->num_feature_groups());
    CHECK_LT(feat_idx_, instance_cref_->feature_group_size(fg_idx_));
    return data_cref_->feature_group_start_idx(fg_idx_) 
        + instance_cref_->get_feature_group_index(fg_idx_, feat_idx_); 
  }

  double value() const {
    CHECK_LT(fg_idx_, instance_cref_->num_feature_groups());
    CHECK_LT(feat_idx_, instance_cref_->feature_group_size(fg_idx_));
    return instance_cref_->get_feature_group_value(fg_idx_, feat_idx_); 
  }

  bool operator == (const instance_iterator& oth) {
    return (data_cref_ == oth.data_cref_) && 
        (instance_cref_ == oth.instance_cref_) && 
        (fg_idx_ == oth.fg_idx_) && 
        (feat_idx_ == oth.feat_idx_);
  }

  bool operator != (const instance_iterator& oth) {
    return ! (*this == oth);
  }

  instance_iterator& operator = (const instance_iterator& oth) {
    data_cref_ = oth.data_cref_;
    instance_cref_ = oth.instance_cref_;
    fg_idx_ = oth.fg_idx_;
    feat_idx_ = oth.feat_idx_;
    return *this;
  }

  instance_iterator& operator ++ () {
    if (fg_idx_ < instance_cref_->num_feature_groups() ) {
      if (feat_idx_ < instance_cref_->feature_group_size(fg_idx_) - 1) {
        ++feat_idx_;
      } else {
        ++fg_idx_;
        feat_idx_ = 0;
      }
    }
    return *this;
  }

  instance_iterator operator ++ (int) {
    instance_iterator tmp = *this;
    ++*this;
    return std::move(tmp);
  }

 private: 
  const Data* data_cref_;
  const Instance* instance_cref_; 
  size_t fg_idx_;
  size_t feat_idx_;

};

Data::instance_iterator Data::begin(size_t idx) const {
  CHECK_LT(idx, size());
  return instance_iterator(*this, instances_[idx], 0, 0);
}

Data::instance_iterator Data::end(size_t idx) const {
  CHECK_LT(idx, size());
  return instance_iterator(*this, instances_[idx], instances_[idx].size(), 0);
}

Data::instance_iterator Data::begin(const Instance& ins) const {
  return instance_iterator(*this, ins, 0, 0);
}

Data::instance_iterator Data::end(const Instance& ins) const {
  return instance_iterator(*this, ins, ins.size(), 0);
}


void Data::shuffle_data() {
  Random::shuffle(std::begin(instances_), std::end(instances_));
}


void Data::random_split(Data& train, Data& test, double test_ratio) const {

  CHECK_LT(test_ratio, 1.0);
  // shuffle_data();
  size_t num_train = static_cast<size_t>((1. - test_ratio) * size());
  size_t num_test = size() - num_train;

  std::vector<size_t> index_vec(size(), 0);
  std::iota(index_vec.begin(), index_vec.end(), 0);
  Random::shuffle(std::begin(index_vec), std::end(index_vec)); 

  std::vector<Instance> train_ins_vec(num_train);
  for(size_t idx = 0; idx < num_train; ++idx) {
    train_ins_vec[idx] = instances_[index_vec[idx]];
  }

  std::vector<Instance> test_ins_vec(this->size() - num_train);
  for(size_t idx = 0; idx < num_test; ++idx) {
    test_ins_vec[idx] = instances_[index_vec[num_train + idx]];
  }

  train = Data(std::move(train_ins_vec), data_info_);
  test = Data(std::move(test_ins_vec), data_info_);
}


void save_train_test_data_split(std::vector<Instance> data_vector, 
                                std::string filename) {

  // open the file
  std::ofstream out;
  out.open(filename, std::ofstream::out); // | std::ofstream::app);
  out.clear(); // reset file
  std::cout << filename << " size: " << data_vector.size() << std::endl;

  // size_t num_lines = std::min(size_t{10}, data_vector.size());
  size_t num_lines = data_vector.size();
  for (size_t line_idx = 0; line_idx < num_lines; ++line_idx) {
    auto& ins = data_vector[line_idx]; // each line in the file is an instance

  // get # of feature groups of an instance  =>  users: 0, items: 1
  // std::cout << ins.num_feature_groups() << std::endl; // 2

    // get user_id of the instance
    size_t fg_idx = 0;
    auto user_id = ins.get_feature_group_index(fg_idx,0); 

    // get item_id of the instance
    fg_idx = 1;
    auto item_id = ins.get_feature_group_index(fg_idx,0); 
    auto rating = ins.label();

    // out << ins << std::endl; 
    // out << "{ user_id: "<< user_id << " item_id: "<< item_id << " rating: " << rating << "} " << std::endl; 
    out << user_id << "\t"<< item_id << "\t" << rating << std::endl; 

  // size_t fg_idx = 0;
  //   out << "{Label: " << ins.label() << "}, " << "{Feature Groups: ["; 
  //   out << ins.num_feature_groups() << std::endl; // 2
  //   for (size_t idx = 0; idx < ins.feature_group_size(0); ++idx) {
  //     out << "{" << ins.get_feature_group_index(0, idx) << ": " << ins.get_feature_group_value(0, idx) << "}\n";
  //     // if (ins.get_feature_group_index(0, idx) < ins.feature_group_size(0); - 1) out << ", ";
  //     // ++fg_idx;
  //   }
  //   out << "]";

  }

  out.close();
}


void Data::random_split_by_feature_group(Data& train, Data& test,
                                         size_t feature_group_idx, double test_ratio, 
                                         std::string dataset) const {

  Timer timer;

  size_t est_num_test = static_cast<size_t>(test_ratio * size());
  size_t est_num_train = size() - est_num_test;

  // allocate vector for train and test instances
  std::vector<Instance> train_ins_vec;
  train_ins_vec.reserve(est_num_train + size() * 0.01);
  std::vector<Instance> test_ins_vec;
  test_ins_vec.reserve(est_num_test + size() * 0.01);

  // get_feature_ins_idx_hashtable for users (feature 0)
  auto fg_idx_ins_idx_hashtable = get_feature_ins_idx_hashtable(feature_group_idx);

  size_t cnt = 0;
  size_t num_test;

  for (auto iter = fg_idx_ins_idx_hashtable.begin(); iter != fg_idx_ins_idx_hashtable.end(); ++iter) {
    auto& tmp_vec = iter->second; // iter->second is the value. The key is iter->first. 

    // randomly select up to test_ratio interactions by user (idx) for test data (test_ins_vec)    
    Random::shuffle(std::begin(tmp_vec), std::end(tmp_vec));

    num_test = static_cast<size_t>(tmp_vec.size() * test_ratio);
    
    for(size_t idx = 0; idx < tmp_vec.size(); ++idx) {
      // std::cout<<"idx: "<<idx<<std::endl;
      if (idx < num_test) {
        test_ins_vec.push_back(instances_[tmp_vec[idx]]); 
      } else {
        train_ins_vec.push_back(instances_[tmp_vec[idx]]);
      }
    }
    ++cnt;
  }
  
  CHECK_EQ(cnt, feature_group_total_dimension(feature_group_idx));
  CHECK_EQ(test_ins_vec.size() + train_ins_vec.size(), size()); // check train + test = data

  std::cout << "train_ins_vec.size() = " << train_ins_vec.size() << std::endl;
  std::cout << "test_ins_vec.size() = " << test_ins_vec.size() << std::endl;
  std::cout << dataset << " size() = " << size() << std::endl;

  Random::shuffle(std::begin(train_ins_vec), std::end(train_ins_vec));
  Random::shuffle(std::begin(test_ins_vec), std::end(test_ins_vec));

  // save train and test data in a non-serialied way 
  std::string dir = "data/bin/";
  std::string train_filename = dataset + "_train.txt";
  save_train_test_data_split(train_ins_vec, dir + train_filename);
  std::string test_filename = dataset + "_test.txt";
  save_train_test_data_split(test_ins_vec, dir + test_filename);

  train = Data(std::move(train_ins_vec), data_info_);
  test = Data(std::move(test_ins_vec), data_info_);

  LOG(INFO) << "Finished splitting data set in " << timer;
}

// void Data::inplace_random_split_by_feature_group(Data& train, Data& test,
//                                          size_t feature_group_idx, double test_ratio,
//                                          std::string dataset)  {

//   Timer timer;

//   size_t est_num_test = static_cast<size_t>(test_ratio * size());
//   size_t est_num_train = size() - est_num_test;

//   std::vector<Instance> train_ins_vec;
//   train_ins_vec.reserve(est_num_train + size() * 0.01);
//   std::vector<Instance> test_ins_vec;
//   test_ins_vec.reserve(est_num_test + size() * 0.01);

//   auto fg_idx_ins_idx_hashtable = get_feature_ins_idx_hashtable(feature_group_idx);
//   size_t cnt = 0;
//   size_t num_test;

//   for (auto iter = fg_idx_ins_idx_hashtable.begin(); iter != fg_idx_ins_idx_hashtable.end(); ++iter) {
//     auto& tmp_vec = iter->second; // iter->second is the value. The key is iter->first 
    
//     std::cout<<"inplace_random_split_by_feature_group\n";
//     std::cout<<tmp_vec<<std::endl;
    
//     Random::shuffle(std::begin(tmp_vec), std::end(tmp_vec));
    
//     // for each user, idx, select iterations up to test_ratio
//     num_test = static_cast<size_t>(tmp_vec.size() * test_ratio);
//     for(size_t idx = 0; idx < tmp_vec.size(); ++idx) {
//       if (idx < num_test) {
//         test_ins_vec.push_back(std::move(instances_[tmp_vec[idx]])); 
//       } else {
//         train_ins_vec.push_back(std::move(instances_[tmp_vec[idx]]));
//       }
//     }
//     ++cnt;
    
//     std::cout << "cnt:" << cnt;

//   }
//   CHECK_EQ(cnt, feature_group_total_dimension(feature_group_idx));
//   CHECK_EQ(test_ins_vec.size() + train_ins_vec.size(), size());

//   Random::shuffle(std::begin(train_ins_vec), std::end(train_ins_vec));
//   Random::shuffle(std::begin(test_ins_vec), std::end(test_ins_vec));

//   train = Data(std::move(train_ins_vec), data_info_);
//   test = Data(std::move(test_ins_vec), data_info_);

//   LOG(INFO) << "Finished splitting data set in " << timer;
// }


std::unordered_map<size_t, std::vector<size_t>> 
Data::get_feature_ins_idx_hashtable(size_t feature_group_idx) const {

  // feature_group_idx is < num_feature_groups() which is 2 (users and items)
  CHECK_LT(feature_group_idx, num_feature_groups());

  // create <fg_idx, ins_id> vector
  // std::pair to store two heterogeneous objects as a single unit.  
  std::vector<std::pair<size_t, size_t>> fg_idx_ins_id_pair_vec;
  fg_idx_ins_id_pair_vec.reserve(size());
  size_t idx = 0;
  size_t ft_idx;

  // begin() => return data();
  // end() => return data() + size();
  for(auto iter = begin(); iter != end(); ++iter) {

    // check whether feature_group_idx size corresponds to a single feature group
    CHECK_EQ(iter->feature_group_size(feature_group_idx), 1); 
	
	  // if feature_group_idx = 0 => it returns the user IDs  		
    ft_idx = iter->get_feature_group_index(feature_group_idx, 0) 
        + feature_group_start_idx(feature_group_idx);  // 0
    
    CHECK_GE(ft_idx, feature_group_start_idx(feature_group_idx));
    
    if (feature_group_idx < num_feature_groups() - 1) {
      CHECK_LT(ft_idx, feature_group_start_idx(feature_group_idx + 1));
    } else {
      CHECK_LT(ft_idx, total_dimensions());
    }
    
    // appends the original feature IDs and the corresponding position in the vector
    fg_idx_ins_id_pair_vec.emplace_back(ft_idx, idx++); 
  }
	  
  // std::cout << "fg_idx_ins_id_pair_vec" << fg_idx_ins_id_pair_vec;
  /*fg_idx_ins_id_pair_vec[(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),...,(18,3242),(52,3243),(226,3244),(173,3245),(118,3246),(169,3247),(118,3248),(110,3249),(122,3250),(105,3251)]
  fg_idx_ins_id_pair_vec[(0,0),(1,1),(2,2),(3,3),(4,4),(5,5),(6,6),(7,7),(8,8),(9,9),...,(52,3243),(226,3244),(173,3245),(118,3246),(169,3247),(118,3248),(110,3249),(122,3250),(105,3251),(26,3252)]*/


  // std::cout << "fg_idx_ins_id_pair_vec" << fg_idx_ins_id_pair_vec;

  CHECK_EQ(idx, size());
  CHECK_EQ(fg_idx_ins_id_pair_vec.size(), size()); 

  std::sort(fg_idx_ins_id_pair_vec.begin(),
            fg_idx_ins_id_pair_vec.end());

  auto pair_vec_iter = fg_idx_ins_id_pair_vec.begin();
  auto pair_vec_internal_iter = fg_idx_ins_id_pair_vec.begin();
  auto pair_vec_iter_end = fg_idx_ins_id_pair_vec.end();
  size_t cnt = 0;

  std::unordered_map<size_t, std::vector<size_t>> rets;
  rets.reserve(feature_group_total_dimension(feature_group_idx));

  std::vector<size_t> tmp_vec;
  while (pair_vec_iter != pair_vec_iter_end) { //.begin() !=  .end()
    
    pair_vec_internal_iter = pair_vec_iter;
    while(++pair_vec_internal_iter != pair_vec_iter_end) {
      if (pair_vec_iter->first != pair_vec_internal_iter->first) {
        break;
      } 
    }

    tmp_vec.resize(std::distance(pair_vec_iter, pair_vec_internal_iter));

    std::transform(pair_vec_iter,           // first1
                   pair_vec_internal_iter,  // last1
                   tmp_vec.begin(),         // result
                   [](const std::pair<size_t, size_t>& v) {     // op
                      return v.second;
                   });
    
    CHECK(std::is_sorted(tmp_vec.begin(), tmp_vec.end()));
    rets[pair_vec_iter->first - feature_group_start_idx(feature_group_idx)] = std::move(tmp_vec);
    pair_vec_iter = pair_vec_internal_iter;
    ++cnt;
  }
  //CHECK_EQ(cnt, feature_group_total_dimension(feature_group_idx));
  return std::move(rets);
}


std::unordered_map<size_t, std::vector<size_t>> 
Data::get_feature_to_vec_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const {
  auto rets = get_feature_ins_idx_hashtable(feature_group_idx_a);
  std::vector<size_t> tmp_vec;
  for (auto outer_iter = rets.begin(); outer_iter != rets.end(); ++outer_iter) {
    tmp_vec.assign(outer_iter->second.size(), 0);
    size_t idx = 0;
    for (auto& v : outer_iter->second) {
      tmp_vec[idx++] = instances_[v].get_feature_group_index(feature_group_idx_b, 0);// + feature_group_start_idx(feature_group_idx_b);
    }
    std::sort(tmp_vec.begin(), tmp_vec.end());
    outer_iter->second = std::move(tmp_vec);
  }
  return std::move(rets);
}

std::unordered_map<size_t, std::unordered_set<size_t>> 
Data::get_feature_to_set_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const {
  std::unordered_map<size_t, std::unordered_set<size_t>> rets;
  std::unordered_set<size_t> tmp_set;
  
  auto feature_ins_table = get_feature_ins_idx_hashtable(feature_group_idx_a);
  rets.reserve(feature_ins_table.size());

  for (auto outer_iter = feature_ins_table.begin(); outer_iter != feature_ins_table.end(); ++outer_iter) {
    tmp_set.clear();
    tmp_set.reserve(outer_iter->second.size());
    for (auto& v : outer_iter->second) {
      tmp_set.insert(instances_[v].get_feature_group_index(feature_group_idx_b, 0));
    }
    rets[outer_iter->first] = std::move(tmp_set);
  }
  CHECK_EQ(rets.size(), feature_ins_table.size());
  return std::move(rets);
}

/*
 * Get user ratings  
*/
std::unordered_map<size_t, std::unordered_map<size_t, double>> 
Data::get_feature_pair_label_hashtable(size_t feature_group_idx_a, 
                                 size_t feature_group_idx_b) const {
  
  auto feat_ins_hashtable = get_feature_ins_idx_hashtable(feature_group_idx_a);
  
  std::unordered_map<size_t, std::unordered_map<size_t, double>> rets; 
  rets.reserve(feat_ins_hashtable.size());
  
  std::unordered_map<size_t, double> tmp_map; // std::pair<const Key, Ty>
  
  for (auto outer_iter = feat_ins_hashtable.begin(); outer_iter != feat_ins_hashtable.end(); ++outer_iter) {
    tmp_map.clear();
    
    for (auto& v : outer_iter->second) {
      //tmp_vec[idx++] = std::make_pair(instances_[v].get_feature_group_index(feature_group_idx_b, 0) + feature_group_start_idx(feature_group_idx_b), instances_[v].label());
      tmp_map.insert(std::make_pair(instances_[v].get_feature_group_index(feature_group_idx_b, 0), instances_[v].label()));
    }

    // display contents of tmp_map
    // for (auto it = tmp_map.begin(); it != tmp_map.end(); ++it)
    //     std::cout << " [" << it->first << ", " << it->second << "]";
    // std::cout << std::endl;
     
    rets[outer_iter->first] = std::move(tmp_map);
  }

  return std::move(rets);
}



} // namesapce 