// Generic pcl
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>


using namespace std;
using namespace pcl;


template<typename FeatureType>
class Features
{
 public:

  // Class constructor
  Features() {}
  Features(typename Feature<PointXYZ, FeatureType>::Ptr feature_extractor);
  Features(typename Feature<PointXYZ, FeatureType>::Ptr feature_extractor,
           const double feat_radius_search,
           const double normal_radius_search);

  // Feature computation
  void compute(const PointCloudXYZPtr cloud,
               const PointCloudXYZPtr keypoints,
               typename PointCloud<FeatureType>::Ptr& descriptors);

  // Search for correspondences
  void findCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                           typename PointCloud<FeatureType>::Ptr target,
                           CorrespondencesPtr& correspondences);

  // Get one direction correspondences
  void getCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                          typename PointCloud<FeatureType>::Ptr target,
                          vector<int>& source2target);

  // Correspondence filtering
  void filterCorrespondences(const PointCloudXYZ::Ptr source,
                             const PointCloudXYZ::Ptr target,
                             const CorrespondencesPtr correspondences,
                             CorrespondencesPtr& filtered_correspondences,
                             Eigen::Matrix4f& transformation,
                             float inlierThreshold=0.015,
                             int nIterations=1000
                             );

  // Set common feature properties
  void setFeatureRadiusSearch(double radius_search);
  void setNormalRadiusSearch(double radius_search);

 private:
  typename Feature<PointXYZ, FeatureType>::Ptr feature_extractor_;
  double feat_radius_search_;
  double normal_radius_search_;

};




/** \brief Class constructor. Initialize the class
  */
template<typename FeatureType>
Features<FeatureType>::Features(typename Feature<PointXYZ, FeatureType>::Ptr feature_extractor)
  : feature_extractor_(feature_extractor)
{
  feat_radius_search_ = 0.08;
  normal_radius_search_ = 0.05;
}
  template<typename FeatureType>
Features<FeatureType>::Features(typename Feature<PointXYZ, FeatureType>::Ptr feature_extractor,
                                const double feat_radius_search,
                                const double normal_radius_search) :
    feature_extractor_(feature_extractor),
    feat_radius_search_(feat_radius_search),
    normal_radius_search_(normal_radius_search) {}


/** \brief Compute descriptors
  * @return
  * \param Input cloud
  * \param Input keypoints, where features will be computed
  * \param Output descriptors
  */
template<typename FeatureType>
void Features<FeatureType>::compute(const PointCloudXYZPtr cloud,
                                    const PointCloudXYZPtr keypoints,
                                    typename PointCloud<FeatureType>::Ptr& descriptors)
{

  typename FeatureFromNormals<PointXYZ, Normal, FeatureType>::Ptr feature_from_normals =
    boost::dynamic_pointer_cast<FeatureFromNormals<PointXYZ, Normal, FeatureType> >(feature_extractor_);

  if(feature_from_normals)
  {
    typename PointCloud<Normal>::Ptr normals (new PointCloud<Normal>);
    Tools::estimateNormals(cloud, normals, normal_radius_search_);
    feature_from_normals->setInputNormals(normals);
  }
  feature_extractor_->setSearchSurface(cloud);
  feature_extractor_->setInputCloud(keypoints);
  search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
  feature_extractor_->setSearchMethod(kdtree);
  feature_extractor_->setRadiusSearch(feat_radius_search_);
  feature_extractor_->compute(*descriptors);
}

/** \brief Sets the feature radius search
  * @return
  * \param Radius search
  */
template<typename FeatureType>
void Features<FeatureType>::setFeatureRadiusSearch(double radius_search)
{
  feat_radius_search_ = radius_search;
}

/** \brief Sets the normals radius search
  * @return
  * \param Radius search
  */
template<typename FeatureType>
void Features<FeatureType>::setNormalRadiusSearch(double radius_search)
{
  normal_radius_search_ = radius_search;
}

/** \brief Find correspondences between features
  * @return
  * \param Source cloud features
  * \param Target cloud features
  * \param Vector of correspondences
  */
template<typename FeatureType>
void Features<FeatureType>::findCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                                                typename PointCloud<FeatureType>::Ptr target,
                                                CorrespondencesPtr& correspondences)
{
  vector<int> source2target;
  vector<int> target2source;

  boost::thread thread1(&Features::getCorrespondences, this, boost::ref(source), boost::ref(target), boost::ref(source2target));
  boost::thread thread2(&Features::getCorrespondences, this, boost::ref(target), boost::ref(source), boost::ref(target2source));

  // Wait until both threads have finished
  thread1.join();
  thread2.join();

  // now populate the correspondences vector
  vector<pair<unsigned, unsigned> > c;
  for (unsigned c_idx = 0; c_idx < source2target.size (); ++c_idx)
    if (target2source[source2target[c_idx]] == c_idx)
      c.push_back(make_pair(c_idx, source2target[c_idx]));

  correspondences->resize(c.size());
  for (unsigned c_idx = 0; c_idx < c.size(); ++c_idx)
  {
    (*correspondences)[c_idx].index_query = c[c_idx].first;
    (*correspondences)[c_idx].index_match = c[c_idx].second;
  }
}

template<typename FeatureType>
void Features<FeatureType>::getCorrespondences(typename PointCloud<FeatureType>::Ptr source,
                                               typename PointCloud<FeatureType>::Ptr target,
                                               vector<int>& source2target)
{
  const int k = 1;
  vector<int> k_indices(k);
  vector<float> k_dist(k);
  source2target.clear();
  KdTreeFLANN<FeatureType> descriptor_kdtree;

  // Find the index of the best match for each keypoint
  // From source to target
  descriptor_kdtree.setInputCloud(target);
  source2target.resize(source->size());
  for (size_t i = 0; i < source->size(); ++i)
  {
    descriptor_kdtree.nearestKSearch(*source, i, k, k_indices, k_dist);
    source2target[i] = k_indices[0];
  }
}

/** \brief Filter the correspondences by RANSAC
  * @return
  * \param Source cloud features
  * \param Target cloud features
  * \param Vector of correspondences
  * \param Output filtered vector of correspondences
  */
template<typename FeatureType>
void Features<FeatureType>::filterCorrespondences(const PointCloudXYZPtr source,
                                                  const PointCloudXYZPtr target,
                                                  CorrespondencesPtr correspondences,
                                                  CorrespondencesPtr& filtered_correspondences,
                                                  Eigen::Matrix4f& transformation,
                                                  float inlierThreshold,
                                                  int nIterations
                                                  )
{
  registration::CorrespondenceRejectorSampleConsensus<PointXYZ> rejector;
  rejector.setInputSource(source);
  rejector.setInputTarget(target);
  rejector.setInputCorrespondences(correspondences);
  rejector.setInlierThreshold(inlierThreshold);
  rejector.setMaximumIterations(nIterations);
  rejector.getCorrespondences(*filtered_correspondences);
  transformation = rejector.getBestTransformation();
}

