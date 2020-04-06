// basics PCL manipulations
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <iostream>

// pcl utils
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/gp3.h>
#include <pcl/filters/filter.h>


typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointCloud<PointXYZ>::Ptr PointCloudXYZPtr;
typedef pcl::PointCloud<PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<pcl::PointXYZ>::ConstPtr PointCloudConstPtr;
typedef pcl::PointXYZ PointXYZ;
typedef pcl::PointCloud<PointXYZ> PointCloudXYZ;
typedef pcl::PointCloud<PointXYZ>::Ptr PointCloudXYZPtr;


using namespace pcl;
using namespace std;


// List of landmarks Bosphorus
static const string NOSE_TIP           = "nose_tip";
static const string EYE_RIGHT_INTERN   = "eye_ri";
static const string EYE_RIGHT_EXTERN   = "eye_re";
static const string EYE_LEFT_INTERN    = "eye_li";
static const string EYE_LEFT_EXTERN    = "eye_le";
static const string EYE_UP_LEFT_EXTERN = "eye_ule";
static const string EYE_UP_LEFT_INTERN = "eye_uli";
static const string EYE_UP_RIGHT_INTERN = "eye_uri";
static const string EYE_UP_RIGHT_EXTERN= "eye_ure";
static const string MOUTH_RIGHT        = "mouth_r";
static const string MOUTH_LEFT         = "mouth_l";
static const string MOUTH_CENTER_UP    = "mouth_cu";
static const string MOUTH_CENTER_DOWN  = "mouth_cd";




class Tools {
 public:
  /** \brief Normal estimation
    * @return
    * \param Cloud where normals will be estimated
    * \param Cloud surface with additional information to estimate the features for every point in the input dataset
    * \param Output cloud with normals
    */
  static void estimateNormals(const PointCloudXYZ::Ptr& cloud,
                              PointCloud<Normal>::Ptr& normals,
                              double radius_search)
  {
    NormalEstimationOMP<PointXYZ, Normal> normal_estimation_omp;
    normal_estimation_omp.setInputCloud(cloud);
    normal_estimation_omp.setRadiusSearch(radius_search);
    search::KdTree<PointXYZ>::Ptr kdtree_omp(new search::KdTree<PointXYZ>);
    normal_estimation_omp.setSearchMethod(kdtree_omp);
    normal_estimation_omp.compute(*normals);
  }
};


/**
 * get indices of keypoints manually marked on bosphorus
 * or other bases with manual marks
 */

void getIndicesKeypoints(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypoints,
        PointIndicesPtr indices,
        string local_point="nose_tip"
        ){

//    std::vector<int> indices2 ;
//    removeNaNFromPointCloud(*cloud_keypoints, *cloud_keypoints,indices2);
    PointCloudXYZPtr cloud_keypoints_choice(new PointCloudXYZ());

    cout<<local_point<<endl;


    // nose tip
    if(local_point == NOSE_TIP)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[13]);

    // eye corner up_left-extern
    else if(local_point == EYE_UP_LEFT_EXTERN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[1])   ;
    //eye corner up_left-intern
    else if(local_point == EYE_UP_LEFT_INTERN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[2]);
    // eye corner up_right-intern
    else if(local_point == EYE_UP_RIGHT_INTERN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[3])   ;
    //eye corner up_right-extern
    else if(local_point == EYE_UP_RIGHT_EXTERN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[4]);

    // eye corner right-extern
    else if(local_point == EYE_RIGHT_EXTERN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[9]);
    //eye corner right-intern
    else if(local_point == EYE_RIGHT_INTERN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[8]);
    // eye corner left-extern
    else if(local_point == EYE_LEFT_EXTERN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[6]);
    // eye corner left-intern
    else if(local_point == EYE_LEFT_INTERN)
       cloud_keypoints_choice->push_back(cloud_keypoints->points[7]);

    // mouth corner left
    else if(local_point == MOUTH_LEFT)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[15]);
    // mouth corner right
    else if(local_point == MOUTH_RIGHT)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[17]);
    // mouth center up
    else if(local_point == MOUTH_CENTER_UP)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[16]);
    // mouth center down
    else if(local_point == MOUTH_CENTER_DOWN)
        cloud_keypoints_choice->push_back(cloud_keypoints->points[20]);


    else {
        cout<<"Choice a valid option for keypoint!\nList Available:"
        <<NOSE_TIP<<"\n"
        << EYE_RIGHT_EXTERN<<"\n"
        << EYE_RIGHT_INTERN<<"\n"
        << EYE_LEFT_EXTERN<<"\n"
        << EYE_LEFT_INTERN<<"\n"
        << MOUTH_LEFT<<"\n"
        << MOUTH_RIGHT<<"\n"
        << MOUTH_CENTER_UP<<"\n"
        << MOUTH_CENTER_DOWN<<"\n"<<endl;

        exit(1);
    }

    // verify if is nan
    if(!isFinite(cloud_keypoints_choice->points[0])){
        return;
    }

    /** get indexes **/
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
    vector<float> dist;
    vector<int> idx;
    for (int i =0;i<cloud_keypoints_choice->points.size();++i){
        tree.nearestKSearch(cloud_keypoints_choice->points [i],1, idx,dist);
        indices->indices.push_back(idx[0]);
    }

}



template<typename FeatureType>
vector<vector<float>> descriptorsVectorToFile(vector<string> output_params,
                                       typename PointCloud<FeatureType>::Ptr features ){

    // num of vectors features
    int n_features = features->points.size();
    // dimension of each vector
    int d_features = features->front().descriptorSize();

    //vector features
    std::ofstream file;
    file.open(output_params[0].c_str(), std::ios_base::app);
    vector<vector<float>> vectors_features;
    vector<float> features_values;
    for(int n=0;n<n_features;++n){
        for(int f=0; f<d_features;++f){
            features_values.push_back(features->points[n].descriptor[f]);
            file<<features->points[n].descriptor[f]<<",";
        }
        vectors_features.push_back(features_values);
        features_values.clear();
        file<<output_params[1]<<"\n";
    }

    file.close();
    return vectors_features;
}



template<typename FeatureType>
vector<vector<float>> histogramsVectorToFile(vector<string> output_params,
                                            typename PointCloud<FeatureType>::Ptr features ){


    // num of vectors features
    int n_features = features->points.size();
    // dimension of each vector
    int d_features = features->front().descriptorSize();

    //vector features
    std::ofstream file;
    file.open(output_params[0].c_str(), std::ios_base::app);
    vector<vector<float>> vectors_features;
    vector<float> features_values;

    // if there's no features, fill with zeros
    if(n_features==0){
        for(int f=0; f<d_features;f++){
            features_values.push_back(0);
            file<<0<<",";
        }
        vectors_features.push_back(features_values);
        features_values.clear();
        file<<output_params[1]<<"\n";
    }

    for(int n=0;n<n_features;n++){
        for(int f=0; f<d_features;f++){
            features_values.push_back(features->points[n].histogram[f]);
            file<<features->points[n].histogram[f]<<",";
        }
        vectors_features.push_back(features_values);
        features_values.clear();
        file<<output_params[1]<<"\n";
    }
//    vectors_features.push_back(features_values);
//    features_values.clear();
//    file<<output_params[1]<<"\n";

    file.close();

    return vectors_features;
}



/** Put a symbol for verification (ex: "-icp" ), a delimiter to separate the parameters(",") and the numbers of parameters
 * separeted by this delimiter. The function return a vector of strings (tokens) where each parameter is a string.
 * example: get_parameters(argc,argv,"-icp", ",");
*/
vector<string> get_parameters(int argc, char **argv ,const char *symbol, const char *delimiter ){

    std::string params_string;
    bool match = pcl::console::parse_argument (argc, argv, symbol, params_string) > 0;
    if (!match){PCL_ERROR ("Couldn't read %s parameters \n (exit is called) \n ", symbol);
        exit(1);
    }

    std::vector<std::string> tokens;
    boost::split (tokens, params_string, boost::is_any_of(delimiter), boost::token_compress_on);

    return tokens;
}

/** Open cloud in pcd. If the cannot open the file, the program finish
*/
void openCloudPCD(const char* file, PointCloudXYZPtr cloud ){

    if (pcl::io::loadPCDFile<pcl::PointXYZ> (file, *cloud) == -1){
        PCL_ERROR ("Couldn't read this path: %s \n (exit is called)\n", file);
        exit(1);
    }
}

/**
 * Save cloud in pcd.
*/
void saveCloudPCD(const char* file,PointCloudXYZPtr cloud){
    cloud->width=cloud->points.size();
    cloud->height = 1;
    pcl::io::savePCDFileASCII (file, *cloud);
}

std::vector<std::string> split(const std::string& s, char delimiter)
{
   std::vector<std::string> tokens;
   std::string token;
   std::istringstream tokenStream(s);
   while (std::getline(tokenStream, token, delimiter))
   {
      tokens.push_back(token);
   }
   return tokens;
}

/**
*
* Save Feature Stogram in file
*/
string bosphorusLabels(string label){
    vector<std::string> results = split(label,'_');
    string r;
    for(int i=0;i<results.size();++i){
       r = i==results.size()-1 ?r + results[i]: r + results[i] + string(",");
    }
    return r;
}


template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}



/// mesh generation (gp3)
void reconstructionGP3(PointCloudXYZPtr cloud, search::KdTree<PointXYZ>::Ptr kdtree,PolygonMesh &triangles,
                       double normalRadiusSearch, double meshSearchRadius,
                       double mu , int maximumNearestNeighbors
                       ){


    // Object for storing the normals.
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // Object for storing both the points and the normals.
    pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointNormal>);

    Tools::estimateNormals(cloud, normals, normalRadiusSearch);


    // Perform triangulation.
    pcl::concatenateFields(*cloud, *normals, *cloudNormals);
    pcl::search::KdTree<pcl::PointNormal>::Ptr kdtree2(new pcl::search::KdTree<pcl::PointNormal>);
    kdtree2->setInputCloud(cloudNormals);

    pcl::GreedyProjectionTriangulation<pcl::PointNormal> triangulation;
    triangulation.setSearchRadius(meshSearchRadius);
    triangulation.setMu(mu);
    triangulation.setMaximumNearestNeighbors(maximumNearestNeighbors);
    triangulation.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees.
    triangulation.setNormalConsistency(false);
    triangulation.setMinimumAngle(M_PI / 18); // 10 degrees.
    triangulation.setMaximumAngle(2 * M_PI / 3); // 120 degrees.
    triangulation.setInputCloud(cloudNormals);
    triangulation.setSearchMethod(kdtree2);
    triangulation.reconstruct(triangles);
}




void getIndicesKeypointsFromNose(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_keypoints,
        PointIndicesPtr indices,
        string local_point="nose_tip"
        ){

    PointCloudXYZPtr cloud_keypoints_choice(new PointCloudXYZ());

    PointXYZ nose = cloud_keypoints->points[13];

    // nose tip
    if(local_point == NOSE_TIP)
        cloud_keypoints_choice->push_back(nose);

    //    // eye corner up_left-extern
    //    else if(local_point == EYE_UP_LEFT_EXTERN)
    //        cloud_keypoints_choice->push_back(cloud_keypoints->points[1])   ;
    //    //eye corner up_left-intern
    //    else if(local_point == EYE_UP_LEFT_INTERN)
    //        cloud_keypoints_choice->push_back(cloud_keypoints->points[2]);
    //    // eye corner up_right-intern
    //    else if(local_point == EYE_UP_RIGHT_INTERN)
    //        cloud_keypoints_choice->push_back(cloud_keypoints->points[3])   ;
    //    //eye corner up_right-extern
    //    else if(local_point == EYE_UP_RIGHT_EXTERN)
    //        cloud_keypoints_choice->push_back(cloud_keypoints->points[4]);

    // eye corner right-extern
    else if(local_point == EYE_RIGHT_EXTERN)
        cloud_keypoints_choice->push_back(PointXYZ(nose.x + 43.15,nose.y + 30.76,nose.z  -50.44));
    //eye corner right-intern
    else if(local_point == EYE_RIGHT_INTERN)
        cloud_keypoints_choice->push_back(PointXYZ(nose.x + 12.75,nose.y + 29.22,nose.z  -42.09));
    // eye corner left-extern
    else if(local_point == EYE_LEFT_EXTERN)
        cloud_keypoints_choice->push_back(PointXYZ(nose.x - 50.45,nose.y + 26.77,nose.z  -45.81));
    // eye corner left-intern
    else if(local_point == EYE_LEFT_INTERN)
       cloud_keypoints_choice->push_back(PointXYZ(nose.x -19.66,nose.y + 27.69,nose.z  -40.48));

    // mouth corner left
    else if(local_point == MOUTH_LEFT)
        cloud_keypoints_choice->push_back(PointXYZ(nose.x -24.48,nose.y -39.36,nose.z  -27.24));
    // mouth corner right
    else if(local_point == MOUTH_RIGHT)
        cloud_keypoints_choice->push_back(PointXYZ(nose.x + 25.27,nose.y - 36.84,nose.z  -29.99));
    // mouth center up
    else if(local_point == MOUTH_CENTER_UP)
        cloud_keypoints_choice->push_back(PointXYZ(nose.x + 0.63,nose.y -29.12,nose.z  -12.6));
    // mouth center down
    else if(local_point == MOUTH_CENTER_DOWN)
        cloud_keypoints_choice->push_back(PointXYZ(nose.x +1.23,nose.y -44.05,nose.z  -14.51));
    else {
        cout<<"Choice a valid option for keypoint!\nList Available:"
        <<NOSE_TIP<<"\n"
        << EYE_RIGHT_EXTERN<<"\n"
        << EYE_RIGHT_INTERN<<"\n"
        << EYE_LEFT_EXTERN<<"\n"
        << EYE_LEFT_INTERN<<"\n"
        << MOUTH_LEFT<<"\n"
        << MOUTH_RIGHT<<"\n"
        << MOUTH_CENTER_UP<<"\n"
        << MOUTH_CENTER_DOWN<<"\n"<<endl;
        exit(1);
    }

    saveCloudPCD("keypoints_from_nose.pcd",cloud_keypoints_choice);

    // verify if is nan
    if(!isFinite(cloud_keypoints_choice->points[0])){
        return;
    }

    /** get indexes **/
    pcl::search::KdTree<pcl::PointXYZ> tree;
    tree.setInputCloud(cloud);
    vector<float> dist;
    vector<int> idx;
    for (int i =0;i<cloud_keypoints_choice->points.size();++i){
        tree.nearestKSearch(cloud_keypoints_choice->points [i],1, idx,dist);
        indices->indices.push_back(idx[0]);
    }


}







