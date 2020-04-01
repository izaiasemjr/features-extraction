#include <pcl/visualization/vtk.h> //http://www.pcl-users.org/pcl-plotter-Error-td4031251.html
#include<pcl/visualization/histogram_visualizer.h>
#include<pcl/visualization/pcl_plotter.h>

// my includes
#include "./include/utils.hpp"
#include "./include/keypoints.hpp"
//#include "./include/features.hpp"
#include "./include/features_impl.hpp"

 using namespace std;

int main (int argc, char** argv){

    vector<string> keypoints_params = get_parameters(argc,argv, "-keypoints",",");
    vector<string> cloud_params = get_parameters(argc,argv, "-cloud",",");
    vector<string> features_params = get_parameters(argc,argv, "-features",",");
    vector<string> output_params = get_parameters(argc,argv, "-output",":");

    /// source
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
    openCloudPCD(cloud_params[0].c_str(),source_cloud);
    cout<<"Cloud ok: "<<source_cloud->points.size()<<" points"<<endl;

    /// Keypoints
    Keypoints keypoints(keypoints_params[0],keypoints_params[1]) ;
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_keypoints (new pcl::PointCloud<pcl::PointXYZ> ());
    keypoints.getKeypointsFromFile(source_keypoints);
    cout<<"keypoints ok: "<<source_keypoints->points.size()<<" keypoints"<<endl;


    vector<vector<float>> features = extractFeatures(source_cloud,source_keypoints,features_params,output_params,keypoints_params[2]);
    cout<<"features ok: "<<features.size()<<" vetores, dim: "<<features[0].size()<<endl;


//    vector<double> x,y ;
//    for(int i ; i<50;i++){
//        x.push_back(i);
//        cout<<"descriptor ["<<i<<"]:"<<1000*source_features->points.front().descriptor[i]<<endl;
//        y.push_back(1+100*source_features->points.front().descriptor[i]);
//    }
//    pcl::visualization::PCLPlotter plotter;
//    plotter.addPlotData(x,y);
//    plotter.setWindowSize(1000,800);
//    plotter.setShowLegend(true);
//    plotter.plot();


    return (0);
}


