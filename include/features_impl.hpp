#include <boost/filesystem.hpp>
#include <boost/assign/list_inserter.hpp>

// pcl features
#include <pcl/features/3dsc.h>
#include <pcl/features/usc.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/rops_estimation.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/moment_invariants.h>
#include <pcl/features/shot.h>
#include <pcl/features/spin_image.h>
#include <pcl/features/vfh.h>
#include <pcl/features/grsd.h>

// surface methode
#include <pcl/surface/gp3.h>


// generic class to compute features
#include "features.hpp"


// List the available descriptors
static const string DESC_SHAPE_CONTEXT = "ShapeContext";
static const string DESC_USC           = "USC";
static const string DESC_BOARD         = "BOARD";
static const string DESC_BOUNDARY      = "Boundary";
static const string DESC_INT_GRAD      = "IntensityGradient";
static const string DESC_INT_SPIN      = "IntensitySpin";
static const string DESC_RIB           = "RIB";
static const string DESC_SPIN_IMAGE    = "SI";
static const string DESC_MOMENT_INV    = "MI";
static const string DESC_CRH           = "CRH";
static const string DESC_DIFF_OF_NORM  = "DifferenceOfNormals";
static const string DESC_ESF           = "ESF";
static const string DESC_GFPFH          = "GFPFH";
static const string DESC_FPFH          = "FPFH";
static const string DESC_NARF          = "NARF";
static const string DESC_VFH           = "VFH";
static const string DESC_CVFH          = "CVFH";
static const string DESC_PFH           = "PFH";
static const string DESC_PPAL_CURV     = "PrincipalCurvatures";
static const string DESC_RIFT          = "RIFT";
static const string DESC_SHOT          = "SHOT";
static const string DESC_SHOT_COLOR    = "SHOTColor";
static const string DESC_SHOT_LRF      = "SHOTLocalReferenceFrame";
static const string DESC_ROPS          = "ROPS";
static const string DESC_GRSD          = "GRSD";



vector<vector<float>>  estimateGRSD(PointCloudXYZPtr cloud,
                                  PointCloudXYZPtr keypoints,
                                  vector<string> params,
                                  vector<string> output_params,
                                  string local_point="nose_tip"
                                  ){



    double normal_radius_search_ = stod(params[1]);
    double feat_radius_search_ = stod(params[2]);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // Object for storing the GRSD descriptors for each point.
    pcl::PointCloud<pcl::GRSDSignature21>::Ptr descriptors(new pcl::PointCloud<pcl::GRSDSignature21>());


    // Note: you would usually perform downsampling now. It has been omitted here
    // for simplicity, but be aware that computation can take a long time.

    // Estimate the normals.
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setRadiusSearch(normal_radius_search_);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    // get indices
    pcl::PointIndicesPtr indices (new pcl::PointIndices);
    getIndicesKeypoints(cloud,keypoints,indices,local_point);

    // GRSD estimation object.
    pcl::GRSDEstimation<pcl::PointXYZ, pcl::Normal, pcl::GRSDSignature21> grsd;
    grsd.setInputCloud(cloud);
    grsd.setInputNormals(normals);
    grsd.setSearchMethod(kdtree);
    grsd.setIndices(indices);
    // Search radius, to look for neighbors. Note: the value given here has to be
    // larger than the radius used to estimate the normals.
    grsd.setRadiusSearch(feat_radius_search_);

    grsd.compute(*descriptors);

    return histogramsVectorToFile<GRSDSignature21>(output_params,descriptors);


}

vector<vector<float>>  estimateVFH(PointCloudXYZPtr cloud,
                                  PointCloudXYZPtr keypoints,
                                  vector<string> params,
                                  vector<string> output_params){


        double normal_radius_search_ = stod(params[1]);
        double feat_radius_search_ = stod(params[2]);

        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        // Object for storing the VFH descriptor.
        pcl::PointCloud<pcl::VFHSignature308>::Ptr descriptors(new pcl::PointCloud<pcl::VFHSignature308>);

        // Estimate the normals.
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(cloud);
        normalEstimation.setRadiusSearch(normal_radius_search_);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);
        normalEstimation.compute(*normals);

        // get indices
        pcl::PointIndicesPtr indices (new pcl::PointIndices);
        getIndicesKeypoints(cloud,keypoints,indices);

        // VFH estimation object.
        pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
        vfh.setInputCloud(cloud);
        vfh.setInputNormals(normals);
        vfh.setSearchMethod(kdtree);
        vfh.setIndices(indices);
        // Optionally, we can normalize the bins of the resulting histogram,
        // using the total number of points.
        vfh.setNormalizeBins(true);
        // Also, we can normalize the SDC with the maximum size found between
        // the centroid and any of the cluster's points.
        vfh.setNormalizeDistance(false);

        vfh.compute(*descriptors);

        return histogramsVectorToFile<VFHSignature308>(output_params,descriptors);

}


vector<vector<float>>  estimateSI(PointCloudXYZPtr cloud,
                                  PointCloudXYZPtr keypoints,
                                  vector<string> params,
                                  vector<string> output_params){


        double normal_radius_search_ = stod(params[1]);
        double feat_radius_search_ = stod(params[2]);
        double image_width_ = stod(params[3]);

        // Object for storing the normals.
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        // Object for storing the spin image for each point.
        pcl::PointCloud<Histogram<153>>::Ptr descriptors(new pcl::PointCloud<Histogram<153>>());

        Tools::estimateNormals(cloud, normals, normal_radius_search_);


        // get indices
        pcl::PointIndicesPtr indices (new pcl::PointIndices);
        getIndicesKeypoints(cloud,keypoints,indices);

        // Spin image estimation object.
        pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, Histogram<153>> si;
        si.setInputCloud(cloud);
        si.setInputNormals(normals);
        si.setIndices(indices);
        // Radius of the support cylinder.
        si.setRadiusSearch(feat_radius_search_);
        // Set the resolution of the spin image (the number of bins along one dimension).
        // Note: you must change the output histogram size to reflect this.
        si.setImageWidth(image_width_);

        si.compute(*descriptors);

        return histogramsVectorToFile<Histogram<153>>(output_params,descriptors);

}

vector<vector<float>> estimateSHOT(PointCloudXYZPtr cloud,
                                                PointCloudXYZPtr keypoints,
                                                vector<string> params,
                                                vector<string> output_params){

        double normal_radius_search_ = stod(params[1]);
        double feat_radius_search_ = stod(params[2]);

         // Object for storing the normals.
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        // Object for storing the SHOT descriptors for each point.
        pcl::PointCloud<pcl::SHOT352>::Ptr descriptors(new pcl::PointCloud<pcl::SHOT352>());


        // Note: you would usually perform downsampling now. It has been omitted here
        // for simplicity, but be aware that computation can take a long time.

        // Estimate the normals.
        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
        normalEstimation.setInputCloud(cloud);
        normalEstimation.setRadiusSearch(normal_radius_search_);
        pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
        normalEstimation.setSearchMethod(kdtree);
        normalEstimation.compute(*normals);


        // SHOT estimation object.
        pcl::SHOTEstimation<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> shot;
        shot.setInputCloud(cloud);
        shot.setInputNormals(normals);

        // get indices
        pcl::PointIndicesPtr indices (new pcl::PointIndices);
        getIndicesKeypoints(cloud,keypoints,indices);
        shot.setIndices(indices);

        // The radius that defines which of the keypoint's neighbors are described.
        // If too large, there may be clutter, and if too small, not enough points may be found.
        shot.setRadiusSearch(feat_radius_search_);

        shot.compute(*descriptors);

    return descriptorsVectorToFile<SHOT352>(output_params,descriptors);

}



vector<vector<float>> estimateMomentsInvariants(PointCloudXYZPtr cloud,
                                                PointCloudXYZPtr keypoints,
                                                vector<string> params,
                                                vector<string> output_params){

    double normal_radius_search_ = stod(params[1]);
    double feat_radius_search_ = stod(params[2]);
    // Intrinsec parameter Feature
    MomentInvariantsEstimation<PointXYZ, MomentInvariants>::Ptr feature_extractor_orig(
    new MomentInvariantsEstimation<PointXYZ, MomentInvariants>);

    // get indices
    pcl::PointIndicesPtr indices (new pcl::PointIndices);
    getIndicesKeypoints(cloud,keypoints,indices);
    feature_extractor_orig->setIndices(indices);

    Feature<PointXYZ, MomentInvariants>::Ptr feature_extractor(feature_extractor_orig);
    PointCloud<MomentInvariants>::Ptr features(new PointCloud<MomentInvariants>);
    Features<MomentInvariants> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
    feat.compute(cloud, keypoints, features);


    // num of vectors features
    int n_features = features->points.size();
    // dimension of each vector
    int d_features = 3;

    //vector features
    std::ofstream file;
    file.open(output_params[0].c_str(), std::ios_base::app);
    vector<vector<float>> vectors_features;
    vector<float> features_values;
    for(int n=0;n<n_features;++n){
        file<<features->points[n].j1
           <<","<<features->points[n].j2
          <<","<<features->points[n].j3
         <<","<<output_params[1]<<"\n";

        vector<float> moments3 {features->points[n].j1,features->points[n].j2,features->points[n].j3};
        vectors_features.push_back(moments3);
    }
    file.close();

    return vectors_features;

}


vector<vector<float>> estimateFPFH(PointCloudXYZPtr cloud,
                     PointCloudXYZPtr keypoints,
                     vector<string> params,
                     vector<string> output_params,
                     string region="nose_tip"
                    ){


    double normal_radius_search_ = stod(params[1]);
    double feat_radius_search_ = stod(params[2]);

    // Object for storing the normals.
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // Object for storing the PFH descriptors for each point.
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr descriptors(new pcl::PointCloud<pcl::FPFHSignature33>());


    // Note: you would usually perform downsampling now. It has been omitted here
    // for simplicity, but be aware that computation can take a long time.


    // Estimate the normals.
    NormalEstimationOMP<PointXYZ, Normal> normal_estimation_omp;
    normal_estimation_omp.setInputCloud(cloud);
    normal_estimation_omp.setRadiusSearch(normal_radius_search_);
    search::KdTree<PointXYZ>::Ptr kdtree_omp(new search::KdTree<PointXYZ>);
    normal_estimation_omp.setSearchMethod(kdtree_omp);
    normal_estimation_omp.compute(*normals);


    // PFH estimation object.
    pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
    fpfh.setInputCloud(cloud);
    fpfh.setInputNormals(normals);
    fpfh.setSearchMethod(kdtree_omp);
    // Search radius, to look for neighbors. Note: the value given here has to be
    // larger than the radius used to estimate the normals.
    fpfh.setRadiusSearch(feat_radius_search_);


    // get indices
    pcl::PointIndicesPtr indices (new pcl::PointIndices);
    getIndicesKeypoints(cloud,keypoints,indices,region);


    fpfh.setIndices(indices);
    fpfh.compute(*descriptors);
    cout<<"compute ok"<<endl;


    return histogramsVectorToFile<FPFHSignature33>(output_params,descriptors);
}


vector<vector<float>> estimatePFH(PointCloudXYZPtr cloud,
                     PointCloudXYZPtr keypoints,
                     vector<string> params,
                     vector<string> output_params){


    double normal_radius_search_ = stod(params[1]);
    double feat_radius_search_ = stod(params[2]);

    // Object for storing the normals.
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    // Object for storing the PFH descriptors for each point.
    pcl::PointCloud<pcl::PFHSignature125>::Ptr descriptors(new pcl::PointCloud<pcl::PFHSignature125>());


    // Note: you would usually perform downsampling now. It has been omitted here
    // for simplicity, but be aware that computation can take a long time.


    // Estimate the normals.
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normalEstimation;
    normalEstimation.setInputCloud(cloud);
    normalEstimation.setRadiusSearch(normal_radius_search_);
    pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree(new pcl::search::KdTree<pcl::PointXYZ>);
    normalEstimation.setSearchMethod(kdtree);
    normalEstimation.compute(*normals);

    // PFH estimation object.
    pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
    pfh.setInputCloud(cloud);
    pfh.setInputNormals(normals);
    pfh.setSearchMethod(kdtree);
    // Search radius, to look for neighbors. Note: the value given here has to be
    // larger than the radius used to estimate the normals.
    pfh.setRadiusSearch(feat_radius_search_);

    // get indices
    pcl::PointIndicesPtr indices (new pcl::PointIndices);
    getIndicesKeypoints(cloud,keypoints,indices);
    pfh.setIndices(indices);

    pfh.compute(*descriptors);

    return histogramsVectorToFile<PFHSignature125>(output_params,descriptors);
}


vector<vector<float>> estimate3DSC(PointCloudXYZPtr cloud,
                     PointCloudXYZPtr keypoints,
                     vector<string> params,
                     vector<string> output_params){

    // Set properties
    double normal_radius_search_ = stod(params[1]);
    double feat_radius_search_ = stod(params[2]);
    double minimalRadius_ = feat_radius_search_ / 10.0;
    double pointDensityRadius_ = feat_radius_search_ / 5.0;
    if (params.size()>3)
        double minimalRadius_ = stod(params[3]);
    if (params.size()>4)
        double pointDensityRadius_ = stod(params[4]);

    // Intrinsec parameter Feature
    ShapeContext3DEstimation<PointXYZ, Normal, ShapeContext1980>::Ptr feature_extractor_orig(
    new ShapeContext3DEstimation<PointXYZ, Normal, ShapeContext1980>);
    feature_extractor_orig->setMinimalRadius(minimalRadius_);
    feature_extractor_orig->setPointDensityRadius(pointDensityRadius_);

    // get indices
    pcl::PointIndicesPtr indices (new pcl::PointIndices);
    getIndicesKeypoints(cloud,keypoints,indices);
    feature_extractor_orig->setIndices(indices);

    // generic class features
    Feature<PointXYZ, ShapeContext1980>::Ptr feature_extractor(feature_extractor_orig);
    Features<ShapeContext1980> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
    PointCloud<ShapeContext1980>::Ptr features(new PointCloud<ShapeContext1980>);
    feat.compute(cloud, keypoints, features);

    return descriptorsVectorToFile<ShapeContext1980>(output_params,features);
}


vector<vector<float>> estimateU3DSC(PointCloudXYZPtr cloud,
                     PointCloudXYZPtr keypoints,
                     vector<string> params,
                     vector<string> output_params){


    // Set properties
    //double normal_radius_search_ = stod(params[1]);
    double feat_radius_search_ = stod(params[1]);
    double minimalRadius_ = feat_radius_search_ / 10.0;
    double pointDensityRadius_ = feat_radius_search_ / 5.0;
    double localRadius_ = feat_radius_search_ / 5.0;

    if (params.size()>2)
        double minimalRadius_ = stod(params[2]);
    if (params.size()>3)
        double pointDensityRadius_ = stod(params[3]);
    if (params.size()>4)
        double localRadius_ = stod(params[4]);



    // USC estimation object.
    UniqueShapeContext<PointXYZ, UniqueShapeContext1960, ReferenceFrame> usc;
    usc.setInputCloud(cloud);


    // get indices
    pcl::PointIndicesPtr indices (new pcl::PointIndices);
    getIndicesKeypoints(cloud,keypoints,indices);
    usc.setIndices(indices);

    // Search radius, to look for neighbors. It will also be the radius of the support sphere.
    usc.setRadiusSearch(feat_radius_search_);
    // The minimal radius value for the search sphere, to avoid being too sensitive
    // in bins close to the center of the sphere.
    usc.setMinimalRadius(minimalRadius_);
    // Radius used to compute the local point density for the neighbors
    // (the density is the number of points within that radius).
    usc.setPointDensityRadius(pointDensityRadius_);
    // Set the radius to compute the Local Reference Frame.
    usc.setLocalRadius(localRadius_);

    // Object for storing the USC descriptors for each point.
    PointCloud<UniqueShapeContext1960>::Ptr descriptors(new PointCloud<UniqueShapeContext1960>());
    usc.compute(*descriptors);

    return descriptorsVectorToFile<UniqueShapeContext1960>(output_params,descriptors);
}


vector<vector<float>> estimateRops(PointCloudXYZPtr cloud,
                     PointCloudXYZPtr keypoints,
                     vector<string> params,
                     vector<string> output_params){


    // Set properties
    double normalRadiusSearch = params.size()>1 ? stod(params[1]): 15 ;
    double meshSearchRadius = params.size()>2 ? stod(params[2]): 15 ;
    double mu = params.size()>3 ? stod(params[3]): 2.5 ;
    int maximumNearestNeighbors = params.size()>4 ? stod(params[4]): 300;
    double featRadiusSearch = params.size()>5 ? stod(params[5]): 20 ;
    int numberOfPartitionBins = params.size()>6 ? stod(params[6]): 15 ;
    int numberOfRotations = params.size()>7 ? stod(params[7]): 3 ;
    double supportRadius = params.size()>8 ? stod(params[8]): 15 ;

    search::KdTree<PointXYZ>::Ptr kdtree(new search::KdTree<PointXYZ>);
    PolygonMesh triangles;
    reconstructionGP3(cloud,kdtree,triangles,normalRadiusSearch,meshSearchRadius,mu,maximumNearestNeighbors);

    // get indices
    pcl::PointIndicesPtr indices (new pcl::PointIndices);
    getIndicesKeypoints(cloud,keypoints,indices);


    // RoPs estimation object.
    ROPSEstimation<pcl::PointXYZ, Histogram<135>> rops;
    rops.setInputCloud(cloud);
    rops.setIndices(indices);
    rops.setSearchMethod(kdtree);
    rops.setRadiusSearch(featRadiusSearch);
    rops.setTriangles(triangles.polygons);
    // Number of partition bins that is used for distribution matrix calculation.
    rops.setNumberOfPartitionBins(numberOfPartitionBins);
    // The greater the number of rotations is, the bigger the resulting descriptor.
    // Make sure to change the histogram size accordingly.
    rops.setNumberOfRotations(numberOfRotations);
    // Support radius that is used to crop the local surface of the point.
    rops.setSupportRadius(supportRadius);

    PointCloud<Histogram<135>>::Ptr descriptors(new pcl::PointCloud<Histogram<135>>());
    rops.compute(*descriptors);


    return histogramsVectorToFile<Histogram<135>>(output_params,descriptors);
}




vector<vector<float>>  extractFeatures(PointCloudXYZPtr cloud,
                      PointCloudXYZPtr keypoints,
                      vector<string> params,
                      vector<string> output_params,
                      string region="nose_tip"

                      ){


    // Extract the features
    if (params[0] == DESC_SHAPE_CONTEXT)
        return estimate3DSC(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_USC)
        return estimateU3DSC(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_ROPS)
        return estimateRops(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_PFH)
        return estimatePFH(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_MOMENT_INV)
        return estimateMomentsInvariants(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_SHOT)
        return estimateSHOT(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_VFH)
        return estimateVFH(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_SPIN_IMAGE)
        return estimateSI(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_GRSD)
        return estimateGRSD(cloud,keypoints,params,output_params);
    else if (params[0] ==  DESC_FPFH)
        return estimateFPFH(cloud,keypoints,params,output_params,region);
    else {
        cout<<"This method is not implemented!"<<endl;
        exit(1);
    }

    // Set properties
   //    else if (desc_type == DESC_USC) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      UniqueShapeContext<PointXYZRGB, ShapeContext1980>::Ptr feature_extractor_orig(
//        new UniqueShapeContext<PointXYZRGB, ShapeContext1980>);

//      // Set properties
//      feature_extractor_orig->setMinimalRadius(feat_radius_search_ / 10.0);
//      feature_extractor_orig->setPointDensityRadius(feat_radius_search_ / 5.0);

//      Feature<PointXYZRGB, ShapeContext1980>::Ptr feature_extractor(feature_extractor_orig);
//      Features<ShapeContext1980> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      PointCloud<ShapeContext1980>::Ptr source_features(new PointCloud<ShapeContext1980>);
//      PointCloud<ShapeContext1980>::Ptr target_features(new PointCloud<ShapeContext1980>);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_BOARD) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      BOARDLocalReferenceFrameEstimation<PointXYZRGB, Normal, ReferenceFrame>::Ptr feature_extractor_orig(
//        new BOARDLocalReferenceFrameEstimation<PointXYZRGB, Normal, ReferenceFrame>);

//      Feature<PointXYZRGB, ReferenceFrame>::Ptr feature_extractor(feature_extractor_orig);
//      PointCloud<ReferenceFrame>::Ptr source_features(new PointCloud<ReferenceFrame>);
//      PointCloud<ReferenceFrame>::Ptr target_features(new PointCloud<ReferenceFrame>);
//      Features<ReferenceFrame> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_BOUNDARY) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      BoundaryEstimation<PointXYZRGB, Normal, Boundary>::Ptr feature_extractor_orig(
//        new BoundaryEstimation<PointXYZRGB, Normal, Boundary>);

//      Feature<PointXYZRGB, Boundary>::Ptr feature_extractor(feature_extractor_orig);
//      PointCloud<Boundary>::Ptr source_features(new PointCloud<Boundary>);
//      PointCloud<Boundary>::Ptr target_features(new PointCloud<Boundary>);
//      Features<Boundary> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_INT_GRAD) {////////////////////////////////////////////////////////////////////////////////////////////
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      PointCloud<PointXYZI>::Ptr source_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr target_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr source_keypoints_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr target_keypoints_intensities(new PointCloud<PointXYZI>);
//      PointCloud<IntensityGradient>::Ptr source_features(new PointCloud<IntensityGradient>);
//      PointCloud<IntensityGradient>::Ptr target_features(new PointCloud<IntensityGradient>);
//      PointCloudXYZRGBtoXYZI(*source_cloud_, *source_intensities);
//      PointCloudXYZRGBtoXYZI(*target_cloud_, *target_intensities);
//      PointCloudXYZRGBtoXYZI(*source_keypoints, *source_keypoints_intensities);
//      PointCloudXYZRGBtoXYZI(*target_keypoints, *target_keypoints_intensities);
//      IntensityGradientEstimation<PointXYZI,
//                                  Normal,
//                                  IntensityGradient,
//                                  common::IntensityFieldAccessor<PointXYZI> > feature_extractor;
//      typename PointCloud<Normal>::Ptr source_normals (new PointCloud<Normal>);
//      typename PointCloud<Normal>::Ptr target_normals (new PointCloud<Normal>);
//      Tools::estimateNormals(source_cloud_, source_normals, normal_radius_search_);
//      Tools::estimateNormals(target_cloud_, target_normals, normal_radius_search_);

//      // Source
//      feature_extractor.setInputNormals(source_normals);
//      feature_extractor.setSearchSurface(source_intensities);
//      feature_extractor.setInputCloud(source_keypoints_intensities);
//      search::KdTree<PointXYZI>::Ptr kdtree(new search::KdTree<PointXYZI>);
//      feature_extractor.setSearchMethod(kdtree);
//      feature_extractor.setRadiusSearch(feat_radius_search_);
//      feature_extractor.compute(*source_features);

//      // Target
//      feature_extractor.setInputNormals(target_normals);
//      feature_extractor.setSearchSurface(target_intensities);
//      feature_extractor.setInputCloud(target_keypoints_intensities);
//      feature_extractor.setSearchMethod(kdtree);
//      feature_extractor.setRadiusSearch(feat_radius_search_);
//      feature_extractor.compute(*target_features);

//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      Features<IntensityGradient> feat;
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_INT_SPIN) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();

//      PointCloud<PointXYZI>::Ptr source_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr target_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr source_keypoints_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr target_keypoints_intensities(new PointCloud<PointXYZI>);
//      PointCloud<Histogram<20> >::Ptr source_features(new PointCloud<Histogram<20> >);
//      PointCloud<Histogram<20> >::Ptr target_features(new PointCloud<Histogram<20> >);
//      PointCloudXYZRGBtoXYZI(*source_cloud_, *source_intensities);
//      PointCloudXYZRGBtoXYZI(*target_cloud_, *target_intensities);
//      PointCloudXYZRGBtoXYZI(*source_keypoints, *source_keypoints_intensities);
//      PointCloudXYZRGBtoXYZI(*target_keypoints, *target_keypoints_intensities);

//      IntensitySpinEstimation<PointXYZI,
//                              Histogram<20> > feature_extractor;
//      // typename PointCloud<Normal>::Ptr source_normals (new PointCloud<Normal>);
//      // typename PointCloud<Normal>::Ptr target_normals (new PointCloud<Normal>);
//      // Tools::estimateNormals(source_cloud_, source_normals, normal_radius_search_);
//      // Tools::estimateNormals(target_cloud_, target_normals, normal_radius_search_);

//      // Source
//      //feature_extractor.setInputNormals(source_normals);
//      feature_extractor.setSearchSurface(source_intensities);
//      feature_extractor.setInputCloud(source_keypoints_intensities);
//      search::KdTree<PointXYZI>::Ptr kdtree(new search::KdTree<PointXYZI>);
//      feature_extractor.setSearchMethod(kdtree);
//      feature_extractor.setRadiusSearch(feat_radius_search_);
//      feature_extractor.compute(*source_features);

//      // Target
//      //feature_extractor.setInputNormals(target_normals);
//      feature_extractor.setSearchSurface(target_intensities);
//      feature_extractor.setInputCloud(target_keypoints_intensities);
//      feature_extractor.setSearchMethod(kdtree);
//      feature_extractor.setRadiusSearch(feat_radius_search_);
//      feature_extractor.compute(*target_features);

//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      Features<Histogram<20> > feat;
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    } else if (desc_type == DESC_SPIN_IMAGE) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      PointCloud<Histogram<153> >::Ptr source_features(new PointCloud<Histogram<153> >);
//      PointCloud<Histogram<153> >::Ptr target_features(new PointCloud<Histogram<153> >);
//      typename PointCloud<Normal>::Ptr source_normals (new PointCloud<Normal>);
//      typename PointCloud<Normal>::Ptr target_normals (new PointCloud<Normal>);
//      Tools::estimateNormals(source_keypoints, source_normals, normal_radius_search_);
//      Tools::estimateNormals(target_keypoints, target_normals, normal_radius_search_);

//      SpinImageEstimation<PointXYZRGB, Normal, Histogram<153> > feature_extractor;

//      // Source
//      feature_extractor.setInputNormals(source_normals);
//      feature_extractor.setSearchSurface(source_cloud_);
//      feature_extractor.setInputCloud(source_keypoints);
//      search::KdTree<PointRGB>::Ptr kdtree(new search::KdTree<PointRGB>);
//      feature_extractor.setSearchMethod(kdtree);
//      feature_extractor.setRadiusSearch(feat_radius_search_);
//      feature_extractor.compute(*source_features);

//      // Target
//      feature_extractor.setInputNormals(target_normals);
//      feature_extractor.setSearchSurface(target_cloud_);
//      feature_extractor.setInputCloud(target_keypoints);
//      feature_extractor.setSearchMethod(kdtree);
//      feature_extractor.setRadiusSearch(feat_radius_search_);
//      feature_extractor.compute(*target_features);

//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      Features<Histogram<153> > feat;
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    } else if (desc_type == DESC_MOMENT_INV) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      Feature<PointXYZRGB, MomentInvariants>::Ptr feature_extractor(new MomentInvariantsEstimation<PointXYZRGB, MomentInvariants>);
//      PointCloud<MomentInvariants>::Ptr source_features(new PointCloud<MomentInvariants>);
//      PointCloud<MomentInvariants>::Ptr target_features(new PointCloud<MomentInvariants>);
//      Features<MomentInvariants> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;


//    } else if (desc_type == DESC_CRH) {
//      // // Compute features
//      // ros::WallTime desc_start = ros::WallTime::now();
//      // Feature<PointXYZRGB, Histogram<90> >::Ptr feature_extractor(new CRHEstimation<PointXYZRGB, Normal, Histogram<90> >);
//      // PointCloud<Histogram<90> >::Ptr source_features(new PointCloud<Histogram<90> >);
//      // PointCloud<Histogram<90> >::Ptr target_features(new PointCloud<Histogram<90> >);
//      // Features<Histogram<90> > feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      // feat.compute(source_cloud_, source_keypoints, source_features);
//      // feat.compute(target_cloud_, target_keypoints, target_features);
//      // desc_runtime = ros::WallTime::now() - desc_start;
//      // source_feat_size = source_features->points.size();
//      // target_feat_size = target_features->points.size();

//      // // Find correspondences
//      // ros::WallTime corr_start = ros::WallTime::now();
//      // feat.findCorrespondences(source_features, target_features, correspondences);
//      // feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      // corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_FPFH)
//    {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      Feature<PointXYZRGB, FPFHSignature33>::Ptr feature_extractor(new FPFHEstimation<PointXYZRGB, Normal, FPFHSignature33>);
//      PointCloud<FPFHSignature33>::Ptr source_features(new PointCloud<FPFHSignature33>);
//      PointCloud<FPFHSignature33>::Ptr target_features(new PointCloud<FPFHSignature33>);
//      Features<FPFHSignature33> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_NARF) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      PointCloud<Narf36>::Ptr source_features(new PointCloud<Narf36>);
//      PointCloud<Narf36>::Ptr target_features(new PointCloud<Narf36>);
//      RangeImagePlanar source_range_image, target_range_image;
//      Tools::convertToRangeImage(source_cloud_, source_range_image);
//      Tools::convertToRangeImage(target_cloud_, target_range_image);

//      // Get the cloud indices
//      vector<int> source_keypoint_indices, target_keypoint_indices;
//      Tools::getIndices(source_cloud_, source_keypoints,
//                        source_keypoint_indices);
//      Tools::getIndices(target_cloud_, target_keypoints,
//                        target_keypoint_indices);

//      NarfDescriptor source_feat(&source_range_image, &source_keypoint_indices);
//      NarfDescriptor target_feat(&source_range_image, &target_keypoint_indices);

//      source_feat.getParameters().support_size = 0.2f;
//      source_feat.getParameters().rotation_invariant = true;
//      source_feat.compute(*source_features);
//      target_feat.getParameters().support_size = 0.2f;
//      target_feat.getParameters().rotation_invariant = true;
//      target_feat.compute(*target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      Features<Narf36> feat;
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_CVFH)
//    {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      CVFHEstimation<PointXYZRGB, Normal, VFHSignature308>::Ptr feature_extractor_orig(
//        new CVFHEstimation<PointXYZRGB, Normal, VFHSignature308>);

//      // Set properties
//      feature_extractor_orig->setRadiusSearch(feat_radius_search_);
//      feature_extractor_orig->setKSearch(0);

//      Feature<PointXYZRGB, VFHSignature308>::Ptr feature_extractor(feature_extractor_orig);
//      PointCloud<VFHSignature308>::Ptr source_features(new PointCloud<VFHSignature308>);
//      PointCloud<VFHSignature308>::Ptr target_features(new PointCloud<VFHSignature308>);
//      Features<VFHSignature308> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_PFH)
//    {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      Feature<PointXYZRGB, PFHSignature125>::Ptr feature_extractor(new PFHEstimation<PointXYZRGB, Normal, PFHSignature125>);
//      PointCloud<PFHSignature125>::Ptr source_features(new PointCloud<PFHSignature125>);
//      PointCloud<PFHSignature125>::Ptr target_features(new PointCloud<PFHSignature125>);
//      Features<PFHSignature125> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_PPAL_CURV)
//    {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      Feature<PointXYZRGB, PrincipalCurvatures>::Ptr feature_extractor(new PrincipalCurvaturesEstimation<PointXYZRGB, Normal, PrincipalCurvatures>);
//      PointCloud<PrincipalCurvatures>::Ptr source_features(new PointCloud<PrincipalCurvatures>);
//      PointCloud<PrincipalCurvatures>::Ptr target_features(new PointCloud<PrincipalCurvatures>);
//      Features<PrincipalCurvatures> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_RIFT) {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();

//      PointCloud<pcl::Histogram<32> >::Ptr source_features(new PointCloud<pcl::Histogram<32> >);
//      PointCloud<pcl::Histogram<32> >::Ptr target_features(new PointCloud<pcl::Histogram<32> >);
//      PointCloud<PointXYZI>::Ptr source_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr source_keypoint_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr target_intensities(new PointCloud<PointXYZI>);
//      PointCloud<PointXYZI>::Ptr target_keypoint_intensities(new PointCloud<PointXYZI>);
//      PointCloud<IntensityGradient>::Ptr source_gradients(new PointCloud<IntensityGradient>);
//      PointCloud<IntensityGradient>::Ptr target_gradients(new PointCloud<IntensityGradient>);
//      PointCloud<Normal>::Ptr source_normals(new PointCloud<Normal>);
//      PointCloud<Normal>::Ptr target_normals(new PointCloud<Normal>);
//      Tools::estimateNormals(source_cloud_, source_normals, normal_radius_search_);
//      Tools::estimateNormals(target_cloud_, target_normals, normal_radius_search_);
//      PointCloudXYZRGBtoXYZI(*source_keypoints, *source_keypoint_intensities);
//      PointCloudXYZRGBtoXYZI(*target_keypoints, *target_keypoint_intensities);
//      PointCloudXYZRGBtoXYZI(*source_cloud_, *source_intensities);
//      PointCloudXYZRGBtoXYZI(*target_cloud_, *target_intensities);
//      Tools::computeGradient(source_intensities, source_normals, source_gradients, normal_radius_search_);
//      Tools::computeGradient(target_intensities, target_normals, target_gradients, normal_radius_search_);

//      // search::KdTree<PointCloudRGB>::Ptr kdtree(new search::KdTree<PointCloudRGB>);
//      RIFTEstimation<PointXYZI, IntensityGradient, pcl::Histogram<32> > rift;
//      rift.setRadiusSearch(feat_radius_search_);
//      rift.setNrDistanceBins(4);
//      rift.setNrGradientBins(8);
//      // rift.setSearchMethod(kdtree);

//      rift.setInputCloud(source_keypoint_intensities);
//      rift.setSearchSurface(source_intensities);
//      rift.setInputGradient(source_gradients);
//      rift.compute(*source_features);

//      rift.setInputCloud(target_keypoint_intensities);
//      rift.setSearchSurface(target_intensities);
//      rift.setInputGradient(target_gradients);
//      rift.compute(*target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      Features<pcl::Histogram<32> > feat;
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_SHOT)
//    {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      Feature<PointXYZRGB, SHOT352>::Ptr feature_extractor(new SHOTEstimationOMP<PointXYZRGB, Normal, SHOT352>);
//      PointCloud<SHOT352>::Ptr source_features(new PointCloud<SHOT352>);
//      PointCloud<SHOT352>::Ptr target_features(new PointCloud<SHOT352>);
//      Features<SHOT352> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_SHOT_COLOR)
//    {
//      // Compute features
//      ros::WallTime desc_start = ros::WallTime::now();
//      Feature<PointXYZRGB, SHOT1344>::Ptr feature_extractor(new SHOTColorEstimationOMP<PointXYZRGB, Normal, SHOT1344>);
//      PointCloud<SHOT1344>::Ptr source_features(new PointCloud<SHOT1344>);
//      PointCloud<SHOT1344>::Ptr target_features(new PointCloud<SHOT1344>);
//      Features<SHOT1344> feat(feature_extractor, feat_radius_search_, normal_radius_search_);
//      feat.compute(source_cloud_, source_keypoints, source_features);
//      feat.compute(target_cloud_, target_keypoints, target_features);
//      desc_runtime = ros::WallTime::now() - desc_start;
//      source_feat_size = source_features->points.size();
//      target_feat_size = target_features->points.size();

//      // Find correspondences
//      ros::WallTime corr_start = ros::WallTime::now();
//      feat.findCorrespondences(source_features, target_features, correspondences);
//      feat.filterCorrespondences(source_keypoints, target_keypoints, correspondences, filtered_correspondences, ransac_tf);
//      corr_runtime = ros::WallTime::now() - corr_start;
//    }
//    else if (desc_type == DESC_SHOT_LRF) {}

//    cout<<params[0]<<" ok; "
//        <<" features: "<<features_return->points.size()
//        <<" values: "<<features_return->front().descriptorSize()
//        <<" time: "<<corr_runtime<<" s"<<endl;

//    return features_return;
}
